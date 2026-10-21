"""CrisisBridge AI - web API and coordinator frontend.

Serves a JSON API over the shared pipeline (src/pipeline.py) and the static web console in
src/interface/web/. Run from the project folder:

    python -m src.interface.api            # then open http://localhost:8000
    python -m src.interface.api --port 9000
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import threading
import time
from collections import Counter
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.config import (
    CACHE_DIR,
    GEMINI_API_KEY,
    GEMINI_MODEL,
    GROQ_API_KEY,
    GROQ_MODEL,
    INDEX_DIR,
    LLM_PROVIDER,
    REDDIT_CLIENT_ID,
    REDDIT_CLIENT_SECRET,
    RELIEFWEB_APPNAME,
    RSS_FEEDS,
    SAMPLE_DATA_PATH,
    get_logger,
)
from src.ingestion.live_sources import SOURCES
from src.pipeline import DATA_MODES, CommanderService, build_system
from src.rag.indexer import CrisisIndex
from src.rag.retriever import RetrievedReport
from src.schema import Report

log = get_logger("interface.api")

WEB_DIR = Path(__file__).resolve().parent / "web"
PRIORITIES = ("Critical", "High", "Medium", "Low")
PROVIDERS = ("auto", "groq", "gemini", "extractive")


# ---------------------------------------------------------------- shared state
class State:
    """Loaded service plus the background build job. All index mutations hold `lock`."""

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.service: CommanderService | None = None
        self.load_error = ""
        self.build = {"state": "idle", "started": None, "finished": None, "message": "", "summary": None}

    def load(self) -> None:
        with self.lock:
            if not CrisisIndex.exists(INDEX_DIR):
                self.service, self.load_error = None, "No index yet. Build one from the System page."
                return
            try:
                self.service = CommanderService.load()
                self.load_error = ""
            except Exception as exc:  # corrupt or stale files: report instead of crashing the server
                self.service, self.load_error = None, f"Could not load the index: {exc}"
                log.warning(self.load_error)

    def require(self) -> CommanderService:
        if self.service is None:
            raise HTTPException(status_code=409, detail=self.load_error or "Index not loaded.")
        return self.service


STATE = State()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    STATE.load()
    yield


app = FastAPI(title="CrisisBridge AI", version="1.0", lifespan=lifespan)


# ---------------------------------------------------------------- serialisation
def report_json(r: Report) -> dict:
    data = r.to_dict()
    data["summary_line"] = r.summary_line()
    return data


def retrieved_json(rank: int, item: RetrievedReport) -> dict:
    return {"rank": rank, "score": item.score, "components": item.components, "report": report_json(item.report)}


def _index_meta() -> dict:
    path = INDEX_DIR / "meta.json"
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except (OSError, ValueError):
        return {}


def _sort_key(r: Report) -> tuple:
    return (r.urgency_score, r.timestamp or "")


# ---------------------------------------------------------------- status / overview
@app.get("/api/status")
def status() -> dict:
    svc = STATE.service
    meta = _index_meta()
    return {
        "index_ready": svc is not None,
        "message": STATE.load_error,
        "index": {
            "count": len(svc.index) if svc else meta.get("count", 0),
            "embedder": meta.get("embedder", ""),
            "dim": meta.get("dim"),
            "built_at": meta.get("built_at", ""),
            "store": svc.index.store_name if svc else "",
        },
        "llm": {
            "default": LLM_PROVIDER,
            "groq": {"key": bool(GROQ_API_KEY), "model": GROQ_MODEL},
            "gemini": {"key": bool(GEMINI_API_KEY), "model": GEMINI_MODEL},
        },
        "build": STATE.build,
        "data_modes": list(DATA_MODES),
    }


@app.get("/api/overview")
def overview() -> dict:
    svc = STATE.require()
    reports = svc.index.reports
    priorities = Counter(r.priority for r in reports)
    categories = Counter(r.category for r in reports if r.category)
    needs = Counter(n for r in reports for n in r.needs)
    locations = Counter(r.primary_location for r in reports if r.primary_location)
    sources = Counter(r.source.split("/")[0] for r in reports)
    top = sorted((r for r in reports if r.priority in ("Critical", "High")), key=_sort_key, reverse=True)[:6]
    return {
        "total": len(reports),
        "priorities": {p: priorities.get(p, 0) for p in PRIORITIES},
        "categories": categories.most_common(),
        "needs": needs.most_common(8),
        "locations": locations.most_common(8),
        "sources": sources.most_common(),
        "with_location": sum(1 for r in reports if r.primary_location),
        "with_need": sum(1 for r in reports if r.needs),
        "with_people": sum(1 for r in reports if r.people_count is not None),
        "top": [report_json(r) for r in top],
    }


# ---------------------------------------------------------------- commander query
class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    k: int = Field(5, ge=1, le=15)
    min_priority: str = ""
    provider: str = "auto"


@app.post("/api/ask")
def ask(req: AskRequest) -> dict:
    svc = STATE.require()
    if req.provider not in PROVIDERS:
        raise HTTPException(400, f"provider must be one of {PROVIDERS}")
    if req.min_priority and req.min_priority not in PRIORITIES:
        raise HTTPException(400, f"min_priority must be empty or one of {PRIORITIES}")
    started = time.perf_counter()
    with STATE.lock:
        answer, results, plan = svc.ask(req.query.strip(), k=req.k, provider=req.provider, min_priority=req.min_priority)
    return {
        "answer": {"text": answer.text, "provider": answer.provider, "notice": answer.notice},
        "plan": {
            "locations": plan.locations,
            "needs": plan.needs,
            "wants_urgent": plan.wants_urgent,
            "wants_recent": plan.wants_recent,
            "location_filter_applied": plan.location_filter_applied,
        },
        "results": [retrieved_json(n, item) for n, item in enumerate(results, 1)],
        "elapsed_ms": round((time.perf_counter() - started) * 1000),
    }


# ---------------------------------------------------------------- priority feed
def _filtered(priority: str, q: str, source: str, need: str) -> list[Report]:
    svc = STATE.require()
    wanted = {p for p in priority.split(",") if p} if priority else set()
    ql = q.strip().lower()
    out = []
    for r in svc.index.reports:
        if wanted and r.priority not in wanted:
            continue
        if source and not r.source.startswith(source):
            continue
        if need and need not in r.needs:
            continue
        if ql and not (ql in r.text.lower() or any(ql in loc.lower() for loc in r.locations)):
            continue
        out.append(r)
    out.sort(key=_sort_key, reverse=True)
    return out


@app.get("/api/reports")
def reports(
    priority: str = "",
    q: str = "",
    source: str = "",
    need: str = "",
    offset: int = Query(0, ge=0),
    limit: int = Query(25, ge=1, le=200),
) -> dict:
    rows = _filtered(priority, q, source, need)
    return {"total": len(rows), "offset": offset, "items": [report_json(r) for r in rows[offset:offset + limit]]}


@app.get("/api/reports.csv")
def reports_csv(priority: str = "", q: str = "", source: str = "", need: str = "") -> StreamingResponse:
    rows = _filtered(priority, q, source, need)
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["priority", "urgency", "location", "need", "people", "category", "time", "source", "text", "id"])
    for r in rows:
        writer.writerow([r.priority, r.urgency_score, r.primary_location, r.primary_need, r.people_count,
                         r.category.replace("_", " "), r.timestamp, r.source, r.text, r.id])
    return StreamingResponse(iter([buf.getvalue()]), media_type="text/csv",
                             headers={"Content-Disposition": "attachment; filename=crisisbridge_reports.csv"})


# ---------------------------------------------------------------- process a message
class ProcessRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)
    add_to_index: bool = False


@app.post("/api/process")
def process(req: ProcessRequest) -> dict:
    svc = STATE.require()
    with STATE.lock:
        if req.add_to_index:
            report = svc.add_text(req.text)
        else:
            report = svc.processor.process_text(req.text)
        total = len(svc.index)
    return {"report": report_json(report), "added": req.add_to_index, "index_count": total}


# ---------------------------------------------------------------- data sources
def _live_config() -> list[dict]:
    reddit_ready = bool(REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET)
    return [
        {"key": "gdacs", "name": "GDACS", "ready": True,
         "desc": "Free real-time global disaster alerts, pre-scored by severity",
         "note": "No key needed. Known to time out from some networks."},
        {"key": "reddit", "name": "Reddit (via PRAW)", "ready": reddit_ready,
         "desc": "Crisis-relevant subreddits, a live, structured-enough text feed",
         "note": "Credentials set." if reddit_ready else "Needs REDDIT_CLIENT_ID / REDDIT_CLIENT_SECRET in .env and praw installed."},
        {"key": "reliefweb", "name": "ReliefWeb API (UN OCHA)", "ready": bool(RELIEFWEB_APPNAME),
         "desc": "Real situation reports, a secondary grounding source",
         "note": "Appname set." if RELIEFWEB_APPNAME else "Needs a pre-approved RELIEFWEB_APPNAME in .env."},
        {"key": "rss", "name": "RSS / Atom feeds", "ready": bool(RSS_FEEDS),
         "desc": "Any feeds in RSS_FEEDS (default: USGS M4.5+ earthquakes)",
         "note": f"{len(RSS_FEEDS)} feed(s) configured."},
        {"key": "ndma", "name": "NDMA / IMD RSS", "ready": False, "unavailable": True,
         "desc": "India-specific flood and cyclone warnings, regional relevance",
         "note": "Not implemented: neither publishes a stable public RSS feed or API."},
    ]


def _curated_status() -> list[dict]:
    humaid_cached = (CACHE_DIR / "humaid_train.parquet").exists()
    return [
        {"name": "HumAID", "desc": "77K tweets, 19 disasters, humanitarian labels, the primary training set",
         "state": "cached" if humaid_cached else "downloads on build",
         "ok": humaid_cached},
        {"name": "Bundled sample", "desc": "50 hand-written labelled messages for the offline demo",
         "state": "available" if SAMPLE_DATA_PATH.exists() else "missing", "ok": SAMPLE_DATA_PATH.exists()},
        {"name": "Kaggle “Disaster Tweets”", "desc": "10K hand-labelled tweets, binary relevance pre-filter",
         "state": "manual download", "ok": False},
        {"name": "CrisisNLP / CrisisBench", "desc": "CrisisLex, SWDM2013, CrisisMMD and AIDR data combined",
         "state": "planned", "ok": False},
        {"name": "IDRISI", "desc": "41 events with location-mention labels, for the NER / geo stage",
         "state": "planned", "ok": False},
    ]


@app.get("/api/sources")
def sources() -> dict:
    return {"live": _live_config(), "curated": _curated_status()}


class LiveRequest(BaseModel):
    sources: list[str] = Field(default_factory=lambda: ["gdacs", "rss"])


@app.post("/api/live")
def live(req: LiveRequest) -> dict:
    svc = STATE.require()
    chosen = [s for s in req.sources if s in SOURCES]
    if not chosen:
        raise HTTPException(400, f"choose at least one of {list(SOURCES)}")
    with STATE.lock:
        added, statuses = svc.ingest_live(chosen)
        total = len(svc.index)
    return {
        "added": added,
        "index_count": total,
        "statuses": [{"name": s.name, "ok": s.ok, "count": s.count, "message": s.message} for s in statuses],
    }


# ---------------------------------------------------------------- build
class BuildRequest(BaseModel):
    data: str = "sample"
    train_limit: int = Field(20000, ge=0, le=100000)
    index_limit: int = Field(2000, ge=0, le=50000)


def _run_build(req: BuildRequest) -> None:
    try:
        _, _, summary = build_system(req.data, req.train_limit or None, req.index_limit or None)
        STATE.load()
        tr = summary.train_result
        STATE.build.update(state="done", message=str(tr), summary={
            "data_used": summary.data_used, "n_indexed": summary.n_indexed, "embedder": summary.embedder,
            "store": summary.store, "notes": summary.messages,
            "eval_name": tr.eval_name, "accuracy": tr.accuracy, "macro_f1": tr.macro_f1,
        })
    except Exception as exc:
        log.warning("Build failed: %s", exc)
        STATE.build.update(state="error", message=f"Build failed: {exc}")
    finally:
        STATE.build["finished"] = datetime.now(timezone.utc).isoformat()


@app.post("/api/build")
def build(req: BuildRequest) -> dict:
    if req.data not in DATA_MODES:
        raise HTTPException(400, f"data must be one of {DATA_MODES}")
    with STATE.lock:
        if STATE.build["state"] == "running":
            raise HTTPException(409, "A build is already running.")
        STATE.build = {"state": "running", "started": datetime.now(timezone.utc).isoformat(), "finished": None,
                       "message": f"Training classifier on '{req.data}' data and building the index...", "summary": None}
    threading.Thread(target=_run_build, args=(req,), daemon=True).start()
    return STATE.build


@app.get("/api/build")
def build_status() -> dict:
    return STATE.build


# ---------------------------------------------------------------- frontend
@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


app.mount("/", StaticFiles(directory=WEB_DIR), name="web")


def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser(description="CrisisBridge AI web console")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    print(f"CrisisBridge AI console: http://{'localhost' if args.host == '127.0.0.1' else args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
