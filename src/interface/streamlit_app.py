"""CrisisBridge AI - Streamlit Commander Console.

Run from the project folder:
    streamlit run src/interface/streamlit_app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src.config import GEMINI_API_KEY, GROQ_API_KEY, INDEX_DIR, LLM_PROVIDER  # noqa: E402
from src.ingestion.live_sources import SOURCES  # noqa: E402
from src.pipeline import CommanderService, build_system  # noqa: E402
from src.rag.indexer import CrisisIndex  # noqa: E402

st.set_page_config(page_title="CrisisBridge AI", page_icon="🛟", layout="wide")

PRIORITY_COLORS = {"Critical": "#d32f2f", "High": "#f57c00", "Medium": "#fbc02d", "Low": "#689f38"}


@st.cache_resource(show_spinner="Loading index and models...")
def get_service() -> CommanderService:
    return CommanderService.load()


def reports_frame(reports) -> pd.DataFrame:
    return pd.DataFrame([{
        "priority": r.priority,
        "urgency": r.urgency_score,
        "location": r.primary_location,
        "need": r.primary_need,
        "people": r.people_count,
        "category": r.category.replace("_", " "),
        "time": r.timestamp,
        "source": r.source,
        "text": r.text,
        "id": r.id,
    } for r in reports])


def priority_badge(priority: str) -> str:
    color = PRIORITY_COLORS.get(priority, "#757575")
    return f"<span style='background:{color};color:white;padding:2px 8px;border-radius:6px;font-size:0.85em'>{priority or 'n/a'}</span>"


# ---------------------------------------------------------------- sidebar
st.sidebar.title("CrisisBridge AI")
st.sidebar.caption("RAG-augmented disaster response prioritisation")

index_ready = CrisisIndex.exists(INDEX_DIR)
with st.sidebar.expander("Build / rebuild system", expanded=not index_ready):
    data_mode = st.selectbox("Training data", ["sample", "auto", "humaid"],
                             help="sample = offline bundled messages; auto = HumAID if downloadable")
    train_limit = st.number_input("HumAID training rows (0 = all)", 0, 60000, 20000, step=5000)
    index_limit = st.number_input("HumAID tweets to index (0 = all)", 0, 20000, 2000, step=500)
    if st.button("Build index", type="primary"):
        with st.spinner("Training classifier and building index..."):
            try:
                _, _, summary = build_system(data_mode, train_limit or None, index_limit or None)
                get_service.clear()
                for m in summary.messages:
                    st.warning(m)
                st.success(f"Built on {summary.data_used}: {summary.n_indexed} reports. {summary.train_result}")
                index_ready = True
            except Exception as exc:
                st.error(f"Build failed: {exc}")

if not index_ready:
    st.title("CrisisBridge AI")
    st.info("No index yet. Open 'Build / rebuild system' in the sidebar and click **Build index** "
            "(or run `python run_pipeline.py` in PowerShell).")
    st.stop()

service = get_service()
meta_path = INDEX_DIR / "meta.json"
meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
st.sidebar.metric("Indexed reports", len(service.index))
st.sidebar.caption(f"Embeddings: {meta.get('embedder', '?')} | store: {service.index.store_name}")

st.sidebar.subheader("Answer generator")
provider_options = ["auto", "groq", "gemini", "extractive"]
provider = st.sidebar.selectbox("Provider", provider_options,
                                index=provider_options.index(LLM_PROVIDER) if LLM_PROVIDER in provider_options else 0)
st.sidebar.caption(f"Groq key: {'set' if GROQ_API_KEY else 'missing'} | Gemini key: {'set' if GEMINI_API_KEY else 'missing'}")
k = st.sidebar.slider("Reports to retrieve", 3, 15, 5)
min_priority = st.sidebar.selectbox("Minimum priority", ["", "Low", "Medium", "High", "Critical"], index=0,
                                    format_func=lambda p: p or "any")

st.sidebar.subheader("Live feeds (optional)")
chosen = st.sidebar.multiselect("Sources", list(SOURCES), default=["gdacs", "rss"])
st.sidebar.caption("Known: GDACS may time out; ReliefWeb needs an approved appname; Reddit needs app credentials.")
if st.sidebar.button("Fetch live reports"):
    with st.spinner("Fetching live feeds..."):
        added, statuses = service.ingest_live(chosen or None)
    for s in statuses:
        (st.sidebar.success if s.ok else st.sidebar.warning)(str(s))
    st.sidebar.info(f"Added {added} new reports to the index.")

# ---------------------------------------------------------------- main
tab_query, tab_board, tab_process = st.tabs(["Commander Query", "Priority Board", "Process a Message"])

with tab_query:
    st.header("Commander Query")
    examples = [
        "What is the most critical situation near XYZ bridge right now?",
        "Where are people trapped and needing rescue?",
        "Which areas need drinking water or food?",
        "Any missing people reported?",
    ]
    example = st.selectbox("Example questions", [""] + examples, format_func=lambda x: x or "(type your own below)")
    query = st.text_input("Ask a question", value=example or examples[0])
    if st.button("Ask", type="primary") and query.strip():
        with st.spinner("Retrieving and generating..."):
            answer, results, plan = service.ask(query, k=k, provider=provider, min_priority=min_priority)
        st.caption(
            f"Parsed: locations={plan.locations or '-'} | needs={plan.needs or '-'} | "
            f"urgency intent={plan.wants_urgent} | recency intent={plan.wants_recent} | "
            f"location filter={'on' if plan.location_filter_applied else 'off'}"
        )
        st.subheader("Answer")
        st.write(answer.text)
        if answer.notice:
            st.info(answer.notice)
        st.caption(f"Generated by: {answer.provider}")
        st.subheader("Cited reports")
        for n, item in enumerate(results, 1):
            r = item.report
            with st.container(border=True):
                st.markdown(f"**[{n}]** {priority_badge(r.priority)} &nbsp; urgency **{r.urgency_score:.0f}**/100 "
                            f"&nbsp; `{r.id}` · {r.source} · {r.timestamp or 'time unknown'}", unsafe_allow_html=True)
                st.write(r.text)
                st.caption(r.summary_line() + (f" | {r.url}" if r.url else ""))
                with st.expander("Why this rank?"):
                    st.json({"retrieval_score": item.score, **item.components, "urgency_reasons": r.urgency_reasons})

with tab_board:
    st.header("Priority Board")
    df = reports_frame(service.index.reports)
    c1, c2, c3, c4 = st.columns(4)
    for col, p in zip((c1, c2, c3, c4), ("Critical", "High", "Medium", "Low")):
        col.metric(p, int((df["priority"] == p).sum()))
    pri_filter = st.multiselect("Priority", ["Critical", "High", "Medium", "Low"], default=["Critical", "High"])
    loc_filter = st.text_input("Location contains")
    view = df[df["priority"].isin(pri_filter)] if pri_filter else df
    if loc_filter:
        view = view[view["location"].str.contains(loc_filter, case=False, na=False) |
                    view["text"].str.contains(loc_filter, case=False, na=False)]
    st.dataframe(view.sort_values("urgency", ascending=False), hide_index=True)
    st.download_button("Download as CSV", view.to_csv(index=False).encode("utf-8"), "crisisbridge_reports.csv", "text/csv")

with tab_process:
    st.header("Process a new message")
    text = st.text_area("Paste a disaster-related post or report",
                        "4 people trapped on the second floor near XYZ bridge, need rescue")
    add_to_index = st.checkbox("Add to the index after processing", value=False)
    if st.button("Process"):
        if add_to_index:
            report = service.add_text(text)
        else:
            report = service.processor.process_text(text)
        st.markdown(f"### {priority_badge(report.priority)} &nbsp; urgency {report.urgency_score:.0f}/100", unsafe_allow_html=True)
        st.code(report.summary_line())
        st.json({
            "category": report.category,
            "category_confidence": report.category_confidence,
            "locations": report.locations,
            "people_count": report.people_count,
            "needs": report.needs,
            "urgency_reasons": report.urgency_reasons,
        })
        if add_to_index:
            st.success(f"Added as {report.id}. Index now holds {len(service.index)} reports.")
