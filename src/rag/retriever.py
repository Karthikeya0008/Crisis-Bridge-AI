"""Commander Query retrieval: semantic search re-ranked by location match, urgency and recency.

final = w_sim * cosine + w_loc * location_match + w_urg * urgency/100 + w_rec * recency
Weights shift with query intent: "most critical" boosts urgency, "right now / latest" boosts recency.
If the query names a place and any reports match it, retrieval is restricted to those reports.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone

from src.nlp.ner import LANDMARK_SUFFIXES, extract_locations, extract_needs
from src.rag.indexer import CrisisIndex
from src.schema import Report

URGENCY_INTENT_RE = re.compile(r"\b(critical|urgent|severe|serious|worst|priority|life|dangerous|emergency)\b", re.I)
RECENCY_INTENT_RE = re.compile(r"\b(now|right now|latest|current(?:ly)?|recent|today|last \d+ hours?|ongoing)\b", re.I)
SUFFIX_ONLY_RE = re.compile(rf"^(?:{LANDMARK_SUFFIXES})$", re.I)
PRIORITY_ORDER = {"Critical": 3, "High": 2, "Medium": 1, "Low": 0, "": 0}


@dataclass
class RetrievedReport:
    report: Report
    score: float
    components: dict[str, float] = field(default_factory=dict)


@dataclass
class QueryPlan:
    query: str
    locations: list[str]
    needs: list[str]
    wants_urgent: bool
    wants_recent: bool
    location_filter_applied: bool = False


def _parse_time(ts: str) -> datetime | None:
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if not SUFFIX_ONLY_RE.match(t) and len(t) > 1}


def location_match(query_locs: list[str], report: Report) -> float:
    """1.0 exact/substring match, partial credit for shared distinctive tokens, else 0."""
    if not query_locs:
        return 0.0
    best = 0.0
    rep_locs = [l.lower() for l in report.locations]
    text = report.text.lower()
    for q in query_locs:
        ql = q.lower().strip()
        if not ql:
            continue
        if any(ql == rl or ql in rl or rl in ql for rl in rep_locs) or re.search(rf"\b{re.escape(ql)}\b", text):
            return 1.0
        qt = _tokens(ql)
        if not qt:
            continue
        for rl in rep_locs:
            overlap = len(qt & _tokens(rl)) / len(qt)
            best = max(best, 0.7 * overlap)
    return best


class CommanderRetriever:
    def __init__(self, index: CrisisIndex):
        self.index = index
        times = [t for t in (_parse_time(r.timestamp) for r in index.reports) if t is not None]
        # "now" = newest report in the index, so recency also works on historical datasets
        self.reference_time = max(times) if times else datetime.now(timezone.utc)

    def plan(self, query: str) -> QueryPlan:
        return QueryPlan(
            query=query,
            locations=extract_locations(query, lenient=True),
            needs=extract_needs(query),
            wants_urgent=bool(URGENCY_INTENT_RE.search(query)),
            wants_recent=bool(RECENCY_INTENT_RE.search(query)),
        )

    def _recency(self, report: Report, half_life_hours: float = 12.0) -> float:
        ts = _parse_time(report.timestamp)
        if ts is None:
            return 0.0
        hours = max(0.0, (self.reference_time - ts).total_seconds() / 3600)
        return math.exp(-math.log(2) * hours / half_life_hours)

    def retrieve(
        self,
        query: str,
        k: int = 5,
        min_priority: str = "",
        sources: list[str] | None = None,
        candidate_pool: int = 100,
    ) -> tuple[list[RetrievedReport], QueryPlan]:
        plan = self.plan(query)
        reports = self.index.reports

        candidates = dict(self.index.search(query, k=candidate_pool))
        loc_scores: dict[int, float] = {}
        if plan.locations:
            for i, r in enumerate(reports):
                s = location_match(plan.locations, r)
                if s > 0:
                    loc_scores[i] = s
            strong = {i for i, s in loc_scores.items() if s >= 1.0}
            if strong:
                plan.location_filter_applied = True
                missing = [i for i in strong if i not in candidates]
                candidates = {i: candidates[i] for i in candidates if i in strong}
                candidates.update(self.index.similarity_to(query, missing))

        w_sim, w_loc, w_urg, w_rec = 0.55, 0.20, 0.15, 0.10
        w_need = 0.25 if plan.needs else 0.0
        if plan.wants_urgent:
            w_sim, w_urg = w_sim - 0.20, w_urg + 0.20
        if plan.wants_recent:
            w_sim, w_rec = w_sim - 0.10, w_rec + 0.10

        min_rank = PRIORITY_ORDER.get(min_priority, 0)
        results = []
        for i, sim in candidates.items():
            r = reports[i]
            if min_rank and PRIORITY_ORDER.get(r.priority, 0) < min_rank:
                continue
            if sources and not any(r.source.startswith(s) for s in sources):
                continue
            comp = {
                "similarity": round(sim, 3),
                "location": round(loc_scores.get(i, 0.0), 3),
                "urgency": round(r.urgency_score / 100, 3),
                "recency": round(self._recency(r), 3),
                "need": 1.0 if set(plan.needs) & set(r.needs) else 0.0,
            }
            score = (w_sim * comp["similarity"] + w_loc * comp["location"] + w_urg * comp["urgency"]
                     + w_rec * comp["recency"] + w_need * comp["need"])
            if r.category == "not_humanitarian":
                score -= 0.2
            results.append(RetrievedReport(r, round(score, 4), comp))

        results.sort(key=lambda x: x.score, reverse=True)
        return results[:k], plan
