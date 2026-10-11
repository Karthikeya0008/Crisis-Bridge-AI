"""Glue layer shared by run_pipeline.py, the CLI and the Streamlit app."""

from __future__ import annotations

from dataclasses import dataclass, field

from src.config import get_logger
from src.ingestion.curated_loader import DataUnavailable, dataframe_to_reports, load_humaid, load_sample_data
from src.ingestion.live_sources import SourceStatus, fetch_all
from src.nlp.classifier import HumanitarianClassifier, TrainResult
from src.nlp.processor import CrisisProcessor
from src.rag.generator import Answer, generate_answer
from src.rag.indexer import CrisisIndex
from src.rag.retriever import CommanderRetriever, QueryPlan, RetrievedReport
from src.schema import Report

log = get_logger("pipeline")

DATA_MODES = ("auto", "humaid", "sample")


@dataclass
class BuildSummary:
    data_used: str
    train_result: TrainResult
    n_indexed: int
    embedder: str
    store: str
    messages: list[str] = field(default_factory=list)


def train_classifier(data: str = "auto", train_limit: int | None = 20000, eval_limit: int | None = 5000) -> tuple[HumanitarianClassifier, TrainResult, str, list[str]]:
    """Train on HumAID when available (or forced), otherwise on the bundled sample data."""
    messages: list[str] = []
    clf = HumanitarianClassifier()
    if data in ("auto", "humaid"):
        try:
            train_df = load_humaid("train", limit=train_limit)
            test_df = load_humaid("test", limit=eval_limit)
            log.info("Training classifier on %d HumAID tweets...", len(train_df))
            result = clf.train(train_df["text"].tolist(), train_df["label"].tolist(),
                               test_df["text"].tolist(), test_df["label"].tolist())
            result.eval_name = f"HumAID test ({len(test_df)} tweets)"
            return clf, result, "humaid", messages
        except DataUnavailable as exc:
            if data == "humaid":
                raise
            log.warning("HumAID unavailable; using bundled sample data instead.")
            messages.append(f"HumAID unavailable, fell back to bundled sample data. Reason: {exc}")
    sample = load_sample_data()
    log.info("Training classifier on %d bundled sample messages (demo quality only)...", len(sample))
    result = clf.train(sample["text"].tolist(), sample["label"].tolist())
    return clf, result, "sample", messages


def load_corpus(data_used: str, index_limit: int | None = 2000) -> list[Report]:
    """Reports to index: always the sample messages, plus a HumAID test subset when HumAID is in use."""
    reports = dataframe_to_reports(load_sample_data(), "sample")
    if data_used == "humaid":
        humaid = load_humaid("test", limit=index_limit)
        reports += dataframe_to_reports(humaid, "humaid")
    return reports


def build_system(data: str = "auto", train_limit: int | None = 20000, index_limit: int | None = 2000) -> tuple[CrisisProcessor, CrisisIndex, BuildSummary]:
    if data not in DATA_MODES:
        raise ValueError(f"data must be one of {DATA_MODES}")
    clf, result, used, messages = train_classifier(data, train_limit)
    clf.save()
    processor = CrisisProcessor(clf)
    reports = processor.process(load_corpus(used, index_limit))
    index = CrisisIndex.build(reports)
    index.save()
    return processor, index, BuildSummary(used, result, len(index), index.embedder.name, index.store_name, messages)


class CommanderService:
    """Loaded system: process new text, ingest live feeds, answer commander queries."""

    def __init__(self, processor: CrisisProcessor, index: CrisisIndex):
        self.processor = processor
        self.index = index
        self.retriever = CommanderRetriever(index)

    @classmethod
    def load(cls) -> "CommanderService":
        return cls(CrisisProcessor.from_saved(), CrisisIndex.load())

    def ask(self, query: str, k: int = 5, provider: str | None = None, min_priority: str = "") -> tuple[Answer, list[RetrievedReport], QueryPlan]:
        results, plan = self.retriever.retrieve(query, k=k, min_priority=min_priority)
        kwargs = {"provider": provider} if provider else {}
        return generate_answer(query, results, plan, **kwargs), results, plan

    def add_reports(self, reports: list[Report], persist: bool = True) -> int:
        processed = self.processor.process(reports)
        added = self.index.add(processed)
        if added:
            self.retriever = CommanderRetriever(self.index)
            if persist:
                self.index.save()
        return added

    def add_text(self, text: str, source: str = "manual") -> Report:
        report = self.processor.process_text(text, source)
        self.add_reports([report])
        return report

    def ingest_live(self, sources: list[str] | None = None) -> tuple[int, list[SourceStatus]]:
        reports, statuses = fetch_all(sources)
        added = self.add_reports(reports) if reports else 0
        return added, statuses
