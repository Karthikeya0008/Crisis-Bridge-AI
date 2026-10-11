"""One object that runs classification, entity extraction and urgency scoring on reports."""

from __future__ import annotations

from src.config import get_logger
from src.nlp.classifier import HumanitarianClassifier
from src.nlp.ner import extract_entities
from src.nlp.preprocess import clean_text
from src.nlp.urgency import score_urgency
from src.schema import Report, make_id

log = get_logger("nlp.processor")


class CrisisProcessor:
    def __init__(self, classifier: HumanitarianClassifier):
        if not classifier.is_trained:
            raise ValueError("CrisisProcessor needs a trained classifier.")
        self.classifier = classifier

    @classmethod
    def from_saved(cls) -> "CrisisProcessor":
        return cls(HumanitarianClassifier.load())

    def process(self, reports: list[Report], batch_size: int = 512) -> list[Report]:
        for start in range(0, len(reports), batch_size):
            batch = reports[start:start + batch_size]
            predictions = self.classifier.predict([r.text for r in batch])
            for report, (label, conf, probs) in zip(batch, predictions):
                ents = extract_entities(report.text)
                urg = score_urgency(report.text, probs, ents.people_count, ents.needs)
                report.category = label
                report.category_confidence = round(conf, 3)
                report.locations = ents.locations
                report.primary_location = ents.primary_location
                report.people_count = ents.people_count
                report.needs = ents.needs
                report.primary_need = ents.primary_need
                report.urgency_score = urg.score
                report.priority = urg.priority
                report.urgency_reasons = urg.reasons
        return reports

    def process_text(self, text: str, source: str = "manual") -> Report:
        text = clean_text(text) or text
        report = Report(id=make_id(source, text), text=text, source=source)
        return self.process([report])[0]
