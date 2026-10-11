"""Humanitarian-category classifier (HumAID label set).

Model: TF-IDF word 1-2 grams + char 3-5 grams, Logistic Regression with balanced class weights.
Runs on CPU in seconds, needs no downloads, and gives calibrated-enough probabilities for the
urgency scorer. Train on HumAID for real accuracy; the 50-message sample only keeps things runnable.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import re

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import FeatureUnion, Pipeline

from src.config import MODEL_DIR, get_logger
from src.nlp.preprocess import normalize_for_model

log = get_logger("nlp.classifier")

HUMAID_LABELS = [
    "caution_and_advice",
    "displaced_people_and_evacuations",
    "infrastructure_and_utility_damage",
    "injured_or_dead_people",
    "missing_or_found_people",
    "not_humanitarian",
    "other_relevant_information",
    "requests_or_urgent_needs",
    "rescue_volunteering_or_donation_effort",
    "sympathy_and_support",
]

DEFAULT_MODEL_PATH = MODEL_DIR / "humanitarian_clf.joblib"

# Keyword prior, blended in only when the model was trained on a tiny set (the 50-message sample).
# With HumAID training data the learned model is used on its own.
CATEGORY_KEYWORDS = {
    "caution_and_advice": r"\b(alert|warn\w*|advis\w*|avoid|stay indoors|do not|don'?t|boil|precaution|beware|safety tips?)\b",
    "displaced_people_and_evacuations": r"\b(evacuat\w*|displaced|shift(?:ed|ing)|relocat\w*|relief camps?|shelters?|homeless|moved to)\b",
    "infrastructure_and_utility_damage": r"\b(power|electricity|transformer|network|bridge|road|damag\w*|collaps\w*|uprooted|caved|outage|supply|blocked|destroy\w*)\b",
    "injured_or_dead_people": r"\b(dead|death|died|killed|injur\w*|bodies|toll|casualt\w*|electrocuted|drowned)\b",
    "missing_or_found_people": r"\b(missing|last seen|found|separated|not returned|whereabouts|untraceable)\b",
    "not_humanitarian": r"\b(coffee|cricket|match|movie|launch|tickets?|party|weekend|holiday|bajji)\b",
    "other_relevant_information": r"\b(mm|rain(?:fall)?|landfall|released|cusecs|closed|suspended|forecast|level|update[sd]?)\b",
    "requests_or_urgent_needs": r"\b(need\w*|urgent\w*|sos|please (?:send|help)|trapped|stuck|stranded|require\w*|help us)\b",
    "rescue_volunteering_or_donation_effort": r"\b(rescued|rescuing|volunteer\w*|donat\w*|collecting|airlifted|ndrf|relief fund|teams?)\b",
    "sympathy_and_support": r"\b(pray\w*|thoughts|condolence\w*|stay (?:safe|strong)|heartbreaking|salute|hearts? (?:go|are))\b",
}
_CATEGORY_RES = {k: re.compile(v, re.IGNORECASE) for k, v in CATEGORY_KEYWORDS.items()}
SMALL_DATA_BLEND = 0.6  # weight of the keyword prior when trained on < 1000 examples


def keyword_prior(texts: list[str], classes: list[str]) -> np.ndarray:
    out = np.zeros((len(texts), len(classes)), dtype=float)
    for i, t in enumerate(texts):
        hits = np.array([len(_CATEGORY_RES[c].findall(t)) if c in _CATEGORY_RES else 0 for c in classes], dtype=float)
        hits += 0.05
        out[i] = hits / hits.sum()
    return out


@dataclass
class TrainResult:
    n_train: int
    eval_name: str
    accuracy: float
    macro_f1: float
    report: str

    def __str__(self) -> str:
        return (
            f"Trained on {self.n_train} messages | {self.eval_name}: "
            f"accuracy={self.accuracy:.3f}, macro-F1={self.macro_f1:.3f}"
        )


def _build_pipeline(small_data: bool) -> Pipeline:
    min_df = 1 if small_data else 2
    features = FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=min_df, max_features=200_000, sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=min_df, max_features=300_000, sublinear_tf=True)),
    ])
    clf = LogisticRegression(max_iter=2000, C=4.0 if not small_data else 10.0, class_weight="balanced")
    return Pipeline([("features", features), ("clf", clf)])


class HumanitarianClassifier:
    def __init__(self, pipeline: Pipeline | None = None, blend: float = 0.0):
        self.pipeline = pipeline
        self.blend = blend

    @property
    def is_trained(self) -> bool:
        return self.pipeline is not None

    @property
    def labels(self) -> list[str]:
        return list(self.pipeline.classes_) if self.pipeline is not None else []

    def train(
        self,
        texts: list[str],
        labels: list[str],
        eval_texts: list[str] | None = None,
        eval_labels: list[str] | None = None,
    ) -> TrainResult:
        x = [normalize_for_model(t) for t in texts]
        small = len(x) < 1000
        self.blend = SMALL_DATA_BLEND if small else 0.0
        pipeline = _build_pipeline(small)

        if eval_texts is not None and eval_labels is not None:
            pipeline.fit(x, labels)
            self.pipeline = pipeline
            preds = [p[0] for p in self.predict(list(eval_texts))]
            truth = list(eval_labels)
            eval_name = f"held-out set ({len(truth)} msgs)"
        else:
            # no held-out set (sample data): cross-validated scores, then fit on everything
            min_class = min(np.unique(labels, return_counts=True)[1])
            folds = max(2, min(5, int(min_class)))
            cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
            probs = cross_val_predict(_build_pipeline(small), x, labels, cv=cv, method="predict_proba")
            classes = sorted(set(labels))
            probs = (1 - self.blend) * probs + self.blend * keyword_prior(x, classes)
            preds = [classes[i] for i in probs.argmax(axis=1)]
            truth = list(labels)
            eval_name = f"{folds}-fold cross-validation" + (" (model + keyword prior)" if self.blend else "")
            pipeline.fit(x, labels)
            self.pipeline = pipeline
        return TrainResult(
            n_train=len(x),
            eval_name=eval_name,
            accuracy=float(accuracy_score(truth, preds)),
            macro_f1=float(f1_score(truth, preds, average="macro", zero_division=0)),
            report=classification_report(truth, preds, zero_division=0),
        )

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        if not self.is_trained:
            raise RuntimeError("Classifier is not trained. Run run_pipeline.py first.")
        x = [normalize_for_model(t) for t in texts]
        probs = self.pipeline.predict_proba(x)
        if self.blend:
            probs = (1 - self.blend) * probs + self.blend * keyword_prior(x, self.labels)
        return probs

    def predict(self, texts: list[str]) -> list[tuple[str, float, dict[str, float]]]:
        """Returns (label, confidence, full probability dict) per text."""
        probs = self.predict_proba(texts)
        classes = self.labels
        out = []
        for row in probs:
            idx = int(np.argmax(row))
            out.append((classes[idx], float(row[idx]), {c: float(p) for c, p in zip(classes, row)}))
        return out

    def save(self, path: Path = DEFAULT_MODEL_PATH) -> Path:
        if not self.is_trained:
            raise RuntimeError("Nothing to save: classifier is not trained.")
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"pipeline": self.pipeline, "blend": self.blend}, path)
        log.info("Saved classifier to %s", path)
        return path

    @classmethod
    def load(cls, path: Path = DEFAULT_MODEL_PATH) -> "HumanitarianClassifier":
        if not path.exists():
            raise FileNotFoundError(f"No trained classifier at {path}. Run: python run_pipeline.py")
        state = joblib.load(path)
        return cls(state["pipeline"], state.get("blend", 0.0))
