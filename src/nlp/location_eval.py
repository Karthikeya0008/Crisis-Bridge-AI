"""Measure the location extractor on IDRISI gold labels, and learn a gazetteer from IDRISI train.

Run from the project folder:
    python -m src.nlp.location_eval            # evaluate, build and save the learned gazetteer
    python -m src.nlp.location_eval --no-save  # evaluate only

Scoring is per tweet on unique place names, case-insensitive. A prediction counts as correct when it
matches a gold mention or one contains the other ("Kerala" vs "Kerala state"), the usual lenient LMR match.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

from src.ingestion.curated_loader import load_idrisi
from src.nlp.ner import build_learned_gazetteer, extract_locations, save_learned_gazetteer, use_learned_gazetteer


@dataclass
class LocationScore:
    name: str
    precision: float
    recall: float
    f1: float
    tweets: int

    def __str__(self) -> str:
        return f"{self.name:46} precision={self.precision:.3f}  recall={self.recall:.3f}  F1={self.f1:.3f}  ({self.tweets} tweets)"


def _matches(a: str, b: str) -> bool:
    return a == b or a in b or b in a


def score_locations(rows: list[dict], name: str) -> LocationScore:
    tp = fp = fn = 0
    for row in rows:
        gold = {m["text"].lower().strip() for m in row.get("location_mentions", [])}
        pred = {p.lower().strip() for p in extract_locations(row["text"])}
        hits = {p for p in pred if any(_matches(p, g) for g in gold)}
        tp += len(hits)
        fp += len(pred - hits)
        fn += sum(1 for g in gold if not any(_matches(g, p) for p in pred))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-9, precision + recall)
    return LocationScore(name, precision, recall, f1, len(rows))


def evaluate(min_count: int = 2, save: bool = True) -> list[LocationScore]:
    train, dev = load_idrisi("train"), load_idrisi("dev")
    names = build_learned_gazetteer(train, min_count)
    results = []
    with use_learned_gazetteer(None):
        results.append(score_locations(dev, "Rules + built-in gazetteer"))
    with use_learned_gazetteer(names):
        results.append(score_locations(dev, f"+ learned gazetteer ({len(names)} names)"))
    # harder test: for each disaster, learn names only from the OTHER disasters
    events = sorted({r["event"] for r in dev})
    unseen = []
    for event in events:
        others = build_learned_gazetteer([r for r in train if r["event"] != event], min_count)
        with use_learned_gazetteer(others):
            unseen += [(r, extract_locations(r["text"])) for r in dev if r["event"] == event]
    tp = fp = fn = 0
    for row, preds in unseen:
        gold = {m["text"].lower().strip() for m in row.get("location_mentions", [])}
        pred = {p.lower().strip() for p in preds}
        hits = {p for p in pred if any(_matches(p, g) for g in gold)}
        tp, fp = tp + len(hits), fp + len(pred - hits)
        fn += sum(1 for g in gold if not any(_matches(g, p) for p in pred))
    p, r = tp / max(1, tp + fp), tp / max(1, tp + fn)
    results.append(LocationScore("+ learned gazetteer, disaster unseen in training", p, r, 2 * p * r / max(1e-9, p + r), len(unseen)))
    if save:
        save_learned_gazetteer(names)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate location extraction on IDRISI and learn a gazetteer")
    parser.add_argument("--min-count", type=int, default=2, help="keep place names annotated at least this often")
    parser.add_argument("--no-save", action="store_true", help="evaluate only; do not write data/models/learned_gazetteer.txt")
    args = parser.parse_args()
    for result in evaluate(args.min_count, save=not args.no_save):
        print(result)
    if not args.no_save:
        print("Saved the learned gazetteer to data/models/learned_gazetteer.txt (used for every new report).")


if __name__ == "__main__":
    main()
