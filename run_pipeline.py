"""CrisisBridge AI end-to-end run.

load data -> train/evaluate classifier -> extract entities -> score urgency -> index -> answer a query

Examples (PowerShell, from the project folder):
    python run_pipeline.py                      # HumAID if reachable, else bundled sample data
    python run_pipeline.py --data sample        # fully offline
    python run_pipeline.py --data humaid --train-limit 0      # full HumAID train split
    python run_pipeline.py --query "where is drinking water needed?"
    python run_pipeline.py --live gdacs rss     # also pull optional live feeds into the index
"""

from __future__ import annotations

import argparse
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

from src.pipeline import CommanderService, build_system  # noqa: E402

DEFAULT_QUERY = "What is the most critical situation near XYZ bridge right now?"


def _limit(value: int) -> int | None:
    return None if value <= 0 else value


def main() -> int:
    parser = argparse.ArgumentParser(description="CrisisBridge AI end-to-end pipeline")
    parser.add_argument("--data", choices=["auto", "humaid", "sample"], default="auto",
                        help="auto = HumAID if downloadable, else sample (default)")
    parser.add_argument("--train-limit", type=int, default=20000, help="HumAID training rows (0 = all 53k)")
    parser.add_argument("--index-limit", type=int, default=2000, help="HumAID test tweets added to the index (0 = all)")
    parser.add_argument("--query", default=DEFAULT_QUERY, help="Commander query to answer at the end")
    parser.add_argument("--k", type=int, default=5, help="reports to retrieve")
    parser.add_argument("--provider", choices=["auto", "groq", "gemini", "extractive"], default=None,
                        help="answer generator (default: LLM_PROVIDER from .env)")
    parser.add_argument("--live", nargs="*", choices=["gdacs", "reliefweb", "reddit", "rss"],
                        help="optional live sources to ingest after indexing")
    args = parser.parse_args()

    t0 = time.time()
    print("=" * 78)
    print("CrisisBridge AI | end-to-end pipeline")
    print("=" * 78)

    print("\n[1-4] Load data, train classifier, extract entities, score urgency, build index")
    processor, index, summary = build_system(args.data, _limit(args.train_limit), _limit(args.index_limit))
    for m in summary.messages:
        print(f"  NOTE: {m}")
    print(f"  data source        : {summary.data_used}")
    print(f"  classifier         : {summary.train_result}")
    print(f"  index              : {summary.n_indexed} reports | embeddings={summary.embedder} | store={summary.store}")

    service = CommanderService(processor, index)

    if args.live is not None:
        print("\n[optional] Live feeds")
        added, statuses = service.ingest_live(args.live or None)
        for s in statuses:
            print(f"  {s}")
        print(f"  added {added} new reports to the index")

    print("\n[sample] Structured extraction (top 5 by urgency)")
    top = sorted(index.reports, key=lambda r: r.urgency_score, reverse=True)[:5]
    for r in top:
        print(f"  [{r.id}] \"{r.text[:70]}\"")
        print(f"      -> {r.summary_line()} | urgency={r.urgency_score:.0f} | category={r.category}")

    counts = {}
    for r in index.reports:
        counts[r.priority] = counts.get(r.priority, 0) + 1
    print("  priority distribution: " + ", ".join(f"{p}={counts.get(p, 0)}" for p in ("Critical", "High", "Medium", "Low")))

    print("\n[5] Commander Query")
    print(f"  Q: {args.query}")
    answer, results, plan = service.ask(args.query, k=args.k, provider=args.provider)
    print(f"  parsed: locations={plan.locations} needs={plan.needs} urgent={plan.wants_urgent} "
          f"recent={plan.wants_recent} location_filter={plan.location_filter_applied}")
    print("-" * 78)
    print(answer.render())
    print("-" * 78)
    print(f"Done in {time.time() - t0:.1f}s. Next: streamlit run src/interface/streamlit_app.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
