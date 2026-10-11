"""CrisisBridge AI command-line fallback (no browser needed).

Run from the project folder:
    python -m src.interface.cli build --data sample
    python -m src.interface.cli ask "What is the most critical situation near XYZ bridge right now?"
    python -m src.interface.cli process "Family of 5 stuck on a roof in Velachery, need a boat"
    python -m src.interface.cli top --n 10
    python -m src.interface.cli live gdacs rss
    python -m src.interface.cli chat
"""

from __future__ import annotations

import argparse
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

from src.ingestion.live_sources import SOURCES  # noqa: E402
from src.pipeline import CommanderService, build_system  # noqa: E402


def _service() -> CommanderService:
    try:
        return CommanderService.load()
    except FileNotFoundError as exc:
        print(f"{exc}\nBuild first: python -m src.interface.cli build --data sample")
        sys.exit(1)


def cmd_build(args) -> None:
    _, _, summary = build_system(args.data, args.train_limit or None, args.index_limit or None)
    for m in summary.messages:
        print(f"NOTE: {m}")
    print(f"Built on {summary.data_used}: {summary.n_indexed} reports indexed ({summary.embedder}, {summary.store}).")
    print(summary.train_result)


def cmd_ask(args) -> None:
    answer, _, plan = _service().ask(args.question, k=args.k, provider=args.provider, min_priority=args.min_priority)
    print(f"parsed: locations={plan.locations} needs={plan.needs} location_filter={plan.location_filter_applied}\n")
    print(answer.render())


def cmd_process(args) -> None:
    svc = _service()
    report = svc.add_text(args.text) if args.add else svc.processor.process_text(args.text)
    print(report.summary_line())
    print(f"category={report.category} ({report.category_confidence:.2f}) | urgency={report.urgency_score:.0f}/100")
    print(f"locations={report.locations} | needs={report.needs}")
    for reason in report.urgency_reasons:
        print(f"  - {reason}")
    if args.add:
        print(f"added to index as {report.id}")


def cmd_top(args) -> None:
    reports = sorted(_service().index.reports, key=lambda r: r.urgency_score, reverse=True)[:args.n]
    for r in reports:
        print(f"{r.urgency_score:5.0f}  {r.priority:<8} {r.summary_line()}")
        print(f"       \"{r.text[:100]}\"  ({r.id})")


def cmd_live(args) -> None:
    added, statuses = _service().ingest_live(args.sources or None)
    for s in statuses:
        print(s)
    print(f"Added {added} new reports.")


def cmd_chat(args) -> None:
    svc = _service()
    print("CrisisBridge Commander chat. Type a question, or 'exit'.")
    while True:
        try:
            q = input("\ncommander> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in {"exit", "quit", "q"}:
            break
        if q:
            answer, _, _ = svc.ask(q, k=args.k, provider=args.provider)
            print(answer.render())


def main() -> None:
    parser = argparse.ArgumentParser(prog="crisisbridge", description="CrisisBridge AI CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    b = sub.add_parser("build", help="train classifier and build the index")
    b.add_argument("--data", choices=["auto", "humaid", "sample"], default="auto")
    b.add_argument("--train-limit", type=int, default=20000)
    b.add_argument("--index-limit", type=int, default=2000)
    b.set_defaults(func=cmd_build)

    providers = ["auto", "groq", "gemini", "extractive"]
    a = sub.add_parser("ask", help="answer a commander query with citations")
    a.add_argument("question")
    a.add_argument("--k", type=int, default=5)
    a.add_argument("--provider", choices=providers, default=None)
    a.add_argument("--min-priority", choices=["", "Low", "Medium", "High", "Critical"], default="")
    a.set_defaults(func=cmd_ask)

    p = sub.add_parser("process", help="classify, extract and score one message")
    p.add_argument("text")
    p.add_argument("--add", action="store_true", help="also add it to the index")
    p.set_defaults(func=cmd_process)

    t = sub.add_parser("top", help="list the most urgent indexed reports")
    t.add_argument("--n", type=int, default=10)
    t.set_defaults(func=cmd_top)

    lv = sub.add_parser("live", help="fetch optional live feeds into the index")
    lv.add_argument("sources", nargs="*", help=f"any of: {', '.join(SOURCES)} (default: all)")
    lv.set_defaults(func=cmd_live)

    c = sub.add_parser("chat", help="interactive question loop")
    c.add_argument("--k", type=int, default=5)
    c.add_argument("--provider", choices=providers, default=None)
    c.set_defaults(func=cmd_chat)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
