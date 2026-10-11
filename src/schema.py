"""Shared data structure for a crisis report as it moves through the pipeline."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def make_id(source: str, text: str) -> str:
    digest = hashlib.sha1(f"{source}|{text}".encode("utf-8")).hexdigest()[:12]
    return f"{source}-{digest}"


@dataclass
class Report:
    id: str
    text: str
    source: str
    timestamp: str = field(default_factory=utc_now_iso)
    url: str = ""
    gold_label: str = ""  # human label when the source provides one (HumAID, sample data)

    # NLP outputs
    category: str = ""
    category_confidence: float = 0.0
    locations: list[str] = field(default_factory=list)
    primary_location: str = ""
    people_count: int | None = None
    needs: list[str] = field(default_factory=list)
    primary_need: str = ""
    urgency_score: float = 0.0
    priority: str = ""
    urgency_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Report":
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**known)

    def summary_line(self) -> str:
        """The one-line structured view, e.g. Location: XYZ bridge | Need: Rescue | People: 4 | Priority: Critical."""
        people = str(self.people_count) if self.people_count is not None else "unknown"
        return (
            f"Location: {self.primary_location or 'unknown'} | "
            f"Need: {self.primary_need or 'none stated'} | "
            f"People: {people} | Priority: {self.priority or 'unscored'}"
        )

    def embedding_text(self) -> str:
        """Text that gets embedded: raw message plus structured fields, so queries on either match."""
        parts = [self.text]
        if self.primary_location:
            parts.append(f"Location: {', '.join(self.locations)}")
        if self.needs:
            parts.append(f"Needs: {', '.join(self.needs)}")
        if self.category:
            parts.append(f"Category: {self.category.replace('_', ' ')}")
        return " | ".join(parts)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)
