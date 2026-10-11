"""Text normalisation shared by the classifier, NER and urgency scorer."""

from __future__ import annotations

import html
import re

URL_RE = re.compile(r"https?://\S+|www\.\S+")
RT_RE = re.compile(r"^\s*RT\s+@\w+:?\s*", re.IGNORECASE)
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#(\w+)")
CAMEL_RE = re.compile(r"(?<=[a-z])(?=[A-Z])")
SPACE_RE = re.compile(r"\s+")


def _split_hashtag(match: re.Match) -> str:
    # #EcuadorEarthquake -> Ecuador Earthquake ; #flood -> flood
    return CAMEL_RE.sub(" ", match.group(1)).replace("_", " ")


def clean_text(text: str) -> str:
    """Readable cleanup: keeps case and punctuation (needed for NER), removes URLs, RT prefixes, @mentions."""
    text = html.unescape(str(text or ""))
    text = RT_RE.sub("", text)
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    text = HASHTAG_RE.sub(_split_hashtag, text)
    return SPACE_RE.sub(" ", text).strip()


def normalize_for_model(text: str) -> str:
    """Lowercased version used as classifier input."""
    return clean_text(text).lower()
