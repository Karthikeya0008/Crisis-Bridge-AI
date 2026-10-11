"""Urgency scoring (0-100) and priority bands.

Transparent, additive scoring so every priority can be explained in a viva:
  category signal  - classifier probabilities weighted by how actionable each HumAID class is
  life threat      - trapped, buried, bleeding, unconscious, in labour, drowning ...
  explicit urgency - SOS, urgent, immediately, please send ...
  people affected  - grows with log2(people)
  need type        - rescue > medical > missing person > water > food > shelter
  resolution       - "found safe", "rescued", "restored" lower the score (situation handled)
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from src.nlp.preprocess import clean_text

CATEGORY_WEIGHTS = {
    "requests_or_urgent_needs": 35,
    "injured_or_dead_people": 30,
    "missing_or_found_people": 25,
    "displaced_people_and_evacuations": 20,
    "infrastructure_and_utility_damage": 15,
    "caution_and_advice": 10,
    "rescue_volunteering_or_donation_effort": 8,
    "other_relevant_information": 5,
    "sympathy_and_support": 0,
    "not_humanitarian": 0,
}
NEED_WEIGHTS = {
    "Rescue": 15,
    "Medical": 12,
    "Missing person": 8,
    "Drinking water": 6,
    "Food": 5,
    "Shelter": 4,
    "Power/Utilities": 3,
    "Infrastructure": 3,
    "Volunteers/Donations": 0,
}

LIFE_THREAT_RE = re.compile(
    r"\btrapped\b|\bstuck\b|\bstranded\b|\bburied\b|\bunder (?:the )?(?:rubble|debris)\b|\bdrowning\b|"
    r"\bbleeding\b|\bunconscious\b|\bcritical(?:ly)?\b|\bin labou?r\b|\bheart (?:condition|attack)\b|"
    r"\bcan'?t breathe\b|\belectrocuted\b|\bwater (?:is )?rising\b|\bon (?:the|a) roof(?:top)?\b|\bcollapsed?\b|"
    r"\bseriously injured\b|\bno (?:food|water) since\b|\bsince yesterday\b",
    re.IGNORECASE,
)
EXPLICIT_URGENCY_RE = re.compile(
    r"\bsos\b|\burgent(?:ly)?\b|\bimmediate(?:ly)?\b|\basap\b|\bemergency\b|\bplease (?:send|help|rescue)\b|"
    r"\bneed(?:s|ed)?\b|\bhelp\b|!{2,}",
    re.IGNORECASE,
)
RESOLVED_RE = re.compile(
    r"\bfound safe\b|\bhas been (?:found|rescued)\b|\b(?:were|was|have been) rescued\b|\brescued \d+|"
    r"\bairlifted\b|\brestored\b|\breopened\b|\bsafe now\b|\ball safe\b|\bno casualties\b",
    re.IGNORECASE,
)

PRIORITY_BANDS = [(70, "Critical"), (45, "High"), (25, "Medium"), (0, "Low")]


@dataclass
class UrgencyResult:
    score: float
    priority: str
    reasons: list[str] = field(default_factory=list)


def priority_from_score(score: float) -> str:
    for threshold, label in PRIORITY_BANDS:
        if score >= threshold:
            return label
    return "Low"


def score_urgency(
    text: str,
    category_probs: dict[str, float] | None = None,
    people_count: int | None = None,
    needs: list[str] | None = None,
) -> UrgencyResult:
    text = clean_text(text)
    reasons: list[str] = []
    score = 0.0

    if category_probs:
        cat_score = sum(CATEGORY_WEIGHTS.get(c, 0) * p for c, p in category_probs.items())
        score += cat_score
        top = max(category_probs, key=category_probs.get)
        reasons.append(f"category {top.replace('_', ' ')} (+{cat_score:.0f})")

    threats = {m.group(0).lower() for m in LIFE_THREAT_RE.finditer(text)}
    if threats:
        add = min(35, 25 + 5 * (len(threats) - 1))
        score += add
        reasons.append(f"life threat: {', '.join(sorted(threats))} (+{add})")

    urgent = {m.group(0).lower() for m in EXPLICIT_URGENCY_RE.finditer(text)}
    if urgent:
        add = min(12, 6 + 3 * (len(urgent) - 1))
        score += add
        reasons.append(f"explicit urgency: {', '.join(sorted(urgent))} (+{add})")

    if people_count:
        add = min(15.0, 5 * math.log2(1 + people_count))
        score += add
        reasons.append(f"{people_count} people affected (+{add:.0f})")

    if needs:
        add = NEED_WEIGHTS.get(needs[0], 0)
        if add:
            score += add
            reasons.append(f"need: {needs[0]} (+{add})")

    if RESOLVED_RE.search(text):
        score -= 25
        reasons.append("situation reported as resolved or handled (-25)")

    score = max(0.0, min(100.0, score))
    return UrgencyResult(score=round(score, 1), priority=priority_from_score(score), reasons=reasons)
