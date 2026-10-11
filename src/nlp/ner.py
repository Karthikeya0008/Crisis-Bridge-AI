"""Crisis entity extraction: locations, number of people affected, and need types.

Rule-based by default (no downloads, works offline, transparent for a course demo). If spaCy and
en_core_web_sm happen to be installed, their GPE/LOC/FAC entities are merged in as extra locations.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache

from src.config import get_logger
from src.nlp.preprocess import clean_text

log = get_logger("nlp.ner")

# ---------------------------------------------------------------- locations
LANDMARK_SUFFIXES = (
    "bridge|road|rd|street|st|salai|highway|nagar|colony|village|district|town|city|hospital|school|college|"
    "university|station|railway station|bus stand|harbour|harbor|port|camp|relief camp|lake|river|dam|"
    "beach|market|temple|church|mosque|airport|flyover|subway|junction|circle|layout|block|sector|"
    "taluk|mandal|tehsil|ward|island|valley|hill|hills|bay|coast|stadium|park|apartments?|towers?|complex"
)
PREPOSITIONS = r"near|at|in|on|from|around|behind|opposite|off|along|outside|inside|towards|toward|between|across|via|of"

# Capitalised tokens (allows ALL-CAPS like XYZ, and Mc/O' style names)
CAP_WORD = r"(?:[A-Z][a-zA-Z'\-]*|[A-Z]{2,})"
CAP_PHRASE = rf"{CAP_WORD}(?:\s+{CAP_WORD}){{0,3}}"

# "near XYZ bridge", "in Velachery", "at Adyar relief camp"
PREP_LOC_RE = re.compile(
    rf"\b(?:{PREPOSITIONS})\s+(?:the\s+)?({CAP_PHRASE}(?:\s+(?:[a-z]+\s+)?(?:{LANDMARK_SUFFIXES})\b)?)"
)
# "Anna Salai", "GST Road", "Chembarambakkam lake" anywhere in the text
SUFFIX_LOC_RE = re.compile(rf"\b({CAP_PHRASE}\s+(?i:{LANDMARK_SUFFIXES}))\b")
# "Chennai, Chengalpattu and Kancheepuram" continuation after a found location
CONTINUATION_RE = re.compile(rf"^(?:\s*,\s*|\s+and\s+|\s*,\s*and\s+)({CAP_PHRASE})")
# lenient pattern for lowercase user queries: "near xyz bridge", "in velachery"
LENIENT_RE = re.compile(
    rf"\b(?:near|at|in|around|behind|opposite|outside)\s+(?:the\s+)?([a-z0-9][\w'\-]*(?:\s+(?!right\b|now\b|today\b|currently\b)[a-z0-9][\w'\-]*){{0,2}})",
    re.IGNORECASE,
)

NON_LOCATION_WORDS = {
    "i", "we", "please", "urgent", "sos", "help", "update", "missing", "found", "breaking", "rt", "the",
    "ndrf", "sdrf", "imd", "ndma", "nss", "un", "unicef", "who", "red cross", "navy", "army", "police",
    "chief minister", "cm", "pm", "government", "monday", "tuesday", "wednesday", "thursday", "friday",
    "saturday", "sunday", "january", "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december", "tamil", "hindi", "english", "god", "least", "all",
    "fire", "fire service", "our", "my", "this", "that", "do", "boil", "residents", "volunteers", "donate",
    "salute", "stay", "praying", "new", "perfect", "cricket", "heartbreaking", "deepest", "schools",
    "suburban", "water", "power", "mobile", "several", "patients", "coastal", "two", "man", "around",
    "relief", "beach",  # bare "Beach" usually means the station in Chennai context; kept out to avoid noise
}
LOWER_STOP = {"right", "now", "today", "currently", "the", "most", "critical", "situation", "situations",
              "area", "areas", "there", "here", "what", "which", "where", "is", "are", "of", "a"}

# small gazetteer so sentence-initial place names ("Chennai received...") are still caught
GAZETTEER = {
    "india", "chennai", "mumbai", "delhi", "new delhi", "kolkata", "bengaluru", "bangalore", "hyderabad",
    "pune", "ahmedabad", "surat", "kochi", "trivandrum", "thiruvananthapuram", "vellore", "madurai",
    "coimbatore", "visakhapatnam", "vijayawada", "guwahati", "bhubaneswar", "puri", "patna", "lucknow",
    "jaipur", "shimla", "dehradun", "srinagar", "wayanad", "kerala", "tamil nadu", "karnataka", "assam",
    "odisha", "bihar", "uttarakhand", "himachal pradesh", "andhra pradesh", "telangana", "maharashtra",
    "gujarat", "west bengal", "sikkim", "manipur", "chengalpattu", "kancheepuram", "tambaram", "velachery",
    "mudichur", "pallikaranai", "saidapet", "guindy", "adyar", "porur", "ennore", "kasimedu", "mamallapuram",
    "kotturpuram", "madipakkam", "perungalathur", "nepal", "bangladesh", "sri lanka", "pakistan", "japan",
    "philippines", "indonesia", "turkey", "ecuador", "mexico", "haiti", "puerto rico", "florida", "texas",
    "california", "houston",
}


def _normalize_location(loc: str) -> str:
    loc = re.sub(r"^(?:the)\s+", "", loc.strip(" ,.;:-'\""), flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", loc)


def _is_valid_location(loc: str) -> bool:
    low = loc.lower()
    if len(loc) < 3 or low in NON_LOCATION_WORDS:
        return False
    first = low.split()[0]
    if first in NON_LOCATION_WORDS and low not in GAZETTEER:
        return False
    return not low.isdigit()


@lru_cache(maxsize=1)
def _spacy_nlp():
    try:
        import spacy

        return spacy.load("en_core_web_sm")
    except Exception:
        return None


def extract_locations(text: str, lenient: bool = False) -> list[str]:
    """Return unique locations in order of appearance. `lenient=True` also handles lowercase queries."""
    text = clean_text(text)
    found: list[tuple[int, str]] = []

    for m in PREP_LOC_RE.finditer(text):
        loc = _normalize_location(m.group(1))
        if _is_valid_location(loc):
            found.append((m.start(1), loc))
            # follow "X, Y and Z" lists
            rest = text[m.end(1):]
            while True:
                c = CONTINUATION_RE.match(rest)
                if not c:
                    break
                nxt = _normalize_location(c.group(1))
                if _is_valid_location(nxt):
                    found.append((m.end(1) + c.start(1), nxt))
                rest = rest[c.end(1):]

    for m in SUFFIX_LOC_RE.finditer(text):
        loc = _normalize_location(m.group(1))
        if _is_valid_location(loc):
            found.append((m.start(1), loc))

    low_text = text.lower()
    for place in GAZETTEER:
        for m in re.finditer(rf"\b{re.escape(place)}\b", low_text):
            found.append((m.start(), text[m.start():m.end()]))

    nlp = _spacy_nlp()
    if nlp is not None:
        for ent in nlp(text).ents:
            if ent.label_ in {"GPE", "LOC", "FAC"} and _is_valid_location(ent.text):
                found.append((ent.start_char, _normalize_location(ent.text)))

    if lenient:
        for m in LENIENT_RE.finditer(text):
            words = [w for w in m.group(1).split() if w.lower() not in LOWER_STOP]
            if words:
                found.append((m.start(1), " ".join(words)))

    # de-duplicate: keep the longest form, drop entries contained in a longer one
    found.sort(key=lambda t: (t[0], -len(t[1])))
    result: list[str] = []
    for _, loc in found:
        low = loc.lower()
        if any(low == r.lower() or low in r.lower().split(" ") or low in r.lower() for r in result):
            continue
        # replace shorter earlier entries that this one contains
        result = [r for r in result if r.lower() not in low]
        result.append(loc)
    return result


# ---------------------------------------------------------------- people count
NUMBER_WORDS = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "fifteen": 15, "twenty": 20,
    "thirty": 30, "forty": 40, "fifty": 50, "hundred": 100, "dozen": 12, "dozens": 24, "hundreds": 200,
    "several": 3, "many": 10, "few": 3, "couple": 2,
}
PERSON_NOUNS = (
    r"people|persons|person|individuals|families|family members|members|children|kids|child|students|"
    r"passengers|residents|fishermen|fisherman|villagers|patients|workers|labourers|tourists|pilgrims|"
    r"men|women|man|woman|boys|girls|boy|girl|elderly|senior citizens|infants|babies|victims|survivors|"
    r"bodies|lives|of us"
)
STATUS_WORDS = r"injured|dead|killed|missing|trapped|stranded|stuck|feared dead|hospitalised|hospitalized|rescued|evacuated|displaced"
NUM = r"\d{1,3}(?:,\d{3})*|\d+|" + "|".join(sorted(NUMBER_WORDS, key=len, reverse=True))
QUANT = r"(?:at least|around|about|nearly|over|more than|approximately|approx\.?|some|almost)?\s*"

PEOPLE_RE = re.compile(
    rf"\b{QUANT}({NUM})\s+(?:[a-z\-]+\s+){{0,2}}?({PERSON_NOUNS})\b", re.IGNORECASE
)
STATUS_COUNT_RE = re.compile(rf"\b{QUANT}({NUM})\s+(?:people\s+)?({STATUS_WORDS})\b", re.IGNORECASE)
FAMILY_OF_RE = re.compile(rf"\bfamily of\s+({NUM})\b", re.IGNORECASE)
SINGLE_PERSON_RE = re.compile(
    r"\b(?:(?:a|an|my|our|one)\s+)?(?:elderly\s+|old\s+|young\s+|pregnant\s+|injured\s+|\d+\s+year\s+old\s+)?"
    r"(man|woman|boy|girl|child|person|father|mother|son|daughter|brother|sister|grandfather|grandmother|baby)\b",
    re.IGNORECASE,
)
AGE_RE = re.compile(r"\b(\d{1,3})\s*(?:-\s*)?(?:year|yr)s?\s*(?:-\s*)?old\b", re.IGNORECASE)
WORD_PERSON_RE = re.compile(
    r"\b(man|woman|boy|girl|father|mother|patient)\b", re.IGNORECASE
)


def _to_int(token: str) -> int | None:
    token = token.lower().replace(",", "").strip()
    if token.isdigit():
        return int(token)
    return NUMBER_WORDS.get(token)


def extract_people_count(text: str) -> int | None:
    """Estimate how many people the message is about. None when nothing is stated."""
    text = clean_text(text)
    spans: list[tuple[int, int, int]] = []  # (start, end, count)

    def add(m: re.Match, value: int | None):
        if value is None or value <= 0 or value > 1_000_000:
            return
        if any(not (m.end() <= s or m.start() >= e) for s, e, _ in spans):
            return  # overlapping mention already counted
        spans.append((m.start(), m.end(), value))

    ages = {m.group(1) for m in AGE_RE.finditer(text)}
    for m in FAMILY_OF_RE.finditer(text):
        add(m, _to_int(m.group(1)))
    for m in PEOPLE_RE.finditer(text):
        if m.group(1) in ages and re.search(rf"{m.group(1)}\s*(?:-\s*)?(?:year|yr)", text[m.start(1):m.start(1) + 12]):
            continue
        noun = m.group(2).lower()
        value = _to_int(m.group(1))
        if noun in {"man", "woman", "boy", "girl", "child", "person", "fisherman"} and value and value > 1 and m.group(1).lower() in {"a", "an"}:
            value = 1
        add(m, value)
    for m in STATUS_COUNT_RE.finditer(text):
        add(m, _to_int(m.group(1)))
    if not spans:
        for m in SINGLE_PERSON_RE.finditer(text):
            add(m, 1)
    if not spans and AGE_RE.search(text) and WORD_PERSON_RE.search(text):
        return 1
    if not spans and ages:
        return 1
    return sum(v for _, _, v in spans) if spans else None


# ---------------------------------------------------------------- needs
NEED_PATTERNS: dict[str, str] = {
    "Rescue": r"\btrapped\b|\bstuck\b|\bstranded\b|\brescue\b|\bevacuat\w*|\bboats?\b|\bairlift\w*|\bsos\b|"
              r"\bdrowning\b|\bburied\b|\bunder (?:the )?(?:rubble|debris)\b|\bon (?:the|a) roof(?:top)?\b|\brooftop\b",
    "Medical": r"\bambulance\b|\bmedicine\w*|\bmedical\b|\bdoctor\w*\b|\binjured\b|\bbleeding\b|\bunconscious\b|"
               r"\bin labou?r\b|\bheart (?:condition|attack|patient)\b|\boxygen\b|\bdialysis\b|\binsulin\b|\bfirst aid\b|\bhospital\b",
    "Missing person": r"\bmissing\b|\blast seen\b|\bnot returned\b|\bunaccounted\b|\bseparated from\b|\bcannot (?:find|reach) (?:my|our)\b|\bwhereabouts\b",
    "Drinking water": r"\bdrinking water\b|\bno water\b|\bneed water\b|\bwater (?:supply|bottles|packets)\b|\bthirsty\b",
    "Food": r"\bfood\b|\bration\w*\b|\bhungry\b|\bmeals?\b|\bbaby food\b|\bmilk\b",
    "Shelter": r"\bshelter\b|\bblankets?\b|\btents?\b|\bhomeless\b|\bovercrowded\b|\bclothes\b|\btarpaulin\w*\b",
    "Power/Utilities": r"\bpower cut\b|\bno (?:power|electricity)\b|\btransformer\b|\bnetwork (?:is )?down\b|"
                       r"\bmobile network\b|\boutage\b|\bfallen (?:live )?wire\b|\bsupply suspended\b",
    "Infrastructure": r"\bcollapsed?\b|\bcaved in\b|\bbridge (?:damaged|collapsed)\b|\broad (?:blocked|damaged)\b|"
                      r"\buprooted\b|\bblocked\b|\bdamaged\b|\bbreach\w*\b",
    "Volunteers/Donations": r"\bvolunteers?\b|\bdonat\w+\b|\bcollecting\b|\brelief fund\b",
}
# order = how life-critical the need is; first match becomes the primary need
NEED_PRIORITY = list(NEED_PATTERNS)
_NEED_RES = {k: re.compile(v, re.IGNORECASE) for k, v in NEED_PATTERNS.items()}


def extract_needs(text: str) -> list[str]:
    text = clean_text(text)
    return [need for need in NEED_PRIORITY if _NEED_RES[need].search(text)]


# ---------------------------------------------------------------- facade
@dataclass
class Entities:
    locations: list[str] = field(default_factory=list)
    people_count: int | None = None
    needs: list[str] = field(default_factory=list)

    @property
    def primary_location(self) -> str:
        return self.locations[0] if self.locations else ""

    @property
    def primary_need(self) -> str:
        return self.needs[0] if self.needs else ""


def extract_entities(text: str) -> Entities:
    locs = extract_locations(text)
    # prefer a specific landmark ("XYZ bridge") over a city as the primary location
    suffix_re = re.compile(rf"\b(?:{LANDMARK_SUFFIXES})$", re.IGNORECASE)
    locs.sort(key=lambda l: 0 if suffix_re.search(l) else 1)
    return Entities(locations=locs, people_count=extract_people_count(text), needs=extract_needs(text))


def spacy_available() -> bool:
    return _spacy_nlp() is not None
