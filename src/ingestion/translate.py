"""Translate regional-language alerts (Telugu, Hindi, Bengali, ...) into English with the configured LLM.

The NLP stages are English-only, so without this about three quarters of NDMA SACHET alerts would be
skipped. Groq is used if GROQ_API_KEY is set, else Gemini; TRANSLATE_REGIONAL=off turns it off.

- Unique texts only: SACHET repeats the same warning for many areas, so each wording is translated once.
- Batches of 20 per request, asking for a JSON array back.
- Translations are cached in data/cache/translations.json, so repeat fetches make no LLM call.
- Never raises: on a missing key or an API error the untranslated alerts are simply left out, with a reason.
"""

from __future__ import annotations

import json
import re

from src.config import CACHE_DIR, GEMINI_API_KEY, GROQ_API_KEY, TRANSLATE_REGIONAL, get_logger

log = get_logger("ingestion.translate")

CACHE_PATH = CACHE_DIR / "translations.json"
BATCH_SIZE = 20
SYSTEM = (
    "You translate official Indian disaster and weather alerts into plain English. Translate faithfully: keep "
    "every place name, district, number, date, time and agency name. Do not add advice or commentary. "
    "Reply with a JSON array only."
)


def in_other_script(text: str) -> bool:
    """True when most letters are outside the Latin alphabet (Tamil, Devanagari, Telugu, Bengali...)."""
    letters = [c for c in text if c.isalpha()]
    return bool(letters) and sum(not c.isascii() for c in letters) / len(letters) > 0.5


def _provider() -> tuple[str, str]:
    """(provider, reason it is unavailable or '')."""
    choice = TRANSLATE_REGIONAL
    if choice == "off":
        return "", "translation is off (TRANSLATE_REGIONAL=off)"
    if choice in ("auto", "groq") and GROQ_API_KEY:
        return "groq", ""
    if choice in ("auto", "gemini") and GEMINI_API_KEY:
        return "gemini", ""
    return "", "no GROQ_API_KEY or GEMINI_API_KEY to translate with"


def _load_cache() -> dict[str, list[str]]:
    try:
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _parse(reply: str, batch: list[str]) -> dict[str, list[str]]:
    match = re.search(r"\[.*\]", reply, re.DOTALL)
    if not match:
        raise ValueError("no JSON array in the reply")
    out = {}
    for item in json.loads(match.group(0)):
        i, english = item.get("i"), str(item.get("english", "")).strip()
        if isinstance(i, int) and 0 <= i < len(batch) and english:
            out[batch[i]] = [english, str(item.get("language", "")).strip() or "a regional language"]
    return out


def translate_to_english(texts: list[str]) -> tuple[dict[str, tuple[str, str]], str]:
    """Map each text to (english, source language). Texts that could not be translated are absent.

    Returns the mapping and a short note on anything that went wrong ('' when everything worked).
    """
    cache = _load_cache()
    todo = list(dict.fromkeys(t for t in texts if t and t not in cache))
    note = ""
    if todo:
        provider, why_not = _provider()
        if not provider:
            note = why_not
        else:
            from src.rag import generator  # imported here: the generator pulls in the retriever and NER

            call = generator._call_groq if provider == "groq" else generator._call_gemini
            for start in range(0, len(todo), BATCH_SIZE):
                batch = todo[start:start + BATCH_SIZE]
                prompt = ("Translate each numbered alert into English. Return a JSON array of objects "
                          '{"i": number, "language": name of the original language, "english": translation}.\n\n'
                          + "\n".join(f"{n}. {t}" for n, t in enumerate(batch)))
                try:
                    cache.update(_parse(call(prompt, system=SYSTEM, max_tokens=4096), batch))
                except Exception as exc:  # network, HTTP, quota or a malformed reply: keep what we have
                    note = f"{provider} translation stopped ({type(exc).__name__}: {str(exc)[:120]})"
                    log.warning(note)
                    break
            try:
                CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
            except OSError as exc:
                log.warning("Could not save the translation cache: %s", exc)
    return {t: tuple(cache[t]) for t in texts if t in cache}, note
