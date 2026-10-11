"""Grounded answer generation with citations.

Providers (plain HTTPS via requests, no SDKs):
- Groq   : OpenAI-compatible chat completions. Default model openai/gpt-oss-20b
           (Groq retired llama-3.1-8b-instant / llama-3.3-70b-versatile on 16 Aug 2026).
- Gemini : generateContent REST API. Default model gemini-2.5-flash.
- Extractive: no LLM. Builds a cited answer straight from the retrieved reports. Always available
              and used automatically when no key is set or the API call fails.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import requests

from src.config import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    GROQ_API_KEY,
    GROQ_MODEL,
    LLM_PROVIDER,
    LLM_TEMPERATURE,
    LLM_TIMEOUT,
    get_logger,
)
from src.rag.retriever import QueryPlan, RetrievedReport

log = get_logger("rag.generator")

SYSTEM_PROMPT = (
    "You are CrisisBridge, a disaster-response operations assistant for an incident commander. "
    "Answer ONLY from the numbered field reports provided. Every factual sentence must cite its "
    "report(s) like [1] or [2][3]. If the reports do not answer the question, say so plainly. "
    "Lead with the most critical, life-threatening situation. State location, need, number of "
    "people and priority when the reports give them. Do not invent places, numbers or times. "
    "Be brief: at most 6 sentences or bullet points. Plain text, no markdown headers."
)


@dataclass
class Answer:
    text: str
    provider: str
    citations: list[RetrievedReport] = field(default_factory=list)
    notice: str = ""  # e.g. why the extractive fallback was used

    def render(self) -> str:
        lines = [self.text.strip(), "", "Sources:"]
        for n, item in enumerate(self.citations, 1):
            r = item.report
            lines.append(f"[{n}] {r.id} | {r.source} | {r.timestamp} | {r.summary_line()}")
            lines.append(f"     \"{r.text[:220]}\"" + (f" {r.url}" if r.url else ""))
        if self.notice:
            lines += ["", f"Note: {self.notice}"]
        lines.append(f"(answer by: {self.provider})")
        return "\n".join(lines)


def build_context(results: list[RetrievedReport]) -> str:
    blocks = []
    for n, item in enumerate(results, 1):
        r = item.report
        blocks.append(
            f"[{n}] time={r.timestamp} source={r.source} category={r.category}\n"
            f"    structured: {r.summary_line()} | urgency={r.urgency_score:.0f}/100\n"
            f"    text: {r.text[:600]}"
        )
    return "\n".join(blocks)


def build_user_prompt(query: str, results: list[RetrievedReport]) -> str:
    return f"Field reports:\n{build_context(results)}\n\nCommander question: {query}\n\nAnswer with citations:"


# ---------------------------------------------------------------- providers
def _call_groq(prompt: str) -> str:
    payload = {
        "model": GROQ_MODEL,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
        "temperature": LLM_TEMPERATURE,
        "max_completion_tokens": 1024,
    }
    if GROQ_MODEL.startswith("openai/gpt-oss"):
        payload["reasoning_effort"] = "low"
    resp = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
        json=payload,
        timeout=LLM_TIMEOUT,
    )
    if resp.status_code >= 400:
        raise RuntimeError(f"Groq HTTP {resp.status_code}: {resp.text[:300]}")
    content = resp.json()["choices"][0]["message"].get("content") or ""
    if not content.strip():
        raise RuntimeError("Groq returned an empty answer")
    return content


def _call_gemini(prompt: str) -> str:
    payload = {
        "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": LLM_TEMPERATURE, "maxOutputTokens": 1024},
    }
    if GEMINI_MODEL.startswith("gemini-2.5-flash"):
        payload["generationConfig"]["thinkingConfig"] = {"thinkingBudget": 0}
    resp = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent",
        headers={"x-goog-api-key": GEMINI_API_KEY, "Content-Type": "application/json"},
        json=payload,
        timeout=LLM_TIMEOUT,
    )
    if resp.status_code >= 400:
        raise RuntimeError(f"Gemini HTTP {resp.status_code}: {resp.text[:300]}")
    data = resp.json()
    try:
        parts = data["candidates"][0]["content"]["parts"]
    except (KeyError, IndexError):
        raise RuntimeError(f"Gemini returned no text (finish/prompt feedback: {str(data)[:300]})")
    text = "".join(p.get("text", "") for p in parts)
    if not text.strip():
        raise RuntimeError("Gemini returned an empty answer")
    return text


def extractive_answer(query: str, results: list[RetrievedReport], plan: QueryPlan | None = None) -> str:
    if not results:
        return "No indexed reports match this question."
    wanted = set(plan.needs) if plan else set()

    def key(i: int) -> tuple:
        r = results[i].report
        return (bool(wanted & set(r.needs)), r.urgency_score)

    ranked = sorted(range(len(results)), key=key, reverse=True)
    top_i = ranked[0]
    top = results[top_i].report
    where = f" near {', '.join(plan.locations)}" if plan and plan.locations else ""
    about = f" involving {' / '.join(sorted(wanted))}" if wanted and wanted & set(top.needs) else ""
    lines = [
        f"Most critical situation{about}{where}: {top.priority} (urgency {top.urgency_score:.0f}/100) [{top_i + 1}]. "
        f"{top.summary_line()}. Report says: \"{top.text[:200]}\" [{top_i + 1}]"
    ]
    others = [i for i in ranked[1:]
              if results[i].report.priority in ("Critical", "High", "Medium") or wanted & set(results[i].report.needs)][:4]
    for i in others:
        r = results[i].report
        lines.append(f"- {r.priority}: {r.summary_line()} [{i + 1}]")
    if plan and plan.locations and not plan.location_filter_applied:
        lines.append(f"No report explicitly mentions {', '.join(plan.locations)}; these are the closest matches.")
    return "\n".join(lines)


def _resolve_provider(requested: str) -> tuple[str, str]:
    requested = (requested or "auto").lower()
    if requested == "auto":
        if GROQ_API_KEY:
            return "groq", ""
        if GEMINI_API_KEY:
            return "gemini", ""
        return "extractive", "No GROQ_API_KEY or GEMINI_API_KEY in .env, so an extractive (no-LLM) answer was used."
    if requested == "groq" and not GROQ_API_KEY:
        return "extractive", "LLM_PROVIDER=groq but GROQ_API_KEY is empty; used the extractive answer."
    if requested == "gemini" and not GEMINI_API_KEY:
        return "extractive", "LLM_PROVIDER=gemini but GEMINI_API_KEY is empty; used the extractive answer."
    if requested not in ("groq", "gemini", "extractive"):
        return "extractive", f"Unknown LLM_PROVIDER '{requested}'; used the extractive answer."
    return requested, ""


def generate_answer(
    query: str,
    results: list[RetrievedReport],
    plan: QueryPlan | None = None,
    provider: str = LLM_PROVIDER,
) -> Answer:
    if not results:
        return Answer("No indexed reports match this question.", "extractive", [])
    chosen, notice = _resolve_provider(provider)
    if chosen in ("groq", "gemini"):
        try:
            prompt = build_user_prompt(query, results)
            text = _call_groq(prompt) if chosen == "groq" else _call_gemini(prompt)
            model = GROQ_MODEL if chosen == "groq" else GEMINI_MODEL
            if not re.search(r"\[\d+\]", text):
                notice = "The model did not cite sources inline; all retrieved reports are listed below."
            return Answer(text.strip(), f"{chosen}:{model}", results, notice)
        except requests.Timeout:
            notice = f"{chosen} timed out after {LLM_TIMEOUT}s; used the extractive answer."
        except requests.RequestException as exc:
            notice = f"{chosen} unreachable ({type(exc).__name__}); used the extractive answer."
        except Exception as exc:
            notice = f"{chosen} failed ({exc}); used the extractive answer."
        log.warning(notice)
    return Answer(extractive_answer(query, results, plan), "extractive", results, notice)
