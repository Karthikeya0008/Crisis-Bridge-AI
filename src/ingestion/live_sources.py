"""Optional live feeds for the RAG index: GDACS, ReliefWeb, Reddit (PRAW) and generic RSS/Atom.

Every source returns (reports, SourceStatus) and never raises. A failed source produces a clear
status message and an empty list, so the rest of the pipeline keeps running.

Known limitations (as of Oct 2026):
- GDACS: public RSS, no key, but it times out from some networks (including some campus networks).
- ReliefWeb API v2: every request needs a pre-approved `appname` (since 1 Nov 2025). Without one the
  API rejects the call, so this source is skipped until RELIEFWEB_APPNAME is set.
- Reddit: needs PRAW plus a registered app. Since Nov 2025, new Reddit API apps go through an approval
  step (Responsible Builder Policy), so credentials may not be instant.
- NDMA / IMD: no stable public RSS/API feed, so they are deliberately not implemented.
- X/Twitter: excluded, the API is not free.
"""

from __future__ import annotations

import html
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

import requests

from src.config import (
    HTTP_TIMEOUT,
    REDDIT_CLIENT_ID,
    REDDIT_CLIENT_SECRET,
    REDDIT_SUBREDDITS,
    REDDIT_USER_AGENT,
    RELIEFWEB_APPNAME,
    RSS_FEEDS,
    get_logger,
)
from src.schema import Report, make_id, utc_now_iso

log = get_logger("ingestion.live")

HEADERS = {"User-Agent": "CrisisBridgeAI/0.1 (student research project)"}
DISASTER_KEYWORDS = re.compile(
    r"\b(flood|flooding|earthquake|quake|cyclone|hurricane|typhoon|storm|landslide|tsunami|wildfire|fire|"
    r"rescue|trapped|stranded|evacuat\w*|relief|disaster|emergency|injured|dead|missing|collapse\w*|"
    r"waterlog\w*|power cut|outage|red alert|orange alert)\b",
    re.IGNORECASE,
)


@dataclass
class SourceStatus:
    name: str
    ok: bool
    count: int
    message: str

    def __str__(self) -> str:
        flag = "OK  " if self.ok else "SKIP"
        return f"[{flag}] {self.name}: {self.message}"


def _clean(text: str) -> str:
    text = html.unescape(text or "")
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _to_iso(date_str: str) -> str:
    if not date_str:
        return utc_now_iso()
    try:
        return parsedate_to_datetime(date_str).isoformat()
    except (TypeError, ValueError):
        return date_str  # Atom dates are already ISO 8601


def _network_error_message(name: str, exc: Exception) -> str:
    if isinstance(exc, requests.Timeout):
        return f"timed out after {HTTP_TIMEOUT}s. The server may be blocked or slow from your network."
    if isinstance(exc, requests.ConnectionError):
        return "could not connect (no internet, DNS failure, or blocked by a firewall/proxy)."
    if isinstance(exc, requests.HTTPError):
        code = exc.response.status_code if exc.response is not None else "?"
        return f"HTTP {code} from {name}."
    return f"{type(exc).__name__}: {exc}"


# ---------------------------------------------------------------- RSS / Atom parsing
def _parse_feed(xml_bytes: bytes, source: str, limit: int) -> list[Report]:
    root = ET.fromstring(xml_bytes)
    atom = "{http://www.w3.org/2005/Atom}"
    reports: list[Report] = []
    items = root.findall(".//item")
    if items:  # RSS 2.0
        for item in items[:limit]:
            title = _clean(item.findtext("title", ""))
            desc = _clean(item.findtext("description", ""))
            text = f"{title}. {desc}".strip(". ") if desc and desc != title else title
            if not text:
                continue
            reports.append(Report(
                id=make_id(source, text), text=text[:1500], source=source,
                timestamp=_to_iso(item.findtext("pubDate", "")), url=item.findtext("link", "") or "",
            ))
    else:  # Atom
        for entry in root.findall(f"{atom}entry")[:limit]:
            title = _clean(entry.findtext(f"{atom}title", ""))
            summary = _clean(entry.findtext(f"{atom}summary", "") or entry.findtext(f"{atom}content", ""))
            text = f"{title}. {summary}".strip(". ") if summary else title
            if not text:
                continue
            link_el = entry.find(f"{atom}link")
            reports.append(Report(
                id=make_id(source, text), text=text[:1500], source=source,
                timestamp=entry.findtext(f"{atom}updated", "") or utc_now_iso(),
                url=link_el.get("href", "") if link_el is not None else "",
            ))
    return reports


# ---------------------------------------------------------------- GDACS
def fetch_gdacs(limit: int = 30) -> tuple[list[Report], SourceStatus]:
    name = "GDACS"
    url = "https://www.gdacs.org/xml/rss.xml"
    try:
        resp = requests.get(url, timeout=HTTP_TIMEOUT, headers=HEADERS)
        resp.raise_for_status()
        reports = _parse_feed(resp.content, "gdacs", limit)
        return reports, SourceStatus(name, True, len(reports), f"fetched {len(reports)} alerts")
    except ET.ParseError as exc:
        return [], SourceStatus(name, False, 0, f"response was not valid XML ({exc}).")
    except Exception as exc:
        msg = _network_error_message(name, exc) + " GDACS is known to time out from some networks."
        return [], SourceStatus(name, False, 0, msg)


# ---------------------------------------------------------------- ReliefWeb
def fetch_reliefweb(limit: int = 20, query: str = "flood OR cyclone OR earthquake OR landslide") -> tuple[list[Report], SourceStatus]:
    name = "ReliefWeb"
    if not RELIEFWEB_APPNAME:
        return [], SourceStatus(
            name, False, 0,
            "skipped: RELIEFWEB_APPNAME is not set. Since 1 Nov 2025 the API only accepts pre-approved "
            "appnames; request one via https://apidoc.reliefweb.int/parameters#appname and put it in .env.",
        )
    url = f"https://api.reliefweb.int/v2/reports?appname={RELIEFWEB_APPNAME}"
    payload = {
        "limit": limit,
        "preset": "latest",
        "profile": "list",
        "query": {"value": query},
        "fields": {"include": ["title", "body", "url", "date.created", "country.name"]},
    }
    try:
        resp = requests.post(url, json=payload, timeout=HTTP_TIMEOUT, headers=HEADERS)
        if resp.status_code in (401, 403):
            return [], SourceStatus(name, False, 0, f"HTTP {resp.status_code}: appname '{RELIEFWEB_APPNAME}' was rejected (not approved yet?).")
        resp.raise_for_status()
        reports = []
        for item in resp.json().get("data", []):
            f = item.get("fields", {})
            countries = ", ".join(c.get("name", "") for c in f.get("country", []))
            body = _clean(f.get("body", ""))[:1200]
            text = f"{f.get('title', '')}. {('Countries: ' + countries + '. ') if countries else ''}{body}".strip()
            reports.append(Report(
                id=f"reliefweb-{item.get('id')}", text=text, source="reliefweb",
                timestamp=f.get("date", {}).get("created", utc_now_iso()), url=f.get("url", ""),
            ))
        return reports, SourceStatus(name, True, len(reports), f"fetched {len(reports)} reports")
    except Exception as exc:
        return [], SourceStatus(name, False, 0, _network_error_message(name, exc))


# ---------------------------------------------------------------- Reddit
def fetch_reddit(limit: int = 50) -> tuple[list[Report], SourceStatus]:
    name = "Reddit"
    try:
        import praw  # optional dependency
    except ImportError:
        return [], SourceStatus(name, False, 0, "skipped: praw is not installed (pip install -r requirements-optional.txt).")
    if not (REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET):
        return [], SourceStatus(
            name, False, 0,
            "skipped: REDDIT_CLIENT_ID / REDDIT_CLIENT_SECRET not set. Create a 'script' app at "
            "https://www.reddit.com/prefs/apps (new apps may need Reddit's approval).",
        )
    try:
        reddit = praw.Reddit(
            client_id=REDDIT_CLIENT_ID, client_secret=REDDIT_CLIENT_SECRET,
            user_agent=REDDIT_USER_AGENT, check_for_async=False,
        )
        reddit.read_only = True
        sub = reddit.subreddit("+".join(REDDIT_SUBREDDITS))
        reports = []
        for post in sub.new(limit=limit * 4):
            text = _clean(f"{post.title}. {post.selftext or ''}")[:1500]
            if not DISASTER_KEYWORDS.search(text):
                continue
            reports.append(Report(
                id=f"reddit-{post.id}", text=text, source=f"reddit/r/{post.subreddit.display_name}",
                timestamp=datetime.fromtimestamp(post.created_utc, tz=timezone.utc).isoformat(),
                url=f"https://www.reddit.com{post.permalink}",
            ))
            if len(reports) >= limit:
                break
        return reports, SourceStatus(name, True, len(reports), f"fetched {len(reports)} disaster-related posts")
    except Exception as exc:
        return [], SourceStatus(name, False, 0, f"failed: {type(exc).__name__}: {exc}")


# ---------------------------------------------------------------- generic RSS
def fetch_rss(feeds: list[str] | None = None, limit_per_feed: int = 20) -> tuple[list[Report], SourceStatus]:
    name = "RSS"
    feeds = feeds if feeds is not None else RSS_FEEDS
    if not feeds:
        return [], SourceStatus(name, False, 0, "skipped: no feeds configured (RSS_FEEDS in .env).")
    reports: list[Report] = []
    problems = []
    for feed in feeds:
        try:
            resp = requests.get(feed, timeout=HTTP_TIMEOUT, headers=HEADERS)
            resp.raise_for_status()
            reports.extend(_parse_feed(resp.content, "rss", limit_per_feed))
        except ET.ParseError:
            problems.append(f"{feed}: not a valid RSS/Atom feed")
        except Exception as exc:
            problems.append(f"{feed}: {_network_error_message('feed', exc)}")
    ok = bool(reports)
    msg = f"fetched {len(reports)} items from {len(feeds) - len(problems)}/{len(feeds)} feeds"
    if problems:
        msg += ". Problems: " + " | ".join(problems)
    return reports, SourceStatus(name, ok, len(reports), msg)


SOURCES = {
    "gdacs": fetch_gdacs,
    "reliefweb": fetch_reliefweb,
    "reddit": fetch_reddit,
    "rss": fetch_rss,
}


def fetch_all(selected: list[str] | None = None) -> tuple[list[Report], list[SourceStatus]]:
    """Run each selected source independently. Returns combined reports and one status per source."""
    selected = selected or list(SOURCES)
    all_reports: list[Report] = []
    statuses: list[SourceStatus] = []
    for key in selected:
        fn = SOURCES.get(key)
        if fn is None:
            statuses.append(SourceStatus(key, False, 0, f"unknown source (choose from {', '.join(SOURCES)})"))
            continue
        reports, status = fn()
        log.debug(str(status))
        all_reports.extend(reports)
        statuses.append(status)
    # de-duplicate by id
    unique = {r.id: r for r in all_reports}
    return list(unique.values()), statuses
