"""Optional live feeds for the RAG index: GDACS, NDMA SACHET, Google News, Bluesky, ReliefWeb, Reddit (PRAW)
and generic RSS/Atom.

Every source returns (reports, SourceStatus) and never raises. A failed source produces a clear
status message and an empty list, so the rest of the pipeline keeps running.

Known limitations (as of Oct 2026):
- GDACS: public RSS, no key, but it times out from some networks (including some campus networks).
- ReliefWeb API v2: every request needs a pre-approved `appname` (since 1 Nov 2025). Without one the
  API rejects the call, so this source is skipped until RELIEFWEB_APPNAME is set.
- Reddit: needs PRAW plus a registered app. Since Nov 2025, new Reddit API apps go through an approval
  step (Responsible Builder Policy), so credentials may not be instant.
- NDMA / IMD: read through NDMA's SACHET CAP alert feed (public RSS, no key), which carries IMD, INCOIS
  and state SDMA warnings. About three quarters of the alerts are in regional languages (mostly Telugu
  and Hindi). The NLP stages are English-only, so those are translated with the LLM (src/ingestion/
  translate.py) and tagged "[translated from <language>]"; without an LLM key they are skipped.
- Google News: public RSS search (no key), one request per query in GOOGLE_NEWS_QUERIES. Headlines only;
  the item description is just a link back to Google News.
- Bluesky: public post search (no key) on api.bsky.app; public.api.bsky.app answers 403 to searches. If
  unauthenticated search is refused, it logs in with BLUESKY_HANDLE + BLUESKY_APP_PASSWORD when set.
  Keyword search is noisy ("landslide victory"), so posts must be English, long enough and mention a
  disaster term.
- Indian newspapers: The Hindu, NDTV, Times of India and Indian Express national RSS (no key). Most
  stories are not about disasters, so only those that mention a disaster term are kept.
- Mastodon: public hashtag timelines on MASTODON_INSTANCE (no key). Search needs a login; tag timelines
  do not. Same English / disaster-term filter as Bluesky, plus posts older than 7 days are dropped.
- NASA EONET and EMSC: structured event feeds (no key) limited to REGION_BBOX; each event becomes one
  short sentence. US "prescribed fire" entries are skipped as planned burns.
- PIB press releases (Hindi titles only), ReliefWeb RSS (bot wall) and Copernicus EMS (feed removed)
  were tried in Oct 2026 and left out.
- X/Twitter: excluded, the API is not free.
"""

from __future__ import annotations

import html
import re
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

import requests

from src.config import (
    BLUESKY_APP_PASSWORD,
    BLUESKY_HANDLE,
    BLUESKY_QUERIES,
    EMSC_MIN_MAGNITUDE,
    GOOGLE_NEWS_COUNTRY,
    GOOGLE_NEWS_QUERIES,
    HTTP_TIMEOUT,
    INDIA_NEWS_FEEDS,
    MASTODON_INSTANCE,
    MASTODON_TAGS,
    NDMA_FEED_URL,
    REGION_BBOX,
    REDDIT_CLIENT_ID,
    REDDIT_CLIENT_SECRET,
    REDDIT_SUBREDDITS,
    REDDIT_USER_AGENT,
    RELIEFWEB_APPNAME,
    RSS_FEEDS,
    get_logger,
)
from src.ingestion.translate import translate_to_english
from src.schema import Report, make_id, utc_now_iso

log = get_logger("ingestion.live")

HEADERS = {"User-Agent": "CrisisBridgeAI/0.1 (student research project)"}
DISASTER_KEYWORDS = re.compile(
    r"\b(flood|flooding|earthquake|quake|cyclone|hurricane|typhoon|storm|landslide|tsunami|wildfire|fire|"
    r"rescue|trapped|stranded|evacuat\w*|relief|disaster|emergency|injured|dead|missing|collapse\w*|"
    r"waterlog\w*|power cut|outage|red alert|orange alert|cloudburst|glacial lake|GLOF|heavy rain\w*|"
    r"inundat\w*|submerged|avalanche|heat ?wave)\b",
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


# ---------------------------------------------------------------- NDMA SACHET (IMD / INCOIS / SDMA alerts)
# Function words that show up in English alerts but not in Latin-script regional ones (e.g. Mizo)
ENGLISH_WORDS = re.compile(
    r"\b(the|is|are|likely|will|over|with|and|of|in|to|for|at|from|next|hours|places|rain|warning|alert)\b",
    re.IGNORECASE,
)


def _is_english(text: str) -> bool:
    if not text or sum(c.isascii() for c in text) / len(text) < 0.9:
        return False
    return len({m.lower() for m in ENGLISH_WORDS.findall(text)}) >= 3


def fetch_ndma(limit: int = 100) -> tuple[list[Report], SourceStatus]:
    name = "NDMA SACHET"
    try:
        resp = requests.get(NDMA_FEED_URL, timeout=HTTP_TIMEOUT, headers=HEADERS)
        resp.raise_for_status()
        items = ET.fromstring(resp.content).findall(".//item")
    except ET.ParseError as exc:
        return [], SourceStatus(name, False, 0, f"response was not valid XML ({exc}).")
    except Exception as exc:
        return [], SourceStatus(name, False, 0, _network_error_message(name, exc))
    alerts = []
    for item in items:
        text = re.sub(r"[\s.]*\.\.$", ".", _clean(item.findtext("title", "")))  # titles end in ". .." when truncated
        # author looks like "controlroom@ndma.gov.in (IMD Raipur)"; keep the issuing agency
        issuer = re.search(r"\(([^)]+)\)", item.findtext("author", "") or "")
        alerts.append((text, f"ndma/{issuer.group(1)}" if issuer else "ndma",
                       _to_iso(item.findtext("pubDate", "")), item.findtext("link", "") or ""))
    regional = [a[0] for a in alerts if a[0] and not _is_english(a[0])]
    translations, note = translate_to_english(regional) if regional else ({}, "")
    found: dict[str, Report] = {}
    english = skipped = 0
    languages: Counter = Counter()
    for text, source, timestamp, url in alerts:
        if not text:
            continue
        if _is_english(text):
            body = text
            english += 1
        elif text in translations:
            translated, language = translations[text]
            body = f"{translated} [translated from {language}]"
            languages[language] += 1
        else:
            skipped += 1
            continue
        # the id comes from the original wording, so the same alert keeps its id whether or not it was translated
        found.setdefault(make_id("ndma", text), Report(id=make_id("ndma", text), text=body[:1500], source=source,
                                                       timestamp=timestamp, url=url))
        if len(found) >= limit:
            break
    reports = list(found.values())
    msg = f"fetched {english} English alerts"
    if languages:
        msg += f" and translated {sum(languages.values())} ({', '.join(f'{n} {lang}' for lang, n in languages.most_common())})"
    if skipped:
        msg += f"; skipped {skipped} in regional languages" + (f" ({note})" if note else "")
    msg += f"; {len(reports)} unique alerts"
    return reports, SourceStatus(name, bool(reports), len(reports), msg)


# ---------------------------------------------------------------- Google News (RSS search)
def _strip_publisher(title: str, publisher: str) -> str:
    """Google News titles end in ' - Publisher'; drop that tail so only the headline is classified."""
    if publisher and " - " in title:
        head, tail = title.rsplit(" - ", 1)
        if tail.strip() and (publisher.startswith(tail.strip()) or tail.strip().startswith(publisher)):
            return head.strip()
    return title


def fetch_google_news(limit: int = 40, queries: list[str] | None = None) -> tuple[list[Report], SourceStatus]:
    name = "Google News"
    queries = queries if queries is not None else GOOGLE_NEWS_QUERIES
    if not queries:
        return [], SourceStatus(name, False, 0, "skipped: no queries configured (GOOGLE_NEWS_QUERIES in .env).")
    country = GOOGLE_NEWS_COUNTRY.upper()
    per_query = max(5, limit // len(queries))
    found: dict[str, Report] = {}
    problems = []
    for query in queries:
        params = {"q": f"{query} when:2d", "hl": f"en-{country}", "gl": country, "ceid": f"{country}:en"}
        try:
            resp = requests.get("https://news.google.com/rss/search", params=params, timeout=HTTP_TIMEOUT, headers=HEADERS)
            resp.raise_for_status()
            items = ET.fromstring(resp.content).findall(".//item")
        except ET.ParseError:
            problems.append(f"'{query}': response was not valid RSS")
            continue
        except Exception as exc:
            problems.append(f"'{query}': {_network_error_message(name, exc)}")
            continue
        for item in items[:per_query]:
            publisher = _clean(item.findtext("source", ""))
            text = _strip_publisher(_clean(item.findtext("title", "")), publisher)
            if not text:
                continue
            report = Report(
                id=make_id("gnews", text), text=text[:1500], source=f"news/{publisher}" if publisher else "news",
                timestamp=_to_iso(item.findtext("pubDate", "")), url=item.findtext("link", "") or "",
            )
            found.setdefault(report.id, report)  # the same headline often comes back for several queries
    reports = list(found.values())[:limit]
    msg = f"fetched {len(reports)} headlines for {len(queries) - len(problems)}/{len(queries)} queries"
    if problems:
        msg += ". Problems: " + " | ".join(problems)
    return reports, SourceStatus(name, bool(reports), len(reports), msg)


# ---------------------------------------------------------------- Bluesky (public post search)
BLUESKY_APPVIEWS = ("https://api.bsky.app", "https://public.api.bsky.app")


def _looks_english(text: str) -> bool:
    """Posts are often emoji- and hashtag-heavy, so judge the letters only, and ask for fewer function words than NDMA."""
    letters = [c for c in text if c.isalpha()]
    if not letters or sum(c.isascii() for c in letters) / len(letters) < 0.9:
        return False
    return len({m.lower() for m in ENGLISH_WORDS.findall(text)}) >= 2


def _bluesky_login() -> str:
    resp = requests.post(
        "https://bsky.social/xrpc/com.atproto.server.createSession",
        json={"identifier": BLUESKY_HANDLE, "password": BLUESKY_APP_PASSWORD}, timeout=HTTP_TIMEOUT, headers=HEADERS,
    )
    resp.raise_for_status()
    return resp.json()["accessJwt"]


def _bluesky_search(query: str, limit: int, token: str | None) -> list[dict]:
    params = {"q": query, "limit": min(100, limit), "lang": "en", "sort": "latest"}
    if token:
        resp = requests.get("https://bsky.social/xrpc/app.bsky.feed.searchPosts", params=params, timeout=HTTP_TIMEOUT,
                            headers={**HEADERS, "Authorization": f"Bearer {token}"})
        resp.raise_for_status()
        return resp.json().get("posts", [])
    refused = None
    for base in BLUESKY_APPVIEWS:
        resp = requests.get(f"{base}/xrpc/app.bsky.feed.searchPosts", params=params, timeout=HTTP_TIMEOUT, headers=HEADERS)
        if resp.status_code in (401, 403):
            refused = resp.status_code
            continue
        resp.raise_for_status()
        return resp.json().get("posts", [])
    raise PermissionError(f"HTTP {refused}: search without login was refused")


def fetch_bluesky(limit: int = 40, queries: list[str] | None = None) -> tuple[list[Report], SourceStatus]:
    name = "Bluesky"
    queries = queries if queries is not None else BLUESKY_QUERIES
    if not queries:
        return [], SourceStatus(name, False, 0, "skipped: no queries configured (BLUESKY_QUERIES in .env).")
    per_query = max(5, limit // len(queries))
    token: str | None = None
    found: dict[str, Report] = {}
    skipped = 0
    problems = []
    for query in queries:
        try:
            try:
                posts = _bluesky_search(query, per_query * 3, token)
            except PermissionError:
                if token or not (BLUESKY_HANDLE and BLUESKY_APP_PASSWORD):
                    raise
                token = _bluesky_login()
                posts = _bluesky_search(query, per_query * 3, token)
        except PermissionError as exc:
            problems.append(f"'{query}': {exc}; set BLUESKY_HANDLE and BLUESKY_APP_PASSWORD in .env")
            continue
        except Exception as exc:
            problems.append(f"'{query}': {_network_error_message(name, exc)}")
            continue
        kept = 0
        for post in posts:
            record = post.get("record", {})
            text = _clean(record.get("text", ""))
            # keyword search also returns "landslide victory" and Wordle scores; keep real English disaster posts
            if len(text) < 40 or not DISASTER_KEYWORDS.search(text) or not _looks_english(text):
                skipped += 1
                continue
            handle = post.get("author", {}).get("handle", "unknown")
            rkey = post.get("uri", "").rsplit("/", 1)[-1]
            report = Report(
                id=f"bluesky-{rkey}", text=text[:1500], source=f"bluesky/@{handle}",
                timestamp=record.get("createdAt") or post.get("indexedAt") or utc_now_iso(),
                url=f"https://bsky.app/profile/{handle}/post/{rkey}",
            )
            if report.id not in found:
                found[report.id] = report
                kept += 1
            if kept >= per_query:
                break
    reports = list(found.values())[:limit]
    msg = f"fetched {len(reports)} English disaster posts (skipped {skipped} off-topic or non-English)"
    if problems:
        msg += ". Problems: " + " | ".join(problems)
    return reports, SourceStatus(name, bool(reports), len(reports), msg)


# ---------------------------------------------------------------- Indian newspapers (RSS)
NEWSPAPERS = {"thehindu": "The Hindu", "ndtv": "NDTV", "timesofindia": "The Times of India", "indianexpress": "Indian Express"}


def _outlet(feed_url: str) -> str:
    return next((name for key, name in NEWSPAPERS.items() if key in feed_url), re.sub(r"^https?://(www\.)?", "", feed_url).split("/")[0])


def fetch_india_news(limit: int = 40, feeds: list[str] | None = None) -> tuple[list[Report], SourceStatus]:
    name = "Indian newspapers"
    feeds = feeds if feeds is not None else INDIA_NEWS_FEEDS
    if not feeds:
        return [], SourceStatus(name, False, 0, "skipped: no feeds configured (INDIA_NEWS_FEEDS in .env).")
    found: dict[str, Report] = {}
    skipped = 0
    problems = []
    for feed in feeds:
        outlet = _outlet(feed)
        try:
            resp = requests.get(feed, timeout=HTTP_TIMEOUT, headers=HEADERS)
            resp.raise_for_status()
            items = ET.fromstring(resp.content).findall(".//item")
        except ET.ParseError:
            problems.append(f"{outlet}: not a valid RSS feed")
            continue
        except Exception as exc:
            problems.append(f"{outlet}: {_network_error_message(outlet, exc)}")
            continue
        for item in items:
            title = _clean(item.findtext("title", ""))
            desc = _clean(item.findtext("description", ""))
            text = f"{title}. {desc}" if desc and desc not in title else title
            if not title or not DISASTER_KEYWORDS.search(text):
                skipped += 1
                continue
            report = Report(
                id=make_id("india_news", title), text=text[:1500], source=f"news/{outlet}",
                timestamp=_to_iso(item.findtext("pubDate", "")), url=item.findtext("link", "") or "",
            )
            found.setdefault(report.id, report)
    reports = list(found.values())[:limit]
    msg = f"fetched {len(reports)} disaster stories from {len(feeds) - len(problems)}/{len(feeds)} newspapers (skipped {skipped} unrelated)"
    if problems:
        msg += ". Problems: " + " | ".join(problems)
    return reports, SourceStatus(name, bool(reports), len(reports), msg)


# ---------------------------------------------------------------- Mastodon (public hashtag timelines)
def fetch_mastodon(limit: int = 40, tags: list[str] | None = None, max_age_days: int = 7) -> tuple[list[Report], SourceStatus]:
    name = "Mastodon"
    tags = tags if tags is not None else MASTODON_TAGS
    if not tags:
        return [], SourceStatus(name, False, 0, "skipped: no hashtags configured (MASTODON_TAGS in .env).")
    cutoff = datetime.now(timezone.utc).timestamp() - max_age_days * 86400
    found: dict[str, Report] = {}
    skipped = 0
    problems = []
    for tag in tags:
        try:
            resp = requests.get(f"{MASTODON_INSTANCE}/api/v1/timelines/tag/{tag}", params={"limit": 40},
                                timeout=HTTP_TIMEOUT, headers=HEADERS)
            resp.raise_for_status()
            statuses = resp.json()
        except Exception as exc:
            problems.append(f"#{tag}: {_network_error_message(name, exc)}")
            continue
        for status in statuses:
            text = re.sub(r"#\s+(\w)", r"#\1", _clean(status.get("content", "")))  # hashtags arrive as "#<span>tag</span>"
            created = status.get("created_at", "")
            try:
                fresh = datetime.fromisoformat(created.replace("Z", "+00:00")).timestamp() >= cutoff
            except ValueError:
                fresh = False
            if (status.get("language") not in (None, "en") or not fresh or len(text) < 40
                    or not DISASTER_KEYWORDS.search(text) or not _looks_english(text)):
                skipped += 1
                continue
            report = Report(
                id=f"mastodon-{status['id']}", text=text[:1500],
                source=f"mastodon/@{status.get('account', {}).get('acct', 'unknown')}",
                timestamp=created, url=status.get("url") or status.get("uri", ""),
            )
            found.setdefault(report.id, report)
    reports = sorted(found.values(), key=lambda r: r.timestamp, reverse=True)[:limit]
    msg = f"fetched {len(reports)} English disaster posts from {len(tags) - len(problems)}/{len(tags)} hashtags (skipped {skipped} old, off-topic or non-English)"
    if problems:
        msg += ". Problems: " + " | ".join(problems)
    return reports, SourceStatus(name, bool(reports), len(reports), msg)


# ---------------------------------------------------------------- NASA EONET (natural events)
def _latlon(lat: float, lon: float) -> str:
    return f"{abs(lat):.1f}°{'N' if lat >= 0 else 'S'}, {abs(lon):.1f}°{'E' if lon >= 0 else 'W'}"


def fetch_eonet(limit: int = 30, days: int = 30) -> tuple[list[Report], SourceStatus]:
    name = "NASA EONET"
    params: dict = {"status": "open", "days": days, "limit": limit * 2}
    if len(REGION_BBOX) == 4:
        min_lon, min_lat, max_lon, max_lat = REGION_BBOX
        params["bbox"] = f"{min_lon},{max_lat},{max_lon},{min_lat}"  # EONET wants upper-left then lower-right
    try:
        resp = requests.get("https://eonet.gsfc.nasa.gov/api/v3/events", params=params, timeout=HTTP_TIMEOUT, headers=HEADERS)
        resp.raise_for_status()
        events = resp.json().get("events", [])
    except Exception as exc:
        return [], SourceStatus(name, False, 0, _network_error_message(name, exc))
    reports: list[Report] = []
    for event in events:
        title = event.get("title", "")
        if not title or title.lower().startswith("prescribed fire"):  # planned burns, not emergencies
            continue
        geometry = (event.get("geometry") or [{}])[-1]
        when = geometry.get("date", "")
        coords = geometry.get("coordinates")
        where = f" near {_latlon(coords[1], coords[0])}" if geometry.get("type") == "Point" and coords else ""
        try:
            day = datetime.fromisoformat(when.replace("Z", "+00:00")).strftime("%d %b %Y")
        except ValueError:
            day = when[:10]
        kinds = ", ".join(c.get("title", "") for c in event.get("categories", []))
        sources = event.get("sources") or [{}]
        text = f"{title}: {kinds.lower() or 'natural event'} tracked by NASA EONET, last observed{where} on {day}."
        reports.append(Report(
            id=f"eonet-{event.get('id')}", text=text, source=f"eonet/{sources[0].get('id', 'nasa')}",
            timestamp=when or utc_now_iso(), url=sources[0].get("url", "") or event.get("link", ""),
        ))
        if len(reports) >= limit:
            break
    area = "in the region" if len(REGION_BBOX) == 4 else "worldwide"
    return reports, SourceStatus(name, True, len(reports), f"fetched {len(reports)} open natural events {area}")


# ---------------------------------------------------------------- EMSC (earthquakes)
def fetch_emsc(limit: int = 30, days: int = 7) -> tuple[list[Report], SourceStatus]:
    name = "EMSC earthquakes"
    start = datetime.fromtimestamp(datetime.now(timezone.utc).timestamp() - days * 86400, tz=timezone.utc)
    params: dict = {"format": "json", "limit": limit, "minmag": EMSC_MIN_MAGNITUDE, "orderby": "time",
                    "start": start.strftime("%Y-%m-%dT%H:%M:%S")}
    if len(REGION_BBOX) == 4:
        params.update(dict(zip(("minlon", "minlat", "maxlon", "maxlat"), REGION_BBOX)))
    try:
        resp = requests.get("https://www.seismicportal.eu/fdsnws/event/1/query", params=params, timeout=HTTP_TIMEOUT, headers=HEADERS)
        if resp.status_code == 204:  # FDSN answers 204 when nothing matched
            return [], SourceStatus(name, True, 0, f"no earthquakes of M{EMSC_MIN_MAGNITUDE:g}+ in the last {days} days")
        resp.raise_for_status()
        features = resp.json().get("features", [])
    except Exception as exc:
        return [], SourceStatus(name, False, 0, _network_error_message(name, exc))
    reports: list[Report] = []
    for feature in features:
        p = feature.get("properties", {})
        try:
            when = datetime.fromisoformat(p["time"].replace("Z", "+00:00"))
            text = (f"M{float(p['mag']):.1f} earthquake in {p.get('flynn_region', 'unknown region').title()}, "
                    f"depth {float(p.get('depth') or 0):.0f} km, at {when:%d %b %Y %H:%M} UTC (reported by EMSC).")
        except (KeyError, TypeError, ValueError):
            continue
        reports.append(Report(
            id=f"emsc-{p.get('unid')}", text=text, source="emsc", timestamp=p["time"],
            url=f"https://www.seismicportal.eu/eventdetails.html?unid={p.get('unid')}",
        ))
    return reports, SourceStatus(name, True, len(reports), f"fetched {len(reports)} earthquakes of M{EMSC_MIN_MAGNITUDE:g}+ from the last {days} days")


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
    "ndma": fetch_ndma,
    "gnews": fetch_google_news,
    "india_news": fetch_india_news,
    "bluesky": fetch_bluesky,
    "mastodon": fetch_mastodon,
    "eonet": fetch_eonet,
    "emsc": fetch_emsc,
    "reliefweb": fetch_reliefweb,
    "reddit": fetch_reddit,
    "rss": fetch_rss,
}
# Pre-selected in the interfaces: every source that works without a key or an approval
DEFAULT_LIVE_SOURCES = ("gdacs", "ndma", "gnews", "india_news", "bluesky", "mastodon", "eonet", "emsc", "rss")


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
