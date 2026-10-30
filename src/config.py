"""Central configuration. Values come from environment variables (.env) with safe defaults."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # python-dotenv is in requirements, but never crash on config import
    load_dotenv = None

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if load_dotenv is not None:
    load_dotenv(PROJECT_ROOT / ".env")


def _env(name: str, default: str = "") -> str:
    value = os.getenv(name, default)
    return value.strip() if isinstance(value, str) else default


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env(name, str(default)))
    except ValueError:
        return default


# ---------------------------------------------------------------- paths
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
INDEX_DIR = DATA_DIR / "index"
MODEL_DIR = DATA_DIR / "models"
SAMPLE_DATA_PATH = PROJECT_ROOT / "sample_data" / "crisis_messages.csv"
_kaggle = Path(_env("KAGGLE_DISASTER_CSV", "data/kaggle/train.csv"))
KAGGLE_DISASTER_CSV = str(_kaggle if _kaggle.is_absolute() else PROJECT_ROOT / _kaggle)

for _d in (DATA_DIR, CACHE_DIR, INDEX_DIR, MODEL_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- datasets
HUMAID_REPO = "QCRI/HumAID-all"
CRISISBENCH_REPO = "QCRI/CrisisBench-english"
IDRISI_ZIP_URL = "https://github.com/rsuwaileh/IDRISI/archive/refs/heads/main.zip"
# Weight of each CrisisBench tweet relative to a HumAID tweet in the humaid+crisisbench training mode
CRISISBENCH_WEIGHT = _env_float("CRISISBENCH_WEIGHT", 0.3)
HTTP_TIMEOUT = _env_int("HTTP_TIMEOUT", 15)

# ---------------------------------------------------------------- embeddings / index
# auto: sentence-transformers if installed and the model loads, otherwise TF-IDF + SVD (offline)
EMBEDDING_BACKEND = _env("EMBEDDING_BACKEND", "auto").lower()
SENTENCE_MODEL = _env("SENTENCE_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
# auto: FAISS if installed, otherwise a NumPy cosine-similarity store
VECTOR_STORE = _env("VECTOR_STORE", "auto").lower()

# ---------------------------------------------------------------- generator
# auto: Groq if GROQ_API_KEY is set, else Gemini if GEMINI_API_KEY is set, else extractive
LLM_PROVIDER = _env("LLM_PROVIDER", "auto").lower()
GROQ_API_KEY = _env("GROQ_API_KEY")
GROQ_MODEL = _env("GROQ_MODEL", "openai/gpt-oss-20b")
GEMINI_API_KEY = _env("GEMINI_API_KEY")
GEMINI_MODEL = _env("GEMINI_MODEL", "gemini-3.5-flash")
LLM_TIMEOUT = _env_int("LLM_TIMEOUT", 60)
LLM_TEMPERATURE = _env_float("LLM_TEMPERATURE", 0.1)

# ---------------------------------------------------------------- live sources
RELIEFWEB_APPNAME = _env("RELIEFWEB_APPNAME")
# NDMA SACHET: official CAP alerts from IMD, INCOIS and state disaster authorities (public, no key)
NDMA_FEED_URL = _env("NDMA_FEED_URL", "https://sachet.ndma.gov.in/cap_public_website/rss/rss_india.xml")
# Regional-language SACHET alerts are translated to English with the LLM: auto (Groq, else Gemini) | groq | gemini | off
TRANSLATE_REGIONAL = _env("TRANSLATE_REGIONAL", "auto").lower()
# Google News RSS search (public, no key): one request per query, English edition of GOOGLE_NEWS_COUNTRY
GOOGLE_NEWS_QUERIES = [q.strip() for q in _env(
    "GOOGLE_NEWS_QUERIES", "flood India,cyclone India,landslide India,earthquake India,heavy rain alert India",
).split(",") if q.strip()]
GOOGLE_NEWS_COUNTRY = _env("GOOGLE_NEWS_COUNTRY", "IN")
# Bluesky public post search (no key); the login is only used if search without one is refused
BLUESKY_QUERIES = [q.strip() for q in _env(
    "BLUESKY_QUERIES", "flood India,cyclone India,landslide India,earthquake India,trapped rescue flood",
).split(",") if q.strip()]
BLUESKY_HANDLE = _env("BLUESKY_HANDLE")
BLUESKY_APP_PASSWORD = _env("BLUESKY_APP_PASSWORD")
# Indian newspaper RSS (public, no key); only stories that mention a disaster term are kept
INDIA_NEWS_FEEDS = [u.strip() for u in _env(
    "INDIA_NEWS_FEEDS",
    "https://www.thehindu.com/news/national/feeder/default.rss,"
    "https://feeds.feedburner.com/ndtvnews-india-news,"
    "https://timesofindia.indiatimes.com/rssfeeds/-2128936835.cms,"
    "https://indianexpress.com/section/india/feed/",
).split(",") if u.strip()]
# Mastodon public hashtag timelines (no key on mastodon.social); search itself needs a login, tags do not
MASTODON_INSTANCE = _env("MASTODON_INSTANCE", "https://mastodon.social").rstrip("/")
MASTODON_TAGS = [t.strip().lstrip("#") for t in _env("MASTODON_TAGS", "flood,floods,flooding,cyclone,earthquake,landslide").split(",") if t.strip()]
# Area for NASA EONET events and EMSC earthquakes, as min_lon,min_lat,max_lon,max_lat (India and neighbours).
# Leave empty for worldwide.
REGION_BBOX = [float(v) for v in _env("REGION_BBOX", "60,0,100,40").split(",") if v.strip()]
EMSC_MIN_MAGNITUDE = _env_float("EMSC_MIN_MAGNITUDE", 3.0)
REDDIT_CLIENT_ID = _env("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = _env("REDDIT_CLIENT_SECRET")
REDDIT_USER_AGENT = _env("REDDIT_USER_AGENT", "CrisisBridgeAI/0.1 (student project)")
REDDIT_SUBREDDITS = [s.strip() for s in _env("REDDIT_SUBREDDITS", "india+chennai+bangalore+mumbai+naturaldisasters+TropicalWeather").split(",") if s.strip()]
RSS_FEEDS = [u.strip() for u in _env(
    "RSS_FEEDS",
    "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/4.5_week.atom",
).split(",") if u.strip()]


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(_env("LOG_LEVEL", "INFO").upper())
        logger.propagate = False
    return logger
