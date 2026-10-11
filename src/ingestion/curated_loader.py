"""Curated datasets: bundled sample data, HumAID (Hugging Face) and the optional Kaggle disaster tweets CSV.

HumAID is fetched as parquet over plain HTTPS (no `datasets` dependency) and cached in data/cache.
If `datasets` happens to be installed it is used as a last resort.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

from src.config import CACHE_DIR, HTTP_TIMEOUT, HUMAID_REPO, KAGGLE_DISASTER_CSV, SAMPLE_DATA_PATH, get_logger
from src.schema import Report

log = get_logger("ingestion.curated")

HUMAID_SPLITS = ("train", "validation", "test")


class DataUnavailable(RuntimeError):
    """Raised when a dataset cannot be loaded; the message says why and what to do."""


# ---------------------------------------------------------------- sample data
def load_sample_data(path: Path = SAMPLE_DATA_PATH) -> pd.DataFrame:
    """Bundled hand-written messages. Columns: id, timestamp, source, text, label."""
    if not path.exists():
        raise DataUnavailable(f"Sample data not found at {path}. Re-extract the project zip.")
    df = pd.read_csv(path, encoding="utf-8")
    missing = {"id", "timestamp", "source", "text", "label"} - set(df.columns)
    if missing:
        raise DataUnavailable(f"Sample data is missing columns: {sorted(missing)}")
    return df


# ---------------------------------------------------------------- HumAID
def _humaid_cache_path(split: str) -> Path:
    return CACHE_DIR / f"humaid_{split}.parquet"


def _parquet_urls(split: str) -> list[str]:
    api = f"https://huggingface.co/api/datasets/{HUMAID_REPO}/parquet/default/{split}"
    try:
        resp = requests.get(api, timeout=HTTP_TIMEOUT)
        resp.raise_for_status()
        urls = resp.json()
        if isinstance(urls, list) and urls:
            return [str(u) for u in urls]
    except (requests.RequestException, ValueError) as exc:
        log.warning("HF parquet listing failed (%s); trying the direct file URL", exc)
    return [f"https://huggingface.co/datasets/{HUMAID_REPO}/resolve/refs%2Fconvert%2Fparquet/default/{split}/0000.parquet"]


def _download_split(split: str) -> pd.DataFrame:
    frames = []
    for i, url in enumerate(_parquet_urls(split)):
        part = CACHE_DIR / f"humaid_{split}_part{i}.parquet"
        resp = requests.get(url, timeout=max(HTTP_TIMEOUT, 60))
        resp.raise_for_status()
        part.write_bytes(resp.content)
        frames.append(pd.read_parquet(part))
        part.unlink(missing_ok=True)
    return pd.concat(frames, ignore_index=True)


def _load_with_datasets_lib(split: str) -> pd.DataFrame:
    from datasets import load_dataset  # optional dependency

    return load_dataset(HUMAID_REPO, split=split).to_pandas()


def load_humaid(split: str = "train", limit: int | None = None, seed: int = 42) -> pd.DataFrame:
    """Return HumAID as a DataFrame with columns text, label. Downloads once, then reads the cache."""
    if split not in HUMAID_SPLITS:
        raise ValueError(f"split must be one of {HUMAID_SPLITS}")
    cache = _humaid_cache_path(split)
    if cache.exists():
        df = pd.read_parquet(cache)
    else:
        log.info("Downloading HumAID %s split from Hugging Face (one time, a few MB)...", split)
        errors = []
        df = None
        for loader in (_download_split, _load_with_datasets_lib):
            try:
                df = loader(split)
                break
            except ImportError:
                errors.append("datasets library not installed")
            except Exception as exc:  # network errors, HTTP errors, parse errors
                errors.append(f"{loader.__name__}: {exc}")
        if df is None:
            raise DataUnavailable(
                "Could not download HumAID (" + "; ".join(errors) + "). "
                "Check internet access to huggingface.co, or run with --data sample."
            )
        df.to_parquet(cache, index=False)
        log.info("Cached HumAID %s split at %s (%d rows)", split, cache, len(df))

    if "tweet_text" not in df.columns or "class_label" not in df.columns:
        raise DataUnavailable(f"Unexpected HumAID columns: {list(df.columns)}")
    df = df.rename(columns={"tweet_text": "text", "class_label": "label"})[["text", "label"]]
    df = df.dropna().drop_duplicates(subset="text").reset_index(drop=True)
    if limit is not None and limit < len(df):
        # stratified-ish sample so every class stays represented
        frac = limit / len(df)
        df = (
            df.groupby("label").sample(frac=frac, random_state=seed)
            .sample(frac=1.0, random_state=seed)
            .reset_index(drop=True)
        )
    return df


# ---------------------------------------------------------------- Kaggle (optional, manual download)
def load_kaggle_disaster_tweets(path: str = KAGGLE_DISASTER_CSV) -> pd.DataFrame:
    """Kaggle 'NLP with Disaster Tweets' train.csv (needs a manual download; Kaggle requires login).

    Returns columns text, label where label is 'disaster' or 'not_disaster'.
    """
    p = Path(path)
    if not p.exists():
        raise DataUnavailable(
            f"Kaggle file not found at {p}. Download train.csv from "
            "https://www.kaggle.com/competitions/nlp-getting-started/data and set KAGGLE_DISASTER_CSV in .env."
        )
    df = pd.read_csv(p, encoding="utf-8")
    df["label"] = df["target"].map({1: "disaster", 0: "not_disaster"})
    return df[["text", "label"]].dropna().reset_index(drop=True)


# ---------------------------------------------------------------- conversion
def dataframe_to_reports(df: pd.DataFrame, source: str) -> list[Report]:
    reports = []
    for i, row in enumerate(df.itertuples(index=False)):
        data = row._asdict()
        text = str(data.get("text", "")).strip()
        if not text:
            continue
        report_id = str(data.get("id") or f"{source}-{i:06d}")
        kwargs = {
            "id": report_id,
            "text": text,
            "source": str(data.get("source") or source),
            "gold_label": str(data.get("label") or ""),
        }
        # curated datasets without timestamps get "" (unknown) so they never look "recent"
        kwargs["timestamp"] = str(data.get("timestamp") or "")
        reports.append(Report(**kwargs))
    return reports
