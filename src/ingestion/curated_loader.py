"""Curated datasets: bundled sample data, HumAID and CrisisBench (Hugging Face), IDRISI (GitHub) and the
optional Kaggle disaster tweets CSV.

HumAID and CrisisBench are fetched as parquet over plain HTTPS (no `datasets` dependency) and cached in
data/cache. If `datasets` happens to be installed it is used as a last resort for HumAID.
- CrisisBench (Alam et al., ICWSM 2021): the English humanitarian task, which combines CrisisLex T6/T26,
  CrisisNLP, CrisisMMD, AIDR, ISCRAM13, SWDM13 and DRD. Its 11 labels are mapped onto HumAID's 10;
  `affected_individual` has no HumAID equivalent and is dropped.
- IDRISI-R (Suwaileh et al., 2023): gold location-mention labels for 19 English disaster events. Only the
  train and dev splits are public (test is held back for a shared task).
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import pandas as pd
import requests

from src.config import (
    CACHE_DIR,
    CRISISBENCH_REPO,
    HTTP_TIMEOUT,
    HUMAID_REPO,
    IDRISI_ZIP_URL,
    KAGGLE_DISASTER_CSV,
    SAMPLE_DATA_PATH,
    get_logger,
)
from src.nlp.preprocess import normalize_for_model
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
    # stratified-ish sample so every class stays represented
    return _stratified_sample(df, limit, seed)


def _stratified_sample(df: pd.DataFrame, limit: int | None, seed: int) -> pd.DataFrame:
    if limit is None or limit >= len(df):
        return df
    frac = limit / len(df)
    return (df.groupby("label").sample(frac=frac, random_state=seed)
            .sample(frac=1.0, random_state=seed).reset_index(drop=True))


# ---------------------------------------------------------------- CrisisBench (humanitarian task, English)
CRISISBENCH_SPLITS = ("train", "dev", "test")
CRISISBENCH_TO_HUMAID = {
    "not_humanitarian": "not_humanitarian",
    "donation_and_volunteering": "rescue_volunteering_or_donation_effort",
    "response_efforts": "rescue_volunteering_or_donation_effort",
    "requests_or_needs": "requests_or_urgent_needs",
    "sympathy_and_support": "sympathy_and_support",
    "infrastructure_and_utilities_damage": "infrastructure_and_utility_damage",
    "caution_and_advice": "caution_and_advice",
    "injured_or_dead_people": "injured_or_dead_people",
    "missing_and_found_people": "missing_or_found_people",
    "displaced_and_evacuations": "displaced_people_and_evacuations",
}


def load_crisisbench(split: str = "train", limit: int | None = None, seed: int = 42,
                     exclude_texts: set[str] | None = None) -> pd.DataFrame:
    """CrisisBench English humanitarian tweets with labels mapped to HumAID's. Columns: text, label.

    `exclude_texts` (normalised texts) drops tweets that also appear in an evaluation set, so a model
    trained on CrisisBench is not scored on tweets it has already seen.
    """
    if split not in CRISISBENCH_SPLITS:
        raise ValueError(f"split must be one of {CRISISBENCH_SPLITS}")
    cache = CACHE_DIR / f"crisisbench_{split}.parquet"
    if cache.exists():
        df = pd.read_parquet(cache)
    else:
        url = f"https://huggingface.co/api/datasets/{CRISISBENCH_REPO}/parquet/humanitarian/{split}/0.parquet"
        log.info("Downloading CrisisBench %s split from Hugging Face (one time, a few MB)...", split)
        try:
            resp = requests.get(url, timeout=max(HTTP_TIMEOUT, 120))
            resp.raise_for_status()
            df = pd.read_parquet(io.BytesIO(resp.content))
        except Exception as exc:
            raise DataUnavailable(f"Could not download CrisisBench ({exc}). Check internet access to huggingface.co.") from exc
        df.to_parquet(cache, index=False)
        log.info("Cached CrisisBench %s split at %s (%d rows)", split, cache, len(df))
    if "text" not in df.columns or "class_label" not in df.columns:
        raise DataUnavailable(f"Unexpected CrisisBench columns: {list(df.columns)}")
    df = df[df["class_label"].isin(CRISISBENCH_TO_HUMAID)]
    df = df.assign(label=df["class_label"].map(CRISISBENCH_TO_HUMAID))[["text", "label"]]
    df = df.dropna().drop_duplicates(subset="text")
    if exclude_texts:
        df = df[~df["text"].map(normalize_for_model).isin(exclude_texts)]
    return _stratified_sample(df.reset_index(drop=True), limit, seed)


# ---------------------------------------------------------------- IDRISI-R (location mentions, English gold)
IDRISI_SPLITS = ("train", "dev")
IDRISI_PREFIX = "IDRISI-main/LMR/data/EN/gold-random-json/"


def load_idrisi(split: str = "dev") -> list[dict]:
    """IDRISI English gold tweets: dicts with text, event and location_mentions [{text, type, ...}]."""
    if split not in IDRISI_SPLITS:
        raise ValueError(f"split must be one of {IDRISI_SPLITS} (the test split is not public)")
    cache = CACHE_DIR / f"idrisi_en_{split}.jsonl"
    if not cache.exists():
        log.info("Downloading IDRISI from GitHub (one time, about 28 MB)...")
        try:
            resp = requests.get(IDRISI_ZIP_URL, timeout=max(HTTP_TIMEOUT, 180))
            resp.raise_for_status()
            archive = zipfile.ZipFile(io.BytesIO(resp.content))
        except Exception as exc:
            raise DataUnavailable(f"Could not download IDRISI ({exc}). Check internet access to github.com.") from exc
        for part in IDRISI_SPLITS:  # keep only the English gold files we use; drop the rest of the archive
            rows = []
            for name in archive.namelist():
                if name.startswith(IDRISI_PREFIX) and name.endswith(f"/{part}.jsonl"):
                    event = name[len(IDRISI_PREFIX):].split("/")[0]
                    for line in archive.read(name).decode("utf-8").splitlines():
                        if line.strip():
                            row = json.loads(line)
                            rows.append({"text": row["text"], "event": event, "location_mentions": row.get("location_mentions", [])})
            (CACHE_DIR / f"idrisi_en_{part}.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
        if not cache.exists():
            raise DataUnavailable("The IDRISI archive did not contain the expected English gold files.")
    return [json.loads(line) for line in cache.read_text(encoding="utf-8").splitlines() if line.strip()]


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
