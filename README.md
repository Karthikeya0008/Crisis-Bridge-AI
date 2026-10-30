# CrisisBridge AI

RAG-augmented disaster response prioritisation. Reads disaster posts and reports, extracts **what, where, who and how urgent**, indexes them, and answers commander questions with answers cited back to the source reports.

```
"4 people trapped on the second floor near XYZ bridge, need rescue"
  -> Location: XYZ bridge | Need: Rescue | People: 4 | Priority: Critical
```

## Pipeline

| Stage | Module | Default (offline, CPU) | Optional upgrade |
|---|---|---|---|
| 1. Ingestion | `src/ingestion/curated_loader.py`, `live_sources.py` | Bundled 50-message sample, HumAID via HTTPS, IDRISI location labels | CrisisBench (extra training data); GDACS, NDMA SACHET (IMD alerts), Google News, Indian newspapers, Bluesky, Mastodon, NASA EONET, EMSC, ReliefWeb, Reddit (PRAW), RSS |
| 2a. Classification | `src/nlp/classifier.py` | TF-IDF (word + char) + Logistic Regression, HumAID's 10 labels | |
| 2b. Entities | `src/nlp/ner.py`, `location_eval.py` | Rule-based locations plus a gazetteer learned from IDRISI, people count, need types | spaCy `en_core_web_sm` if installed |
| 2c. Urgency | `src/nlp/urgency.py` | Explainable additive score 0-100 -> Critical / High / Medium / Low | |
| 3. Indexing | `src/rag/embeddings.py`, `indexer.py` | TF-IDF + SVD vectors, NumPy cosine store | all-MiniLM-L6-v2, FAISS |
| 4. RAG | `src/rag/retriever.py`, `generator.py` | Semantic + location + urgency + recency re-ranking, extractive cited answer | Groq (`openai/gpt-oss-20b`) or Gemini (`gemini-3.5-flash`) |
| 5. Interface | `src/interface/api.py`, `web/`, `streamlit_app.py`, `cli.py` | Web console (FastAPI + HTML/JS), Streamlit console, CLI | |

`run_pipeline.py` runs stages 1 to 4 end to end and answers a sample query. `src/pipeline.py` is the shared glue used by all entry points.

## Setup (Windows PowerShell, Python 3.11)

Run each line separately.

```powershell
cd E:\Crisis_Bridge_AI
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

If `Activate.ps1` is blocked, run this once and then activate again:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

## Run

1. **Offline run** on the bundled sample data. Needs no internet, keys or GPU.

   ```powershell
   python run_pipeline.py --data sample
   ```

2. **Offline tests.** All network calls are mocked.

   ```powershell
   python -m unittest -v tests.test_smoke
   ```

3. **HumAID run.** Downloads about 7 MB once to `data\cache`, trains on 20k tweets, evaluates on the HumAID test split, and indexes 2,000 test tweets plus the sample data.

   ```powershell
   python run_pipeline.py --data humaid
   ```

   To train on the full 53k-tweet training split, add `--train-limit 0`.

4. **Web console** (recommended; opens on http://localhost:8000).

   ```powershell
   python -m src.interface.api
   ```

   Six views that follow the project deck: Overview (detect, extract, prioritize and the system flow),
   Commander Query (cited answers with a per-report "why this rank" breakdown), Live Priority Feed
   (filter, auto-refresh, CSV export), Process a Message, Data Sources (curated vs live, with a live fetch)
   and System (index details, LLM keys, build or rebuild). The JSON API is documented at
   http://localhost:8000/docs. Use `--port 9000` to change the port.

5. **Streamlit console** (opens http://localhost:8501).

   ```powershell
   streamlit run src/interface/streamlit_app.py
   ```

6. **CLI fallback.**

   ```powershell
   python -m src.interface.cli ask "What is the most critical situation near XYZ bridge right now?"
   python -m src.interface.cli process "Family of 5 stuck on a roof in Velachery, need a boat"
   python -m src.interface.cli top --n 10
   python -m src.interface.cli live gdacs ndma gnews india_news bluesky mastodon eonet emsc rss
   python -m src.interface.cli chat
   ```

By default, `run_pipeline.py` uses `--data auto`: it tries HumAID first and falls back to the sample data with a printed reason. Other flags: `--query "..."`, `--k 5`, `--provider groq|gemini|extractive`, and `--live gdacs ndma gnews india_news bluesky mastodon eonet emsc rss`.

### LLM answers (optional)

Put a free key in `.env` as `GROQ_API_KEY=...` (https://console.groq.com/keys) or `GEMINI_API_KEY=...` (https://aistudio.google.com/apikey). With no key, or if the API call fails, the system falls back to an extractive answer that still cites its sources, and it prints the reason. Note that free-tier Gemini prompts may be used by Google to improve its products.

### Neural embeddings and FAISS (optional)

```powershell
pip install -r requirements-optional.txt
python run_pipeline.py --data humaid
```

This downloads all-MiniLM-L6-v2 (about 90 MB) on first use. It runs on CPU, and the 16 GB of RAM is plenty. If the model cannot load, the pipeline falls back to TF-IDF and logs why. Rebuild the index after switching backends.

## Live sources: actual status

| Source | Status |
|---|---|
| GDACS RSS | No key needed. **Known to time out from some networks**; it then reports a timeout and is skipped. |
| ReliefWeb API v2 | **Needs a pre-approved `appname`** (required since 1 Nov 2025). Skipped until `RELIEFWEB_APPNAME` is set. |
| Reddit (PRAW) | Needs `praw` and a Reddit "script" app. Since Nov 2025, new API apps may need Reddit's approval. Skipped until credentials are set. |
| RSS / Atom | Any feeds listed in `RSS_FEEDS`. The default is the USGS M4.5+ weekly earthquake feed. |
| NDMA / IMD | Read through NDMA's **SACHET** alert feed (IMD, INCOIS, CWC and state SDMA warnings). No key needed. About three quarters of alerts are in regional languages (on 30 Oct 2026: 53 Telugu, 12 Hindi, 7 Bengali, 3 Odia, 3 Marathi out of 100). The NLP stages are English-only, so `src/ingestion/translate.py` translates them with the configured LLM (Groq, else Gemini; `TRANSLATE_REGIONAL`), 20 per request, and tags each one `[translated from <language>]`. Each wording is translated once and cached in `data/cache/translations.json`, so a repeat fetch makes no LLM call. Without an LLM key they are skipped, as before. |
| Google News | Public RSS search, no key. One request per search in `GOOGLE_NEWS_QUERIES` (default: flood, cyclone, landslide, earthquake and heavy-rain searches for India, last 2 days). Headlines only; the source is stored as `news/<publisher>`. |
| Bluesky | Public post search on `api.bsky.app`, no key (`public.api.bsky.app` refuses searches with HTTP 403). Searches come from `BLUESKY_QUERIES`. Keyword search is noisy, so a post is kept only if it is English, at least 40 characters, and mentions a disaster term. If Bluesky ever requires a login for search, set `BLUESKY_HANDLE` and an app password in `BLUESKY_APP_PASSWORD`. |
| Indian newspapers (`india_news`) | The Hindu, NDTV, Times of India and Indian Express national RSS (`INDIA_NEWS_FEEDS`), no key. Most stories are not about disasters, so only those that mention a disaster term are kept. |
| Mastodon | Public hashtag timelines on `MASTODON_INSTANCE` (default mastodon.social), no key; search itself needs a login. Tags come from `MASTODON_TAGS`. Same English and disaster-term filter as Bluesky, and posts older than 7 days are dropped. |
| NASA EONET | Open natural events (cyclones, floods, wildfires, volcanoes), no key, limited to `REGION_BBOX` (default: India and neighbours). US "prescribed fire" entries are skipped. |
| EMSC earthquakes | Earthquakes from the European-Mediterranean Seismological Centre, no key, M3+ (`EMSC_MIN_MAGNITUDE`) from the last 7 days inside `REGION_BBOX`. |
| Not used | PIB press-release RSS only returned Hindi titles, ReliefWeb RSS sits behind a bot wall, and the Copernicus EMS feed was removed (all checked Oct 2026). |
| X / Twitter | Excluded because the API is paid. |

A source that fails never stops the pipeline. Each one prints `[OK]` or `[SKIP]` with the reason.

## Extra training data

### IDRISI: location extraction (in use)

IDRISI-R gives human-annotated place names for 19 English disasters (14,392 train and 2,056 dev tweets; the test split is not public). It is downloaded once from GitHub (about 28 MB) and used two ways: place names annotated at least twice in **train** become a learned gazetteer (683 names, saved to `data/models/learned_gazetteer.txt`), and the extractor is scored on **dev**.

```powershell
python -m src.nlp.location_eval            # score, then build and save the learned gazetteer
python -m src.nlp.location_eval --no-save  # score only
```

| Location extractor, IDRISI dev (2,056 tweets) | Precision | Recall | F1 |
|---|---|---|---|
| Rules + built-in 100-place gazetteer | 0.676 | 0.587 | 0.628 |
| + learned gazetteer | 0.686 | 0.865 | **0.765** |
| + learned gazetteer, each disaster left out of the gazetteer it is scored with | 0.668 | 0.710 | **0.689** |

The last row is the honest number for a disaster the system has not seen. Learned names match case-sensitively, so "mobile network" is not read as Mobile, Alabama. New reports use the gazetteer as soon as the file exists; restart the web console after building it, and rebuild the index if existing reports should be re-tagged.

### CrisisBench: classifier training (optional, off by default)

CrisisBench's English humanitarian task (61,235 train tweets from CrisisLex T6/T26, CrisisNLP, CrisisMMD, AIDR, ISCRAM13, SWDM13 and DRD) is downloaded from Hugging Face (`QCRI/CrisisBench-english`). Its 11 labels map onto HumAID's 10; `affected_individual` has no equivalent and is dropped, and tweets that also appear in the HumAID test split are removed so the score is not inflated.

```powershell
python run_pipeline.py --data humaid+crisisbench
```

Each CrisisBench tweet counts `CRISISBENCH_WEIGHT` (default 0.3) times as much as a HumAID tweet. Every model below was trained on the same 20,000 HumAID tweets, and is scored on 5,000 HumAID test tweets and on CrisisBench's own test split (16,672 tweets from the other datasets, with labels mapped the same way):

| Training data | HumAID test: accuracy | macro-F1 | CrisisBench test: accuracy | macro-F1 |
|---|---|---|---|---|
| HumAID 20k (default) | **0.728** | **0.716** | 0.537 | 0.467 |
| + CrisisBench, weight 0.1 | 0.727 | 0.707 | | |
| + CrisisBench, weight 0.3 | 0.723 | 0.698 | **0.819** | **0.672** |
| + CrisisBench, weight 1.0 | 0.710 | 0.680 | 0.827 | 0.682 |

CrisisBench never improves the HumAID score. The datasets label tweets differently; for example, CrisisBench has no "other relevant information" class. But the HumAID-only model transfers poorly to the other datasets (0.537 accuracy). At weight 0.3, the model gives up half a point on HumAID and gains 28 points on tweets from other sources, which is closer to what the live feeds bring in. HumAID alone stays the default, so the reported results do not change; build with `humaid+crisisbench` for the more robust model.

### Kaggle (not used)

- **Kaggle "NLP with Disaster Tweets"**: download `train.csv` by hand (requires a Kaggle login) to `data\kaggle\train.csv`. Load it with `src.ingestion.curated_loader.load_kaggle_disaster_tweets()`. Its labels are binary (disaster or not), so it is not used to train the 10-class model.

## Honest limitations of v1

- **Sample-data classifier.** Fifty messages cannot train a 10-class model on their own, so in sample mode the classifier is blended with a keyword prior. The printed cross-validation score is **optimistic**, because the keywords were written alongside the sample. Use `--data humaid` for a real evaluation.
- **Rule-based entity extraction.** It works well on the sample style ("near X bridge", "in Velachery"). On IDRISI's real disaster tweets it reaches F1 0.69 for an unseen disaster with the learned gazetteer (0.63 without it). A trained sequence tagger is the next step.
- **Urgency weights are hand-set** (see `src/nlp/urgency.py`), so every score can be explained. Calibrating them against labelled data is a future step.
- **"Right now"** is measured relative to the newest report in the index, so recency also works on historical data. HumAID tweets have no timestamps and never count as recent.

## Layout

```
run_pipeline.py              end-to-end run
sample_data/crisis_messages.csv   50 hand-written labelled messages
src/config.py  src/schema.py  src/pipeline.py
src/ingestion/  curated_loader.py  live_sources.py
src/nlp/        preprocess.py  classifier.py  ner.py  urgency.py  processor.py
src/rag/        embeddings.py  indexer.py  retriever.py  generator.py
src/interface/  api.py  web/ (index.html, styles.css, app.js)  streamlit_app.py  cli.py
tests/test_smoke.py
data/                        created at runtime: cache, models, index (git-ignored)
```

HumAID: Alam et al., ICWSM 2021, CC BY-NC-SA 4.0.
