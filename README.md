# CrisisBridge AI

RAG-augmented disaster response prioritisation. Reads disaster posts and reports, extracts **what, where, who and how urgent**, indexes them, and answers commander questions with answers cited back to the source reports.

```
"4 people trapped on the second floor near XYZ bridge, need rescue"
  -> Location: XYZ bridge | Need: Rescue | People: 4 | Priority: Critical
```

## Pipeline

| Stage | Module | Default (offline, CPU) | Optional upgrade |
|---|---|---|---|
| 1. Ingestion | `src/ingestion/curated_loader.py`, `live_sources.py` | Bundled 50-message sample, HumAID via HTTPS | GDACS, ReliefWeb, Reddit (PRAW), RSS |
| 2a. Classification | `src/nlp/classifier.py` | TF-IDF (word + char) + Logistic Regression, HumAID's 10 labels | |
| 2b. Entities | `src/nlp/ner.py` | Rule-based locations, people count, need types | spaCy `en_core_web_sm` if installed |
| 2c. Urgency | `src/nlp/urgency.py` | Explainable additive score 0-100 -> Critical / High / Medium / Low | |
| 3. Indexing | `src/rag/embeddings.py`, `indexer.py` | TF-IDF + SVD vectors, NumPy cosine store | all-MiniLM-L6-v2, FAISS |
| 4. RAG | `src/rag/retriever.py`, `generator.py` | Semantic + location + urgency + recency re-ranking, extractive cited answer | Groq (`openai/gpt-oss-20b`) or Gemini (`gemini-2.5-flash`) |
| 5. Interface | `src/interface/streamlit_app.py`, `cli.py` | Streamlit console + CLI | |

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

4. **Streamlit console** (opens http://localhost:8501).

   ```powershell
   streamlit run src/interface/streamlit_app.py
   ```

5. **CLI fallback.**

   ```powershell
   python -m src.interface.cli ask "What is the most critical situation near XYZ bridge right now?"
   python -m src.interface.cli process "Family of 5 stuck on a roof in Velachery, need a boat"
   python -m src.interface.cli top --n 10
   python -m src.interface.cli live gdacs rss
   python -m src.interface.cli chat
   ```

By default, `run_pipeline.py` uses `--data auto`: it tries HumAID first and falls back to the sample data with a printed reason. Other flags: `--query "..."`, `--k 5`, `--provider groq|gemini|extractive`, and `--live gdacs rss`.

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
| NDMA / IMD | **Not implemented**: neither has a stable public RSS feed or API. |
| X / Twitter | Excluded because the API is paid. |

A source that fails never stops the pipeline. Each one prints `[OK]` or `[SKIP]` with the reason.

## Optional datasets

- **Kaggle "NLP with Disaster Tweets"**: download `train.csv` by hand (requires a Kaggle login) to `data\kaggle\train.csv`. Load it with `src.ingestion.curated_loader.load_kaggle_disaster_tweets()`. Its labels are binary (disaster or not), so it is not used to train the 10-class model.
- **IDRISI** (location mentions): not wired in yet. It is the planned dataset for evaluating the location extractor.

## Honest limitations of v1

- **Sample-data classifier.** Fifty messages cannot train a 10-class model on their own, so in sample mode the classifier is blended with a keyword prior. The printed cross-validation score is **optimistic**, because the keywords were written alongside the sample. Use `--data humaid` for a real evaluation.
- **Rule-based entity extraction.** It works well on the sample style ("near X bridge", "in Velachery"). Expect misses on noisy tweets; this is the obvious next component to improve.
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
src/interface/  streamlit_app.py  cli.py
tests/test_smoke.py
data/                        created at runtime: cache, models, index (git-ignored)
```

HumAID: Alam et al., ICWSM 2021, CC BY-NC-SA 4.0.
