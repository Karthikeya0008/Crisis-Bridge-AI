"""Text embedders.

- SentenceTransformerEmbedder: all-MiniLM-L6-v2 (384-d, ~90 MB, CPU friendly). Needs the optional
  sentence-transformers package and a one-time model download from Hugging Face.
- TfidfSvdEmbedder: TF-IDF (word + char n-grams) reduced with SVD. Pure scikit-learn, no download,
  always available. For small corpora it keeps full rank, so ranking matches plain TF-IDF cosine.

get_embedder("auto") tries sentence-transformers first and falls back to TF-IDF with a clear message.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion

from src.config import EMBEDDING_BACKEND, SENTENCE_MODEL, get_logger
from src.nlp.preprocess import normalize_for_model

log = get_logger("rag.embeddings")


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (x / norms).astype(np.float32)


class TfidfSvdEmbedder:
    name = "tfidf-svd"
    needs_fit = True

    def __init__(self, max_dim: int = 256):
        self.max_dim = max_dim
        self.vectorizer: FeatureUnion | None = None
        self.svd: TruncatedSVD | None = None

    def fit(self, texts: list[str]) -> "TfidfSvdEmbedder":
        docs = [normalize_for_model(t) for t in texts]
        self.vectorizer = FeatureUnion([
            ("word", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, max_features=100_000)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, max_features=100_000)),
        ])
        matrix = self.vectorizer.fit_transform(docs)
        n_components = max(1, min(self.max_dim, matrix.shape[0] - 1, matrix.shape[1] - 1))
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        self.svd.fit(matrix)
        return self

    def encode(self, texts: list[str]) -> np.ndarray:
        if self.vectorizer is None or self.svd is None:
            raise RuntimeError("TfidfSvdEmbedder must be fit before encode.")
        matrix = self.vectorizer.transform([normalize_for_model(t) for t in texts])
        return _l2_normalize(self.svd.transform(matrix))

    @property
    def dim(self) -> int:
        return int(self.svd.n_components) if self.svd is not None else 0

    def save(self, path: Path) -> None:
        joblib.dump({"vectorizer": self.vectorizer, "svd": self.svd, "max_dim": self.max_dim}, path)

    @classmethod
    def load(cls, path: Path) -> "TfidfSvdEmbedder":
        state = joblib.load(path)
        obj = cls(state["max_dim"])
        obj.vectorizer, obj.svd = state["vectorizer"], state["svd"]
        return obj


class SentenceTransformerEmbedder:
    name = "sentence-transformers"
    needs_fit = False

    def __init__(self, model_name: str = SENTENCE_MODEL):
        from sentence_transformers import SentenceTransformer  # optional dependency

        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device="cpu")

    def fit(self, texts: list[str]) -> "SentenceTransformerEmbedder":
        return self

    def encode(self, texts: list[str]) -> np.ndarray:
        vecs = self.model.encode(texts, batch_size=64, show_progress_bar=len(texts) > 500, normalize_embeddings=True)
        return np.asarray(vecs, dtype=np.float32)

    @property
    def dim(self) -> int:
        return int(self.model.get_sentence_embedding_dimension())

    def save(self, path: Path) -> None:
        path.write_text(self.model_name, encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "SentenceTransformerEmbedder":
        return cls(path.read_text(encoding="utf-8").strip())


def get_embedder(backend: str = EMBEDDING_BACKEND):
    backend = (backend or "auto").lower()
    if backend in ("auto", "sentence-transformers", "st"):
        try:
            emb = SentenceTransformerEmbedder()
            log.info("Embeddings: sentence-transformers (%s)", emb.model_name)
            return emb
        except ImportError:
            msg = "sentence-transformers not installed"
        except Exception as exc:  # model download blocked, no internet, etc.
            msg = f"could not load {SENTENCE_MODEL} ({type(exc).__name__}: {exc})"
        if backend != "auto":
            raise RuntimeError(f"EMBEDDING_BACKEND={backend} requested but {msg}.")
        hint = " Install requirements-optional.txt for neural embeddings." if "not installed" in msg else ""
        log.info("Embeddings: TF-IDF + SVD fallback (%s).%s", msg, hint)
        return TfidfSvdEmbedder()
    if backend in ("tfidf", "tfidf-svd"):
        log.info("Embeddings: TF-IDF + SVD")
        return TfidfSvdEmbedder()
    raise ValueError(f"Unknown EMBEDDING_BACKEND '{backend}' (use auto, sentence-transformers or tfidf)")


def load_embedder(name: str, path: Path):
    if name == TfidfSvdEmbedder.name:
        return TfidfSvdEmbedder.load(path)
    if name == SentenceTransformerEmbedder.name:
        return SentenceTransformerEmbedder.load(path)
    raise ValueError(f"Unknown saved embedder '{name}'")
