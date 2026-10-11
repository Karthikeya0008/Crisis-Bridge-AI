"""Vector index for processed reports, persisted to data/index/.

Backend: FAISS (IndexFlatIP on L2-normalised vectors = cosine) when faiss-cpu is installed,
otherwise an equivalent NumPy store. Both give exact results; NumPy is fast enough for tens of
thousands of reports on a laptop.

Files written: vectors.npy, reports.jsonl, meta.json, embedder.bin
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.config import INDEX_DIR, VECTOR_STORE, get_logger
from src.rag.embeddings import get_embedder, load_embedder
from src.schema import Report

log = get_logger("rag.indexer")


def _try_faiss():
    if VECTOR_STORE in ("numpy",):
        return None
    try:
        import faiss  # optional dependency

        return faiss
    except ImportError:
        if VECTOR_STORE == "faiss":
            raise RuntimeError("VECTOR_STORE=faiss but faiss-cpu is not installed.")
        return None


class CrisisIndex:
    def __init__(self, embedder, reports: list[Report], vectors: np.ndarray):
        self.embedder = embedder
        self.reports = reports
        self.vectors = vectors.astype(np.float32)
        self._faiss = _try_faiss()
        self._faiss_index = None
        if self._faiss is not None and len(self.reports):
            self._faiss_index = self._faiss.IndexFlatIP(self.vectors.shape[1])
            self._faiss_index.add(self.vectors)

    # ------------------------------------------------------------ construction
    @classmethod
    def build(cls, reports: list[Report], embedder=None) -> "CrisisIndex":
        if not reports:
            raise ValueError("Cannot build an index with zero reports.")
        embedder = embedder or get_embedder()
        texts = [r.embedding_text() for r in reports]
        if embedder.needs_fit:
            embedder.fit(texts)
        vectors = embedder.encode(texts)
        index = cls(embedder, reports, vectors)
        log.info("Indexed %d reports (%s, dim=%d, store=%s)", len(reports), embedder.name, vectors.shape[1], index.store_name)
        return index

    def add(self, new_reports: list[Report]) -> int:
        """Add reports (skips ids already present). TF-IDF embedder is refit on the full corpus."""
        existing = {r.id for r in self.reports}
        fresh = [r for r in new_reports if r.id not in existing]
        if not fresh:
            return 0
        all_reports = self.reports + fresh
        if self.embedder.needs_fit:
            rebuilt = CrisisIndex.build(all_reports, embedder=self.embedder)
            self.__dict__.update(rebuilt.__dict__)
        else:
            vecs = self.embedder.encode([r.embedding_text() for r in fresh])
            self.reports = all_reports
            self.vectors = np.vstack([self.vectors, vecs]).astype(np.float32)
            if self._faiss_index is not None:
                self._faiss_index.add(vecs)
        return len(fresh)

    # ------------------------------------------------------------ search
    @property
    def store_name(self) -> str:
        return "faiss" if self._faiss_index is not None else "numpy"

    def __len__(self) -> int:
        return len(self.reports)

    def search(self, query: str, k: int = 50) -> list[tuple[int, float]]:
        """Return (report_index, cosine_similarity) pairs, best first."""
        if not self.reports:
            return []
        k = min(k, len(self.reports))
        qvec = self.embedder.encode([query])
        if self._faiss_index is not None:
            scores, idx = self._faiss_index.search(qvec, k)
            return [(int(i), float(s)) for i, s in zip(idx[0], scores[0]) if i >= 0]
        sims = self.vectors @ qvec[0]
        top = np.argpartition(-sims, k - 1)[:k]
        top = top[np.argsort(-sims[top])]
        return [(int(i), float(sims[i])) for i in top]

    def similarity_to(self, query: str, indices: list[int]) -> dict[int, float]:
        qvec = self.embedder.encode([query])[0]
        return {i: float(self.vectors[i] @ qvec) for i in indices}

    # ------------------------------------------------------------ persistence
    def save(self, directory: Path = INDEX_DIR) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "vectors.npy", self.vectors)
        with open(directory / "reports.jsonl", "w", encoding="utf-8") as f:
            for r in self.reports:
                f.write(r.to_json() + "\n")
        self.embedder.save(directory / "embedder.bin")
        meta = {
            "embedder": self.embedder.name,
            "count": len(self.reports),
            "dim": int(self.vectors.shape[1]),
            "built_at": datetime.now(timezone.utc).isoformat(),
        }
        (directory / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        log.info("Saved index to %s", directory)
        return directory

    @classmethod
    def load(cls, directory: Path = INDEX_DIR) -> "CrisisIndex":
        meta_path = directory / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"No index at {directory}. Run: python run_pipeline.py")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        embedder = load_embedder(meta["embedder"], directory / "embedder.bin")
        with open(directory / "reports.jsonl", encoding="utf-8") as f:
            reports = [Report.from_dict(json.loads(line)) for line in f if line.strip()]
        vectors = np.load(directory / "vectors.npy")
        return cls(embedder, reports, vectors)

    @staticmethod
    def exists(directory: Path = INDEX_DIR) -> bool:
        return (directory / "meta.json").exists()
