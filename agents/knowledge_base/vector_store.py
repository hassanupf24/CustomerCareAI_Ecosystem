"""
Vector Store — FAISS index management (load, search, update).

Maintains an in-memory FAISS index backed by optional persistent storage.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import structlog

logger = structlog.get_logger()

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("faiss_not_available_using_brute_force")


class VectorStore:
    """FAISS-backed vector store with brute-force fallback."""

    def __init__(self, dimension: int = 384, index_path: Optional[str] = None) -> None:
        self.dimension = dimension
        self.index_path = index_path
        self.vectors: list[np.ndarray] = []
        self.metadata: list[dict] = []
        self.index = None

        if FAISS_AVAILABLE:
            self.index = faiss.IndexFlatIP(dimension)  # inner-product (cosine after normalization)

        if index_path:
            self._load(index_path)

    def _load(self, path: str) -> None:
        """Load persisted index and metadata."""
        idx_file = Path(path) / "index.faiss"
        meta_file = Path(path) / "metadata.json"
        if idx_file.exists() and FAISS_AVAILABLE:
            self.index = faiss.read_index(str(idx_file))
            logger.info("faiss_index_loaded", path=str(idx_file))
        if meta_file.exists():
            with open(meta_file, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)

    def save(self, path: str) -> None:
        """Persist index and metadata to disk."""
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        if FAISS_AVAILABLE and self.index is not None and self.index.ntotal > 0:
            faiss.write_index(self.index, str(p / "index.faiss"))
        with open(p / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)

    def add(self, vectors: np.ndarray, metadata_list: list[dict]) -> None:
        """Add vectors and associated metadata."""
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)

        # L2-normalize for cosine similarity via inner product
        norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-9
        vectors = vectors / norms

        if FAISS_AVAILABLE and self.index is not None:
            self.index.add(vectors.astype(np.float32))
        else:
            for v in vectors:
                self.vectors.append(v)

        self.metadata.extend(metadata_list)

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> list[dict]:
        """
        Search for top-k most similar vectors.
        Returns metadata dicts augmented with 'score'.
        """
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)

        norm = np.linalg.norm(query_vector) + 1e-9
        query_vector = query_vector / norm

        if FAISS_AVAILABLE and self.index is not None and self.index.ntotal > 0:
            scores, indices = self.index.search(query_vector.astype(np.float32), min(top_k, self.index.ntotal))
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.metadata) and idx >= 0:
                    entry = dict(self.metadata[idx])
                    entry["score"] = float(score)
                    results.append(entry)
            return results
        elif self.vectors:
            # Brute-force fallback
            mat = np.array(self.vectors)
            scores = mat @ query_vector.flatten()
            top_idx = np.argsort(scores)[::-1][:top_k]
            results = []
            for idx in top_idx:
                entry = dict(self.metadata[idx])
                entry["score"] = float(scores[idx])
                results.append(entry)
            return results
        return []

    @property
    def size(self) -> int:
        if FAISS_AVAILABLE and self.index is not None:
            return self.index.ntotal
        return len(self.vectors)
