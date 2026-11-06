"""Local vector indexes: FAISS when installed, otherwise an exact numpy cosine index.

Both expect L2-normalised vectors, so inner product == cosine similarity.
"""

from __future__ import annotations

import numpy as np


class NumpyIndex:
    backend = "numpy"

    def __init__(self, vectors: np.ndarray) -> None:
        self.vectors = np.asarray(vectors, dtype=np.float32)

    def search(self, query: np.ndarray, k: int) -> list[tuple[int, float]]:
        if len(self.vectors) == 0:
            return []
        scores = self.vectors @ np.asarray(query, dtype=np.float32).ravel()
        k = min(k, len(scores))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top], kind="stable")]
        return [(int(i), float(scores[i])) for i in top]


class FaissIndex:
    backend = "faiss"

    def __init__(self, vectors: np.ndarray) -> None:
        import faiss  # optional dependency

        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)

    def search(self, query: np.ndarray, k: int) -> list[tuple[int, float]]:
        if self.index.ntotal == 0:
            return []
        q = np.ascontiguousarray(np.asarray(query, dtype=np.float32).reshape(1, -1))
        scores, ids = self.index.search(q, min(k, self.index.ntotal))
        return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i >= 0]


def build_index(vectors: np.ndarray, prefer: str = "auto"):
    """Build a FAISS index if available (and requested), else a numpy index."""
    if prefer in ("auto", "faiss") and len(vectors):
        try:
            return FaissIndex(vectors)
        except ImportError:
            if prefer == "faiss":
                raise
    return NumpyIndex(vectors)
