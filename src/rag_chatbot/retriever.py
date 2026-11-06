"""Document store + retriever with on-disk persistence.

The store keeps the chunk texts and their embeddings. Adding documents re-embeds
the whole corpus when the embedder must be fitted (TF-IDF); dense embedders only
embed the new chunks.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .embeddings import Embedder, get_embedder
from .ingest import Chunk
from .vector_store import build_index


@dataclass(frozen=True)
class SearchResult:
    chunk: Chunk
    score: float


class Retriever:
    def __init__(self, embedder: Embedder | None = None, index_backend: str = "auto") -> None:
        self.embedder = embedder or get_embedder("auto")
        self.index_backend = index_backend
        self.chunks: list[Chunk] = []
        self.vectors = np.zeros((0, 0), dtype=np.float32)
        self._index = None

    # ------------------------------------------------------------------ build
    def add_chunks(self, chunks: list[Chunk]) -> int:
        """Add chunks (replacing any existing chunks from the same source files). Returns count added."""
        if not chunks:
            return 0
        new_sources = {c.source for c in chunks}
        kept = [c for c in self.chunks if c.source not in new_sources]
        replaced = len(kept) != len(self.chunks)
        self.chunks = kept + list(chunks)
        if self.embedder.needs_fit or replaced or self.vectors.size == 0:
            if self.embedder.needs_fit:
                self.embedder.fit([c.text for c in self.chunks])
            self.vectors = self.embedder.embed([c.text for c in self.chunks])
        else:
            self.vectors = np.vstack([self.vectors, self.embedder.embed([c.text for c in chunks])])
        self._index = build_index(self.vectors, self.index_backend)
        return len(chunks)

    @property
    def sources(self) -> list[str]:
        return sorted({c.source for c in self.chunks})

    def __len__(self) -> int:
        return len(self.chunks)

    # ----------------------------------------------------------------- search
    def search(self, query: str, k: int = 4) -> list[SearchResult]:
        if not self.chunks or not query.strip():
            return []
        q = self.embedder.embed([query])[0]
        return [SearchResult(self.chunks[i], score) for i, score in self._index.search(q, k)]

    @property
    def index_name(self) -> str:
        return getattr(self._index, "backend", "none")

    # ------------------------------------------------------------ persistence
    def save(self, directory: str | Path) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "vectors.npy", self.vectors)
        (d / "chunks.json").write_text(json.dumps([c.to_dict() for c in self.chunks], ensure_ascii=False), encoding="utf-8")
        meta = {"embedder": self.embedder.name, "embedder_state": self.embedder.state()}
        (d / "meta.json").write_text(json.dumps(meta), encoding="utf-8")

    @classmethod
    def load(cls, directory: str | Path, embedder: Embedder | None = None, index_backend: str = "auto") -> "Retriever":
        """Load a saved index. If the current embedder differs from the saved one, chunks are re-embedded."""
        d = Path(directory)
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        chunks = [Chunk.from_dict(x) for x in json.loads((d / "chunks.json").read_text(encoding="utf-8"))]
        if embedder is None:
            try:
                embedder = get_embedder(meta["embedder"])
            except Exception:  # saved with a backend that is no longer installed
                embedder = get_embedder("auto")
        r = cls(embedder, index_backend)
        if embedder.name == meta["embedder"]:
            embedder.load_state(meta.get("embedder_state", {}))
            r.chunks = chunks
            r.vectors = np.load(d / "vectors.npy")
            r._index = build_index(r.vectors, index_backend)
        else:
            r.add_chunks(chunks)
        return r

    @staticmethod
    def exists(directory: str | Path) -> bool:
        return (Path(directory) / "meta.json").exists()
