"""Text embedders.

Two backends behind one small interface:
  * SentenceTransformerEmbedder - dense all-MiniLM-L6-v2 vectors (optional dependency)
  * TfidfEmbedder               - sparse-ish TF-IDF vectors in pure numpy (always available)

All embedders return L2-normalised float32 rows, so cosine similarity is a dot product.
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter

import numpy as np

log = logging.getLogger(__name__)

DEFAULT_ST_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

_TOKEN = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")
STOPWORDS = frozenset(
    """a about above after again all am an and any are as at be because been before being below between both
    but by can could did do does doing down during each few for from further had has have having he her here
    hers him his how i if in into is it its itself just me more most my no nor not now of off on once only or
    other our ours out over own same she should so some such than that the their theirs them then there these
    they this those through to too under until up very was we were what when where which while who whom why
    will with would you your yours""".split()
)


def _light_stem(token: str) -> str:
    """Very small plural/3rd-person normaliser: batteries->battery, lasts->last, glass stays glass."""
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 3 and token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    return [_light_stem(t) for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS]


def _normalise(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (m / norms).astype(np.float32)


class Embedder:
    name = "base"
    needs_fit = False

    def fit(self, texts: list[str]) -> None:  # pragma: no cover - default no-op
        pass

    def embed(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError

    def state(self) -> dict:
        return {}

    def load_state(self, state: dict) -> None:
        pass


class TfidfEmbedder(Embedder):
    """Smoothed TF-IDF (sklearn-style idf) with sublinear term frequency, fitted on the corpus."""

    name = "tfidf"
    needs_fit = True

    def __init__(self) -> None:
        self.vocab: dict[str, int] = {}
        self.idf: np.ndarray = np.zeros(0, dtype=np.float32)

    def fit(self, texts: list[str]) -> None:
        df: Counter[str] = Counter()
        for t in texts:
            df.update(set(tokenize(t)))
        self.vocab = {term: i for i, term in enumerate(sorted(df))}
        n = len(texts)
        self.idf = np.array([math.log((1 + n) / (1 + df[t])) + 1.0 for t in sorted(df)], dtype=np.float32)

    def embed(self, texts: list[str]) -> np.ndarray:
        if not self.vocab:
            raise RuntimeError("TfidfEmbedder must be fitted before embedding")
        m = np.zeros((len(texts), len(self.vocab)), dtype=np.float32)
        for row, text in enumerate(texts):
            for term, count in Counter(tokenize(text)).items():
                col = self.vocab.get(term)
                if col is not None:
                    m[row, col] = (1.0 + math.log(count)) * self.idf[col]
        return _normalise(m)

    def state(self) -> dict:
        return {"vocab": self.vocab, "idf": self.idf.tolist()}

    def load_state(self, state: dict) -> None:
        self.vocab = dict(state["vocab"])
        self.idf = np.asarray(state["idf"], dtype=np.float32)


class SentenceTransformerEmbedder(Embedder):
    name = "sentence-transformers"

    def __init__(self, model_name: str = DEFAULT_ST_MODEL) -> None:
        from sentence_transformers import SentenceTransformer  # optional dependency

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> np.ndarray:
        vecs = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return _normalise(np.asarray(vecs, dtype=np.float32))


def get_embedder(kind: str = "auto") -> Embedder:
    """Return the requested embedder; 'auto' prefers sentence-transformers and falls back to TF-IDF."""
    kind = (kind or "auto").lower()
    if kind == "tfidf":
        return TfidfEmbedder()
    if kind in ("auto", "sentence-transformers", "st"):
        try:
            return SentenceTransformerEmbedder()
        except Exception as exc:  # ImportError, or model download failure when offline
            if kind != "auto":
                raise
            log.info("sentence-transformers unavailable (%s); using TF-IDF embeddings", exc.__class__.__name__)
            return TfidfEmbedder()
    raise ValueError(f"Unknown embedder: {kind!r}")
