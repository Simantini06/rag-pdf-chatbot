"""Runtime settings, read from environment variables (and an optional .env file)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:  # python-dotenv is optional at runtime
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    return int(raw) if raw else default


@dataclass
class Settings:
    model: str = field(default_factory=lambda: os.getenv("RAG_MODEL", "claude-sonnet-5-5"))
    embedder: str = field(default_factory=lambda: os.getenv("RAG_EMBEDDER", "auto").strip().lower())
    chunk_size: int = field(default_factory=lambda: _int_env("RAG_CHUNK_SIZE", 800))
    chunk_overlap: int = field(default_factory=lambda: _int_env("RAG_CHUNK_OVERLAP", 150))
    top_k: int = field(default_factory=lambda: _int_env("RAG_TOP_K", 4))
    index_dir: Path = field(default_factory=lambda: Path(os.getenv("RAG_INDEX_DIR", ".rag_index")))

    @property
    def api_key(self) -> str | None:
        key = os.getenv("ANTHROPIC_API_KEY", "").strip()
        return key or None
