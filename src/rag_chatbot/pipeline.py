"""High-level RAG pipeline used by both the CLI and the Streamlit app."""

from __future__ import annotations

from pathlib import Path
from typing import BinaryIO

from .config import Settings
from .embeddings import get_embedder
from .generator import Answer, answer_question
from .ingest import chunk_pages, load_pdf
from .retriever import Retriever


class RAGChatbot:
    def __init__(self, settings: Settings | None = None, retriever: Retriever | None = None) -> None:
        self.settings = settings or Settings()
        self.retriever = retriever or Retriever(get_embedder(self.settings.embedder))

    @classmethod
    def from_disk(cls, settings: Settings | None = None) -> "RAGChatbot":
        settings = settings or Settings()
        if Retriever.exists(settings.index_dir):
            return cls(settings, Retriever.load(settings.index_dir, get_embedder(settings.embedder)))
        return cls(settings)

    def ingest(self, file: str | Path | BinaryIO, source_name: str | None = None) -> int:
        """Parse, chunk and index one PDF. Returns the number of chunks added."""
        pages = load_pdf(file, source_name)
        chunks = chunk_pages(pages, self.settings.chunk_size, self.settings.chunk_overlap)
        return self.retriever.add_chunks(chunks)

    def save(self) -> None:
        self.retriever.save(self.settings.index_dir)

    def ask(self, question: str, history: list[dict] | None = None, k: int | None = None) -> Answer:
        results = self.retriever.search(question, k or self.settings.top_k)
        return answer_question(question, results, self.settings.api_key, self.settings.model, history)
