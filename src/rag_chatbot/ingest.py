"""PDF parsing and overlapping chunking.

Chunks never span a page boundary, so every chunk can be cited as (file, page).
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import BinaryIO, Iterable

from pypdf import PdfReader


@dataclass(frozen=True)
class Page:
    source: str  # file name shown in citations
    page: int  # 1-indexed page number
    text: str


@dataclass(frozen=True)
class Chunk:
    id: str
    source: str
    page: int
    text: str

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Chunk":
        return cls(**d)


_WS = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Normalise whitespace and re-join words hyphenated across line breaks."""
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    return _WS.sub(" ", text).strip()


def load_pdf(file: str | Path | BinaryIO, source_name: str | None = None) -> list[Page]:
    """Extract the text of every page that has any. Accepts a path or a file-like object."""
    if source_name is None:
        source_name = Path(file).name if isinstance(file, (str, Path)) else getattr(file, "name", "document.pdf")
    reader = PdfReader(file)
    pages: list[Page] = []
    for i, pdf_page in enumerate(reader.pages, start=1):
        text = clean_text(pdf_page.extract_text() or "")
        if text:
            pages.append(Page(source=Path(source_name).name, page=i, text=text))
    return pages


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> list[str]:
    """Split text into ~chunk_size-character chunks on word boundaries.

    Consecutive chunks share roughly `overlap` characters so that a sentence cut
    at a boundary still appears whole in at least one chunk. A single word longer
    than chunk_size becomes its own chunk rather than being split mid-word.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must satisfy 0 <= overlap < chunk_size")

    words = text.split()
    chunks: list[str] = []
    start = 0
    while start < len(words):
        end, length = start, 0
        while end < len(words):
            add = len(words[end]) + (1 if end > start else 0)
            if length + add > chunk_size and end > start:
                break
            length += add
            end += 1
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        # Step back from `end` to re-include up to `overlap` characters, always advancing.
        back, olen = end, 0
        while back > start + 1 and olen + len(words[back - 1]) + 1 <= overlap:
            back -= 1
            olen += len(words[back]) + 1
        start = back
    return chunks


def chunk_pages(pages: Iterable[Page], chunk_size: int = 800, overlap: int = 150) -> list[Chunk]:
    chunks: list[Chunk] = []
    for page in pages:
        for n, piece in enumerate(chunk_text(page.text, chunk_size, overlap)):
            chunks.append(Chunk(id=f"{page.source}:p{page.page}:c{n}", source=page.source, page=page.page, text=piece))
    return chunks
