from pathlib import Path

import pytest

from rag_chatbot.ingest import Page, chunk_pages, chunk_text, clean_text, load_pdf

SAMPLE_PDF = Path(__file__).resolve().parent.parent / "data" / "sample_weather_station_guide.pdf"


def make_text(n_words: int) -> str:
    return " ".join(f"word{i}" for i in range(n_words))


def test_short_text_is_single_chunk():
    assert chunk_text("hello world", chunk_size=100, overlap=10) == ["hello world"]


def test_empty_text_gives_no_chunks():
    assert chunk_text("   ", chunk_size=100, overlap=10) == []


def test_chunks_respect_size_limit():
    chunks = chunk_text(make_text(500), chunk_size=200, overlap=40)
    assert len(chunks) > 1
    assert all(len(c) <= 200 for c in chunks)


def test_consecutive_chunks_overlap():
    chunks = chunk_text(make_text(300), chunk_size=120, overlap=30)
    for a, b in zip(chunks, chunks[1:]):
        tail = a.split()[-1]
        assert tail in b.split(), "last word of a chunk should reappear in the next one"


def test_zero_overlap_partitions_the_words():
    words = make_text(250).split()
    chunks = chunk_text(" ".join(words), chunk_size=100, overlap=0)
    assert " ".join(chunks).split() == words


def test_all_words_covered_with_overlap():
    words = make_text(400).split()
    chunks = chunk_text(" ".join(words), chunk_size=150, overlap=50)
    assert set(" ".join(chunks).split()) == set(words)
    assert chunks[0].split()[0] == words[0] and chunks[-1].split()[-1] == words[-1]


def test_word_longer_than_chunk_is_kept_whole():
    long_word = "x" * 50
    assert chunk_text(f"a {long_word} b", chunk_size=20, overlap=5) == ["a", long_word, "b"]


@pytest.mark.parametrize("size,overlap", [(0, 0), (100, 100), (100, -1)])
def test_invalid_parameters(size, overlap):
    with pytest.raises(ValueError):
        chunk_text("some text", chunk_size=size, overlap=overlap)


def test_chunk_pages_keeps_page_metadata():
    pages = [Page("a.pdf", 1, make_text(100)), Page("a.pdf", 2, "short page")]
    chunks = chunk_pages(pages, chunk_size=200, overlap=20)
    assert {c.page for c in chunks} == {1, 2}
    assert all(c.source == "a.pdf" for c in chunks)
    assert len({c.id for c in chunks}) == len(chunks)


def test_clean_text_joins_hyphenation_and_whitespace():
    assert clean_text("rain-\nfall   is\n\nhigh") == "rainfall is high"


@pytest.mark.skipif(not SAMPLE_PDF.exists(), reason="sample PDF not generated")
def test_load_sample_pdf():
    pages = load_pdf(SAMPLE_PDF)
    assert [p.page for p in pages] == [1, 2, 3]
    assert pages[0].source == SAMPLE_PDF.name
    assert "Orbit-7" in pages[0].text
