import pytest

from rag_chatbot.embeddings import TfidfEmbedder
from rag_chatbot.ingest import Chunk
from rag_chatbot.retriever import Retriever


@pytest.fixture
def corpus() -> list[Chunk]:
    texts = [
        ("guide.pdf", 1, "Mount the sensor mast at least 1.5 metres above the ground, away from walls and trees."),
        ("guide.pdf", 2, "With lithium cells the sensor mast battery lasts about 18 months. Alkaline cells last 8 months."),
        ("guide.pdf", 3, "The warranty covers manufacturing defects for two years but not lightning damage."),
        ("notes.pdf", 1, "Clean the rain gauge funnel with water every three months to keep readings accurate."),
    ]
    return [Chunk(id=f"{s}:p{p}:c0", source=s, page=p, text=t) for s, p, t in texts]


@pytest.fixture
def retriever(corpus) -> Retriever:
    r = Retriever(TfidfEmbedder(), index_backend="numpy")
    r.add_chunks(corpus)
    return r


@pytest.fixture(autouse=True)
def no_api_key(monkeypatch):
    """Tests must never hit the network: make sure no real key leaks in from the environment."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
