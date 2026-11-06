import numpy as np

from rag_chatbot.embeddings import TfidfEmbedder, tokenize
from rag_chatbot.ingest import Chunk
from rag_chatbot.retriever import Retriever
from rag_chatbot.vector_store import NumpyIndex


def test_tokenize_drops_stopwords_and_lowercases():
    assert tokenize("The Battery is LOW") == ["battery", "low"]


def test_tfidf_vectors_are_normalised(corpus):
    emb = TfidfEmbedder()
    emb.fit([c.text for c in corpus])
    vecs = emb.embed([c.text for c in corpus])
    assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0, atol=1e-5)


def test_numpy_index_orders_by_cosine():
    vecs = np.array([[1, 0], [0.6, 0.8], [0, 1]], dtype=np.float32)
    hits = NumpyIndex(vecs).search(np.array([0, 1], dtype=np.float32), k=3)
    assert [i for i, _ in hits] == [2, 1, 0]
    assert hits[0][1] >= hits[1][1] >= hits[2][1]


def test_numpy_index_k_larger_than_corpus():
    vecs = np.eye(2, dtype=np.float32)
    assert len(NumpyIndex(vecs).search(np.array([1, 0], dtype=np.float32), k=10)) == 2


def test_relevant_chunk_ranks_first(retriever):
    assert retriever.search("How long does the battery last?", k=2)[0].chunk.page == 2
    assert retriever.search("lightning warranty", k=2)[0].chunk.page == 3
    top = retriever.search("cleaning the rain gauge funnel", k=2)[0].chunk
    assert (top.source, top.page) == ("notes.pdf", 1)


def test_scores_are_descending(retriever):
    scores = [r.score for r in retriever.search("sensor mast battery", k=4)]
    assert scores == sorted(scores, reverse=True)


def test_reingesting_a_file_replaces_its_chunks(retriever, corpus):
    before = len(retriever)
    retriever.add_chunks([Chunk("notes.pdf:p1:c0", "notes.pdf", 1, "Completely new note about solar panels.")])
    assert len(retriever) == before  # old notes.pdf chunk replaced, not duplicated
    assert retriever.search("solar panels", k=1)[0].chunk.source == "notes.pdf"


def test_empty_query_and_empty_index():
    assert Retriever(TfidfEmbedder(), "numpy").search("anything") == []


def test_save_and_load_roundtrip(retriever, tmp_path):
    retriever.save(tmp_path)
    loaded = Retriever.load(tmp_path, TfidfEmbedder(), index_backend="numpy")
    q = "warranty for manufacturing defects"
    assert [r.chunk.id for r in loaded.search(q, 3)] == [r.chunk.id for r in retriever.search(q, 3)]


def test_tokenize_normalises_plurals():
    assert tokenize("batteries lasts glass") == ["battery", "last", "glass"]
