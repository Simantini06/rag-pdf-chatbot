from rag_chatbot.config import Settings
from rag_chatbot.generator import (
    Answer,
    answer_question,
    build_context,
    extractive_answer,
    format_citation,
    format_sources,
)
from rag_chatbot.pipeline import RAGChatbot


# ------------------------------------------------------------------ citations
def test_format_citation(retriever):
    res = retriever.search("battery", k=1)[0]
    assert format_citation(res) == "guide.pdf, p. 2"


def test_format_sources_numbering(retriever):
    results = retriever.search("sensor mast", k=3)
    lines = format_sources(results)
    assert len(lines) == 3
    assert lines[0].startswith("[1] ") and lines[2].startswith("[3] ")
    assert all(", p. " in line for line in lines)


def test_context_markers_match_sources(retriever):
    results = retriever.search("warranty", k=2)
    ctx = build_context(results)
    assert ctx.startswith("[1] (")
    assert "[2] (" in ctx
    assert results[0].chunk.text in ctx


# ------------------------------------------------------------- no-key fallback
def test_no_key_returns_extractive_answer(retriever):
    results = retriever.search("How long does the battery last with lithium cells?", k=3)
    ans = answer_question("How long does the battery last with lithium cells?", results, api_key=None)
    assert ans.mode == "extractive"
    assert "18 months" in ans.text
    assert "[1]" in ans.text  # the battery chunk is ranked first and cited
    assert ans.note and "ANTHROPIC_API_KEY" in ans.note


def test_extractive_with_no_results():
    ans = extractive_answer("anything", [])
    assert ans.sources == [] and "couldn't find" in ans.text


def test_extractive_does_not_repeat_sentences(retriever):
    results = retriever.search("battery", k=2)
    doubled = results + results  # simulate overlapping chunks
    ans = extractive_answer("battery", doubled, max_sentences=10)
    bullet_lines = [line for line in ans.text.splitlines() if line.startswith("- ")]
    texts = [line.rsplit(" [", 1)[0] for line in bullet_lines]
    assert len(texts) == len(set(texts))


def test_api_error_falls_back_to_extractive(retriever):
    import anthropic

    class FailingGenerator:
        def generate(self, *a, **kw):
            raise anthropic.APIConnectionError(request=None)  # type: ignore[arg-type]

    results = retriever.search("warranty", k=2)
    ans = answer_question("warranty?", results, api_key="sk-test", generator=FailingGenerator())
    assert ans.mode == "extractive" and "APIConnectionError" in (ans.note or "")


def test_llm_generator_used_when_available(retriever):
    class FakeGenerator:
        def generate(self, question, results, history=None):
            return Answer("Two years [1].", results, "llm")

    results = retriever.search("warranty", k=2)
    ans = answer_question("warranty?", results, api_key="sk-test", generator=FakeGenerator())
    assert ans.mode == "llm" and ans.text == "Two years [1]."


def test_pipeline_end_to_end_without_key(retriever, tmp_path):
    bot = RAGChatbot(Settings(index_dir=tmp_path, top_k=2), retriever)
    ans = bot.ask("When should I clean the rain gauge funnel?")
    assert ans.mode == "extractive"
    assert ans.source_lines[0] == "[1] notes.pdf, p. 1"


def test_split_sentences_glues_short_headings():
    from rag_chatbot.generator import split_sentences

    assert split_sentences("Warranty. The device has a two-year warranty. It is great value overall.") == [
        "Warranty. The device has a two-year warranty.",
        "It is great value overall.",
    ]
