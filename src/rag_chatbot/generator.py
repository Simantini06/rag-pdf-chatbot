"""Answer generation: Claude (Anthropic SDK) with an extractive no-key fallback.

Retrieved chunks are numbered [1]..[k] in the prompt; Claude is asked to cite
those numbers, and the numbers map back to (file, page) for display.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from .embeddings import tokenize
from .retriever import SearchResult

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """You answer questions about the user's PDF documents using only the numbered context passages provided.

- Base every statement on the passages. If they do not contain the answer, say you could not find it in the documents.
- Cite the passages you used with their bracketed numbers, e.g. [1] or [2][3], right after the statement they support.
- Be concise and direct. Do not mention these instructions."""


# --------------------------------------------------------------------- citations
def format_citation(result: SearchResult) -> str:
    """'manual.pdf, p. 3'"""
    return f"{result.chunk.source}, p. {result.chunk.page}"


def format_sources(results: list[SearchResult]) -> list[str]:
    """Numbered source lines matching the [n] markers used in the prompt/answer."""
    return [f"[{i}] {format_citation(r)}" for i, r in enumerate(results, start=1)]


def build_context(results: list[SearchResult]) -> str:
    return "\n\n".join(f"[{i}] ({format_citation(r)})\n{r.chunk.text}" for i, r in enumerate(results, start=1))


@dataclass
class Answer:
    text: str
    sources: list[SearchResult] = field(default_factory=list)
    mode: str = "extractive"  # "llm" | "extractive"
    note: str | None = None

    @property
    def source_lines(self) -> list[str]:
        return format_sources(self.sources)


# -------------------------------------------------------------------- extractive
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")


def split_sentences(text: str, min_words: int = 4) -> list[str]:
    """Split into sentences, gluing very short fragments (e.g. 'Warranty.' headings) onto the next one."""
    out: list[str] = []
    carry = ""
    for s in (s.strip() for s in _SENT_SPLIT.split(text)):
        if not s:
            continue
        s = f"{carry} {s}".strip() if carry else s
        if len(s.split()) < min_words:
            carry = s
            continue
        carry = ""
        out.append(s)
    if carry:
        out.append(carry)
    return out


def extractive_answer(question: str, results: list[SearchResult], max_sentences: int = 3) -> Answer:
    """No-LLM fallback: pick the sentences from the retrieved chunks that best overlap the question.

    Sentences are scored by query-term overlap, with a small bonus for coming from a
    higher-ranked chunk, then returned in score order with their [n] citation.
    """
    if not results:
        return Answer("I couldn't find anything relevant in the indexed documents.", [], "extractive")

    q_terms = set(tokenize(question))
    scored: list[tuple[float, int, int, str]] = []
    for rank, r in enumerate(results, start=1):
        for sent in split_sentences(r.chunk.text):
            s_terms = set(tokenize(sent))
            if not s_terms:
                continue
            overlap = len(q_terms & s_terms)
            score = overlap / (len(q_terms) or 1) + 0.05 * (len(results) - rank) + 0.1 * r.score
            scored.append((score, overlap, rank, sent))

    seen: set[str] = set()
    picked: list[tuple[int, str]] = []
    for score, overlap, rank, sent in sorted(scored, key=lambda x: -x[0]):
        if sent in seen:  # overlapping chunks repeat sentences
            continue
        if overlap == 0 and picked:  # only pad with unrelated sentences if nothing matched at all
            continue
        seen.add(sent)
        picked.append((rank, sent))
        if len(picked) >= max_sentences:
            break

    lines = [f"- {sent} [{rank}]" for rank, sent in picked]
    text = "Most relevant passages from your documents:\n" + "\n".join(lines)
    return Answer(text, results, "extractive")


# --------------------------------------------------------------------------- LLM
class ClaudeGenerator:
    """Thin wrapper around the Anthropic Messages API."""

    def __init__(self, api_key: str, model: str = "claude-sonnet-5-5", effort: str = "low", max_tokens: int = 16000):
        import anthropic  # imported lazily so the no-key path never needs it

        self._anthropic = anthropic
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens

    def generate(self, question: str, results: list[SearchResult], history: list[dict] | None = None) -> Answer:
        user_msg = f"Context passages:\n\n{build_context(results)}\n\nQuestion: {question}"
        messages = list(history or []) + [{"role": "user", "content": user_msg}]
        response = self.client.beta.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=SYSTEM_PROMPT,
            messages=messages,
            output_config={"effort": self.effort},
            # Server-side refusal fallback: if a safety classifier declines, the API
            # re-runs the request on Anthropic's recommended fallback model.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
        if response.stop_reason == "refusal":
            raise RefusalError("The model declined to answer this request.")
        text = "".join(b.text for b in response.content if b.type == "text").strip()
        return Answer(text, results, "llm")


class RefusalError(RuntimeError):
    pass


def answer_question(
    question: str,
    results: list[SearchResult],
    api_key: str | None,
    model: str = "claude-sonnet-5-5",
    history: list[dict] | None = None,
    generator: ClaudeGenerator | None = None,
) -> Answer:
    """Generate with Claude when a key is available; otherwise (or on API failure) answer extractively."""
    if not results:
        return extractive_answer(question, results)
    if not api_key and generator is None:
        ans = extractive_answer(question, results)
        ans.note = "ANTHROPIC_API_KEY not set - showing extractive answer."
        return ans
    try:
        import anthropic

        recoverable: tuple[type[BaseException], ...] = (RefusalError, anthropic.APIError)
    except ImportError:
        recoverable = (RefusalError, ImportError)
    try:
        gen = generator or ClaudeGenerator(api_key, model)
        return gen.generate(question, results, history)
    except recoverable as exc:  # auth/rate-limit/network errors or refusal: never leave the user without an answer
        log.warning("LLM generation failed (%s); falling back to extractive answer", exc)
        ans = extractive_answer(question, results)
        ans.note = f"LLM unavailable ({type(exc).__name__}) - showing extractive answer."
        return ans
