"""Command-line interface.

    ragbot ingest data/*.pdf          # build / update the local index
    ragbot ask "How long do the batteries last?"
    ragbot chat                       # interactive session
    ragbot info                       # what's indexed and which backends are active
"""

from __future__ import annotations

import argparse
import glob
import sys

from .config import Settings
from .generator import Answer
from .pipeline import RAGChatbot


def _print_answer(ans: Answer, show_scores: bool = False) -> None:
    if ans.note:
        print(f"({ans.note})\n")
    print(ans.text)
    if ans.sources:
        print("\nSources:")
        for line, r in zip(ans.source_lines, ans.sources):
            print(f"  {line}" + (f"  (score {r.score:.3f})" if show_scores else ""))


def _expand(patterns: list[str]) -> list[str]:
    files: list[str] = []
    for p in patterns:  # Windows shells don't expand globs for us
        files.extend(sorted(glob.glob(p)) or [p])
    return files


def cmd_ingest(bot: RAGChatbot, args: argparse.Namespace) -> int:
    for path in _expand(args.pdfs):
        n = bot.ingest(path)
        print(f"Indexed {path}: {n} chunks")
    bot.save()
    r = bot.retriever
    print(f"Index saved to {bot.settings.index_dir} ({len(r)} chunks, embedder={r.embedder.name}, index={r.index_name})")
    return 0


def cmd_ask(bot: RAGChatbot, args: argparse.Namespace) -> int:
    if not len(bot.retriever):
        print("The index is empty. Run `ragbot ingest <file.pdf>` first.", file=sys.stderr)
        return 1
    _print_answer(bot.ask(args.question, k=args.k), args.scores)
    return 0


def cmd_chat(bot: RAGChatbot, args: argparse.Namespace) -> int:
    if not len(bot.retriever):
        print("The index is empty. Run `ragbot ingest <file.pdf>` first.", file=sys.stderr)
        return 1
    print("Ask about your documents (empty line or Ctrl+C to quit).")
    history: list[dict] = []
    while True:
        try:
            q = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            break
        ans = bot.ask(q, history=history[-6:], k=args.k)
        print()
        _print_answer(ans, args.scores)
        history += [{"role": "user", "content": q}, {"role": "assistant", "content": ans.text}]
    return 0


def cmd_info(bot: RAGChatbot, args: argparse.Namespace) -> int:
    r, s = bot.retriever, bot.settings
    print(f"Index dir : {s.index_dir}")
    print(f"Documents : {', '.join(r.sources) or '(none)'}")
    print(f"Chunks    : {len(r)}")
    print(f"Embedder  : {r.embedder.name}")
    print(f"Index     : {r.index_name}")
    print(f"Generator : {s.model if s.api_key else 'extractive (no ANTHROPIC_API_KEY)'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ragbot", description="Chat with your PDFs (RAG with page-level citations).")
    sub = p.add_subparsers(dest="command", required=True)

    pi = sub.add_parser("ingest", help="parse, chunk, embed and index PDF files")
    pi.add_argument("pdfs", nargs="+", help="PDF paths or glob patterns")
    pi.set_defaults(func=cmd_ingest)

    for name, func, helptext in (("ask", cmd_ask, "ask a single question"), ("chat", cmd_chat, "interactive chat")):
        sp = sub.add_parser(name, help=helptext)
        if name == "ask":
            sp.add_argument("question")
        sp.add_argument("-k", type=int, default=None, help="number of chunks to retrieve (default: RAG_TOP_K)")
        sp.add_argument("--scores", action="store_true", help="show retrieval similarity scores")
        sp.set_defaults(func=func)

    sub.add_parser("info", help="show index status").set_defaults(func=cmd_info)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    bot = RAGChatbot.from_disk(Settings())
    return args.func(bot, args)


if __name__ == "__main__":
    raise SystemExit(main())
