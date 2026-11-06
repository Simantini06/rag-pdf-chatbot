"""Streamlit chat UI.   Run with:  streamlit run app.py"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))  # allow running without `pip install -e .`

from rag_chatbot.config import Settings  # noqa: E402
from rag_chatbot.pipeline import RAGChatbot  # noqa: E402

st.set_page_config(page_title="RAG PDF Chatbot", page_icon=":books:", layout="wide")


@st.cache_resource
def get_bot() -> RAGChatbot:
    return RAGChatbot.from_disk(Settings())


bot = get_bot()
st.session_state.setdefault("messages", [])

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Documents")
    uploads = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
    if uploads and st.button("Index uploaded PDFs", type="primary"):
        with st.spinner("Parsing, chunking and embedding..."):
            for f in uploads:
                n = bot.ingest(f, source_name=f.name)
                st.success(f"{f.name}: {n} chunks")
            bot.save()

    r, s = bot.retriever, bot.settings
    st.caption(
        f"**Indexed:** {', '.join(r.sources) or 'nothing yet'}  \n"
        f"**Chunks:** {len(r)} · **Embedder:** {r.embedder.name} · **Index:** {r.index_name}  \n"
        f"**Generator:** {s.model if s.api_key else 'extractive (no API key)'}"
    )
    s.top_k = st.slider("Chunks to retrieve (top-k)", 1, 10, s.top_k)
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

st.title("Chat with your PDFs")
if not len(bot.retriever):
    st.info("Upload one or more PDFs in the sidebar to get started.")


def render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})"):
        for src in sources:
            st.markdown(f"**{src['label']}** · score {src['score']:.3f}")
            st.caption(src["text"])


for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        render_sources(m.get("sources", []))

if question := st.chat_input("Ask a question about your documents", disabled=not len(bot.retriever)):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages[:-1]][-6:]
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            ans = bot.ask(question, history=history)
        if ans.note:
            st.caption(ans.note)
        st.markdown(ans.text)
        sources = [
            {"label": label, "score": res.score, "text": res.chunk.text}
            for label, res in zip(ans.source_lines, ans.sources)
        ]
        render_sources(sources)
    st.session_state.messages.append({"role": "assistant", "content": ans.text, "sources": sources})
