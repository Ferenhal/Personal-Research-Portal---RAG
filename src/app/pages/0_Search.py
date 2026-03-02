# src/app/pages/0_Search.py
from __future__ import annotations
from pathlib import Path
import streamlit as st

from src.retrieve import retrieve_topk

st.set_page_config(page_title="Search", page_icon="🔎", layout="wide")
st.title("🔎 Search")

with st.sidebar:
    index_dir = st.text_input("Index dir", value="data/index")
    chunks_path = st.text_input("Chunks path", value="data/chunks/chunks.jsonl")
    k = st.slider("Top-k", 1, 50, 10)

q = st.text_input("Search query", placeholder="keywords or a question…")

run = st.button("Search", type="primary", disabled=not bool(q.strip()))

if run:
    out = retrieve_topk(
        question=q.strip(),
        k=int(k),
        index_dir=Path(index_dir),
        chunks_path=Path(chunks_path),
    )
    st.session_state["search_out"] = out

out = st.session_state.get("search_out")
if not out:
    st.info("Enter a query and click **Search**.")
    st.stop()

results = out.get("results", [])
st.caption(f"Retrieved {len(results)} chunks (model: {out.get('embedding_model')})")

# Summary table
st.dataframe(
    [
        {
            "rank": r["rank"],
            "score": round(r["score"], 4),
            "chunk_id": r["chunk_id"],
            "source_id": r["source_id"],
            "title": r.get("title", ""),
            "section": r.get("section", ""),
        }
        for r in results
    ],
    use_container_width=True,
    hide_index=True,
)

st.subheader("Results")
for r in results:
    header = f"{r['chunk_id']} | score={r['score']:.3f} | {r.get('title','')} — {r.get('section','')}"
    with st.expander(header, expanded=False):
        st.code(r.get("text", ""), language="text")