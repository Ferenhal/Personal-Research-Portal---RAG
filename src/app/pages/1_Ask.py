from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.query import run_query
from src.app.utils import (
    extract_citations,
    group_query_logs_by_thread,
    now_slug,
    read_jsonl,
    safe_thread_id,
)

st.set_page_config(page_title="Ask", page_icon="💬", layout="wide")

LOG_PATH_DEFAULT = Path("logs/query.jsonl")

st.title("💬 Ask")

# ----------------------------
# Sidebar: corpus + model settings
# ----------------------------
with st.sidebar:
    st.header("Settings")

    index_dir = st.text_input("Index dir", value="data/index")
    chunks_path = st.text_input("Chunks path", value="data/chunks/chunks.jsonl")
    manifest_path = st.text_input("Manifest (enriched)", value="data/manifest/manifest.enriched.csv")
    log_dir = st.text_input("Log dir", value="logs")

    st.divider()

    k = st.slider("Top-k", min_value=1, max_value=12, value=5, step=1)

    st.divider()

    st.subheader("Local generator (Ollama)")
    local_model = st.text_input("Model", value="llama3.2")
    ollama_base_url = st.text_input("Base URL", value="http://localhost:11434")
    num_ctx = st.number_input("Context window (num_ctx)", min_value=1024, max_value=32768, value=4096, step=512)
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.05)
    timeout_s = st.number_input("Timeout (seconds)", min_value=10, max_value=600, value=180, step=10)

    st.divider()

    # Thread selection (based on existing logs)
    log_path = Path(log_dir) / "query.jsonl"
    rows = read_jsonl(log_path) if log_path.exists() else []
    by_thread = group_query_logs_by_thread(rows)
    thread_ids = sorted(by_thread.keys())

    thread_choice = st.selectbox(
        "Thread",
        options=["(new thread)"] + thread_ids,
        index=0,
        help="Threads are grouped by thread_id in logs/query.jsonl",
    )

    if thread_choice == "(new thread)":
        thread_label = st.text_input("New thread label", value="my-research")
        thread_id = f"{safe_thread_id(thread_label)}_{now_slug()}"
        st.caption(f"Will use thread_id: `{thread_id}`")
    else:
        thread_id = thread_choice

# ----------------------------
# Main: question input + answer
# ----------------------------
question = st.text_area("Question", height=100, placeholder="Ask something grounded in the corpus…")

col_run, col_clear = st.columns([1, 1])

with col_run:
    run_clicked = st.button("Run", type="primary", disabled=not bool(question.strip()))

with col_clear:
    if st.button("Clear"):
        st.session_state.pop("last_result", None)
        st.rerun()

if run_clicked:
    with st.spinner("Retrieving evidence and generating answer…"):
        out = run_query(
            question=question.strip(),
            k=int(k),
            index_dir=index_dir,
            chunks_path=chunks_path,
            log_dir=log_dir,
            manifest_path=manifest_path,
            local_model=local_model,
            ollama_base_url=ollama_base_url,
            num_ctx=int(num_ctx),
            temperature=float(temperature),
            timeout_s=int(timeout_s),
            thread_id=thread_id,
        )
    st.session_state["last_result"] = out

out = st.session_state.get("last_result")

if out:
    st.subheader("Answer")
    st.markdown(out["answer"])

    # ----------------------------
    # Trust UX: suggest next retrieval steps when missing evidence
    # ----------------------------
    answer_text = (out.get("answer") or "").strip().lower()
    evidence = out.get("evidence") or []
    cited_now = extract_citations(out.get("answer", ""))

    is_not_found = (
        answer_text.startswith("not found in the corpus")
        or answer_text.startswith("not found")
        or (len(evidence) == 0 and len(cited_now) == 0)
    )

    if is_not_found:
        st.warning("This question didn't have enough support in the current corpus/index.")
        st.markdown(
            """
            **Next retrieval steps**
            - Try broader keywords (remove dates, specific names, or extra constraints).
            - Increase **Top-k** (try 10-12) and rerun.
            - Use the **Search** page to find the best terms (titles/sections) and then re-ask.
            - If you believe the answer should exist, add more documents and re-run ingest → chunk → index.
            """
            )

        # Optional convenience: one-click rerun with higher Top-k
        if int(k) < 12 and st.button("Rerun with Top-k = 12"):
            with st.spinner("Retrying with higher Top-k…"):
                out2 = run_query(
                    question=question.strip(),
                    k=12,
                    index_dir=index_dir,
                    chunks_path=chunks_path,
                    log_dir=log_dir,
                    manifest_path=manifest_path,
                    local_model=local_model,
                    ollama_base_url=ollama_base_url,
                    num_ctx=int(num_ctx),
                    temperature=float(temperature),
                    timeout_s=int(timeout_s),
                    thread_id=thread_id,
                )
            st.session_state["last_result"] = out2
            st.rerun()

    st.divider()

    cited = extract_citations(out.get("answer", ""))
    valid = out.get("valid_citations") or []
    invalid = out.get("invalid_citations") or []

    c1, c2, c3 = st.columns(3)
    c1.metric("Citations", value=str(len(cited)))
    c2.metric("Valid", value=str(len(valid)))
    c3.metric("Invalid", value=str(len(invalid)))

    if invalid:
        st.warning(f"Invalid citations detected: {', '.join(invalid)}")

    st.subheader("Retrieved evidence")
    evidence = out.get("evidence") or []

    for ev in evidence:
        cid = ev.get("chunk_id")
        title = ev.get("title", "")
        section = ev.get("section", "")
        score = ev.get("score")

        header = f"{cid} | score={score:.3f} | {title} — {section}" if isinstance(score, (int, float)) else f"{cid} | {title} — {section}"
        with st.expander(header, expanded=False):
            st.code(ev.get("text", ""), language="text")

    st.caption(f"Saved to logs with thread_id = `{out['log_entry'].get('thread_id')}`")
else:
    st.info("Enter a question and hit **Run**.")
