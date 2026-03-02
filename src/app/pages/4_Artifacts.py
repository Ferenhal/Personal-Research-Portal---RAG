# src/app/pages/4_Artifacts.py
from __future__ import annotations
from pathlib import Path
import streamlit as st

from src.app.utils import read_jsonl, group_query_logs_by_thread, now_slug
from src.artifacts.evidence_table import (
    build_evidence_table_from_run,
    to_markdown,
    to_csv_bytes,
    to_pdf_bytes,
)

st.set_page_config(page_title="Artifacts", page_icon="📄", layout="wide")
st.title("📄 Artifacts")

log_dir = st.sidebar.text_input("Log dir", value="logs")
log_path = Path(log_dir) / "query.jsonl"

rows = read_jsonl(log_path)
if not rows:
    st.info("No logs yet. Run something in **Ask** first.")
    st.stop()

by_thread = group_query_logs_by_thread(rows)
thread_id = st.selectbox("Thread", options=sorted(by_thread.keys()), index=0)

items = by_thread[thread_id]
run_choice = st.selectbox(
    "Pick a run (most recent first)",
    options=list(range(len(items))),
    format_func=lambda i: f"{items[i].get('timestamp','')} — {items[i].get('question','')}",
)

run_log = items[run_choice]

if st.button("Generate evidence table", type="primary"):
    table = build_evidence_table_from_run(run_log)
    st.session_state["artifact_rows"] = table

table = st.session_state.get("artifact_rows")
if not table:
    st.stop()

st.subheader("Evidence table (preview)")
st.dataframe(table, use_container_width=True, hide_index=True)

title = f"Evidence Table — {thread_id}"
md = to_markdown(table, title=title)
csv_bytes = to_csv_bytes(table)
pdf_bytes = to_pdf_bytes(table, title=title)

out_dir = Path("outputs")
out_dir.mkdir(parents=True, exist_ok=True)
stamp = now_slug()

md_path = out_dir / f"artifact_evidence_table_{thread_id}_{stamp}.md"
csv_path = out_dir / f"artifact_evidence_table_{thread_id}_{stamp}.csv"
pdf_path = out_dir / f"artifact_evidence_table_{thread_id}_{stamp}.pdf"

# Write only when user clicks export to avoid creating files on every rerun
st.subheader("Export")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Write Markdown"):
        md_path.write_text(md, encoding="utf-8")
        st.success(f"Wrote {md_path}")
    st.download_button("Download Markdown", data=md.encode("utf-8"), file_name=md_path.name, mime="text/markdown")

with col2:
    if st.button("Write CSV"):
        csv_path.write_bytes(csv_bytes)
        st.success(f"Wrote {csv_path}")
    st.download_button("Download CSV", data=csv_bytes, file_name=csv_path.name, mime="text/csv")

with col3:
    if st.button("Write PDF"):
        pdf_path.write_bytes(pdf_bytes)
        st.success(f"Wrote {pdf_path}")
    st.download_button("Download PDF", data=pdf_bytes, file_name=pdf_path.name, mime="application/pdf")