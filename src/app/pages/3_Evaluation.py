from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path
from statistics import mean
from src.app.utils import read_jsonl

import streamlit as st

st.set_page_config(page_title="Evaluation", page_icon="🧪", layout="wide")

st.title("🧪 Evaluation")

with st.sidebar:
    st.header("Eval settings")
    queries_path = st.text_input("Queries JSONL", value="data/eval/queries.jsonl")
    index_dir = st.text_input("Index dir", value="data/index")
    chunks_path = st.text_input("Chunks path", value="data/chunks/chunks.jsonl")
    log_dir = st.text_input("Log dir", value="logs")
    local_model = st.text_input("Ollama model", value="llama3.2")
    manifest_path = st.text_input("Manifest (enriched)", value="data/manifest/manifest.enriched.csv")

st.markdown(
    """
This page runs the existing evaluation script (`python -m src.eval`) against the queries file.

It will write:
- `logs/eval.jsonl`
- `logs/eval_summary.csv`

and then render the CSV here.
    """
)

run = st.button("Run evaluation", type="primary")

if run:
    cmd = [
        sys.executable,
        "-m",
        "src.eval",
        "--queries",
        queries_path,
        "--index_dir",
        index_dir,
        "--chunks_path",
        chunks_path,
        "--log_dir",
        log_dir,
        "--local_model",
        local_model,
        "--manifest_path",
        manifest_path,
    ]

    with st.spinner("Running eval… this can take a while (multiple queries)."):
        proc = subprocess.run(cmd, capture_output=True, text=True)

    if proc.returncode != 0:
        st.error("Eval failed.")
        st.code(proc.stderr or proc.stdout, language="text")
    else:
        st.success("Eval finished.")
        if proc.stdout:
            st.code(proc.stdout, language="text")

summary_csv = Path(log_dir) / "eval_summary.csv"

if not summary_csv.exists():
    st.info("No eval summary found yet. Click **Run evaluation**.")
    st.stop()

rows = []
with summary_csv.open("r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for r in reader:
        rows.append(r)

if not rows:
    st.warning("eval_summary.csv exists but is empty.")
    st.stop()

# Compute quick aggregates
precisions = [float(r.get("citation_precision", 0.0) or 0.0) for r in rows]
coverages = [float(r.get("citation_coverage_proxy", 0.0) or 0.0) for r in rows]
invalids = [int(float(r.get("num_invalid_citations", 0) or 0)) for r in rows]

c1, c2, c3, c4 = st.columns(4)

c1.metric("Queries", value=str(len(rows)))
c2.metric("Mean citation precision", value=f"{mean(precisions):.2f}")
c3.metric("Mean citation coverage", value=f"{mean(coverages):.2f}")
c4.metric("Total invalid citations", value=str(sum(invalids)))

st.subheader("eval_summary.csv")
st.dataframe(rows, use_container_width=True, hide_index=True)

st.caption(f"Loaded from `{summary_csv}`")

st.divider()
st.subheader("Representative examples")

query_log = Path(log_dir) / "query.jsonl"
query_rows = read_jsonl(query_log) if query_log.exists() else []

# Map question -> most recent matching query log entry
qmap = {}
for r in sorted(query_rows, key=lambda x: x.get("timestamp", ""), reverse=True):
    q = r.get("question")
    if q and q not in qmap:
        qmap[q] = r

def ffloat(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default

def fint(x, default=0):
    try:
        return int(float(x))
    except Exception:
        return default

def show_examples(title: str, picked: list[dict]):
    st.markdown(f"### {title}")
    if not picked:
        st.info("No matching examples found.")
        return

    for r in picked:
        q = r.get("question", "")
        pid = r.get("id", "")
        prec = ffloat(r.get("citation_precision", 0.0))
        cov = ffloat(r.get("citation_coverage_proxy", 0.0))
        inv = fint(r.get("num_invalid_citations", 0))

        label = f"{pid} — precision={prec:.2f}, coverage={cov:.2f}, invalid={inv} — {q}"
        with st.expander(label, expanded=False):
            qr = qmap.get(q)

            if not qr:
                st.warning("No matching entry found in logs/query.jsonl for this question.")
                continue

            ans = qr.get("generation", {}).get("answer", "")
            invalid_list = qr.get("generation", {}).get("citation_validation", {}).get("invalid_citations", [])
            evidence = qr.get("retrieval", {}).get("evidence", []) or []

            st.markdown("**Answer**")
            st.markdown(ans if ans else "_(empty answer)_")

            if invalid_list:
                st.markdown("**Invalid citations (from query log):**")
                st.code(", ".join(invalid_list), language="text")

            st.markdown("**Evidence (top 3 shown):**")
            for ev in evidence[:3]:
                cid = ev.get("chunk_id", "")
                title2 = ev.get("title", "")
                section = ev.get("section", "")
                score = ev.get("score", None)
                hdr = f"{cid} | score={score:.3f} | {title2} — {section}" if isinstance(score, (int, float)) else f"{cid} | {title2} — {section}"
                with st.expander(hdr, expanded=False):
                    st.code(ev.get("text", ""), language="text")

# Build three “representative” slices from eval_summary.csv (already loaded into `rows`)
worst_precision = sorted(rows, key=lambda r: ffloat(r.get("citation_precision", 1.0)))[:5]
has_invalid = sorted(
    [r for r in rows if fint(r.get("num_invalid_citations", 0)) > 0],
    key=lambda r: fint(r.get("num_invalid_citations", 0)),
    reverse=True,
)[:5]
worst_coverage = sorted(rows, key=lambda r: ffloat(r.get("citation_coverage_proxy", 1.0)))[:5]

show_examples("Worst 5 by citation precision", worst_precision)
show_examples("Top 5 with invalid citations", has_invalid)
show_examples("Worst 5 by citation coverage (proxy)", worst_coverage)
