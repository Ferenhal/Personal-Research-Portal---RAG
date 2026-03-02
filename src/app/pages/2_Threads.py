from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.app.utils import group_query_logs_by_thread, now_slug, read_jsonl, render_thread_markdown

st.set_page_config(page_title="Threads", page_icon="🧵", layout="wide")

st.title("🧵 Threads")

log_dir = st.sidebar.text_input("Log dir", value="logs")
log_path = Path(log_dir) / "query.jsonl"

rows = read_jsonl(log_path)
by_thread = group_query_logs_by_thread(rows)
thread_ids = sorted(by_thread.keys())

if not rows:
    st.info("No query logs found yet. Run a question in **Ask** first.")
    st.stop()

thread_id = st.selectbox("Choose a thread", options=thread_ids, index=0)
items = by_thread.get(thread_id, [])

st.caption(f"Loaded {len(items)} runs from `{log_path}`")

# Summary table
summary = []
for r in items:
    ans = r.get("generation", {}).get("answer", "")
    ts = r.get("timestamp", "")
    q = r.get("question", "")
    cited = r.get("generation", {}).get("citation_validation", {}).get("cited_chunk_ids", [])
    invalid = r.get("generation", {}).get("citation_validation", {}).get("invalid_citations", [])
    summary.append(
        {
            "timestamp": ts,
            "question": q,
            "citations": len(cited),
            "invalid": len(invalid),
            "answer_preview": (ans[:140] + "…") if len(ans) > 140 else ans,
        }
    )

st.subheader("Runs")
st.dataframe(summary, use_container_width=True, hide_index=True)

st.divider()

st.subheader("Details")

for r in items:
    ts = r.get("timestamp", "")
    q = r.get("question", "")
    ans = r.get("generation", {}).get("answer", "")
    ev = r.get("retrieval", {}).get("evidence", [])

    with st.expander(f"{ts} — {q}", expanded=False):
        st.markdown(ans)
        st.markdown("**Evidence (trimmed):**")
        for e in ev:
            cid = e.get("chunk_id")
            title = e.get("title", "")
            section = e.get("section", "")
            with st.expander(f"{cid} | {title} — {section}", expanded=False):
                st.code(e.get("text", ""), language="text")

st.divider()

st.subheader("Export")

md = render_thread_markdown(thread_id=thread_id, items=items)

# ---- helpers: thread CSV + PDF ----
def thread_summary_rows(thread_items: list[dict]) -> list[dict]:
    out = []
    for r in thread_items:
        ans = r.get("generation", {}).get("answer", "")
        cited = r.get("generation", {}).get("citation_validation", {}).get("cited_chunk_ids", [])
        invalid = r.get("generation", {}).get("citation_validation", {}).get("invalid_citations", [])
        out.append(
            {
                "timestamp": r.get("timestamp", ""),
                "question": r.get("question", ""),
                "citations": len(cited),
                "invalid": len(invalid),
                "answer": ans,
            }
        )
    return out

def thread_to_csv_bytes(thread_items: list[dict]) -> bytes:
    import io, csv
    rows2 = thread_summary_rows(thread_items)
    if not rows2:
        return b""
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(rows2[0].keys()))
    w.writeheader()
    w.writerows(rows2)
    return buf.getvalue().encode("utf-8")

def thread_markdown_to_pdf_bytes(markdown_text: str, title: str) -> bytes:
    """
    Minimal "text PDF" export (no fancy markdown rendering).
    Requires reportlab!
    """
    from io import BytesIO
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    bio = BytesIO()
    c = canvas.Canvas(bio, pagesize=letter)
    width, height = letter

    y = height - 50
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y, title[:90])
    y -= 30

    c.setFont("Helvetica", 9)

    # naive wrapping
    def wrap_line(s: str, max_chars: int = 110) -> list[str]:
        s = s.replace("\t", " ").rstrip()
        return [s[i : i + max_chars] for i in range(0, len(s), max_chars)] or [""]

    for line in markdown_text.splitlines():
        for chunk in wrap_line(line):
            if y < 60:
                c.showPage()
                c.setFont("Helvetica", 9)
                y = height - 50
            c.drawString(50, y, chunk)
            y -= 12

    c.save()
    return bio.getvalue()

# ---- stable filenames per selected thread (avoid changing on rerun) ----
if (
    st.session_state.get("thread_export_thread_id") != thread_id
    or "thread_export_stamp" not in st.session_state
):
    st.session_state["thread_export_thread_id"] = thread_id
    st.session_state["thread_export_stamp"] = now_slug()

stamp = st.session_state["thread_export_stamp"]

out_dir = Path("outputs")
out_dir.mkdir(parents=True, exist_ok=True)

md_name = f"thread_{thread_id}_{stamp}.md"
csv_name = f"thread_{thread_id}_{stamp}.csv"
pdf_name = f"thread_{thread_id}_{stamp}.pdf"

md_path = out_dir / md_name
csv_path = out_dir / csv_name
pdf_path = out_dir / pdf_name

csv_bytes = thread_to_csv_bytes(items)
pdf_bytes = thread_markdown_to_pdf_bytes(md, title=f"Thread export — {thread_id}")

c1, c2, c3 = st.columns(3)

with c1:
    if st.button("Write Markdown", type="primary"):
        md_path.write_text(md, encoding="utf-8")
        st.success(f"Wrote `{md_path}`")
    st.download_button(
        label="Download Markdown",
        data=md.encode("utf-8"),
        file_name=md_name,
        mime="text/markdown",
    )

with c2:
    if st.button("Write CSV"):
        csv_path.write_bytes(csv_bytes)
        st.success(f"Wrote `{csv_path}`")
    st.download_button(
        label="Download CSV",
        data=csv_bytes,
        file_name=csv_name,
        mime="text/csv",
    )

with c3:
    if st.button("Write PDF"):
        pdf_path.write_bytes(pdf_bytes)
        st.success(f"Wrote `{pdf_path}`")
    st.download_button(
        label="Download PDF",
        data=pdf_bytes,
        file_name=pdf_name,
        mime="application/pdf",
    )

st.caption(f"Exports go to `outputs/` (files are only written when you click **Write**).")
