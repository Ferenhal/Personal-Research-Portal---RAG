# src/artifacts/evidence_table.py
from __future__ import annotations

import re
from typing import Dict, List, Any
import csv
from io import StringIO, BytesIO

CIT_RE = re.compile(r"\[(S\d{3}::c\d{5})\]")

def extract_citations(text: str) -> List[str]:
    seen, out = set(), []
    for c in CIT_RE.findall(text or ""):
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out

def build_evidence_table_from_run(run_log: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    run_log is one record from logs/query.jsonl
    """
    ans = run_log.get("generation", {}).get("answer", "") or ""
    evidence = run_log.get("retrieval", {}).get("evidence", []) or []

    # map chunk_id -> (text, score)
    score_map = {}
    text_map = {}
    for e in evidence:
        cid = e.get("chunk_id")
        if not cid:
            continue
        text_map[cid] = (e.get("text") or "").strip()
        # score is on evidence items in your query logging
        score_map[cid] = e.get("score")

    rows: List[Dict[str, Any]] = []
    # use non-empty lines as “claims” (safer than naive sentence splitting)
    for line in [ln.strip() for ln in ans.splitlines() if ln.strip()]:
        cits = extract_citations(line)
        if not cits:
            continue  # skip non-claim lines (like "References:")

        # pick first citation as primary evidence for snippet/confidence
        primary = cits[0]
        snippet = text_map.get(primary, "")
        snippet = snippet[:240].replace("\n", " ").strip()

        score = score_map.get(primary)
        conf = ""
        if isinstance(score, (int, float)):
            # normalize-ish display only; keep it simple
            conf = f"{score:.3f}"

        rows.append(
            {
                "Claim": line,
                "Evidence snippet": snippet,
                "Citation": " ".join([f"[{c}]" for c in cits]),
                "Confidence": conf,
                "Notes": "",
            }
        )
    return rows

def to_markdown(rows: List[Dict[str, Any]], title: str) -> str:
    if not rows:
        return f"# {title}\n\n(no rows)\n"

    headers = list(rows[0].keys())
    md = [f"# {title}\n"]
    md.append("| " + " | ".join(headers) + " |")
    md.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for r in rows:
        md.append("| " + " | ".join((str(r.get(h,""))).replace("\n"," ") for h in headers) + " |")
    return "\n".join(md) + "\n"

def to_csv_bytes(rows: List[Dict[str, Any]]) -> bytes:
    if not rows:
        return b""
    buf = StringIO()
    w = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue().encode("utf-8")

def to_pdf_bytes(rows: List[Dict[str, Any]], title: str) -> bytes:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    bio = BytesIO()
    c = canvas.Canvas(bio, pagesize=letter)
    width, height = letter

    y = height - 50
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y, title)
    y -= 30

    c.setFont("Helvetica", 9)
    for r in rows:
        line = f"- {r['Claim']}"
        # wrap roughly
        for chunk in [line[i:i+110] for i in range(0, len(line), 110)]:
            if y < 60:
                c.showPage()
                c.setFont("Helvetica", 9)
                y = height - 50
            c.drawString(50, y, chunk)
            y -= 12

        cite = f"  Evidence: {r['Evidence snippet']} {r['Citation']}"
        for chunk in [cite[i:i+110] for i in range(0, len(cite), 110)]:
            if y < 60:
                c.showPage()
                c.setFont("Helvetica", 9)
                y = height - 50
            c.drawString(50, y, chunk)
            y -= 12

        y -= 8

    c.save()
    return bio.getvalue()