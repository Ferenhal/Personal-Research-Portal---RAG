# src/chunk.py
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd
from tqdm import tqdm


CANON_HEADINGS = {
    "abstract": "Abstract",
    "introduction": "Introduction",
    "background": "Background",
    "related work": "Related Work",
    "literature review": "Related Work",
    "methods": "Methods",
    "method": "Methods",
    "methodology": "Methods",
    "approach": "Methods",
    "experimental setup": "Methods",
    "experiments": "Experiments",
    "results": "Results",
    "discussion": "Discussion",
    "limitations": "Limitations",
    "conclusion": "Conclusion",
    "conclusions": "Conclusion",
    "future work": "Future Work",
    "references": "References",
    "bibliography": "References",
    "appendix": "Appendix",
    "acknowledgements": "Acknowledgements",
    "acknowledgments": "Acknowledgements",
    "supplementary material": "Supplementary",
    "supplementary materials": "Supplementary",
}

# Sections that usually add noise for retrieval (reference lists, appendices, etc.)
SKIP_SECTIONS_PREFIX = (
    "references",
    "bibliography",
    "appendix",
    "acknowledgements",
    "acknowledgments",
    "supplementary",
)

def should_skip_section(section_name: str) -> bool:
    """
    Skip sections that commonly contain irrelevant lexical noise for QA retrieval.
    Handles variants like "References", "References and Notes", "Appendix A", etc.
    """
    s = (section_name or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    # remove leading numbering like "7 References"
    s = re.sub(r"^\s*\d+(\.\d+)*\s*", "", s).strip()
    return any(s.startswith(prefix) for prefix in SKIP_SECTIONS_PREFIX)


def normalize_heading(line: str) -> Optional[str]:
    s = line.strip()
    if not s:
        return None

    # Reject obvious non-headings.
    if len(s) > 80:
        return None
    if s.endswith("."):
        return None

    lower = s.lower()

    # Direct canonical match (e.g., "Abstract", "1 Introduction", "2. Methods")
    lower_clean = re.sub(r"^\s*\d+(\.\d+)*\s*", "", lower).strip()
    lower_clean = re.sub(r"\s+", " ", lower_clean)

    if lower_clean in CANON_HEADINGS:
        return CANON_HEADINGS[lower_clean]

    # Heuristic: numbered headings with title-ish text: "3.2 Experimental Setup"
    m = re.match(r"^\s*\d+(\.\d+)*\s+([A-Za-z][A-Za-z0-9 \-]{3,})\s*$", s)
    if m:
        candidate = m.group(2).strip()

        cand_lower = candidate.lower()
        cand_lower = re.sub(r"\s+", " ", cand_lower)
        if cand_lower in CANON_HEADINGS:
            return CANON_HEADINGS[cand_lower]

        if len(candidate) <= 60:
            return candidate

    # Heuristic: ALL CAPS short line (headings in extracted PDFs)
    if s.isupper() and 3 <= len(s) <= 60 and sum(c.isalpha() for c in s) >= 6:
        return s.title()

    return None


def compute_line_spans(lines: List[str]) -> List[Tuple[int, int]]:
    spans = []
    pos = 0
    for line in lines:
        start = pos
        end = pos + len(line)
        spans.append((start, end))
        pos = end + 1  # account for '\n'
    return spans


def split_into_sections(text: str) -> List[Dict]:
    """
    Returns a list of sections with absolute offsets into the original text.
    Each item: {section, start_char, end_char, text}
    """
    lines = text.splitlines()
    spans = compute_line_spans(lines)

    heading_idxs: List[Tuple[int, str]] = []
    for i, line in enumerate(lines):
        h = normalize_heading(line)
        if h:
            heading_idxs.append((i, h))

    if not heading_idxs:
        return [{
            "section": "Body",
            "start_char": 0,
            "end_char": len(text),
            "text": text.strip(),
        }]

    sections = []

    first_i, _ = heading_idxs[0]
    if first_i > 0:
        start = 0
        end = spans[first_i][0] if first_i < len(spans) else len(text)
        front_text = "\n".join(lines[:first_i]).strip()
        if front_text:
            sections.append({
                "section": "FrontMatter",
                "start_char": start,
                "end_char": end,
                "text": front_text,
            })

    for idx, (i, sec_name) in enumerate(heading_idxs):
        j = heading_idxs[idx + 1][0] if idx + 1 < len(heading_idxs) else len(lines)
        start = spans[i][0] if i < len(spans) else 0
        end = spans[j - 1][1] + 1 if j - 1 < len(spans) else len(text)
        sec_text = "\n".join(lines[i:j]).strip()
        if sec_text:
            sections.append({
                "section": sec_name,
                "start_char": start,
                "end_char": min(end, len(text)),
                "text": sec_text,
            })

    return sections


def sliding_chunks(section_text: str, chunk_size: int, overlap: int) -> List[Tuple[int, int, str]]:
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    s = section_text
    n = len(s)
    if n <= chunk_size:
        return [(0, n, s)]

    chunks = []
    start = 0
    while start < n:
        end = min(start + chunk_size, n)
        chunk = s[start:end].strip()
        if chunk:
            chunks.append((start, end, chunk))
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks


def write_jsonl(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--chunk_size_chars", type=int, default=3200)
    ap.add_argument("--overlap_chars", type=int, default=400)
    ap.add_argument("--min_chunk_chars", type=int, default=300)
    ap.add_argument("--log_dir", type=str, required=True)
    args = ap.parse_args()

    manifest_path = Path(args.manifest)
    out_path = Path(args.out)
    log_dir = Path(args.log_dir)

    df = pd.read_csv(manifest_path, encoding="utf-8")
    created_at = datetime.now(timezone.utc).isoformat()

    chunk_rows: List[Dict] = []
    log_rows: List[Dict] = []

    global_chunk_idx = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Chunking"):
        source_id = str(row.get("source_id", "")).strip()
        text_path = str(row.get("text_path", "")).strip()

        if not source_id or not text_path:
            log_rows.append({
                "timestamp": created_at,
                "source_id": source_id,
                "status": "skip",
                "reason": "missing source_id or text_path",
            })
            continue

        tp = Path(text_path)
        if not tp.exists():
            log_rows.append({
                "timestamp": created_at,
                "source_id": source_id,
                "status": "skip",
                "reason": f"text file not found: {text_path}",
            })
            continue

        text = tp.read_text(encoding="utf-8", errors="replace").strip()
        if len(text) < args.min_chunk_chars:
            log_rows.append({
                "timestamp": created_at,
                "source_id": source_id,
                "status": "skip",
                "reason": "text too short",
                "num_chars": len(text),
            })
            continue

        sections = split_into_sections(text)

        per_source_chunks = 0
        per_source_sections = 0
        per_source_sections_skipped = 0
        per_source_chunks_skipped = 0

        for sec in sections:
            sec_name = sec["section"]
            if should_skip_section(sec_name):
                per_source_sections_skipped += 1
                continue

            sec_start_abs = int(sec["start_char"])
            sec_text = sec["text"]
            per_source_sections += 1

            rel_chunks = sliding_chunks(sec_text, args.chunk_size_chars, args.overlap_chars)

            for rel_start, rel_end, chunk_text in rel_chunks:
                if len(chunk_text) < args.min_chunk_chars:
                    per_source_chunks_skipped += 1
                    continue

                chunk_id = f"{source_id}::c{global_chunk_idx:05d}"
                approx_tokens = max(1, len(chunk_text) // 4)

                chunk_rows.append({
                    "chunk_id": chunk_id,
                    "source_id": source_id,
                    "section": sec_name,
                    "chunk_index_global": global_chunk_idx,
                    "text": chunk_text,
                    "char_start": sec_start_abs + rel_start,
                    "char_end": sec_start_abs + rel_end,
                    "num_chars": len(chunk_text),
                    "approx_tokens": approx_tokens,
                    "chunk_size_chars": args.chunk_size_chars,
                    "overlap_chars": args.overlap_chars,
                    "created_at": created_at,
                    "title": row.get("title", ""),
                    "year": row.get("year", ""),
                    "type": row.get("type", ""),
                    "link_or_doi": row.get("link_or_doi", ""),
                })

                global_chunk_idx += 1
                per_source_chunks += 1

        log_rows.append({
            "timestamp": created_at,
            "source_id": source_id,
            "status": "ok",
            "sections_detected": per_source_sections,
            "sections_skipped": per_source_sections_skipped,
            "chunks_written": per_source_chunks,
            "chunks_skipped_short": per_source_chunks_skipped,
        })

    write_jsonl(out_path, chunk_rows)

    log_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(log_dir / "chunk.jsonl", log_rows)

    print(f"Wrote chunks: {out_path} ({len(chunk_rows)} chunks)")
    print(f"Wrote chunk log: {log_dir / 'chunk.jsonl'}")


if __name__ == "__main__":
    main()
