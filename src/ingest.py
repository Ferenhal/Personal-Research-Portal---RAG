# Created by venv; see https://docs.python.org/3/library/venv.html

# src/ingest.py
from __future__ import annotations

import argparse
import csv
import json
import re
import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import pandas as pd
import fitz  # PyMuPDF
from tqdm import tqdm


REQUIRED_COLS = ["source_id", "title", "authors", "year", "type", "venue", "link_or_doi", "relevance_note"]


def read_manifest_csv(path: Path) -> pd.DataFrame:
    """
    Reads Corpus.csv robustly and normalizes columns.
    """
    # Try UTF-8 first, then fall back.
    try:
        df = pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="ISO-8859-1")

    # Drop unnamed junk columns from exports.
    df = df.loc[:, ~df.columns.str.contains(r"^Unnamed", regex=True)]

    # Normalize your current schema to the required schema.
    # File uses: data_id, title, authors, year, type, venue, DOI, relevance_note
    rename_map = {
        "data_id": "source_id",
        "DOI": "link_or_doi",
    }
    df = df.rename(columns=rename_map)

    # Ensure required columns exist (venue can be blank if not applicable).
    for col in ["source_id", "title", "authors", "year", "type", "venue", "link_or_doi", "relevance_note"]:
        if col not in df.columns:
            df[col] = ""

    # Make source_id stable and clearly a string. Example: S001, S002, ...
    def normalize_source_id(x) -> str:
        s = str(x).strip()
        if s.isdigit():
            return f"S{int(s):03d}"
        if re.match(r"^S\d+$", s):
            return f"S{int(s[1:]):03d}"
        return s

    df["source_id"] = df["source_id"].apply(normalize_source_id)

    # Coerce year to int where possible.
    def coerce_year(x):
        try:
            return int(str(x).strip())
        except Exception:
            return None

    df["year"] = df["year"].apply(coerce_year)

    # Strip whitespace.
    for col in ["title", "authors", "type", "venue", "link_or_doi", "relevance_note"]:
        df[col] = df[col].astype(str).str.strip()

    return df


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def extract_text_pymupdf(pdf_path: Path) -> Tuple[str, int]:
    doc = fitz.open(pdf_path)
    pages = doc.page_count
    chunks: List[str] = []
    for i in range(pages):
        page = doc.load_page(i)
        chunks.append(page.get_text("text"))
    doc.close()
    return "\n".join(chunks), pages


def clean_text(text: str) -> str:
    # Fix hyphenation across line breaks: "trans-\nformer" -> "transformer"
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    # Normalize line breaks (keep paragraph-ish structure, but remove excessive blank lines)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove trailing spaces on lines
    text = "\n".join(line.rstrip() for line in text.split("\n"))

    # Collapse weird whitespace runs
    text = re.sub(r"[ \t]{2,}", " ", text)

    return text.strip()


def guess_pdf_for_row(raw_dir: Path, source_id: str, link_or_doi: str, title: str) -> Optional[Path]:
    """
    Tries multiple strategies to locate the correct PDF in data/raw.

    Have to add a 'local_file' column in the manifest later, but this gets it moving now.
    """
    pdfs = list(raw_dir.glob("*.pdf"))
    if not pdfs:
        return None

    # 1) If link_or_doi contains an arXiv id, try exact match like 2307.03875v2.pdf
    m = re.search(r"arXiv:(\d{4}\.\d{5}(v\d+)?)", link_or_doi, flags=re.IGNORECASE)
    if m:
        candidate = raw_dir / f"{m.group(1)}.pdf"
        if candidate.exists():
            return candidate
        # fallback: any file containing that id
        arxiv_id = m.group(1)
        hits = [p for p in pdfs if arxiv_id in p.name]
        if len(hits) == 1:
            return hits[0]

    # 2) If DOI-like, match by digits/slug fragments
    doi_fragment = re.sub(r"[^a-zA-Z0-9]+", "", link_or_doi.lower())
    if doi_fragment:
        hits = [p for p in pdfs if doi_fragment[:12] and doi_fragment[:12] in re.sub(r"[^a-zA-Z0-9]+", "", p.name.lower())]
        if len(hits) == 1:
            return hits[0]

    # 3) Weak heuristic: match by a few title keywords in filename
    keywords = [w.lower() for w in re.findall(r"[A-Za-z]{5,}", title)[:6]]
    if keywords:
        scored = []
        for p in pdfs:
            name = p.name.lower()
            score = sum(1 for w in keywords if w in name)
            if score > 0:
                scored.append((score, p))
        scored.sort(reverse=True, key=lambda x: x[0])
        if scored and scored[0][0] >= 2:
            return scored[0][1]

    return None


def write_jsonl(path: Path, records: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=str, required=True, help="Path to corpus manifest CSV (e.g., data/manifest/Corpus.csv)")
    ap.add_argument("--raw_dir", type=str, required=True, help="Directory containing raw PDFs (e.g., data/raw)")
    ap.add_argument("--out_dir", type=str, required=True, help="Directory to write processed .txt files (e.g., data/processed)")
    ap.add_argument("--log_dir", type=str, required=True, help="Directory to write logs (e.g., logs)")
    args = ap.parse_args()

    manifest_path = Path(args.manifest)
    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    log_dir = Path(args.log_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    df = read_manifest_csv(manifest_path)

    enriched_records = []
    run_records = []

    ingested_at = datetime.now(timezone.utc).isoformat()

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Ingesting"):
        source_id = row["source_id"]
        title = row["title"]
        link_or_doi = row["link_or_doi"]

        pdf_path = guess_pdf_for_row(raw_dir, source_id, link_or_doi, title)
        status = "ok"
        error = None

        if pdf_path is None:
            status = "missing_pdf"
            error = f"No PDF found for {source_id}"
            run_records.append(
                {"timestamp": ingested_at, "source_id": source_id, "status": status, "error": error}
            )
            enriched = {**row.to_dict(), "raw_path": "", "text_path": "", "sha256": "", "num_pages": None, "num_chars": 0, "ingested_at": ingested_at}
            enriched_records.append(enriched)
            continue

        try:
            raw_hash = sha256_file(pdf_path)
            raw_text, num_pages = extract_text_pymupdf(pdf_path)
            cleaned = clean_text(raw_text)

            text_path = out_dir / f"{source_id}.txt"
            text_path.write_text(cleaned, encoding="utf-8")

            run_records.append(
                {"timestamp": ingested_at, "source_id": source_id, "status": status, "raw_path": str(pdf_path), "text_path": str(text_path)}
            )

            enriched = {
                **row.to_dict(),
                "raw_path": str(pdf_path),
                "text_path": str(text_path),
                "sha256": raw_hash,
                "num_pages": int(num_pages),
                "num_chars": int(len(cleaned)),
                "ingested_at": ingested_at,
            }
            enriched_records.append(enriched)

        except Exception as e:
            status = "error"
            error = repr(e)
            run_records.append(
                {"timestamp": ingested_at, "source_id": source_id, "status": status, "raw_path": str(pdf_path), "error": error}
            )
            enriched = {**row.to_dict(), "raw_path": str(pdf_path), "text_path": "", "sha256": "", "num_pages": None, "num_chars": 0, "ingested_at": ingested_at}
            enriched_records.append(enriched)

    # Write enriched manifest outputs
    enriched_df = pd.DataFrame(enriched_records)
    out_csv = manifest_path.parent / "manifest.enriched.csv"
    enriched_df.to_csv(out_csv, index=False, encoding="utf-8")

    out_jsonl = manifest_path.parent / "manifest.enriched.jsonl"
    write_jsonl(out_jsonl, enriched_records)

    # Write machine-readable run log for ingestion step
    ingest_log = log_dir / "ingest.jsonl"
    write_jsonl(ingest_log, run_records)

    print(f"Wrote enriched manifest: {out_csv}")
    print(f"Wrote enriched manifest JSONL: {out_jsonl}")
    print(f"Wrote ingest log: {ingest_log}")


if __name__ == "__main__":
    main()
