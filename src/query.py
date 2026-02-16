# src/query.py
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import re
import pandas as pd

from src.generators.ollama_local import ollama_generate
from src.retrieve import retrieve_topk


PROMPT_TEMPLATE = """You are a research assistant answering strictly from the provided evidence snippets.

Rules:
- Use ONLY the evidence snippets below.
- Every factual claim must include at least one citation in square brackets like [S001::c00000].
- You may cite multiple chunks: [S001::c00000][S003::c00123]
- Do NOT invent citations. If evidence is missing or conflicting, say so explicitly.
- If the corpus does not contain evidence, answer exactly: Not found in the corpus.
- If evidence is irrelevant to the question, answer exactly: Not found in the corpus.
- Evidence may contain unrelated examples or benchmark Q&A. Do not answer those; answer only the user's question.
- No additional commentary. Do not include notes, apologies, or filler.

Question:
{question}

Evidence snippets (each has an ID you must cite):
{evidence}

Write:
1) Answer (with inline citations)
2) "Cited chunks" list: each cited chunk_id with a one-line description (title/section)
"""

STRICT_PROMPT_SUFFIX = """

CRITICAL CITATION RULES:
- Every sentence MUST include at least one chunk citation like [S001::c00002].
- You MUST cite only chunk IDs that appear in the evidence headers.
- If you cannot comply, output exactly: Not found in the corpus.
"""

# Sections that frequently contain irrelevant lexical noise for QA (reference lists, tables, appendices)
SKIP_SECTIONS = {
    "references",
    "bibliography",
    "acknowledgements",
    "acknowledgments",
    "appendix",
    "supplementary material",
    "supplementary materials",
}

MAX_EVIDENCE_CHARS = 1400  # helps local models stay focused


def sha8(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:8]


CITATION_RE = re.compile(r"\[(S\d{3}::c\d{5})\]")


def extract_citations(text: str) -> List[str]:
    # Returns unique citations in appearance order
    seen = set()
    ordered = []
    for cid in CITATION_RE.findall(text or ""):
        if cid not in seen:
            seen.add(cid)
            ordered.append(cid)
    return ordered


def is_abstention(text: str) -> bool:
    return (text or "").strip().lower() == "not found in the corpus."


def build_manifest_map(manifest_path: Path) -> dict:
    df = pd.read_csv(manifest_path, encoding="utf-8")
    df["source_id"] = df["source_id"].astype(str).str.strip()
    return {row["source_id"]: row for _, row in df.iterrows()}


def format_references(cited_chunk_ids: List[str], manifest_map: dict) -> str:
    source_ids = []
    seen = set()
    for cid in cited_chunk_ids:
        sid = cid.split("::")[0]
        if sid not in seen:
            seen.add(sid)
            source_ids.append(sid)

    lines = []
    for sid in source_ids:
        row = manifest_map.get(sid)
        if row is None:
            lines.append(f"[{sid}] (missing from manifest)")
            continue

        title = str(row.get("title", "")).strip()
        authors = str(row.get("authors", "")).strip()
        year = str(row.get("year", "")).strip()
        venue = str(row.get("venue", "")).strip()
        link = str(row.get("link_or_doi", "")).strip()
        typ = str(row.get("type", "")).strip()

        venue_part = f"{venue}." if venue else ""
        type_part = f"{typ}." if typ else ""
        link_part = f"{link}" if link else ""

        lines.append(
            f"[{sid}] {title} — {authors} ({year}). {venue_part} {type_part} {link_part}".strip()
        )

    if not lines:
        return "References:\n(none)"
    return "References:\n" + "\n".join(lines)


def validate_citations(answer_text: str, allowed_chunk_ids: set) -> Tuple[List[str], List[str], List[str]]:
    cited = extract_citations(answer_text)
    invalid = [cid for cid in cited if cid not in allowed_chunk_ids]
    valid = [cid for cid in cited if cid in allowed_chunk_ids]
    return cited, valid, invalid

def contains_abstention_line(text: str) -> bool:
    lines = [(l or "").strip().lower() for l in (text or "").splitlines()]
    return any(l in {"not found in the corpus.", "not found in corpus.", "not found in the corpus", "not found in corpus"} for l in lines)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--question", type=str, required=True)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--index_dir", type=str, default="data/index")
    ap.add_argument("--chunks_path", type=str, default="data/chunks/chunks.jsonl")
    ap.add_argument("--log_dir", type=str, default="logs")
    ap.add_argument("--manifest_path", type=str, default="data/manifest/manifest.enriched.csv")

    # Local generator settings
    ap.add_argument("--local_model", type=str, default="llama3.2")
    ap.add_argument("--ollama_base_url", type=str, default="http://localhost:11434")
    ap.add_argument("--num_ctx", type=int, default=4096)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--timeout_s", type=int, default=180)

    args = ap.parse_args()

    ts = datetime.now(timezone.utc).isoformat()
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    r = retrieve_topk(
        question=args.question,
        k=args.k,
        index_dir=Path(args.index_dir),
        chunks_path=Path(args.chunks_path),
    )

    # Build evidence block (skip noisy sections; trim text)
    evidence_lines: List[str] = []
    retrieved_chunks_for_log: List[Dict] = []

    for item in r["results"]:
        cid = item["chunk_id"]
        title = item.get("title", "")
        section = (item.get("section", "") or "").strip()
        section_norm = section.lower()

        # Query-time filtering of common noise sections
        if section_norm in SKIP_SECTIONS:
            continue

        text = (item.get("text", "") or "")[:MAX_EVIDENCE_CHARS]

        evidence_lines.append(f"[{cid}] ({title} — {section})\n{text}\n")

        retrieved_chunks_for_log.append({
            "rank": item.get("rank"),
            "score": item.get("score"),
            "chunk_id": cid,
            "source_id": item.get("source_id"),
            "title": title,
            "section": section,
            "text": text,
        })

    # If filtering removed everything, fall back to original (better than empty prompt)
    if not evidence_lines:
        for item in r["results"]:
            cid = item["chunk_id"]
            title = item.get("title", "")
            section = (item.get("section", "") or "").strip()
            text = (item.get("text", "") or "")[:MAX_EVIDENCE_CHARS]
            evidence_lines.append(f"[{cid}] ({title} — {section})\n{text}\n")

    prompt = PROMPT_TEMPLATE.format(
        question=args.question,
        evidence="\n---\n".join(evidence_lines),
    )
    prompt_id = f"query_v1_{sha8(PROMPT_TEMPLATE)}"
    prompt_sha = sha8(prompt)

    allowed_chunk_ids = {x["chunk_id"] for x in r["results"]}
    manifest_map = build_manifest_map(Path(args.manifest_path))

    # Attempt 1
    gen1 = ollama_generate(
        prompt=prompt,
        model=args.local_model,
        base_url=args.ollama_base_url,
        num_ctx=args.num_ctx,
        temperature=args.temperature,
        timeout_s=args.timeout_s,
    )
    answer1 = gen1["answer"] if gen1["status"] == "ok" else ""

    cited1, valid1, invalid1 = validate_citations(answer1, allowed_chunk_ids)

    # If invalid citations, refuse immediately
    retry_used = False
    final_answer_text = answer1
    cited_chunk_ids = cited1
    valid_citations = valid1
    invalid_citations = invalid1
    gen2 = None

    # If no valid citations and not abstaining, do one strict retry
    if (not invalid_citations) and (not valid_citations) and (not is_abstention(answer1)):
        retry_used = True
        strict_prompt = prompt + STRICT_PROMPT_SUFFIX
        gen2 = ollama_generate(
            prompt=strict_prompt,
            model=args.local_model,
            base_url=args.ollama_base_url,
            num_ctx=args.num_ctx,
            temperature=0.0,
            timeout_s=args.timeout_s,
        )
        answer2 = gen2["answer"] if gen2["status"] == "ok" else ""
        cited2, valid2, invalid2 = validate_citations(answer2, allowed_chunk_ids)

        final_answer_text = answer2
        cited_chunk_ids = cited2
        valid_citations = valid2
        invalid_citations = invalid2

    # Final trust gate:
    # 1) If the model includes an abstention line anywhere, treat as abstention (no references)
    # 2) If it cites non-retrieved chunks, refuse
    # 3) If it provides no valid citations, abstain
    # 4) Else: accept + append structured references

    if contains_abstention_line(final_answer_text):
        answer_final = "Not found in the corpus."
        references_block = "References:\n(none)"

    elif invalid_citations:
        answer_final = (
            "I can't answer reliably because the generator cited chunk IDs that were not in the retrieved evidence.\n"
            f"Invalid citations: {', '.join(invalid_citations)}\n\n"
            "Try increasing top-k, refining the question, or rerunning.\n"
            "Not found in the corpus."
        )
        references_block = "References:\n(none)"

    elif not valid_citations:
        answer_final = "Not found in the corpus."
        references_block = "References:\n(none)"

    else:
        references_block = format_references(valid_citations, manifest_map)
        answer_final = final_answer_text.strip() + "\n\n" + references_block


    log_entry = {
        "timestamp": ts,
        "question": args.question,
        "k": args.k,
        "retrieval": {
            "embedding_model": r.get("embedding_model"),
            "metric": r.get("metric"),
            "results": [
                {
                    "rank": x["rank"],
                    "score": x["score"],
                    "chunk_id": x["chunk_id"],
                    "source_id": x["source_id"],
                    "title": x.get("title", ""),
                    "section": x.get("section", ""),
                } for x in r["results"]
            ],
            "evidence": retrieved_chunks_for_log,
        },
        "generation": {
            "provider": "ollama",
            "model": args.local_model,
            "prompt_id": prompt_id,
            "prompt_sha8": prompt_sha,
            "num_ctx": args.num_ctx,
            "temperature": args.temperature,
            "retry_used": retry_used,

            "attempt1": {
                "status": gen1.get("status"),
                "error": gen1.get("error"),
                "answer": answer1,
                "cited_chunk_ids": cited1,
                "valid_citations": valid1,
                "invalid_citations": invalid1,
            },
            "attempt2": None if gen2 is None else {
                "status": gen2.get("status"),
                "error": gen2.get("error"),
                "answer": final_answer_text,
                "cited_chunk_ids": cited_chunk_ids,
                "valid_citations": valid_citations,
                "invalid_citations": invalid_citations,
            },

            "answer": answer_final,
            "citation_validation": {
                "allowed_chunk_ids": [x["chunk_id"] for x in r["results"]],
                "cited_chunk_ids": cited_chunk_ids,
                "valid_citations": valid_citations,
                "invalid_citations": invalid_citations,
            },
        },
    }

    log_path = log_dir / "query.jsonl"
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

    print(answer_final)
    print(f"\n[log saved: {log_path}]")
    print(f"[prompt_id: {prompt_id}]")
    print(f"[local_model: {args.local_model}]")


if __name__ == "__main__":
    main()
