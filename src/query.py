# src/query.py
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

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

# Avoid sections that are typically pure noise for QA.
# (Do NOT skip Introduction by default; it often contains the definitions you need.)
SKIP_SECTIONS = {
    "references",
    "introduction",
    "related work",
    "bibliography",
    "acknowledgements",
    "acknowledgments",
    "appendix",
    "supplementary material",
    "supplementary materials",
}

MAX_EVIDENCE_CHARS = 1400  # helps local models stay focused

CITATION_RE = re.compile(r"\[(S\d{3}::c\d{5})\]")


def sha8(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:8]


def extract_citations(text: str) -> List[str]:
    """Return unique citations in appearance order."""
    seen = set()
    ordered: List[str] = []
    for cid in CITATION_RE.findall(text or ""):
        if cid not in seen:
            seen.add(cid)
            ordered.append(cid)
    return ordered


def normalize_abstention_line(line: str) -> str:
    return (line or "").strip().lower().rstrip(".")


def is_abstention(text: str) -> bool:
    return normalize_abstention_line(text or "") == "not found in the corpus"


def contains_abstention_line(text: str) -> bool:
    lines = [(l or "").strip() for l in (text or "").splitlines()]
    return any(normalize_abstention_line(l) in {"not found in the corpus", "not found in corpus"} for l in lines)


def validate_citations(answer_text: str, allowed_chunk_ids: set) -> Tuple[List[str], List[str], List[str]]:
    cited = extract_citations(answer_text)
    invalid = [cid for cid in cited if cid not in allowed_chunk_ids]
    valid = [cid for cid in cited if cid in allowed_chunk_ids]
    return cited, valid, invalid


def build_manifest_map(manifest_path: Path) -> Dict[str, Dict[str, str]]:
    """
    Reads manifest.enriched.csv into a {source_id -> row-dict} map.
    Uses csv.DictReader to avoid a hard dependency on pandas.
    """
    if not manifest_path.exists():
        return {}

    out: Dict[str, Dict[str, str]] = {}
    try:
        with manifest_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sid = (row.get("source_id") or "").strip()
                if not sid:
                    continue
                out[sid] = {k: (v or "").strip() for k, v in row.items()}
    except Exception:
        # Don't crash query if manifest is malformed; references will degrade gracefully.
        return {}

    return out


def format_references(cited_chunk_ids: Sequence[str], manifest_map: Dict[str, Dict[str, str]]) -> str:
    # Preserve first-appearance order by source_id
    source_ids: List[str] = []
    seen = set()
    for cid in cited_chunk_ids:
        sid = cid.split("::")[0]
        if sid not in seen:
            seen.add(sid)
            source_ids.append(sid)

    lines: List[str] = []
    for sid in source_ids:
        row = manifest_map.get(sid)
        if not row:
            lines.append(f"[{sid}] (missing from manifest)")
            continue

        title = row.get("title", "").strip()
        authors = row.get("authors", "").strip()
        year = row.get("year", "").strip()
        venue = row.get("venue", "").strip()
        link = row.get("link_or_doi", "").strip()
        typ = row.get("type", "").strip()

        venue_part = f"{venue}." if venue else ""
        type_part = f"{typ}." if typ else ""
        link_part = f"{link}" if link else ""

        # Keep it compact and readable
        lines.append(
            f"[{sid}] {title} — {authors} ({year}). {venue_part} {type_part} {link_part}".strip()
        )

    if not lines:
        return "References:\n(none)"
    return "References:\n" + "\n".join(lines)


def _build_evidence_from_results(results: Sequence[Dict[str, Any]]) -> Tuple[List[str], List[Dict[str, Any]], List[str]]:
    """
    Returns:
      - evidence_lines for prompt
      - evidence_for_log (includes trimmed text)
      - evidence_chunk_ids (IDs that appear in the prompt headers)
    """
    kept_lines: List[str] = []
    kept_for_log: List[Dict[str, Any]] = []
    kept_ids: List[str] = []

    # First pass: apply SKIP_SECTIONS
    candidates: List[Dict[str, Any]] = []
    for item in results:
        section = (item.get("section", "") or "").strip()
        if section.lower() in SKIP_SECTIONS:
            continue
        candidates.append(item)

    # If filtering removed everything (or nearly everything), fall back to all results.
    use_items = candidates if len(candidates) >= 1 else list(results)

    for item in use_items:
        cid = item["chunk_id"]
        title = item.get("title", "") or ""
        section = (item.get("section", "") or "").strip()
        text = (item.get("text", "") or "")[:MAX_EVIDENCE_CHARS]

        kept_lines.append(f"[{cid}] ({title} — {section})\n{text}\n")
        kept_ids.append(cid)

        kept_for_log.append(
            {
                "rank": item.get("rank"),
                "score": item.get("score"),
                "chunk_id": cid,
                "source_id": item.get("source_id"),
                "title": title,
                "section": section,
                "text": text,
            }
        )

    return kept_lines, kept_for_log, kept_ids


def run_query(
    *,
    question: str,
    k: int = 5,
    index_dir: Path | str = "data/index",
    chunks_path: Path | str = "data/chunks/chunks.jsonl",
    log_dir: Path | str | None = "logs",
    manifest_path: Path | str = "data/manifest/manifest.enriched.csv",
    # Local generator settings
    local_model: str = "llama3.2",
    ollama_base_url: str = "http://localhost:11434",
    num_ctx: int = 4096,
    temperature: float = 0.2,
    timeout_s: int = 180,
    # Product-layer metadata (used by the UI)
    thread_id: str | None = None,
) -> Dict[str, Any]:
    """Importable entry point for Phase 3 UI + CLI.

    Returns a structured dict with:
      - answer (final: references appended when trusted)
      - answer_text (raw model answer before references)
      - evidence (trimmed evidence chunks shown to the model)
      - citation validation details
      - log_entry
      - system_warnings (e.g., generator error)
    """
    ts = datetime.now(timezone.utc).isoformat()
    index_dir = Path(index_dir)
    chunks_path = Path(chunks_path)
    manifest_path = Path(manifest_path)

    log_dir_path: Optional[Path]
    if log_dir is None:
        log_dir_path = None
    else:
        log_dir_path = Path(log_dir)
        log_dir_path.mkdir(parents=True, exist_ok=True)

    # Retrieval
    r = retrieve_topk(
        question=question,
        k=k,
        index_dir=index_dir,
        chunks_path=chunks_path,
    )

    results = r.get("results") or []
    if not results:
        # No retrieved evidence -> immediate abstention
        answer_final = "Not found in the corpus."
        log_entry = {
            "timestamp": ts,
            "thread_id": thread_id,
            "question": question,
            "k": k,
            "retrieval": {
                "embedding_model": r.get("embedding_model"),
                "metric": r.get("metric"),
                "results": [],
                "evidence": [],
            },
            "generation": {
                "provider": "ollama",
                "model": local_model,
                "prompt_id": f"query_v1_{sha8(PROMPT_TEMPLATE)}",
                "prompt_sha8": sha8(""),
                "num_ctx": num_ctx,
                "temperature": temperature,
                "retry_used": False,
                "attempt1": {"status": "skipped", "error": "no_retrieval_results", "answer": "", "cited_chunk_ids": [], "valid_citations": [], "invalid_citations": []},
                "attempt2": None,
                "answer": answer_final,
                "citation_validation": {"allowed_chunk_ids": [], "cited_chunk_ids": [], "valid_citations": [], "invalid_citations": []},
            },
        }
        if log_dir_path is not None:
            (log_dir_path / "query.jsonl").open("a", encoding="utf-8").write(json.dumps(log_entry, ensure_ascii=False) + "\n")

        return {
            "answer": answer_final,
            "answer_text": "",
            "references": "References:\n(none)",
            "cited_chunk_ids": [],
            "valid_citations": [],
            "invalid_citations": [],
            "prompt_id": log_entry["generation"]["prompt_id"],
            "prompt_sha8": log_entry["generation"]["prompt_sha8"],
            "retrieval": r,
            "evidence": [],
            "log_entry": log_entry,
            "system_warnings": ["no_retrieval_results"],
        }

    # Evidence shown to the model
    evidence_lines, evidence_for_log, evidence_ids = _build_evidence_from_results(results)
    evidence_block = "\n---\n".join(evidence_lines)

    prompt = PROMPT_TEMPLATE.format(question=question, evidence=evidence_block)
    prompt_id = f"query_v1_{sha8(PROMPT_TEMPLATE)}"
    prompt_sha = sha8(prompt)

    # IMPORTANT: allowed_chunk_ids must match chunk IDs actually shown in the prompt headers
    allowed_chunk_ids = set(evidence_ids)

    manifest_map = build_manifest_map(manifest_path)

    system_warnings: List[str] = []

    # Attempt 1
    gen1 = ollama_generate(
        prompt=prompt,
        model=local_model,
        base_url=ollama_base_url,
        num_ctx=num_ctx,
        temperature=temperature,
        timeout_s=timeout_s,
    )
    answer1 = gen1["answer"] if gen1.get("status") == "ok" else ""
    if gen1.get("status") != "ok":
        system_warnings.append(f"generator_error_attempt1:{gen1.get('error')}")

    cited1, valid1, invalid1 = validate_citations(answer1, allowed_chunk_ids)

    # Attempt 2 (strict retry)
    retry_used = False
    gen2 = None

    final_answer_text = answer1
    cited_chunk_ids = cited1
    valid_citations = valid1
    invalid_citations = invalid1

    # Retry when: generator produced text but did not cite correctly OR cited invalid IDs.
    # Skip retry only if it already abstained.
    should_retry = (not is_abstention(answer1)) and (invalid1 or not valid1)

    if should_retry:
        retry_used = True
        strict_prompt = prompt + STRICT_PROMPT_SUFFIX
        gen2 = ollama_generate(
            prompt=strict_prompt,
            model=local_model,
            base_url=ollama_base_url,
            num_ctx=num_ctx,
            temperature=0.0,
            timeout_s=timeout_s,
        )
        answer2 = gen2["answer"] if gen2.get("status") == "ok" else ""
        if gen2.get("status") != "ok":
            system_warnings.append(f"generator_error_attempt2:{gen2.get('error')}")

        cited2, valid2, invalid2 = validate_citations(answer2, allowed_chunk_ids)

        final_answer_text = answer2
        cited_chunk_ids = cited2
        valid_citations = valid2
        invalid_citations = invalid2

    # Final trust gate
    if contains_abstention_line(final_answer_text):
        answer_final = "Not found in the corpus."
        references_block = "References:\n(none)"
    elif invalid_citations:
        # Do not surface system internals in the answer; keep strict abstention behavior.
        answer_final = "Not found in the corpus."
        references_block = "References:\n(none)"
        system_warnings.append(f"invalid_citations:{','.join(invalid_citations)}")
    elif not valid_citations:
        answer_final = "Not found in the corpus."
        references_block = "References:\n(none)"
        system_warnings.append("no_valid_citations")
    else:
        references_block = format_references(valid_citations, manifest_map)
        answer_final = final_answer_text.strip() + "\n\n" + references_block

    log_entry: Dict[str, Any] = {
        "timestamp": ts,
        "thread_id": thread_id,
        "question": question,
        "k": k,
        "retrieval": {
            "embedding_model": r.get("embedding_model"),
            "metric": r.get("metric"),
            "results": [
                {
                    "rank": x.get("rank"),
                    "score": x.get("score"),
                    "chunk_id": x.get("chunk_id"),
                    "source_id": x.get("source_id"),
                    "title": x.get("title", ""),
                    "section": x.get("section", ""),
                }
                for x in results
            ],
            "evidence": evidence_for_log,
        },
        "generation": {
            "provider": "ollama",
            "model": local_model,
            "prompt_id": prompt_id,
            "prompt_sha8": prompt_sha,
            "num_ctx": num_ctx,
            "temperature": temperature,
            "retry_used": retry_used,
            "system_warnings": system_warnings,
            "attempt1": {
                "status": gen1.get("status"),
                "error": gen1.get("error"),
                "answer": answer1,
                "cited_chunk_ids": cited1,
                "valid_citations": valid1,
                "invalid_citations": invalid1,
            },
            "attempt2": None
            if gen2 is None
            else {
                "status": gen2.get("status"),
                "error": gen2.get("error"),
                "answer": final_answer_text,
                "cited_chunk_ids": cited_chunk_ids,
                "valid_citations": valid_citations,
                "invalid_citations": invalid_citations,
            },
            "answer": answer_final,
            "citation_validation": {
                "allowed_chunk_ids": list(allowed_chunk_ids),
                "cited_chunk_ids": cited_chunk_ids,
                "valid_citations": valid_citations,
                "invalid_citations": invalid_citations,
            },
        },
    }

    if log_dir_path is not None:
        log_path = log_dir_path / "query.jsonl"
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

    return {
        "answer": answer_final,
        "answer_text": final_answer_text,
        "references": references_block,
        "cited_chunk_ids": cited_chunk_ids,
        "valid_citations": valid_citations,
        "invalid_citations": invalid_citations,
        "prompt_id": prompt_id,
        "prompt_sha8": prompt_sha,
        "retrieval": r,
        "evidence": evidence_for_log,
        "log_entry": log_entry,
        "system_warnings": system_warnings,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--question", type=str, required=True)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--index_dir", type=str, default="data/index")
    ap.add_argument("--chunks_path", type=str, default="data/chunks/chunks.jsonl")
    ap.add_argument("--log_dir", type=str, default="logs")
    ap.add_argument("--manifest_path", type=str, default="data/manifest/manifest.enriched.csv")
    ap.add_argument("--thread_id", type=str, default=None)

    ap.add_argument("--local_model", type=str, default="llama3.2")
    ap.add_argument("--ollama_base_url", type=str, default="http://localhost:11434")
    ap.add_argument("--num_ctx", type=int, default=4096)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--timeout_s", type=int, default=180)

    ap.add_argument("--debug", action="store_true", help="Print generator warnings to stderr")

    args = ap.parse_args()

    out = run_query(
        question=args.question,
        k=args.k,
        index_dir=args.index_dir,
        chunks_path=args.chunks_path,
        log_dir=args.log_dir,
        manifest_path=args.manifest_path,
        local_model=args.local_model,
        ollama_base_url=args.ollama_base_url,
        num_ctx=args.num_ctx,
        temperature=args.temperature,
        timeout_s=args.timeout_s,
        thread_id=args.thread_id,
    )

    # Main output: final answer only
    print(out["answer"])

    # Helpful CLI metadata (does not affect the answer text)
    if args.log_dir is not None:
        log_path = Path(args.log_dir) / "query.jsonl"
        print(f"\n[log saved: {log_path}]")
    print(f"[prompt_id: {out['prompt_id']}]")
    print(f"[local_model: {args.local_model}]")

    if args.debug and out.get("system_warnings"):
        print(f"[warnings: {out['system_warnings']}]", file=sys.stderr)


if __name__ == "__main__":
    main()