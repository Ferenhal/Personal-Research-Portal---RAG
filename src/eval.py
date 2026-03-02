# src/eval.py
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

CITATION_RE = re.compile(r"\[(S\d{3}::c\d{5})\]")

def read_jsonl(path: Path) -> List[Dict]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def last_jsonl_line(path: Path) -> Dict:
    rows = read_jsonl(path)
    return rows[-1] if rows else {}

def extract_citations(text: str) -> List[str]:
    seen = set()
    out = []
    for cid in CITATION_RE.findall(text or ""):
        if cid not in seen:
            seen.add(cid)
            out.append(cid)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", type=str, required=True)
    ap.add_argument("--index_dir", type=str, default="data/index")
    ap.add_argument("--chunks_path", type=str, default="data/chunks/chunks.jsonl")
    ap.add_argument("--log_dir", type=str, default="logs")
    ap.add_argument("--local_model", type=str, default="llama3.2")
    ap.add_argument("--manifest_path", type=str, default="data/manifest/manifest.enriched.csv")
    args = ap.parse_args()

    queries_path = Path(args.queries)
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    eval_log_path = log_dir / "eval.jsonl"
    query_log_path = log_dir / "query.jsonl"
    summary_csv_path = log_dir / "eval_summary.csv"

    queries = read_jsonl(queries_path)

    summary_rows = []
    for q in queries:
        qid = q["id"]
        question = q["question"]
        k = int(q.get("k", 5))

        # Run query pipeline
        cmd = [
            sys.executable,
            "-m", "src.query",
            "--question", question,
            "--k", str(k),
            "--index_dir", args.index_dir,
            "--chunks_path", args.chunks_path,
            "--log_dir", args.log_dir,
            "--local_model", args.local_model,
            "--manifest_path", args.manifest_path,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        stdout = proc.stdout.strip()
        stderr = proc.stderr.strip()

        # Grab latest query log entry to compute automatic metrics
        last = last_jsonl_line(query_log_path)

        retrieved = last.get("retrieval", {}).get("results", [])
        allowed_chunk_ids = {r.get("chunk_id") for r in retrieved if r.get("chunk_id")}
        raw_answer = last.get("generation", {}).get("answer", "")

        cited = extract_citations(raw_answer)
        valid = [c for c in cited if c in allowed_chunk_ids]
        invalid = [c for c in cited if c not in allowed_chunk_ids]

        citation_precision = (len(valid) / len(cited)) if cited else 1.0
        # "Coverage" proxy: % of non-empty lines containing at least one citation
        lines = [ln for ln in raw_answer.splitlines() if ln.strip()]
        cited_lines = sum(1 for ln in lines if CITATION_RE.search(ln))
        citation_coverage = (cited_lines / len(lines)) if lines else 0.0

        eval_row = {
            "id": qid,
            "type": q.get("type"),
            "k": k,
            "question": question,
            "exit_code": proc.returncode,
            "stderr": stderr,
            "num_citations": len(cited),
            "num_invalid_citations": len(invalid),
            "citation_precision": citation_precision,
            "citation_coverage_proxy": citation_coverage,
        }

        with eval_log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(eval_row, ensure_ascii=False) + "\n")

        summary_rows.append(eval_row)

        # Print lightweight progress to see it running
        print(f"{qid} done (code={proc.returncode}) | citation_precision={citation_precision:.2f} | invalid={len(invalid)}")

    # CSV summary for report
    import csv
    fieldnames = list(summary_rows[0].keys()) if summary_rows else []
    with summary_csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in summary_rows:
            w.writerow(r)

    print(f"\nSaved eval log: {eval_log_path}")
    print(f"Saved summary CSV: {summary_csv_path}")

if __name__ == "__main__":
    main()
