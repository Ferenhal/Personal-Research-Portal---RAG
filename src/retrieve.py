# src/retrieve.py
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import faiss

import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

from sentence_transformers import SentenceTransformer


def read_jsonl(path: Path) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return x / norms


def load_chunk_text_map(chunks_path: Path) -> Dict[str, str]:
    m = {}
    with chunks_path.open("r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            m[r["chunk_id"]] = r["text"]
    return m


_WORD_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> List[str]:
    return _WORD_RE.findall((text or "").lower())


def lexical_overlap_score(question: str, doc_text: str) -> float:
    q_toks = set(tokenize(question))
    if not q_toks:
        return 0.0
    d_toks = set(tokenize(doc_text))
    return len(q_toks & d_toks) / max(1, len(q_toks))


def retrieve_topk(
    question: str,
    k: int,
    index_dir: Path,
    chunks_path: Path,
    candidate_k: int = 40,
    min_score_ratio: float = 0.90,
    min_score_abs_delta: float = 0.05,
    lexical_rerank: bool = True,
    lexical_boost: float = 0.15,
) -> Dict:
    """
    Retrieval improvements:
    - retrieve candidate_k from FAISS, not just k
    - filter by similarity threshold relative to best match
    - optional lexical rerank/boost for relevance
    """
    cfg = json.loads((index_dir / "index_config.json").read_text(encoding="utf-8"))
    embedding_model = cfg["embedding_model"]
    metric = cfg["metric"]

    index = faiss.read_index(str(index_dir / "faiss.index"))
    meta_rows = read_jsonl(index_dir / "chunks_meta.jsonl")
    chunk_text = load_chunk_text_map(chunks_path)

    model = SentenceTransformer(embedding_model)
    q = model.encode([question], convert_to_numpy=True).astype(np.float32)

    cosine_mode = "cosine" in metric.lower()
    if cosine_mode:
        q = l2_normalize(q)

    # Candidate search
    cand_k = max(candidate_k, k)
    scores, ids = index.search(q, cand_k)

    raw = []
    for row_id, score in zip(ids[0].tolist(), scores[0].tolist()):
        if row_id < 0:
            continue
        m = meta_rows[row_id]
        cid = m["chunk_id"]
        raw.append({
            "row_id": int(row_id),
            "score": float(score),
            "chunk_id": cid,
            "source_id": m["source_id"],
            "section": m.get("section", ""),
            "title": m.get("title", ""),
            "text": chunk_text.get(cid, ""),
        })

    if not raw:
        return {
            "question": question,
            "k": k,
            "metric": metric,
            "embedding_model": embedding_model,
            "results": [],
        }

    # Similarity thresholding (keep only candidates close to the best)
    top_score = raw[0]["score"]
    # For cosine similarity (via IP on normalized vectors), higher is better.
    # Threshold: keep if score >= max(top_score*ratio, top_score-abs_delta).
    thresh = max(top_score * min_score_ratio, top_score - min_score_abs_delta)
    filtered = [r for r in raw if r["score"] >= thresh]

    # If filtering is too aggressive, keep at least k by falling back
    if len(filtered) < min(k, len(raw)):
        filtered = raw[:max(k, min(10, len(raw)))]

    # Optional lexical rerank/boost to downweight “semantically close but irrelevant” chunks
    if lexical_rerank:
        for r in filtered:
            r["lexical"] = lexical_overlap_score(question, r["text"])
            r["combined_score"] = r["score"] + lexical_boost * r["lexical"]
        filtered.sort(key=lambda x: x["combined_score"], reverse=True)
    else:
        for r in filtered:
            r["lexical"] = None
            r["combined_score"] = r["score"]

    # Take top-k after rerank/filter
    picked = filtered[:k]

    results = []
    for rank, r in enumerate(picked, start=1):
        results.append({
            "rank": rank,
            "score": float(r["score"]),
            "combined_score": float(r["combined_score"]),
            "lexical": None if r["lexical"] is None else float(r["lexical"]),
            "row_id": int(r["row_id"]),
            "chunk_id": r["chunk_id"],
            "source_id": r["source_id"],
            "section": r.get("section", ""),
            "title": r.get("title", ""),
            "text": r.get("text", ""),
        })

    return {
        "question": question,
        "k": k,
        "metric": metric,
        "embedding_model": embedding_model,
        "candidate_k": cand_k,
        "threshold": thresh,
        "results": results,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--question", type=str, required=True)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--index_dir", type=str, default="data/index")
    ap.add_argument("--chunks_path", type=str, default="data/chunks/chunks.jsonl")

    ap.add_argument("--candidate_k", type=int, default=40)
    ap.add_argument("--min_score_ratio", type=float, default=0.90)
    ap.add_argument("--min_score_abs_delta", type=float, default=0.05)
    ap.add_argument("--no_lexical_rerank", action="store_true")
    ap.add_argument("--lexical_boost", type=float, default=0.15)

    args = ap.parse_args()

    out = retrieve_topk(
        question=args.question,
        k=args.k,
        index_dir=Path(args.index_dir),
        chunks_path=Path(args.chunks_path),
        candidate_k=args.candidate_k,
        min_score_ratio=args.min_score_ratio,
        min_score_abs_delta=args.min_score_abs_delta,
        lexical_rerank=(not args.no_lexical_rerank),
        lexical_boost=args.lexical_boost,
    )
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
