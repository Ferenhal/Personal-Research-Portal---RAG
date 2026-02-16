# src/retrieve.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import faiss
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


def retrieve_topk(
    question: str,
    k: int,
    index_dir: Path,
    chunks_path: Path,
) -> Dict:
    cfg = json.loads((index_dir / "index_config.json").read_text(encoding="utf-8"))
    embedding_model = cfg["embedding_model"]
    metric = cfg["metric"]

    index = faiss.read_index(str(index_dir / "faiss.index"))
    meta_rows = read_jsonl(index_dir / "chunks_meta.jsonl")
    chunk_text = load_chunk_text_map(chunks_path)

    model = SentenceTransformer(embedding_model)
    q = model.encode([question], convert_to_numpy=True).astype(np.float32)

    if "cosine" in metric:
        q = l2_normalize(q)

    scores, ids = index.search(q, k)

    results = []
    for rank, (row_id, score) in enumerate(zip(ids[0].tolist(), scores[0].tolist()), start=1):
        if row_id < 0:
            continue
        m = meta_rows[row_id]
        cid = m["chunk_id"]
        results.append({
            "rank": rank,
            "score": float(score),
            "row_id": int(row_id),
            "chunk_id": cid,
            "source_id": m["source_id"],
            "section": m.get("section", ""),
            "title": m.get("title", ""),
            "text": chunk_text.get(cid, ""),
        })

    return {
        "question": question,
        "k": k,
        "metric": metric,
        "embedding_model": embedding_model,
        "results": results,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--question", type=str, required=True)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--index_dir", type=str, default="data/index")
    ap.add_argument("--chunks_path", type=str, default="data/chunks/chunks.jsonl")
    args = ap.parse_args()

    out = retrieve_topk(
        question=args.question,
        k=args.k,
        index_dir=Path(args.index_dir),
        chunks_path=Path(args.chunks_path),
    )
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
