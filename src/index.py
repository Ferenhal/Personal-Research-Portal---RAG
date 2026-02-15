# src/index.py
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def read_jsonl(path: Path) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return x / norms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", type=str, default="data/chunks/chunks.jsonl")
    ap.add_argument("--index_dir", type=str, default="data/index")
    ap.add_argument("--log_dir", type=str, default="logs")
    ap.add_argument("--model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--use_cosine", action="store_true", help="Use cosine similarity (via normalized vectors + inner product)")
    args = ap.parse_args()

    chunks_path = Path(args.chunks)
    index_dir = Path(args.index_dir)
    log_dir = Path(args.log_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    created_at = datetime.now(timezone.utc).isoformat()

    chunk_rows = read_jsonl(chunks_path)
    if not chunk_rows:
        raise RuntimeError(f"No chunks found in {chunks_path}")

    # Minimum needed for citation resolution and debugging.
    meta_rows = []
    texts = []
    for r in chunk_rows:
        meta_rows.append({
            "row_id": len(meta_rows),
            "chunk_id": r["chunk_id"],
            "source_id": r["source_id"],
            "section": r.get("section", ""),
            "char_start": r.get("char_start", None),
            "char_end": r.get("char_end", None),
            "title": r.get("title", ""),
            "year": r.get("year", ""),
            "type": r.get("type", ""),
            "link_or_doi": r.get("link_or_doi", ""),
        })
        texts.append(r["text"])

    model = SentenceTransformer(args.model)

    # Embed in batches.
    all_vecs = []
    for i in tqdm(range(0, len(texts), args.batch_size), desc="Embedding"):
        batch = texts[i:i + args.batch_size]
        vecs = model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
        vecs = vecs.astype(np.float32)
        all_vecs.append(vecs)
    X = np.vstack(all_vecs)

    # Build FAISS index.
    d = X.shape[1]

    if args.use_cosine:
        # FAISS supports inner product search and cosine via normalization.
        X = l2_normalize(X)
        index = faiss.IndexFlatIP(d)
        metric = "cosine (via IP on normalized vectors)"
    else:
        index = faiss.IndexFlatL2(d)
        metric = "L2"

    index.add(X)

    # Persist.
    faiss_path = index_dir / "faiss.index"
    faiss.write_index(index, str(faiss_path))

    meta_path = index_dir / "chunks_meta.jsonl"
    write_jsonl(meta_path, meta_rows)

    config_path = index_dir / "index_config.json"
    config = {
        "created_at": created_at,
        "embedding_model": args.model,
        "metric": metric,
        "num_vectors": int(index.ntotal),
        "dimension": int(d),
        "chunks_path": str(chunks_path),
    }
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    # Machine-readable log.
    log_path = log_dir / "index.jsonl"
    write_jsonl(log_path, [{
        "timestamp": created_at,
        "status": "ok",
        "faiss_index": str(faiss_path),
        "chunks_meta": str(meta_path),
        "index_config": str(config_path),
        "num_vectors": int(index.ntotal),
        "dimension": int(d),
        "metric": metric,
        "embedding_model": args.model,
    }])

    print(f"Wrote FAISS index: {faiss_path}")
    print(f"Wrote chunk metadata store: {meta_path}")
    print(f"Wrote index config: {config_path}")
    print(f"Wrote index log: {log_path}")


if __name__ == "__main__":
    main()
