from __future__ import annotations

import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

CITATION_RE = re.compile(r"\[(S\d{3}::c\d{5})\]")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def group_query_logs_by_thread(rows: Iterable[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    by_thread: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        tid = r.get("thread_id") or "default"
        by_thread[tid].append(r)
    # newest-first within each thread
    for tid, items in by_thread.items():
        items.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    return dict(by_thread)


def extract_citations(text: str) -> List[str]:
    seen = set()
    out: List[str] = []
    for cid in CITATION_RE.findall(text or ""):
        if cid not in seen:
            seen.add(cid)
            out.append(cid)
    return out


def now_slug() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def safe_thread_id(label: str) -> str:
    # lightweight: keep alnum, dash, underscore
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "-", (label or "").strip())
    cleaned = re.sub(r"-+", "-", cleaned).strip("-")
    if not cleaned:
        cleaned = "thread"
    return cleaned[:48]


def render_thread_markdown(thread_id: str, items: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    lines.append(f"# Research thread: {thread_id}\n")

    for i, row in enumerate(reversed(items), start=1):
        ts = row.get("timestamp", "")
        q = row.get("question", "")
        ans = row.get("generation", {}).get("answer", "")
        lines.append(f"## {i}. {q}")
        if ts:
            lines.append(f"*Timestamp:* {ts}\n")
        lines.append(ans)
        lines.append("\n---\n")

    return "\n".join(lines)
