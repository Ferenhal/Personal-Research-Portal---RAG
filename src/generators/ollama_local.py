from __future__ import annotations

import json
from typing import Dict, Optional

import requests


def ollama_generate(
    prompt: str,
    model: str = "llama3.2",
    base_url: str = "http://localhost:11434",
    num_ctx: int = 4096,
    temperature: float = 0.2,
    timeout_s: int = 180,
) -> Dict:
    """
    Calls local Ollama /api/generate and returns:
      {"status": "ok"|"error", "answer": str, "error": optional dict}
    """
    url = f"{base_url}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_ctx": num_ctx,
            "temperature": temperature,
        },
    }

    try:
        r = requests.post(url, json=payload, timeout=timeout_s)
        r.raise_for_status()
        data = r.json()
        return {"status": "ok", "answer": data.get("response", ""), "error": None, "raw": data}
    except Exception as e:
        return {"status": "error", "answer": "", "error": {"type": e.__class__.__name__, "message": str(e)}}
