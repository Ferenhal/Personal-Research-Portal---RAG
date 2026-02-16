# Personal Research Portal — Research-Grade RAG

This repository implements a baseline Retrieval-Augmented Generation (RAG) pipeline over a local PDF corpus, with production-minded patterns for reproducibility, logging, traceable citations, and evaluation.

The system ingests PDFs into cleaned text, chunks them, embeds and indexes chunks with FAISS, retrieves top-k evidence chunks for a query, and generates an answer using a local LLM via Ollama. Answers include inline chunk citations (e.g., '[S001::c00002]') and a deterministic References section built from the manifest.

## What this RAG does

Given a question:

1. Retrieves the top-k most similar chunks from the corpus (FAISS vector search over sentence-transformer embeddings).
2. Generates an answer using a local model ('llama3.2') constrained to the retrieved evidence snippets.
3. Validates citations: if the model cites chunk IDs not in the retrieved set, the system refuses to answer rather than invent citations.
4. Adds structured references: it appends a 'References:' section containing the source metadata from the manifest for cited sources.

All runs are logged to 'logs/' (queries, retrieved chunks, prompt IDs, answers, and evaluation summaries).

---

## Repository structure

* 'data/raw/' — raw PDF sources (the corpus)
* 'data/processed/' — cleaned extracted text per source
* 'data/manifest/'

  * 'Corpus.csv' — initial corpus listing
  * 'manifest.enriched.csv' — enriched manifest used by the pipeline
  * 'manifest.enriched.jsonl' — JSONL manifest (if present)
* 'data/chunks/chunks.jsonl' — chunked corpus output
* 'data/index/' — FAISS index + metadata

  * 'faiss.index'
  * 'chunks_meta.jsonl'
  * 'index_config.json'
* 'data/eval/queries.jsonl' — evaluation query set (>=20 queries)
* 'logs/' — machine-readable logs (ingest/chunk/index/query/eval)
* 'src/' — pipeline code

  * 'ingest.py', 'chunk.py', 'index.py', 'retrieve.py', 'query.py', 'eval.py'
  * 'generators/ollama_local.py' — local generation wrapper

---

## Requirements

* macOS (tested on Apple M1 Silicon)
* Python 3.12
* Ollama (for local generation)
* A pulled Ollama model: 'llama3.2'

---

## 1) Setup Python environment

Create a virtual environment and install dependencies:

'''bash
make setup
'''

This will create '.venv/' and install pinned packages from 'requirements.txt'.

If you need to activate the environment manually:

'''bash
source .venv/bin/activate
'''

---

## 2) Install and set up Ollama (local LLM)

This project uses Ollama for local generation (no cloud API required). Ollama runs a local model and exposes an HTTP API at 'http://localhost:11434'.

### Install Ollama

Install Ollama from the official site and ensure the CLI is available:

'''bash
ollama --version
'''

### Pull the model used by this repo

This repo defaults to:

'''bash
ollama pull llama3.2
'''

Optional: verify it responds:

'''bash
ollama run llama3.2
'''

If you want to use a different Ollama model, you can override it at runtime (see "Asking questions" below).

---

## 3) Build the RAG pipeline

Run the pipeline steps in order.

### A) Ingest (PDF --> cleaned text + enriched manifest)

'''bash
make ingest
'''

Outputs:

* 'data/processed/' cleaned text files
* 'data/manifest/manifest.enriched.csv'
* 'logs/ingest.jsonl'

### B) Chunk (section-aware heuristic + overlap)

'''bash
make chunk
'''

Outputs:

* 'data/chunks/chunks.jsonl'
* 'logs/chunk.jsonl'

Chunking parameters are defined in 'Makefile':

* 'chunk_size_chars=3200'
* 'overlap_chars=400'

### C) Index (embeddings + FAISS)

'''bash
make index
'''

Outputs:

* 'data/index/faiss.index'
* 'data/index/chunks_meta.jsonl'
* 'data/index/index_config.json'
* 'logs/index.jsonl'

---

## 4) Asking questions

### One-command query

Use the Makefile 'query' target:

'''bash
make query Q="What is OptiGuide and what problem does it address?"
'''

What happens:

* Retrieves top-k chunks ('k=5' by default in the Makefile target)
* Generates an answer using Ollama ('llama3.2')
* Validates citations (refuses if citations are invalid)
* Appends a 'References:' section based on the manifest
* Writes a log entry to 'logs/query.jsonl'

### Change the local model

Override the default model from the command line:

'''bash
make query Q="..." OLLAMA_MODEL=llama3.2
'''

(Replace with any model you have pulled via 'ollama pull <model>'.)

### Where outputs and traceability live

Each query appends a record to:

* 'logs/query.jsonl'

This includes:

* query text
* retrieved chunk IDs, scores, and evidence text
* prompt hash / version ID
* generator status
* citation validation results (valid/invalid citations)
* final answer (with structured references)

---

## 5) Evaluating the system (≥20 queries)

The evaluation query set is stored at:

* 'data/eval/queries.jsonl'

Run evaluation:

'''bash
make eval
'''

Outputs:

* 'logs/eval.jsonl' — per-run metadata and metric values
* 'logs/eval_summary.csv' — summary metrics per query

### Metrics

The evaluation currently includes:

* Groundedness / faithfulness (structural) via citation validity:

  * invalid-citation count (must be 0)
  * citation coverage proxy (fraction of output lines containing citations)
  
* Additional metric: citation precision
  * fraction of cited chunk IDs that are in the retrieved set

Because a citation validator is implemented, invalid citation rate should be near zero; remaining failure modes tend to be "missing citations" or "insufficient evidence retrieved."

### How to inspect failures

Open:

* 'logs/eval_summary.csv' to identify low-performing queries (e.g., low coverage / no citations).

Then locate the corresponding entry in:

* 'logs/query.jsonl' (search by question or timestamp).

---

## Citation behavior and enhancement

This repo implements a "Structured citations" enhancement:

1. Inline citations in the answer body: '[S###::c#####]' (chunk IDs)
2. Deterministic References section appended by the program from the manifest ('manifest.enriched.csv'), not generated by the model.

Additionally, a citation validator is enforced:

* If the model cites any chunk IDs that were not in the retrieved top-k evidence for that query, the system refuses rather than hallucinate citations.

---

## Makefile targets

Common commands:

'''bash
make setup     # create .venv and install deps
make ingest    # parse PDFs -> cleaned text + enriched manifest
make chunk     # chunk cleaned texts -> chunks.jsonl
make index     # embed + build FAISS index
make query Q="..."  # ask a question
make eval      # run evaluation set
make clean     # remove generated processed + logs (use with care)
'''

---

## Notes and troubleshooting

### Ollama isn’t running

If the Python code can’t reach 'http://localhost:11434', open the Ollama app or run a model once:

'''bash
ollama run llama3.2
'''

### Hugging Face warning (unauthenticated)

You may see warnings about unauthenticated Hugging Face requests when sentence-transformers downloads embedding models. This is normal; adding an HF token is optional and only affects rate limits.

### Performance tips (Apple Silicon laptops)

* Use smaller 'k' for direct queries (5 is fine) and larger 'k' for synthesis queries (7–10).
* Keep the context window at 4096 unless you see truncation. Increasing it can slow generation and increase memory usage.

---

## Reproducibility

* Python deps are pinned in 'requirements.txt'.
* One-command run paths exist via 'make ingest', 'make chunk', 'make index', 'make query', 'make eval'.
* Logs are machine-readable and stored in 'logs/'.