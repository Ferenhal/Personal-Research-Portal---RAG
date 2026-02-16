SHELL := /bin/bash

VENV := .venv
PYTHON := /opt/homebrew/opt/python@3.12/bin/python3.12
PY := $(VENV)/bin/python
PIP := $(PY) -m pip
OLLAMA_MODEL ?= llama3.2

MANIFEST := data/manifest/Corpus.csv
MANIFEST_ENRICHED := data/manifest/manifest.enriched.csv

EVAL_QUERIES := data/eval/queries.jsonl

RAW_DIR := data/raw
PROCESSED_DIR := data/processed
LOG_DIR := logs
CHUNKS_DIR := data/chunks
CHUNKS_PATH := $(CHUNKS_DIR)/chunks.jsonl
INDEX_DIR := data/index


.PHONY: help venv setup ingest clean

help:
	@echo "make venv    - create virtual environment"
	@echo "make setup   - install pinned dependencies"
	@echo "make ingest  - parse PDFs -> cleaned text + updated manifest"
	@echo "make clean   - remove generated artifacts"

venv:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip

setup: venv
	$(PIP) install -r requirements.txt

ingest:
	$(PY) -m src.ingest --manifest $(MANIFEST) --raw_dir $(RAW_DIR) --out_dir $(PROCESSED_DIR) --log_dir $(LOG_DIR)

clean:
	rm -rf $(PROCESSED_DIR) $(LOG_DIR)

chunk:
	mkdir -p $(CHUNKS_DIR)
	$(PY) -m src.chunk --manifest data/manifest/manifest.enriched.csv --out $(CHUNKS_PATH) --chunk_size_chars 3200 --overlap_chars 400 --log_dir $(LOG_DIR)

index:
	mkdir -p $(INDEX_DIR)
	$(PY) -m src.index --chunks data/chunks/chunks.jsonl --index_dir $(INDEX_DIR) --log_dir $(LOG_DIR) --use_cosine

query:
	$(PY) -m src.query --question "$(Q)" --k 5 --index_dir data/index --chunks_path data/chunks/chunks.jsonl --log_dir $(LOG_DIR) --local_model "$(OLLAMA_MODEL)" --manifest_path "$(MANIFEST_ENRICHED)"

eval:
	$(PY) -m src.eval --queries $(EVAL_QUERIES) --index_dir $(INDEX_DIR) --chunks_path $(CHUNKS_PATH) --log_dir $(LOG_DIR) --local_model "$(OLLAMA_MODEL)" --manifest_path "$(MANIFEST_ENRICHED)"
