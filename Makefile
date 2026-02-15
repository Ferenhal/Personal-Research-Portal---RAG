SHELL := /bin/bash

VENV := .venv
PYTHON := /opt/homebrew/opt/python@3.12/bin/python3.12
PY := $(VENV)/bin/python
PIP := $(PY) -m pip

MANIFEST := data/manifest/Corpus.csv
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
