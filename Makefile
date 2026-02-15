SHELL := /bin/bash

VENV := .venv
PY := $(VENV)/bin/python
PIP := $(PY) -m pip

MANIFEST := data/manifest/Corpus.csv
RAW_DIR := data/raw
PROCESSED_DIR := data/processed
LOG_DIR := logs

.PHONY: help venv setup ingest clean

help:
	@echo "make venv    - create virtual environment"
	@echo "make setup   - install pinned dependencies"
	@echo "make ingest  - parse PDFs -> cleaned text + updated manifest"
	@echo "make clean   - remove generated artifacts"

venv:
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip

setup: venv
	$(PIP) install -r requirements.txt

ingest:
	$(PY) -m src.ingest --manifest $(MANIFEST) --raw_dir $(RAW_DIR) --out_dir $(PROCESSED_DIR) --log_dir $(LOG_DIR)

clean:
	rm -rf $(PROCESSED_DIR) $(LOG_DIR)
