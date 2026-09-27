PY ?= python

.PHONY: install install-ml test lint data ingest ingest-fast serve redteam cache-bench eval-retrieval gen-questions

install:        ## core + dev dependencies
	$(PY) -m pip install -e ".[dev]"

install-ml:     ## + sentence-transformers (local embeddings, reranker)
	$(PY) -m pip install -e ".[dev,ml]"

test:
	$(PY) -m pytest

lint:
	ruff check src tests

data:           ## download + extract the default 20 mailboxes
	clearance download

ingest:         ## real embeddings (needs install-ml)
	clearance ingest --embedder local

ingest-fast:    ## no model download, ~2 min
	clearance ingest --embedder hash

serve:
	clearance serve

redteam:
	clearance eval redteam --n 200

cache-bench:
	clearance eval cache --n 80

gen-questions:
	clearance gen-questions --method offline --out eval_data/generated.jsonl

eval-retrieval:
	clearance eval retrieval --gold eval_data/gold.jsonl
