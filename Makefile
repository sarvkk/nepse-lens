.PHONY: install fixture features train train-full test lint api dash preview phrasebank reproduce vercel-db clean

install:
	python -m pip install -e ".[full,dev]"

fixture:
	python scripts/fetch_data.py --source fixture

features: fixture
	python scripts/build_features.py

train: features
	python scripts/train.py

phrasebank:
	python scripts/fetch_phrasebank.py

train-full: features phrasebank
	python scripts/train.py --phrasebank data/external/FinancialPhraseBank-v1.0/Sentences_AllAgree.txt

vercel-db: train
	cp data/nepse.duckdb data/nepse-demo.duckdb

test:
	pytest

lint:
	ruff check src scripts tests

api:
	uvicorn nepse_lens.api:app --reload --port 8000

dash:
	streamlit run src/nepse_lens/dashboard.py --server.port 8501

preview: train
	python scripts/make_preview.py

reproduce: lint test train
	@echo "Reports written to reports/"

clean:
	rm -f data/nepse.duckdb data/nepse.duckdb.wal
	rm -rf reports/*.json reports/*.csv
