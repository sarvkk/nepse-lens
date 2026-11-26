.PHONY: install fixture features train test lint api dash reproduce clean

install:
	python -m pip install -e ".[dev]"

fixture:
	python scripts/fetch_data.py --source fixture

features: fixture
	python scripts/build_features.py

train: features
	python scripts/train.py

test:
	pytest

lint:
	ruff check src tests scripts

api:
	uvicorn nepse_lens.api:app --reload --port 8000

dash:
	streamlit run src/nepse_lens/dashboard.py --server.port 8501

reproduce: lint test train
	@echo "Reports written to reports/"

clean:
	rm -f data/nepse.duckdb data/nepse.duckdb.wal
	rm -rf reports/*.json reports/*.csv
