from __future__ import annotations

import pandas as pd
import pytest

from nepse_lens import store
from nepse_lens.ingest.fixtures import generate_headlines, generate_prices


@pytest.fixture(scope="session")
def fixture_data():
    prices, labels = generate_prices(n_days=400)
    headlines = generate_headlines(pd.DatetimeIndex(sorted(prices["date"].unique())))
    return prices, headlines, labels


@pytest.fixture(scope="session")
def small_db(tmp_path_factory, fixture_data):
    prices, headlines, labels = fixture_data
    path = tmp_path_factory.mktemp("db") / "test.duckdb"
    store.init_db(path)
    con = store.connect(path)
    try:
        store.write_prices(con, prices)
        store.write_headlines(con, headlines)
        store.write_anomaly_labels(con, labels)
    finally:
        con.close()
    return path


@pytest.fixture(scope="session")
def trained(small_db, tmp_path_factory):
    import nepse_lens.pipeline as pipeline

    reports = tmp_path_factory.mktemp("reports")
    original = pipeline.REPORTS_DIR
    pipeline.REPORTS_DIR = reports
    try:
        metrics = pipeline.run_pipeline(small_db)
    finally:
        pipeline.REPORTS_DIR = original
    return small_db, metrics, reports
