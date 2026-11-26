from __future__ import annotations

import pandas as pd

from nepse_lens.ingest.fixtures import (
    SEED,
    UNIVERSE,
    generate_headlines,
    generate_prices,
    write_fixtures,
)


def test_universe_is_consistent():
    assert len(UNIVERSE) >= 40
    assert len({sector for sector, _ in UNIVERSE.values()}) >= 8
    assert all(price > 0 for _, price in UNIVERSE.values())


def test_prices_are_deterministic():
    first, _ = generate_prices(n_days=60, seed=SEED)
    second, _ = generate_prices(n_days=60, seed=SEED)
    pd.testing.assert_frame_equal(first, second)


def test_price_schema_and_invariants():
    prices, labels = generate_prices(n_days=60)
    assert set(prices.columns) == {
        "symbol", "date", "open", "high", "low", "close", "volume", "turnover", "sector"
    }
    assert prices["symbol"].nunique() == len(UNIVERSE)
    assert (prices["high"] >= prices["low"]).all()
    assert (prices["high"] >= prices["close"]).all()
    assert (prices["low"] <= prices["close"]).all()
    assert (prices["volume"] > 0).all()
    assert (prices["close"] > 0).all()

    keys = set(zip(labels["symbol"], labels["date"], strict=False))
    price_keys = set(zip(prices["symbol"], prices["date"], strict=False))
    assert keys.issubset(price_keys)


def test_headlines_have_valid_labels():
    prices, _ = generate_prices(n_days=40)
    headlines = generate_headlines(pd.DatetimeIndex(sorted(prices["date"].unique())))
    assert set(headlines["label"]) <= {"positive", "negative", "neutral"}
    assert headlines["id"].is_unique
    assert headlines["headline"].str.len().min() > 10


def test_write_fixtures_roundtrip(tmp_path):
    paths = write_fixtures(tmp_path)
    for path in paths.values():
        assert path.exists()
        assert path.stat().st_size > 0
