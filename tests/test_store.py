from __future__ import annotations

import pandas as pd

from nepse_lens import store


def test_roundtrip_prices(tmp_path):
    db = tmp_path / "roundtrip.duckdb"
    store.init_db(db)
    con = store.connect(db)
    try:
        frame = pd.DataFrame(
            {
                "symbol": ["AAA", "AAA"],
                "date": pd.to_datetime(["2026-01-01", "2026-01-02"]),
                "open": [100.0, 101.0],
                "high": [102.0, 103.0],
                "low": [99.0, 100.0],
                "close": [101.0, 102.0],
                "volume": [1000, 1100],
                "turnover": [101000.0, 112200.0],
                "sector": ["Test", "Test"],
            }
        )
        assert store.write_prices(con, frame) == 2
        back = store.read_prices(con)
        assert len(back) == 2
        assert back["close"].tolist() == [101.0, 102.0]
        assert store.table_counts(con)["prices"] == 2
    finally:
        con.close()


def test_writes_are_idempotent(tmp_path):
    db = tmp_path / "idempotent.duckdb"
    store.init_db(db)
    con = store.connect(db)
    try:
        frame = pd.DataFrame(
            {
                "symbol": ["AAA"],
                "date": pd.to_datetime(["2026-01-01"]),
                "open": [100.0],
                "high": [102.0],
                "low": [99.0],
                "close": [101.0],
                "volume": [1000],
                "turnover": [101000.0],
                "sector": ["Test"],
            }
        )
        store.write_prices(con, frame)
        frame.loc[0, "close"] = 105.0
        store.write_prices(con, frame)
        back = store.read_prices(con)
        assert len(back) == 1
        assert back.iloc[0]["close"] == 105.0
    finally:
        con.close()
