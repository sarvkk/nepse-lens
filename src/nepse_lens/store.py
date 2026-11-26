"""DuckDB persistence layer.

DuckDB gives us analytical SQL over a single file with no server to run, which
keeps local dev and CI identical. Tables are idempotent: re-running the pipeline
replaces rows instead of duplicating them.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from .config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS prices (
    symbol VARCHAR NOT NULL,
    date DATE NOT NULL,
    open DOUBLE,
    high DOUBLE,
    low DOUBLE,
    close DOUBLE,
    volume BIGINT,
    turnover DOUBLE,
    sector VARCHAR,
    PRIMARY KEY (symbol, date)
);
CREATE TABLE IF NOT EXISTS headlines (
    id VARCHAR PRIMARY KEY,
    date DATE,
    source VARCHAR,
    headline VARCHAR,
    symbol VARCHAR,
    label VARCHAR
);
CREATE TABLE IF NOT EXISTS anomaly_labels (
    symbol VARCHAR NOT NULL,
    date DATE NOT NULL,
    PRIMARY KEY (symbol, date)
);
CREATE TABLE IF NOT EXISTS anomalies (
    symbol VARCHAR NOT NULL,
    date DATE NOT NULL,
    method VARCHAR NOT NULL,
    score DOUBLE,
    is_anomaly BOOLEAN,
    PRIMARY KEY (symbol, date, method)
);
CREATE TABLE IF NOT EXISTS predictions (
    symbol VARCHAR NOT NULL,
    date DATE NOT NULL,
    model VARCHAR NOT NULL,
    prob_up DOUBLE,
    PRIMARY KEY (symbol, date, model)
);
CREATE TABLE IF NOT EXISTS sentiment (
    date DATE NOT NULL,
    symbol VARCHAR NOT NULL,
    model VARCHAR NOT NULL,
    score DOUBLE,
    label VARCHAR,
    PRIMARY KEY (date, symbol, model)
);
"""

PRICE_COLS = ["symbol", "date", "open", "high", "low", "close", "volume", "turnover", "sector"]


def connect(path: Path | str = DB_PATH, read_only: bool = False) -> duckdb.DuckDBPyConnection:
    return duckdb.connect(str(path), read_only=read_only)


def init_db(path: Path | str = DB_PATH) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    con = connect(path)
    try:
        con.execute(SCHEMA)
    finally:
        con.close()


def _replace(con: duckdb.DuckDBPyConnection, table: str, df: pd.DataFrame, cols: list[str]) -> int:
    if df.empty:
        return 0
    frame = df[cols].copy()
    con.register("_incoming", frame)
    try:
        con.execute(f"INSERT OR REPLACE INTO {table} SELECT * FROM _incoming")
    finally:
        con.unregister("_incoming")
    return len(frame)


def write_prices(con: duckdb.DuckDBPyConnection, df: pd.DataFrame) -> int:
    return _replace(con, "prices", df, PRICE_COLS)


def write_headlines(con: duckdb.DuckDBPyConnection, df: pd.DataFrame) -> int:
    return _replace(con, "headlines", df, ["id", "date", "source", "headline", "symbol", "label"])


def write_anomaly_labels(con: duckdb.DuckDBPyConnection, df: pd.DataFrame) -> int:
    return _replace(con, "anomaly_labels", df, ["symbol", "date"])


def write_anomalies(con: duckdb.DuckDBPyConnection, df: pd.DataFrame) -> int:
    return _replace(con, "anomalies", df, ["symbol", "date", "method", "score", "is_anomaly"])


def write_predictions(con: duckdb.DuckDBPyConnection, df: pd.DataFrame) -> int:
    return _replace(con, "predictions", df, ["symbol", "date", "model", "prob_up"])


def write_sentiment(con: duckdb.DuckDBPyConnection, df: pd.DataFrame) -> int:
    return _replace(con, "sentiment", df, ["date", "symbol", "model", "score", "label"])


def read_prices(con: duckdb.DuckDBPyConnection, symbols: list[str] | None = None) -> pd.DataFrame:
    query = "SELECT * FROM prices"
    params: list[object] = []
    if symbols:
        placeholders = ", ".join(["?"] * len(symbols))
        query += f" WHERE symbol IN ({placeholders})"
        params = list(symbols)
    query += " ORDER BY symbol, date"
    df = con.execute(query, params).df()
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
    return df


def read_headlines(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    df = con.execute("SELECT * FROM headlines ORDER BY date").df()
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
    return df


def read_table(con: duckdb.DuckDBPyConnection, table: str) -> pd.DataFrame:
    df = con.execute(f"SELECT * FROM {table}").df()
    if "date" in df.columns and not df.empty:
        df["date"] = pd.to_datetime(df["date"])
    return df


def table_counts(con: duckdb.DuckDBPyConnection) -> dict[str, int]:
    tables = ["prices", "headlines", "anomaly_labels", "anomalies", "predictions", "sentiment"]
    return {t: int(con.execute(f"SELECT count(*) FROM {t}").fetchone()[0]) for t in tables}
