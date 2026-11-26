"""Read-only FastAPI service over the DuckDB store."""

from __future__ import annotations

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query

from . import __version__, store
from .config import DB_PATH

app = FastAPI(
    title="NEPSE Lens API",
    version=__version__,
    description="Market intelligence for the Nepal Stock Exchange: prices, anomalies, sentiment, predictions.",
)


def _connect():
    if not DB_PATH.exists():
        raise HTTPException(status_code=503, detail="Database not found. Run `make train` first.")
    return store.connect(DB_PATH, read_only=True)


def _records(frame: pd.DataFrame) -> list[dict]:
    if frame.empty:
        return []
    clean = frame.replace({np.nan: None})
    if "date" in clean.columns:
        clean["date"] = clean["date"].astype(str)
    return clean.to_dict(orient="records")


@app.get("/")
def root() -> dict:
    return {
        "name": "NEPSE Lens API",
        "version": __version__,
        "docs": "/docs",
        "endpoints": ["/health", "/tickers", "/ohlcv/{symbol}", "/anomalies", "/sentiment", "/predictions", "/market/summary"],
    }


@app.get("/health")
def health() -> dict:
    con = _connect()
    try:
        return {"status": "ok", "version": __version__, "tables": store.table_counts(con)}
    finally:
        con.close()


@app.get("/tickers")
def tickers() -> list[dict]:
    con = _connect()
    try:
        frame = con.execute(
            "SELECT symbol, any_value(sector) AS sector, min(date) AS first_date, "
            "max(date) AS last_date, count(*) AS sessions FROM prices GROUP BY symbol ORDER BY symbol"
        ).df()
        return _records(frame)
    finally:
        con.close()


@app.get("/ohlcv/{symbol}")
def ohlcv(symbol: str, limit: int = Query(180, ge=1, le=2000)) -> list[dict]:
    con = _connect()
    try:
        frame = con.execute(
            "SELECT symbol, date, open, high, low, close, volume, turnover, sector "
            "FROM prices WHERE symbol = ? ORDER BY date DESC LIMIT ?",
            [symbol.upper(), limit],
        ).df()
        if frame.empty:
            raise HTTPException(status_code=404, detail=f"Unknown symbol: {symbol}")
        return _records(frame.sort_values("date"))
    finally:
        con.close()


@app.get("/anomalies")
def anomalies(
    method: str = Query("robust_zscore"),
    limit: int = Query(50, ge=1, le=500),
    only_flagged: bool = Query(True),
) -> list[dict]:
    con = _connect()
    try:
        clause = "AND is_anomaly" if only_flagged else ""
        frame = con.execute(
            f"SELECT symbol, date, method, score, is_anomaly FROM anomalies "
            f"WHERE method = ? {clause} ORDER BY score DESC LIMIT ?",
            [method, limit],
        ).df()
        return _records(frame)
    finally:
        con.close()


@app.get("/sentiment")
def sentiment(
    symbol: str = Query("MARKET"),
    limit: int = Query(90, ge=1, le=1000),
) -> list[dict]:
    con = _connect()
    try:
        frame = con.execute(
            "SELECT date, symbol, model, score, label FROM sentiment "
            "WHERE symbol = ? ORDER BY date DESC LIMIT ?",
            [symbol.upper(), limit],
        ).df()
        return _records(frame.sort_values("date"))
    finally:
        con.close()


@app.get("/predictions")
def predictions(limit: int = Query(50, ge=1, le=500)) -> list[dict]:
    con = _connect()
    try:
        frame = con.execute(
            "SELECT symbol, date, model, prob_up FROM predictions "
            "WHERE date = (SELECT max(date) FROM predictions) ORDER BY prob_up DESC LIMIT ?",
            [limit],
        ).df()
        return _records(frame)
    finally:
        con.close()


@app.get("/market/summary")
def market_summary() -> dict:
    con = _connect()
    try:
        last_date = con.execute("SELECT max(date) FROM prices").fetchone()[0]
        if last_date is None:
            raise HTTPException(status_code=404, detail="No prices loaded")

        rows = con.execute(
            "SELECT symbol, close, volume, turnover, sector FROM prices WHERE date = ?", [last_date]
        ).df()
        previous = con.execute(
            "SELECT symbol, close FROM prices WHERE date = ("
            "SELECT max(date) FROM prices WHERE date < ?)", [last_date]
        ).df()
        merged = rows.merge(previous, on="symbol", suffixes=("", "_prev"))
        merged["ret_1d"] = merged["close"] / merged["close_prev"] - 1.0

        sentiment_row = con.execute(
            "SELECT score, label FROM sentiment WHERE symbol = 'MARKET' AND date = ?", [last_date]
        ).fetchone()

        return {
            "date": str(last_date),
            "symbols": int(len(merged)),
            "market_return": float(merged["ret_1d"].mean()),
            "breadth": float((merged["ret_1d"] > 0).mean()),
            "turnover": float(merged["turnover"].sum()),
            "sentiment_score": float(sentiment_row[0]) if sentiment_row else None,
            "sentiment_label": sentiment_row[1] if sentiment_row else None,
            "top_gainers": _records(
                merged.sort_values("ret_1d", ascending=False).head(5)[["symbol", "close", "ret_1d"]]
            ),
            "top_losers": _records(
                merged.sort_values("ret_1d").head(5)[["symbol", "close", "ret_1d"]]
            ),
        }
    finally:
        con.close()
