"""Feature engineering.

Every feature at row ``t`` uses data available at the close of ``t``. The target
is the sign of the next session's close-to-close return. Keeping that boundary
strict is the whole point: leakage makes a backtest look brilliant and a live
model useless.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "ret_1d",
    "ret_5d",
    "ret_20d",
    "vol_20d",
    "vol_60d",
    "volume_z_20",
    "turnover_z_20",
    "ma_gap_20",
    "rsi_14",
    "range_pct",
    "gap_pct",
    "market_ret",
    "breadth",
    "avg_volume_z",
]

MARKET_COLUMNS = ["market_ret", "breadth", "avg_volume_z", "n_symbols"]


def _rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window, min_periods=window).mean()
    loss = (-delta.clip(upper=0)).rolling(window, min_periods=window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)


def add_price_features(prices: pd.DataFrame) -> pd.DataFrame:
    df = prices.sort_values(["symbol", "date"]).reset_index(drop=True).copy()
    grouped = df.groupby("symbol", group_keys=False)

    df["ret_1d"] = grouped["close"].pct_change()
    df["ret_5d"] = grouped["close"].pct_change(5)
    df["ret_20d"] = grouped["close"].pct_change(20)
    df["vol_20d"] = grouped["ret_1d"].transform(lambda s: s.rolling(20, min_periods=20).std())
    df["vol_60d"] = grouped["ret_1d"].transform(lambda s: s.rolling(60, min_periods=40).std())

    for col in ("volume", "turnover"):
        mean = grouped[col].transform(lambda s: s.rolling(20, min_periods=20).mean())
        std = grouped[col].transform(lambda s: s.rolling(20, min_periods=20).std())
        df[f"{col}_z_20"] = (df[col] - mean) / std.replace(0, np.nan)

    ma20 = grouped["close"].transform(lambda s: s.rolling(20, min_periods=20).mean())
    df["ma_gap_20"] = df["close"] / ma20 - 1.0
    df["rsi_14"] = grouped["close"].transform(lambda s: _rsi(s, 14))
    df["range_pct"] = (df["high"] - df["low"]) / df["close"].replace(0, np.nan)
    df["gap_pct"] = df["open"] / grouped["close"].shift(1) - 1.0

    next_close = grouped["close"].shift(-1)
    df["target_up"] = np.where(next_close.notna(), (next_close > df["close"]).astype(float), np.nan)
    return df


def add_market_features(df: pd.DataFrame) -> pd.DataFrame:
    market = (
        df.groupby("date")
        .agg(
            market_ret=("ret_1d", "mean"),
            breadth=("ret_1d", lambda s: float((s > 0).mean())),
            avg_volume_z=("volume_z_20", "mean"),
            n_symbols=("symbol", "nunique"),
        )
        .reset_index()
    )
    return df.merge(market, on="date", how="left")


def build_features(prices: pd.DataFrame) -> pd.DataFrame:
    return add_market_features(add_price_features(prices))


def training_frame(features: pd.DataFrame) -> pd.DataFrame:
    frame = features.dropna(subset=[*FEATURE_COLUMNS, "target_up"]).copy()
    frame["target_up"] = frame["target_up"].astype(int)
    return frame.sort_values(["date", "symbol"]).reset_index(drop=True)


def sector_daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    df = prices.sort_values(["symbol", "date"]).copy()
    df["ret_1d"] = df.groupby("symbol")["close"].pct_change()
    return (
        df.dropna(subset=["ret_1d"])
        .groupby(["date", "sector"])["ret_1d"]
        .mean()
        .reset_index()
    )
