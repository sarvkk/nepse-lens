#!/usr/bin/env python
"""Render a static dashboard preview (docs/preview.png) from the DuckDB store."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from nepse_lens import store
from nepse_lens.config import DB_PATH
from nepse_lens.features import sector_daily_returns


def main() -> int:
    con = store.connect(DB_PATH, read_only=True)
    try:
        prices = store.read_prices(con)
        anomalies = store.read_table(con, "anomalies")
        sentiment = store.read_table(con, "sentiment")
    finally:
        con.close()

    symbol = "NABIL"
    bars = prices[prices["symbol"] == symbol].sort_values("date").tail(180).copy()
    bars["ma20"] = bars["close"].rolling(20).mean()
    flags = anomalies[
        (anomalies["method"] == "robust_zscore")
        & (anomalies["is_anomaly"])
        & (anomalies["symbol"] == symbol)
    ].merge(bars[["date", "close"]], on="date", how="inner")

    sector = sector_daily_returns(prices)
    pivot = sector.pivot(index="sector", columns="date", values="ret_1d").iloc[:, -30:]

    wide = prices.pivot(index="date", columns="symbol", values="close").sort_index()
    returns = wide.pct_change()
    breadth = (returns > 0).where(returns.notna()).mean(axis=1).reset_index()
    breadth.columns = ["date", "breadth"]
    breadth = breadth.tail(180)

    market = sentiment[sentiment["symbol"] == "MARKET"].sort_values("date").tail(180)

    figure = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            f"{symbol} — daily bars, MA20, anomalies",
            "Sector returns — last 30 sessions (%)",
            "Market breadth — share of advancers",
            "Daily headline sentiment index",
        ),
        specs=[[{"type": "xy"}, {"type": "xy"}], [{"type": "xy"}, {"type": "xy"}]],
        vertical_spacing=0.14,
        horizontal_spacing=0.08,
    )

    figure.add_trace(
        go.Candlestick(
            x=bars["date"], open=bars["open"], high=bars["high"], low=bars["low"],
            close=bars["close"], name=symbol, showlegend=False,
        ),
        row=1, col=1,
    )
    figure.add_trace(
        go.Scatter(x=bars["date"], y=bars["ma20"], mode="lines", name="MA20",
                   line={"color": "#f59e0b", "width": 1.4}, showlegend=False),
        row=1, col=1,
    )
    if not flags.empty:
        figure.add_trace(
            go.Scatter(x=flags["date"], y=flags["close"], mode="markers", name="anomaly",
                       marker={"symbol": "triangle-down", "size": 10, "color": "#dc2626"},
                       showlegend=False),
            row=1, col=1,
        )

    figure.add_trace(
        go.Heatmap(
            z=pivot.to_numpy() * 100, x=pivot.columns, y=pivot.index,
            colorscale="RdYlGn", zmid=0, showscale=False,
        ),
        row=1, col=2,
    )

    figure.add_trace(
        go.Scatter(x=breadth["date"], y=breadth["breadth"], mode="lines",
                   line={"color": "#059669", "width": 2}, showlegend=False),
        row=2, col=1,
    )
    figure.add_trace(
        go.Bar(
            x=market["date"], y=market["score"],
            marker_color=["#059669" if s >= 0 else "#dc2626" for s in market["score"]],
            showlegend=False,
        ),
        row=2, col=2,
    )
    figure.add_hline(y=0, line_width=1, line_color="#64748b", row=2, col=2)

    figure.update_layout(
        template="plotly_white",
        title={"text": "NEPSE Lens — market intelligence dashboard", "x": 0.01, "font": {"size": 22}},
        height=820,
        width=1440,
        margin={"l": 110, "r": 30, "t": 90, "b": 40},
    )
    figure.update_xaxes(rangeslider_visible=False)
    figure.update_yaxes(title_text="NPR", row=1, col=1)
    figure.update_yaxes(title_text="share advancing", row=2, col=1)
    figure.update_yaxes(title_text="score", row=2, col=2)

    out = Path("docs/preview.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.write_image(out, scale=2)
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
