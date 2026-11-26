"""Streamlit dashboard.

Run with:
    streamlit run src/nepse_lens/dashboard.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from nepse_lens import store
from nepse_lens.config import DB_PATH
from nepse_lens.features import sector_daily_returns

st.set_page_config(page_title="NEPSE Lens", page_icon="📈", layout="wide")


@st.cache_data(show_spinner=False)
def load_all(db_mtime: float) -> dict[str, pd.DataFrame]:
    con = store.connect(DB_PATH, read_only=True)
    try:
        return {
            "prices": store.read_prices(con),
            "anomalies": store.read_table(con, "anomalies"),
            "sentiment": store.read_table(con, "sentiment"),
            "predictions": store.read_table(con, "predictions"),
        }
    finally:
        con.close()


def line_chart(frame: pd.DataFrame, x: str, y: str, title: str, color: str = "#2563eb") -> go.Figure:
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=frame[x], y=frame[y], mode="lines", line={"color": color, "width": 2}))
    figure.update_layout(title=title, height=280, margin={"l": 10, "r": 10, "t": 40, "b": 10})
    return figure


st.title("NEPSE Lens")
st.caption(
    "Market intelligence for the Nepal Stock Exchange — prices, anomaly detection, "
    "financial-news sentiment, and next-session probabilities."
)

if not Path(DB_PATH).exists():
    st.error("No database found. Run `make train` (or `python scripts/fetch_data.py && python scripts/train.py`).")
    st.stop()

bundle = load_all(Path(DB_PATH).stat().st_mtime)
prices = bundle["prices"]
if prices.empty:
    st.error("The database is empty. Run `make train` first.")
    st.stop()

last_date = prices["date"].max()
first_date = prices["date"].min()

with st.sidebar:
    st.header("Controls")
    symbols = sorted(prices["symbol"].unique())
    symbol = st.selectbox("Ticker", symbols, index=symbols.index("NABIL") if "NABIL" in symbols else 0)
    lookback = st.slider("Sessions shown", 60, 360, 180, step=20)
    st.divider()
    st.metric("Universe", f"{prices['symbol'].nunique()} symbols")
    st.metric("Sessions", f"{prices['date'].nunique():,}")
    st.caption(f"{first_date.date()} → {last_date.date()}")
    st.divider()
    st.caption("Fixture data by default. See MODEL_CARD.md for limits.")

symbol_prices = prices[prices["symbol"] == symbol].sort_values("date").tail(lookback).copy()
symbol_prices["ma20"] = (
    prices[prices["symbol"] == symbol].sort_values("date")["close"].rolling(20).mean().tail(lookback).to_numpy()
)
latest = symbol_prices.iloc[-1]
prev = symbol_prices.iloc[-2] if len(symbol_prices) > 1 else latest
ret_1d = latest["close"] / prev["close"] - 1.0
vol_20d = symbol_prices["close"].pct_change().tail(20).std()

market_sentiment = bundle["sentiment"]
market_sentiment = market_sentiment[market_sentiment["symbol"] == "MARKET"].sort_values("date")
latest_sentiment = market_sentiment.iloc[-1] if not market_sentiment.empty else None

recent_anomalies = bundle["anomalies"]
recent_anomalies = recent_anomalies[
    (recent_anomalies["method"] == "robust_zscore") & (recent_anomalies["is_anomaly"])
]
recent_count = int((recent_anomalies["date"] >= last_date - pd.Timedelta(days=30)).sum())

cols = st.columns(5)
cols[0].metric(f"{symbol} close", f"Rs {latest['close']:,.2f}", f"{ret_1d * 100:+.2f}%")
cols[1].metric("20d volatility", f"{vol_20d * 100:.2f}%")
cols[2].metric("Market sentiment", f"{latest_sentiment['score']:+.2f}" if latest_sentiment is not None else "n/a",
               latest_sentiment["label"] if latest_sentiment is not None else None)
cols[3].metric("Anomalies (30d)", recent_count)
cols[4].metric("Price rows", f"{len(prices):,}")

candle = go.Figure()
candle.add_trace(
    go.Candlestick(
        x=symbol_prices["date"],
        open=symbol_prices["open"],
        high=symbol_prices["high"],
        low=symbol_prices["low"],
        close=symbol_prices["close"],
        name=symbol,
    )
)
candle.add_trace(
    go.Scatter(x=symbol_prices["date"], y=symbol_prices["ma20"], mode="lines", name="MA20",
               line={"color": "#f59e0b", "width": 1.5})
)
flags = recent_anomalies[recent_anomalies["symbol"] == symbol].merge(
    symbol_prices[["date", "close"]], on="date", how="inner"
)
if not flags.empty:
    candle.add_trace(
        go.Scatter(
            x=flags["date"], y=flags["close"], mode="markers", name="anomaly",
            marker={"symbol": "triangle-down", "size": 11, "color": "#dc2626"},
        )
    )
candle.update_layout(
    title=f"{symbol} — daily bars",
    height=440,
    xaxis_rangeslider_visible=False,
    margin={"l": 10, "r": 10, "t": 50, "b": 10},
)
st.plotly_chart(candle, width="stretch")

left, right = st.columns(2)

with left:
    sector_returns = sector_daily_returns(prices)
    pivot = sector_returns.pivot(index="sector", columns="date", values="ret_1d").iloc[:, -30:]
    heatmap = go.Figure(
        go.Heatmap(
            z=pivot.to_numpy() * 100,
            x=pivot.columns,
            y=pivot.index,
            colorscale="RdYlGn",
            zmid=0,
            colorbar={"title": "%"},
        )
    )
    heatmap.update_layout(
        title="Sector returns — last 30 sessions (%)",
        height=340,
        margin={"l": 10, "r": 10, "t": 50, "b": 10},
    )
    st.plotly_chart(heatmap, width="stretch")

with right:
    wide = prices.pivot(index="date", columns="symbol", values="close").sort_index()
    returns = wide.pct_change()
    breadth = (returns > 0).mean(axis=1)
    breadth_frame = breadth.reset_index()
    breadth_frame.columns = ["date", "breadth"]
    st.plotly_chart(
        line_chart(breadth_frame.tail(lookback), "date", "breadth", "Market breadth — share of advancers", "#059669"),
        width="stretch",
    )

if not market_sentiment.empty:
    sentiment_frame = market_sentiment.tail(lookback)
    figure = go.Figure()
    figure.add_trace(
        go.Bar(
            x=sentiment_frame["date"],
            y=sentiment_frame["score"],
            marker_color=["#059669" if s >= 0 else "#dc2626" for s in sentiment_frame["score"]],
            name="sentiment",
        )
    )
    figure.add_hline(y=0, line_width=1, line_color="#64748b")
    figure.update_layout(
        title="Daily headline sentiment index (finance lexicon)",
        height=280,
        margin={"l": 10, "r": 10, "t": 50, "b": 10},
    )
    st.plotly_chart(figure, width="stretch")

st.subheader("Anomaly feed")
feed = recent_anomalies.sort_values("score", ascending=False).head(12).merge(
    prices[["symbol", "date", "close", "sector"]], on=["symbol", "date"], how="left"
)
feed = feed[["date", "symbol", "sector", "close", "score"]].rename(
    columns={"date": "Date", "symbol": "Symbol", "sector": "Sector", "close": "Close", "score": "Score"}
)
st.dataframe(feed, width="stretch", hide_index=True)

predictions = bundle["predictions"]
if not predictions.empty:
    st.subheader("Next-session up-probability (latest fit)")
    latest_preds = predictions[predictions["date"] == predictions["date"].max()].sort_values(
        "prob_up", ascending=False
    )
    st.dataframe(
        latest_preds[["symbol", "prob_up"]].rename(columns={"symbol": "Symbol", "prob_up": "P(up)"}).head(15),
        width="stretch",
        hide_index=True,
    )

with st.expander("What this dashboard is and is not"):
    st.markdown(
        """
        - Prices come from the bundled **NEPSE-shaped fixture** unless live ingestion was configured.
        - Anomaly labels are injected events, so detector precision measures recovery, not alpha.
        - Direction probabilities come from a walk-forward model; on synthetic data the edge is small
          by construction. Treat them as a demonstration of the pipeline, not investment advice.
        - The sentiment index is a finance lexicon baseline; `MODEL_CARD.md` documents the FinBERT path.
        """
    )
