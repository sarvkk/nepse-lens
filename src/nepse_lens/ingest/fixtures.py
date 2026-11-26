"""Deterministic NEPSE-like fixture data.

Why synthetic data is checked in:

* NEPSE's public JSON API requires a short-lived ``Salter`` bearer token that is
  minted by the site's Angular app. CI cannot hold one, and a demo that needs
  credentials is not reproducible.
* The fixture generator produces the *shape* of real NEPSE data — 45 tickers,
  ten sectors, 760 trading days, fat-tailed returns, volume that co-moves with
  absolute return, and labelled anomaly events — so the whole pipeline,
  dashboard, API, and test suite run offline.

The generator deliberately includes a small AR(1) momentum component. That
makes short-horizon direction mildly predictable so the walk-forward harness can
be validated end to end. It is a sanity check, **not** evidence of a market
edge; see MODEL_CARD.md.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..config import FIXTURE_DIR

SEED = 7
N_DAYS = 760
END_DATE = "2026-11-25"

# symbol -> (sector, base price in NPR)
UNIVERSE: dict[str, tuple[str, float]] = {
    "NABIL": ("Commercial Banks", 720),
    "SCB": ("Commercial Banks", 610),
    "NICA": ("Commercial Banks", 540),
    "ADBL": ("Commercial Banks", 380),
    "EBL": ("Commercial Banks", 560),
    "HBL": ("Commercial Banks", 260),
    "SANIMA": ("Commercial Banks", 420),
    "PRVU": ("Commercial Banks", 300),
    "MBL": ("Commercial Banks", 240),
    "NMB": ("Commercial Banks", 220),
    "KBL": ("Commercial Banks", 210),
    "GBIME": ("Commercial Banks", 230),
    "SBI": ("Commercial Banks", 430),
    "CIT": ("Commercial Banks", 190),
    "NBB": ("Commercial Banks", 250),
    "BOKL": ("Commercial Banks", 180),
    "MEGA": ("Commercial Banks", 200),
    "MMFDB": ("Development Banks", 160),
    "JBBL": ("Development Banks", 210),
    "MNBBL": ("Development Banks", 320),
    "GRDBL": ("Development Banks", 190),
    "EDBL": ("Development Banks", 200),
    "BSL": ("Finance", 900),
    "STC": ("Finance", 700),
    "RIDI": ("Finance", 400),
    "GFCL": ("Finance", 500),
    "SHIVM": ("Hydro Power", 520),
    "UNL": ("Hydro Power", 280),
    "HDL": ("Hydro Power", 210),
    "NHPC": ("Hydro Power", 150),
    "API": ("Hydro Power", 240),
    "CHCL": ("Hydro Power", 380),
    "BPCL": ("Hydro Power", 190),
    "UPPER": ("Hydro Power", 260),
    "JOSHI": ("Hydro Power", 300),
    "NLIC": ("Life Insurance", 900),
    "LICN": ("Life Insurance", 700),
    "SICL": ("Life Insurance", 420),
    "ALICL": ("Life Insurance", 380),
    "NLICL": ("Life Insurance", 600),
    "NICL": ("Non-Life Insurance", 800),
    "NIL": ("Non-Life Insurance", 300),
    "PRIN": ("Non-Life Insurance", 500),
    "SAGIC": ("Non-Life Insurance", 400),
    "IGI": ("Non-Life Insurance", 450),
    "RLI": ("Non-Life Insurance", 350),
    "NFS": ("Microfinance", 900),
    "MFIL": ("Microfinance", 600),
    "CBBL": ("Microfinance", 500),
    "GMFIL": ("Microfinance", 400),
    "JSLBB": ("Microfinance", 700),
    "BNL": ("Manufacturing", 300),
    "GCIL": ("Manufacturing", 400),
    "OHL": ("Hotels", 450),
    "TRH": ("Hotels", 300),
    "NIFRA": ("Investment", 220),
    "HIDCL": ("Investment", 180),
    "NTC": ("Others", 900),
    "NRIC": ("Others", 700),
}

SOURCES = ["NEPSE Daily", "Kathmandu Post", "Sharesansar", "Bizmandu"]

POSITIVE = [
    "{sym} posts {p}% profit growth in Q{q}",
    "{sym} declares {p}% cash dividend",
    "{sym} reports record revenue for the quarter",
    "Analysts upgrade {sym} on a stronger balance sheet",
    "{sym} wins a new hydropower contract worth Rs {p} crore",
    "Foreign investment inflows lift {sym}",
    "{sym} beats profit estimates as loan growth accelerates",
    "Shares of {sym} surge {p}% on heavy volume",
    "{sym} announces {p}% bonus share proposal",
    "Brokerages raise target price for {sym}",
    "{sym} expands branch network after strong deposit growth",
    "NEPSE gains {p} points as banking stocks rally",
    "Market closes higher as buying interest returns to {sym}",
    "{sym} posts improvement in net interest margin",
]

NEGATIVE = [
    "{sym} profit falls {p}% on rising operating costs",
    "{sym} shares slump after weak quarterly earnings",
    "{sym} faces regulatory probe over loan classification",
    "Analysts downgrade {sym} amid a liquidity crunch",
    "{sym} writes off bad loans in Q{q}",
    "Selling pressure drags {sym} lower for a third session",
    "{sym} fails to meet profit expectations",
    "NEPSE drops {p} points on political uncertainty",
    "{sym} reports a rise in non-performing loans",
    "Credit rating of {sym} cut on capital shortfall",
    "{sym} posts a net loss for the quarter",
    "Trading halts in {sym} after a sharp decline",
    "NEPSE falls as investors exit {sym} and other heavyweights",
    "{sym} flags going-concern risk in annual report",
]

NEUTRAL = [
    "{sym} holds its annual general meeting",
    "NEPSE turnover reaches Rs {p} crore",
    "{sym} appoints a new director to its board",
    "Board meeting of {sym} scheduled for Friday",
    "{sym} publishes its quarterly financial report",
    "NEPSE lists {p} new securities this month",
    "{sym} updates its registered office address",
    "Shareholders of {sym} approve the auditor appointment",
    "{sym} announces the book closure date",
    "NEPSE publishes the daily trading summary",
    "{sym} files its annual report with the regulator",
    "Market observes normal trading volume on Sunday",
]

TEMPLATES = {"positive": POSITIVE, "negative": NEGATIVE, "neutral": NEUTRAL}


def _sector_returns(rng: np.random.Generator, sectors: list[str], n: int) -> dict[str, np.ndarray]:
    return {s: rng.normal(0.0, 0.006, n) for s in sectors}


def generate_prices(seed: int = SEED, n_days: int = N_DAYS, end: str = END_DATE) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=end, periods=n_days)
    n = len(dates)
    market = rng.normal(0.0002, 0.0095, n)
    sectors = sorted({s for s, _ in UNIVERSE.values()})
    sector_ret = _sector_returns(rng, sectors, n)

    rows: list[dict[str, object]] = []
    anomalies: list[dict[str, object]] = []

    for symbol, (sector, base) in UNIVERSE.items():
        beta = rng.uniform(0.7, 1.3)
        idio = rng.normal(0.0, 0.010, n)
        ret = np.zeros(n)
        for t in range(1, n):
            ret[t] = 0.08 * ret[t - 1] + beta * market[t] + sector_ret[sector][t] + idio[t]

        base_volume = float(rng.lognormal(mean=np.log(45_000), sigma=0.5))
        volume_mult = np.ones(n)

        # Two labelled anomaly events per symbol, after a warm-up window.
        # Small fixtures (used in tests) fall back to a shorter warm-up.
        candidates = np.arange(60, n - 2)
        if len(candidates) < 2:
            candidates = np.arange(max(1, n - 6), n - 2)
        if len(candidates) >= 2:
            for t in rng.choice(candidates, size=2, replace=False):
                shock = float(rng.choice([-1.0, 1.0]) * rng.uniform(0.045, 0.11))
                ret[t] += shock
                volume_mult[t] = float(rng.uniform(6.0, 15.0))
                anomalies.append({"symbol": symbol, "date": dates[t]})

        close = base * np.exp(np.cumsum(ret))
        close = np.maximum(close, 10.0)

        gap = rng.normal(0.0, 0.004, n)
        open_ = np.empty(n)
        open_[0] = base
        open_[1:] = close[:-1] * (1.0 + gap[1:])

        hi_spread = np.abs(rng.normal(0.0, 0.006, n))
        lo_spread = np.abs(rng.normal(0.0, 0.006, n))
        high = np.maximum(open_, close) * (1.0 + hi_spread)
        low = np.minimum(open_, close) * (1.0 - lo_spread)

        volume = base_volume * np.exp(rng.normal(0.0, 0.35, n)) * (1.0 + 3.5 * np.abs(ret)) * volume_mult
        volume = np.maximum(volume, 100.0).astype(np.int64)

        for t in range(n):
            rows.append(
                {
                    "symbol": symbol,
                    "date": dates[t],
                    "open": round(float(open_[t]), 2),
                    "high": round(float(high[t]), 2),
                    "low": round(float(low[t]), 2),
                    "close": round(float(close[t]), 2),
                    "volume": int(volume[t]),
                    "turnover": round(float(close[t] * volume[t]), 2),
                    "sector": sector,
                }
            )

    prices = pd.DataFrame(rows)
    labels = pd.DataFrame(anomalies).drop_duplicates()
    return prices, labels


def generate_headlines(dates: pd.DatetimeIndex, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 1)
    symbols = list(UNIVERSE)
    rows: list[dict[str, object]] = []
    idx = 0
    for date in dates:
        for _ in range(int(rng.integers(3, 7))):
            label = str(rng.choice(["positive", "negative", "neutral"], p=[0.36, 0.32, 0.32]))
            template = str(rng.choice(TEMPLATES[label]))
            symbol = str(rng.choice(symbols))
            headline = template.format(sym=symbol, p=int(rng.integers(3, 40)), q=int(rng.integers(1, 5)))
            rows.append(
                {
                    "id": f"h{idx:06d}",
                    "date": date,
                    "source": str(rng.choice(SOURCES)),
                    "headline": headline,
                    "symbol": symbol,
                    "label": label,
                }
            )
            idx += 1
    return pd.DataFrame(rows)


def write_fixtures(out_dir: Path = FIXTURE_DIR) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    prices, labels = generate_prices()
    headlines = generate_headlines(pd.DatetimeIndex(sorted(prices["date"].unique())))

    paths = {
        "prices": out_dir / "nepse_daily.csv",
        "headlines": out_dir / "headlines.csv",
        "anomaly_labels": out_dir / "anomaly_labels.csv",
    }
    prices.to_csv(paths["prices"], index=False)
    headlines.to_csv(paths["headlines"], index=False)
    labels.to_csv(paths["anomaly_labels"], index=False)
    return paths


def load_fixtures(fixture_dir: Path = FIXTURE_DIR) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prices = pd.read_csv(fixture_dir / "nepse_daily.csv", parse_dates=["date"])
    headlines = pd.read_csv(fixture_dir / "headlines.csv", parse_dates=["date"])
    labels = pd.read_csv(fixture_dir / "anomaly_labels.csv", parse_dates=["date"])
    return prices, headlines, labels
