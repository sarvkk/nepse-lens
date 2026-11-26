"""End-to-end orchestration: features → models → NLP → report.

Kept as a module (not a script) so tests can run a miniature pipeline and the
CLI stays a thin wrapper.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from . import store
from .config import DB_PATH, REPORTS_DIR
from .features import build_features, training_frame
from .models.anomaly import (
    attach_labels,
    evaluate_anomaly,
    isolation_forest_anomaly,
    lstm_autoencoder_anomaly,
    robust_zscore_anomaly,
)
from .models.direction import evaluate_direction, predict_latest
from .models.torch_models import HAS_TORCH
from .nlp import phrasebank
from .nlp.sentiment import (
    daily_sentiment_index,
    evaluate_sentiment,
    label_from_score,
    score_headlines,
)
from .report import render_report

SENTIMENT_MODEL = "lexicon"


def _data_summary(prices: pd.DataFrame, headlines: pd.DataFrame) -> dict[str, Any]:
    return {
        "symbols": int(prices["symbol"].nunique()),
        "sessions": int(prices["date"].nunique()),
        "start": str(prices["date"].min().date()),
        "end": str(prices["date"].max().date()),
        "headlines": int(len(headlines)),
    }


def run_pipeline(
    db_path: Path | str = DB_PATH,
    use_torch: bool = False,
    phrasebank_path: Path | str | None = None,
    write: bool = True,
) -> dict[str, Any]:
    con = store.connect(db_path)
    try:
        prices = store.read_prices(con)
        headlines = store.read_headlines(con)
        labels = store.read_table(con, "anomaly_labels")
        if prices.empty:
            raise RuntimeError("No prices in the database. Run scripts/fetch_data.py first.")

        features = build_features(prices)
        training = training_frame(features)

        direction = evaluate_direction(training)
        latest_predictions = predict_latest(features)

        detectors: dict[str, pd.DataFrame] = {
            "robust_zscore": robust_zscore_anomaly(features),
            "isolation_forest": isolation_forest_anomaly(features),
        }
        if use_torch and HAS_TORCH:
            detectors["lstm_autoencoder"] = lstm_autoencoder_anomaly(features)

        anomaly_metrics: dict[str, Any] = {}
        anomaly_rows: list[pd.DataFrame] = []
        for name, scored in detectors.items():
            evaluated = attach_labels(scored, labels) if not labels.empty else scored.assign(is_injected=False)
            anomaly_metrics[name] = evaluate_anomaly(evaluated)
            rows = scored[["symbol", "date", "score", "is_anomaly"]].copy()
            rows["method"] = name
            anomaly_rows.append(rows)

        scored_headlines = score_headlines(headlines, SENTIMENT_MODEL)
        sentiment_metrics: dict[str, Any] = {
            "synthetic templates": {"model": SENTIMENT_MODEL, **evaluate_sentiment(scored_headlines)}
        }

        if phrasebank_path:
            bench = phrasebank.load(phrasebank_path)
            scored_bench = score_headlines(bench, SENTIMENT_MODEL)
            sentiment_metrics["Financial PhraseBank (all-agree)"] = {
                "model": SENTIMENT_MODEL,
                **evaluate_sentiment(scored_bench),
            }

        if write:
            store.write_predictions(con, latest_predictions)
            store.write_anomalies(con, pd.concat(anomaly_rows, ignore_index=True))

            per_symbol = (
                scored_headlines.groupby(["date", "symbol"], as_index=False)["score"].mean()
            )
            per_symbol["model"] = SENTIMENT_MODEL
            per_symbol["label"] = [label_from_score(s) for s in per_symbol["score"]]

            market = daily_sentiment_index(scored_headlines)
            market["symbol"] = "MARKET"
            market["model"] = SENTIMENT_MODEL

            combined = pd.concat(
                [
                    per_symbol[["date", "symbol", "model", "score", "label"]],
                    market[["date", "symbol", "model", "score", "label"]],
                ],
                ignore_index=True,
            )
            store.write_sentiment(con, combined)

        metrics: dict[str, Any] = {
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "data": _data_summary(prices, headlines),
            "direction": direction,
            "anomaly": anomaly_metrics,
            "sentiment": sentiment_metrics,
            "notes": [
                "Prices are a deterministic NEPSE-shaped fixture unless live ingestion was used.",
                "Direction accuracy near the base rate is the honest result on synthetic data.",
                "Anomaly labels are injected events, so precision@k measures recovery, not alpha.",
                "The lexicon is a baseline; FinBERT is the production path (pip install -e '.[nlp]').",
            ],
        }

        if write:
            REPORTS_DIR.mkdir(parents=True, exist_ok=True)
            (REPORTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
            (REPORTS_DIR / "report.md").write_text(render_report(metrics))
        return metrics
    finally:
        con.close()


def evaluate_phrasebank_only(phrasebank_path: Path | str, model: str = "lexicon") -> dict[str, Any]:
    bench = phrasebank.load(phrasebank_path)
    scored = score_headlines(bench, model)
    return {"model": model, **evaluate_sentiment(scored)}
