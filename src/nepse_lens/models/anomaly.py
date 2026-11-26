"""Anomaly detection over daily bars.

Three detectors, deliberately different in inductive bias:

* robust z-score — interpretable, per-feature, cheap.
* Isolation Forest — multivariate, non-parametric.
* LSTM autoencoder — sequence-aware; optional, needs PyTorch.

Scores are oriented so that **higher means more anomalous** for every method.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score

from .torch_models import HAS_TORCH, make_windows, reconstruction_scores, train_lstm_autoencoder

ANOMALY_FEATURES = ["ret_1d", "volume_z_20", "turnover_z_20", "range_pct", "gap_pct"]


def robust_zscore_anomaly(
    features: pd.DataFrame,
    cols: list[str] | None = None,
    window: int = 60,
    threshold: float = 4.0,
) -> pd.DataFrame:
    cols = cols or ANOMALY_FEATURES
    parts: list[pd.DataFrame] = []
    for _, group in features.groupby("symbol", sort=False):
        frame = group.sort_values("date").copy()
        zs = []
        for col in cols:
            median = frame[col].rolling(window, min_periods=20).median()
            mad = (frame[col] - median).abs().rolling(window, min_periods=20).median()
            zs.append((0.6745 * (frame[col] - median) / mad.replace(0, np.nan)).abs())
        frame["score"] = pd.concat(zs, axis=1).max(axis=1).fillna(0.0)
        frame["is_anomaly"] = frame["score"] >= threshold
        parts.append(frame)
    return pd.concat(parts, ignore_index=True)


def isolation_forest_anomaly(
    features: pd.DataFrame,
    cols: list[str] | None = None,
    contamination: float = 0.01,
    random_state: int = 42,
) -> pd.DataFrame:
    cols = cols or ANOMALY_FEATURES
    frame = features.copy()
    matrix = frame[cols].fillna(0.0)
    model = IsolationForest(
        n_estimators=300,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(matrix)
    frame["score"] = -model.score_samples(matrix)
    cutoff = np.quantile(frame["score"], 1.0 - contamination)
    frame["is_anomaly"] = frame["score"] >= cutoff
    return frame


def lstm_autoencoder_anomaly(
    features: pd.DataFrame,
    cols: list[str] | None = None,
    seq_len: int = 20,
    epochs: int = 15,
    seed: int = 42,
) -> pd.DataFrame:
    """Score the last step of each rolling window by reconstruction error.

    Windows never cross symbol boundaries, so the model cannot learn one
    ticker's price level as another's context.
    """
    if not HAS_TORCH:
        raise ImportError("LSTM autoencoder needs PyTorch: pip install -e '.[torch]'")

    cols = cols or ANOMALY_FEATURES
    scored_parts: list[pd.DataFrame] = []
    for _, group in features.groupby("symbol", sort=False):
        frame = group.sort_values("date").copy()
        matrix = frame[cols].fillna(0.0).to_numpy(dtype=np.float32)
        if len(matrix) <= seq_len + 5:
            frame["score"] = 0.0
            frame["is_anomaly"] = False
            scored_parts.append(frame)
            continue

        windows = make_windows(matrix, seq_len)
        model = train_lstm_autoencoder(windows, epochs=epochs, seed=seed)
        errors = reconstruction_scores(model, windows)
        frame["score"] = 0.0
        frame.loc[frame.index[seq_len - 1 :], "score"] = errors
        cutoff = np.quantile(frame["score"], 0.99)
        frame["is_anomaly"] = frame["score"] >= cutoff
        scored_parts.append(frame)
    return pd.concat(scored_parts, ignore_index=True)


def attach_labels(scored: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Left-join injected/known anomaly events as ``is_injected``."""
    frame = scored.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    label_keys = labels[["symbol", "date"]].copy()
    label_keys["date"] = pd.to_datetime(label_keys["date"])
    label_keys["is_injected"] = True
    frame = frame.merge(label_keys, on=["symbol", "date"], how="left")
    frame["is_injected"] = frame["is_injected"].fillna(False).astype(bool)
    return frame


def evaluate_anomaly(
    frame: pd.DataFrame,
    label_col: str = "is_injected",
    score_col: str = "score",
    k: int | None = None,
) -> dict[str, float | int]:
    y = frame[label_col].to_numpy(dtype=bool)
    scores = frame[score_col].fillna(0.0).to_numpy(dtype=float)
    positives = int(y.sum())
    if positives == 0:
        return {"positives": 0, "average_precision": float("nan"), "roc_auc": float("nan")}

    k = k or positives
    top = np.argsort(-scores)[:k]
    precision_at_k = float(y[top].mean())
    prevalence = float(y.mean())
    return {
        "positives": positives,
        "average_precision": float(average_precision_score(y, scores)),
        "roc_auc": float(roc_auc_score(y, scores)),
        "precision_at_k": precision_at_k,
        "recall_at_k": float(y[top].sum() / positives),
        "lift_at_k": float(precision_at_k / prevalence) if prevalence else float("nan"),
        "k": int(k),
    }
