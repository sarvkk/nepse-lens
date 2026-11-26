"""Anomaly detection over daily bars.

Three detectors, deliberately different in inductive bias:

* robust z-score — interpretable, per-feature, cheap, trailing-window only.
* Isolation Forest — multivariate, non-parametric, fit on the first half.
* LSTM autoencoder — sequence-aware; optional, needs PyTorch, fit on the first half.

Scores are oriented so that **higher means more anomalous** for every method.

Fitted detectors never see the evaluation half. ``in_sample`` marks the rows a
detector was fit on, and the pipeline scores only ``~in_sample`` so the reported
precision@k is out-of-sample.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score

from .torch_models import HAS_TORCH, make_windows, reconstruction_scores, train_lstm_autoencoder

ANOMALY_FEATURES = ["ret_1d", "volume_z_20", "turnover_z_20", "range_pct", "gap_pct"]


def chronological_cutoff(dates: pd.Series, fit_fraction: float = 0.5) -> pd.Timestamp:
    """Return the last date of the fit window.

    Uses unique sessions, not rows, so a day with more listed symbols does not
    move the boundary.
    """
    if not 0.0 < fit_fraction < 1.0:
        raise ValueError("fit_fraction must be between 0 and 1")
    unique = np.sort(pd.unique(pd.to_datetime(dates)))
    if len(unique) < 4:
        raise ValueError("Need at least four sessions to split fit/evaluation")
    index = min(len(unique) - 2, max(1, int(len(unique) * fit_fraction) - 1))
    return pd.Timestamp(unique[index])


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
        frame["in_sample"] = False  # trailing window: every score is out-of-sample
        parts.append(frame)
    return pd.concat(parts, ignore_index=True)


def isolation_forest_anomaly(
    features: pd.DataFrame,
    cols: list[str] | None = None,
    contamination: float = 0.01,
    random_state: int = 42,
    fit_fraction: float = 0.5,
) -> pd.DataFrame:
    cols = cols or ANOMALY_FEATURES
    frame = features.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    matrix = frame[cols].fillna(0.0)
    cutoff_date = chronological_cutoff(frame["date"], fit_fraction)
    fit_mask = (frame["date"] <= cutoff_date).to_numpy()

    model = IsolationForest(
        n_estimators=300,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(matrix[fit_mask])
    frame["score"] = -model.score_samples(matrix)
    frame["in_sample"] = fit_mask

    # Threshold on out-of-sample scores so the flag rate is not driven by the
    # rows the forest memorised.
    reference = frame.loc[~frame["in_sample"], "score"]
    if reference.empty:
        reference = frame["score"]
    cutoff = float(np.quantile(reference, 1.0 - contamination))
    frame["is_anomaly"] = frame["score"] >= cutoff
    return frame


def lstm_autoencoder_anomaly(
    features: pd.DataFrame,
    cols: list[str] | None = None,
    seq_len: int = 20,
    epochs: int = 15,
    seed: int = 42,
    fit_fraction: float = 0.5,
) -> pd.DataFrame:
    """Score the last step of each rolling window by reconstruction error.

    Windows never cross symbol boundaries, and training uses only windows that
    end on or before the fit cutoff.
    """
    if not HAS_TORCH:
        raise ImportError("LSTM autoencoder needs PyTorch: pip install -e '.[torch]'")

    cols = cols or ANOMALY_FEATURES
    cutoff_date = chronological_cutoff(features["date"], fit_fraction)
    scored_parts: list[pd.DataFrame] = []

    for _, group in features.groupby("symbol", sort=False):
        frame = group.sort_values("date").copy()
        frame["date"] = pd.to_datetime(frame["date"])
        matrix = frame[cols].fillna(0.0).to_numpy(dtype=np.float32)

        if len(matrix) <= seq_len + 5:
            frame["score"] = 0.0
            frame["is_anomaly"] = False
            frame["in_sample"] = frame["date"] <= cutoff_date
            scored_parts.append(frame)
            continue

        windows = make_windows(matrix, seq_len)
        window_end_dates = frame["date"].iloc[seq_len - 1 :].to_numpy()
        train_mask = window_end_dates <= np.datetime64(cutoff_date)
        if train_mask.sum() < 20:
            train_mask = np.zeros(len(windows), dtype=bool)
            train_mask[: max(20, len(windows) // 2)] = True

        model = train_lstm_autoencoder(windows[train_mask], epochs=epochs, seed=seed)
        errors = reconstruction_scores(model, windows)

        frame["score"] = 0.0
        frame.loc[frame.index[seq_len - 1 :], "score"] = errors
        frame["in_sample"] = frame["date"] <= cutoff_date

        reference = frame.loc[~frame["in_sample"], "score"]
        if reference.empty:
            reference = frame["score"]
        cutoff = float(np.quantile(reference, 0.99))
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

    if positives == 0 or positives == len(y):
        return {
            "positives": positives,
            "n": int(len(y)),
            "average_precision": float("nan"),
            "roc_auc": float("nan"),
            "precision_at_k": float("nan"),
            "recall_at_k": float("nan"),
            "lift_at_k": float("nan"),
            "k": int(k or positives),
        }

    k = k or positives
    top = np.argsort(-scores)[:k]
    precision_at_k = float(y[top].mean())
    prevalence = float(y.mean())
    return {
        "positives": positives,
        "n": int(len(y)),
        "average_precision": float(average_precision_score(y, scores)),
        "roc_auc": float(roc_auc_score(y, scores)),
        "precision_at_k": precision_at_k,
        "recall_at_k": float(y[top].sum() / positives),
        "lift_at_k": float(precision_at_k / prevalence) if prevalence else float("nan"),
        "k": int(k),
    }
