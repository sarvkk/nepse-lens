"""Next-session direction models with walk-forward evaluation.

The split is expanding-window and strictly time-ordered: fold *k* trains on all
sessions before its test block and predicts the block after. No shuffling, no
random K-fold, no peeking at the future.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..features import FEATURE_COLUMNS


def model_factories() -> dict[str, object]:
    return {
        "naive_prior": lambda: DummyClassifier(strategy="prior"),
        "logistic": lambda: Pipeline(
            [
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
            ]
        ),
        "hist_gbm": lambda: HistGradientBoostingClassifier(
            max_iter=250,
            learning_rate=0.05,
            max_depth=3,
            l2_regularization=1.0,
            random_state=42,
        ),
    }


def walk_forward_splits(
    dates: pd.Series | np.ndarray,
    n_splits: int = 5,
    min_train: int = 150,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    unique = np.sort(pd.unique(np.asarray(dates)))
    if len(unique) <= min_train + n_splits:
        raise ValueError(f"Need more than {min_train + n_splits} sessions, got {len(unique)}")

    fold_size = (len(unique) - min_train) // n_splits
    for fold in range(n_splits):
        train_end = min_train + fold * fold_size
        test_end = len(unique) if fold == n_splits - 1 else train_end + fold_size
        yield unique[:train_end], unique[train_end:test_end]


def _safe_auc(y: np.ndarray, proba: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, proba))


def evaluate_direction(
    frame: pd.DataFrame,
    feature_cols: list[str] | None = None,
    n_splits: int = 5,
    min_train: int = 150,
) -> dict[str, object]:
    feature_cols = feature_cols or FEATURE_COLUMNS
    data = frame.dropna(subset=[*feature_cols, "target_up"]).copy()
    data["target_up"] = data["target_up"].astype(int)

    splits = list(walk_forward_splits(data["date"], n_splits=n_splits, min_train=min_train))
    report: dict[str, object] = {
        "n_folds": len(splits),
        "n_rows": int(len(data)),
        "n_sessions": int(data["date"].nunique()),
        "features": feature_cols,
        "models": {},
    }

    for name, factory in model_factories().items():
        folds: list[dict[str, float | int]] = []
        for fold, (train_dates, test_dates) in enumerate(splits):
            train = data[data["date"].isin(train_dates)]
            test = data[data["date"].isin(test_dates)]
            if train["target_up"].nunique() < 2:
                continue

            model = factory()
            model.fit(train[feature_cols], train["target_up"])
            proba = model.predict_proba(test[feature_cols])[:, 1]
            pred = (proba >= 0.5).astype(int)
            y = test["target_up"].to_numpy()

            folds.append(
                {
                    "fold": fold,
                    "train_end": str(pd.Timestamp(train_dates.max()).date()),
                    "test_start": str(pd.Timestamp(test_dates.min()).date()),
                    "test_end": str(pd.Timestamp(test_dates.max()).date()),
                    "n_test": int(len(test)),
                    "accuracy": float(accuracy_score(y, pred)),
                    "roc_auc": _safe_auc(y, proba),
                    "brier": float(brier_score_loss(y, proba)),
                    "log_loss": float(log_loss(y, np.clip(proba, 1e-6, 1 - 1e-6), labels=[0, 1])),
                    "base_rate": float(y.mean()),
                }
            )

        if not folds:
            continue
        report["models"][name] = {
            "accuracy_mean": float(np.nanmean([f["accuracy"] for f in folds])),
            "accuracy_std": float(np.nanstd([f["accuracy"] for f in folds])),
            "roc_auc_mean": float(np.nanmean([f["roc_auc"] for f in folds])),
            "brier_mean": float(np.nanmean([f["brier"] for f in folds])),
            "log_loss_mean": float(np.nanmean([f["log_loss"] for f in folds])),
            "n_predictions": int(sum(f["n_test"] for f in folds)),
            "folds": folds,
        }
    return report


def predict_latest(
    features: pd.DataFrame,
    feature_cols: list[str] | None = None,
    model_name: str = "hist_gbm",
) -> pd.DataFrame:
    """Fit on everything before the last session and score the last session."""
    feature_cols = feature_cols or FEATURE_COLUMNS
    usable = features.dropna(subset=feature_cols).copy()
    last_date = usable["date"].max()
    train = usable[usable["date"] < last_date].dropna(subset=["target_up"])
    latest = usable[usable["date"] == last_date]
    if train.empty or latest.empty:
        return pd.DataFrame(columns=["symbol", "date", "model", "prob_up"])

    model = model_factories()[model_name]()
    model.fit(train[feature_cols], train["target_up"].astype(int))
    proba = model.predict_proba(latest[feature_cols])[:, 1]
    return pd.DataFrame(
        {
            "symbol": latest["symbol"].to_numpy(),
            "date": latest["date"].to_numpy(),
            "model": model_name,
            "prob_up": proba,
        }
    )
