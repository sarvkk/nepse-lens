from __future__ import annotations

import numpy as np
import pandas as pd

from nepse_lens.features import build_features, training_frame
from nepse_lens.models.anomaly import (
    attach_labels,
    chronological_cutoff,
    evaluate_anomaly,
    isolation_forest_anomaly,
    robust_zscore_anomaly,
)
from nepse_lens.models.direction import evaluate_direction, walk_forward_splits


def test_walk_forward_splits_are_time_ordered(fixture_data):
    prices, _, _ = fixture_data
    frame = training_frame(build_features(prices))
    splits = list(walk_forward_splits(frame["date"], n_splits=4, min_train=100))
    assert len(splits) == 4
    previous_test_end = None
    for train, test in splits:
        assert train.max() < test.min()
        if previous_test_end is not None:
            assert test.min() > previous_test_end
        previous_test_end = test.max()


def test_robust_zscore_flags_extreme_volume(fixture_data):
    prices, _, labels = fixture_data
    features = build_features(prices)
    scored = attach_labels(robust_zscore_anomaly(features), labels)
    flagged = scored[scored["is_anomaly"]]
    assert len(flagged) > 0
    assert flagged["score"].max() > 4.0
    assert evaluate_anomaly(scored)["positives"] > 0


def test_isolation_forest_scores_are_oriented(fixture_data):
    prices, _, labels = fixture_data
    features = build_features(prices)
    scored = attach_labels(isolation_forest_anomaly(features), labels)
    assert scored["is_anomaly"].sum() > 0
    result = evaluate_anomaly(scored)
    assert 0.0 <= result["average_precision"] <= 1.0
    assert result["precision_at_k"] >= 0.0


def test_evaluate_anomaly_perfect_separation():
    frame = pd.DataFrame({"is_injected": [True, True, False, False], "score": [9.0, 8.0, 1.0, 0.0]})
    result = evaluate_anomaly(frame)
    assert result["precision_at_k"] == 1.0
    assert result["recall_at_k"] == 1.0
    assert result["roc_auc"] == 1.0


def test_evaluate_anomaly_single_class_is_nan():
    all_positive = pd.DataFrame({"is_injected": [True, True], "score": [0.9, 0.1]})
    result = evaluate_anomaly(all_positive)
    assert result["positives"] == 2
    assert np.isnan(result["roc_auc"])
    assert np.isnan(result["precision_at_k"])


def test_chronological_cutoff_splits_sessions():
    dates = pd.Series(pd.bdate_range("2026-01-01", periods=100))
    cutoff = chronological_cutoff(dates, 0.5)
    assert dates.min() < cutoff < dates.max()
    assert (dates <= cutoff).sum() == 50


def test_isolation_forest_fits_only_on_first_half(fixture_data):
    prices, _, _ = fixture_data
    features = build_features(prices)
    scored = isolation_forest_anomaly(features, fit_fraction=0.5)
    cutoff = chronological_cutoff(features["date"], 0.5)
    assert scored.loc[scored["date"] <= cutoff, "in_sample"].all()
    assert not scored.loc[scored["date"] > cutoff, "in_sample"].any()


def test_direction_report_is_well_formed(fixture_data):
    prices, _, _ = fixture_data
    report = evaluate_direction(training_frame(build_features(prices)), n_splits=3, min_train=100)
    assert set(report["models"]) == {"naive_prior", "logistic", "hist_gbm"}
    for model in report["models"].values():
        assert 0.0 <= model["accuracy_mean"] <= 1.0
        assert model["n_predictions"] > 0
    assert np.isfinite(report["models"]["hist_gbm"]["roc_auc_mean"])
