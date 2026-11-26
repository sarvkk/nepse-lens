from __future__ import annotations

from nepse_lens.report import render_report


def test_report_contains_all_sections():
    metrics = {
        "generated_at": "2026-11-26T00:00:00+00:00",
        "data": {"symbols": 59, "sessions": 760, "start": "2023-12-28", "end": "2026-11-25", "headlines": 3422},
        "direction": {
            "models": {
                "naive_prior": {"accuracy_mean": 0.5, "accuracy_std": 0.01, "roc_auc_mean": 0.5, "brier_mean": 0.25, "n_predictions": 100},
            }
        },
        "anomaly": {
            "robust_zscore": {
                "average_precision": 0.5,
                "roc_auc": 0.9,
                "precision_at_k": 0.5,
                "recall_at_k": 0.5,
                "lift_at_k": 100.0,
            }
        },
        "sentiment": {"synthetic templates": {"model": "lexicon", "accuracy": 0.95, "macro_f1": 0.94, "n": 100}},
        "notes": ["fixture data"],
    }
    report = render_report(metrics)
    assert "# NEPSE Lens — run report" in report
    assert "Next-session direction" in report
    assert "Anomaly detection" in report
    assert "Headline sentiment" in report
    assert "naive_prior" in report
    assert "fixture data" in report


def test_report_handles_missing_values():
    report = render_report({})
    assert "n/a" in report
