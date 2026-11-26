from __future__ import annotations

from nepse_lens import store


def test_pipeline_writes_metrics_and_tables(trained):
    db_path, metrics, reports = trained

    assert metrics["data"]["symbols"] > 0
    assert metrics["data"]["sessions"] > 0
    assert set(metrics["direction"]["models"]) == {"naive_prior", "logistic", "hist_gbm"}
    assert "robust_zscore" in metrics["anomaly"]
    assert "isolation_forest" in metrics["anomaly"]
    assert "synthetic templates" in metrics["sentiment"]

    assert (reports / "metrics.json").exists()
    report = (reports / "report.md").read_text()
    assert "Next-session direction" in report
    assert "Anomaly detection" in report

    con = store.connect(db_path)
    try:
        counts = store.table_counts(con)
        assert counts["predictions"] > 0
        assert counts["anomalies"] > 0
        assert counts["sentiment"] > 0
    finally:
        con.close()
