from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from nepse_lens.features import build_features  # noqa: E402
from nepse_lens.models.anomaly import lstm_autoencoder_anomaly  # noqa: E402
from nepse_lens.models.torch_models import (  # noqa: E402
    HAS_TORCH,
    make_windows,
    predict_tcn,
    reconstruction_scores,
    train_lstm_autoencoder,
    train_tcn,
)


def test_torch_is_available():
    assert HAS_TORCH


def test_make_windows_shape():
    matrix = np.arange(30, dtype=np.float32).reshape(10, 3)
    windows = make_windows(matrix, seq_len=4)
    assert windows.shape == (7, 4, 3)
    assert np.array_equal(windows[0], matrix[0:4])


def test_lstm_autoencoder_trains_and_scores():
    rng = np.random.default_rng(0)
    windows = rng.normal(size=(64, 6, 4)).astype(np.float32)
    model = train_lstm_autoencoder(windows, epochs=2, batch_size=32)
    scores = reconstruction_scores(model, windows)
    assert scores.shape == (64,)
    assert np.isfinite(scores).all()
    assert (scores >= 0).all()


def test_tcn_outputs_probabilities():
    rng = np.random.default_rng(1)
    windows = rng.normal(size=(64, 6, 4)).astype(np.float32)
    labels = rng.integers(0, 2, size=64).astype(np.float32)
    model = train_tcn(windows, labels, epochs=2, batch_size=32)
    probabilities = predict_tcn(model, windows)
    assert probabilities.shape == (64,)
    assert ((probabilities >= 0) & (probabilities <= 1)).all()


def test_lstm_anomaly_runs_end_to_end(fixture_data):
    prices, _, _ = fixture_data
    subset = prices[prices["symbol"].isin(["NABIL", "SCB"])]
    features = build_features(subset)
    scored = lstm_autoencoder_anomaly(features, epochs=1, seq_len=10)
    assert len(scored) == len(features)
    assert scored["score"].notna().all()
    assert scored["in_sample"].any()
