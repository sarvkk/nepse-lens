from __future__ import annotations

import pandas as pd

from nepse_lens.features import (
    FEATURE_COLUMNS,
    build_features,
    sector_daily_returns,
    training_frame,
)


def test_feature_columns_exist(fixture_data):
    prices, _, _ = fixture_data
    features = build_features(prices)
    missing = set(FEATURE_COLUMNS) - set(features.columns)
    assert not missing


def test_target_matches_next_close(fixture_data):
    prices, _, _ = fixture_data
    features = build_features(prices)
    row = features[features["symbol"] == "NABIL"].sort_values("date").reset_index(drop=True)
    for i in (100, 200, 300):
        expected = float(row.loc[i + 1, "close"] > row.loc[i, "close"])
        assert row.loc[i, "target_up"] == expected


def test_last_session_per_symbol_has_no_target(fixture_data):
    prices, _, _ = fixture_data
    features = build_features(prices)
    last = features.sort_values("date").groupby("symbol").tail(1)
    assert last["target_up"].isna().all()


def test_no_future_leakage(fixture_data):
    prices, _, _ = fixture_data
    baseline = build_features(prices)

    tampered = prices.copy()
    future_dates = sorted(tampered["date"].unique())[-5:]
    mask = tampered["date"].isin(future_dates)
    for column in ("open", "high", "low", "close"):
        tampered.loc[mask, column] = tampered.loc[mask, column] * 3.0

    changed = build_features(tampered)
    cutoff = sorted(tampered["date"].unique())[-6]

    left = baseline[baseline["date"] <= cutoff].set_index(["symbol", "date"])[FEATURE_COLUMNS]
    right = changed[changed["date"] <= cutoff].set_index(["symbol", "date"])[FEATURE_COLUMNS]
    pd.testing.assert_frame_equal(left, right, check_exact=False, rtol=1e-9)


def test_training_frame_drops_incomplete_rows(fixture_data):
    prices, _, _ = fixture_data
    frame = training_frame(build_features(prices))
    assert not frame[FEATURE_COLUMNS].isna().any().any()
    assert frame["target_up"].isin([0, 1]).all()


def test_sector_returns_cover_every_sector(fixture_data):
    prices, _, _ = fixture_data
    returns = sector_daily_returns(prices)
    assert set(returns["sector"]) == set(prices["sector"])
    assert returns["ret_1d"].notna().all()
