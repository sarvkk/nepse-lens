#!/usr/bin/env python
"""Materialise the feature table for inspection.

    python scripts/build_features.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nepse_lens import store
from nepse_lens.config import DB_PATH
from nepse_lens.features import FEATURE_COLUMNS, build_features, training_frame


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=str(DB_PATH))
    parser.add_argument("--out", default="data/processed/features.parquet")
    args = parser.parse_args()

    con = store.connect(args.db)
    try:
        prices = store.read_prices(con)
    finally:
        con.close()

    features = build_features(prices)
    trainable = training_frame(features)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(out, index=False)

    print(f"Features: {len(features):,} rows x {len(FEATURE_COLUMNS)} model features")
    print(f"Trainable (target known): {len(trainable):,} rows")
    print(f"Up rate: {trainable['target_up'].mean():.3f}")
    print(f"Written: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
