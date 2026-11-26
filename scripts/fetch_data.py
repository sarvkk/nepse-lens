#!/usr/bin/env python
"""Load price, headline, and anomaly-label data into DuckDB.

    python scripts/fetch_data.py --source fixture
    NEPSE_TOKEN=... python scripts/fetch_data.py --source live
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nepse_lens import store
from nepse_lens.config import DB_PATH, FIXTURE_DIR
from nepse_lens.ingest import NepseClient, fixtures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["fixture", "live"], default="fixture")
    parser.add_argument("--db", default=str(DB_PATH))
    parser.add_argument("--headlines", default=None, help="Optional CSV of headlines for live mode")
    parser.add_argument("--regenerate", action="store_true", help="Rebuild the fixture CSVs from the seed")
    args = parser.parse_args()

    store.init_db(args.db)
    con = store.connect(args.db)

    try:
        if args.source == "fixture":
            if args.regenerate or not (FIXTURE_DIR / "nepse_daily.csv").exists():
                print("Generating fixture snapshot...")
                fixtures.write_fixtures()
            prices, headlines, labels = fixtures.load_fixtures()
            store.write_prices(con, prices)
            store.write_headlines(con, headlines)
            store.write_anomaly_labels(con, labels)
            print(f"Loaded fixture: {len(prices):,} bars, {len(headlines):,} headlines, {len(labels):,} labels")
        else:
            client = NepseClient()
            snapshot = client.fetch_daily_snapshot()
            store.write_prices(con, snapshot)
            print(f"Loaded live snapshot: {len(snapshot):,} symbols")
            if args.headlines:
                import pandas as pd

                headlines = pd.read_csv(args.headlines, parse_dates=["date"])
                store.write_headlines(con, headlines)
                print(f"Loaded {len(headlines):,} headlines from {args.headlines}")

        print("Table counts:", store.table_counts(con))
    finally:
        con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
