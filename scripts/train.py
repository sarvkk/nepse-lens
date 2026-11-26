#!/usr/bin/env python
"""Train/evaluate every model and write reports.

    python scripts/train.py
    python scripts/train.py --torch --phrasebank data/external/FinancialPhraseBank-v1.0/Sentences_AllAgree.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nepse_lens.config import DB_PATH
from nepse_lens.pipeline import run_pipeline


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=str(DB_PATH))
    parser.add_argument("--torch", action="store_true", help="Include the LSTM autoencoder (needs PyTorch)")
    parser.add_argument("--phrasebank", default=None, help="Path to Sentences_AllAgree.txt")
    args = parser.parse_args()

    metrics = run_pipeline(args.db, use_torch=args.torch, phrasebank_path=args.phrasebank)

    print("=" * 60)
    print("NEPSE Lens — results")
    print("=" * 60)
    print(f"Data: {metrics['data']['symbols']} symbols, {metrics['data']['sessions']} sessions")
    print("\nDirection (walk-forward):")
    for name, result in metrics["direction"]["models"].items():
        print(f"  {name:14s} acc={result['accuracy_mean']:.3f}  auc={result['roc_auc_mean']:.3f}")
    print("\nAnomaly detection:")
    for name, result in metrics["anomaly"].items():
        print(f"  {name:18s} AP={result['average_precision']:.3f}  P@k={result['precision_at_k']:.3f}")
    print("\nSentiment:")
    for name, result in metrics["sentiment"].items():
        print(f"  {name:34s} acc={result['accuracy']:.3f}  macroF1={result['macro_f1']:.3f}")
    print("\nReports: reports/metrics.json, reports/report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
