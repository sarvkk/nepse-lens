#!/usr/bin/env python
"""Score Financial PhraseBank with the sentiment model.

    python scripts/evaluate.py
    python scripts/evaluate.py --model finbert
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nepse_lens.nlp import phrasebank
from nepse_lens.pipeline import evaluate_phrasebank_only


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["lexicon", "finbert"], default="lexicon")
    parser.add_argument("--path", default=None)
    args = parser.parse_args()

    path = Path(args.path) if args.path else phrasebank.SENTENCES
    if not path.exists():
        path = phrasebank.download()

    result = evaluate_phrasebank_only(path, model=args.model)
    print(json.dumps({k: v for k, v in result.items() if k not in {"per_class", "confusion"}}, indent=2))
    print("\nConfusion matrix (rows=true, cols=pred), labels:", result["labels"])
    for row in result["confusion"]:
        print(" ", row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
