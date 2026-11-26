#!/usr/bin/env python
"""Download Financial PhraseBank into data/external/ (gitignored).

Licence: CC BY-NC-SA 3.0, Malo et al. (2014). Not redistributed here.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nepse_lens.nlp import phrasebank


def main() -> int:
    path = phrasebank.download()
    frame = phrasebank.load(path)
    print(f"PhraseBank: {len(frame):,} sentences -> {path}")
    print(frame["label"].value_counts().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
