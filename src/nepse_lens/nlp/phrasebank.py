"""Financial PhraseBank loader.

Malo et al. (2014), CC BY-NC-SA 3.0. The dataset is *not* vendored into this
repo; ``scripts/fetch_phrasebank.py`` downloads it to ``data/external/`` which is
gitignored. That keeps the licence boundary clean while still giving the
sentiment model a real, human-labelled benchmark.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd
import requests

URL = "https://huggingface.co/datasets/takala/financial_phrasebank/resolve/main/data/FinancialPhraseBank-v1.0.zip"
DEST_DIR = Path("data/external")
ARCHIVE = DEST_DIR / "FinancialPhraseBank-v1.0.zip"
SENTENCES = DEST_DIR / "FinancialPhraseBank-v1.0" / "Sentences_AllAgree.txt"


def download(dest_dir: Path = DEST_DIR, force: bool = False) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    archive = dest_dir / ARCHIVE.name
    if force or not archive.exists():
        response = requests.get(URL, timeout=60)
        response.raise_for_status()
        archive.write_bytes(response.content)
    target = dest_dir / "FinancialPhraseBank-v1.0"
    if not target.exists():
        with zipfile.ZipFile(archive) as bundle:
            bundle.extractall(dest_dir)
    return target / "Sentences_AllAgree.txt"


def load(path: Path | str = SENTENCES) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run scripts/fetch_phrasebank.py first.")
    rows: list[dict[str, str]] = []
    for line in path.read_text(encoding="latin-1").splitlines():
        if "@" not in line:
            continue
        sentence, label = line.rsplit("@", 1)
        label = label.strip().lower()
        if label not in {"positive", "negative", "neutral"}:
            continue
        rows.append({"headline": sentence.strip(), "label": label})
    return pd.DataFrame(rows)
