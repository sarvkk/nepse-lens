from __future__ import annotations

import pytest

from nepse_lens.nlp import phrasebank


def test_load_parses_labels_and_skips_junk(tmp_path):
    path = tmp_path / "sentences.txt"
    path.write_text(
        "Profit rose sharply .@positive\n"
        "Losses widened .@negative\n"
        "The board met on Friday .@neutral\n"
        "Malformed line without a label\n"
        "Unknown label .@mixed\n",
        encoding="latin-1",
    )
    frame = phrasebank.load(path)
    assert len(frame) == 3
    assert set(frame["label"]) == {"positive", "negative", "neutral"}
    assert frame.iloc[0]["headline"] == "Profit rose sharply ."


def test_load_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        phrasebank.load(tmp_path / "nope.txt")


def test_default_paths_are_absolute():
    assert phrasebank.DEST_DIR.is_absolute()
    assert phrasebank.SENTENCES.is_absolute()
