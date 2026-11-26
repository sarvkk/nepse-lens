from __future__ import annotations

import pandas as pd

from nepse_lens.nlp.sentiment import (
    daily_sentiment_index,
    evaluate_sentiment,
    label_from_score,
    score_headlines,
    score_text,
)


def test_positive_and_negative_phrases():
    assert score_text("NABIL posts strong profit growth") > 0
    assert score_text("NABIL profit falls on bad loans") < 0
    assert score_text("NEPSE gains as banking stocks rally") > 0


def test_negation_flips_polarity():
    assert score_text("not a bad quarter for NABIL") > 0
    assert score_text("NABIL fails to meet profit expectations") < 0


def test_neutral_headline_scores_zero():
    assert score_text("NABIL holds its annual general meeting") == 0.0
    assert label_from_score(0.0) == "neutral"


def test_label_thresholds():
    assert label_from_score(1.0) == "positive"
    assert label_from_score(-1.0) == "negative"
    assert label_from_score(0.1) == "neutral"


def test_score_headlines_adds_predictions():
    frame = pd.DataFrame(
        {
            "headline": [
                "NABIL declares a cash dividend",
                "NABIL shares slump after weak earnings",
                "NABIL announces the book closure date",
            ],
            "label": ["positive", "negative", "neutral"],
        }
    )
    scored = score_headlines(frame)
    assert list(scored["pred_label"]) == ["positive", "negative", "neutral"]
    assert scored["model"].unique().tolist() == ["lexicon"]


def test_evaluate_sentiment_perfect_predictions():
    frame = pd.DataFrame(
        {"label": ["positive", "negative", "neutral"], "pred_label": ["positive", "negative", "neutral"]}
    )
    result = evaluate_sentiment(frame)
    assert result["accuracy"] == 1.0
    assert result["macro_f1"] == 1.0


def test_daily_index_aggregates_by_date():
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-01", "2026-01-01", "2026-01-02"]),
            "headline": ["profit growth", "record dividend", "bad loans"],
            "label": ["positive", "positive", "negative"],
        }
    )
    scored = score_headlines(frame)
    index = daily_sentiment_index(scored)
    assert len(index) == 2
    assert index.iloc[0]["n_headlines"] == 2
    assert index.iloc[0]["label"] == "positive"
