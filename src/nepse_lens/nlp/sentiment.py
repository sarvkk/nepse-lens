"""Headline sentiment: a finance lexicon baseline and an optional FinBERT path.

The lexicon is the default because it needs no model download and runs in
milliseconds, which keeps CI hermetic. FinBERT is the production path: it is
better on negation and on phrasing the lexicon has never seen, at the cost of a
~440 MB download and a GPU for anything large.
"""

from __future__ import annotations

import re

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from .lexicon import (
    INTENSIFIERS,
    NEGATIVE,
    NEGATIVE_THRESHOLD,
    NEGATORS,
    POSITIVE,
    POSITIVE_THRESHOLD,
)

LABELS = ["positive", "negative", "neutral"]


def _tokenize(text: str) -> list[str]:
    # Join hyphenated compounds so "non-performing" and "write-off" match the
    # lexicon instead of being split into two harmless words.
    return re.findall(r"[a-z]+", str(text).lower().replace("-", ""))


def score_text(text: str) -> float:
    tokens = _tokenize(text)
    if not tokens:
        return 0.0

    total = 0.0
    for i, token in enumerate(tokens):
        weight = POSITIVE.get(token, 0.0) - NEGATIVE.get(token, 0.0)
        if weight == 0.0:
            continue
        context = tokens[max(0, i - 3) : i]
        if any(word in NEGATORS for word in context):
            weight = -weight * 1.15
        if any(word in INTENSIFIERS for word in context):
            weight *= 1.35
        total += weight
    return round(total, 4)


def label_from_score(score: float) -> str:
    if score >= POSITIVE_THRESHOLD:
        return "positive"
    if score <= NEGATIVE_THRESHOLD:
        return "negative"
    return "neutral"


class FinBertScorer:
    """Thin wrapper around ProsusAI/finbert. Optional dependency."""

    def __init__(self, model_name: str = "ProsusAI/finbert", device: int = -1):
        try:
            from transformers import pipeline
        except ImportError as exc:  # pragma: no cover
            raise ImportError("FinBERT needs transformers: pip install -e '.[nlp]'") from exc
        self.model_name = model_name
        self.pipe = pipeline("sentiment-analysis", model=model_name, device=device)

    def score(self, texts: list[str]) -> tuple[list[float], list[str]]:  # pragma: no cover
        results = self.pipe(list(texts), truncation=True, batch_size=32)
        scores: list[float] = []
        labels: list[str] = []
        for result in results:
            label = str(result["label"]).lower()
            confidence = float(result["score"])
            if label == "positive":
                scores.append(confidence)
            elif label == "negative":
                scores.append(-confidence)
            else:
                scores.append(0.0)
            labels.append(label)
        return scores, labels


def score_headlines(headlines: pd.DataFrame, model: str = "lexicon") -> pd.DataFrame:
    frame = headlines.copy()
    if model == "lexicon":
        frame["score"] = [score_text(text) for text in frame["headline"]]
        frame["pred_label"] = [label_from_score(score) for score in frame["score"]]
    elif model == "finbert":  # pragma: no cover
        scores, labels = FinBertScorer().score(frame["headline"].tolist())
        frame["score"] = scores
        frame["pred_label"] = labels
    else:
        raise ValueError(f"Unknown sentiment model: {model}")
    frame["model"] = model
    return frame


def evaluate_sentiment(
    scored: pd.DataFrame,
    true_col: str = "label",
    pred_col: str = "pred_label",
) -> dict[str, object]:
    truth = scored[true_col].astype(str)
    pred = scored[pred_col].astype(str)
    return {
        "n": int(len(scored)),
        "accuracy": float(accuracy_score(truth, pred)),
        "macro_f1": float(f1_score(truth, pred, average="macro", labels=LABELS, zero_division=0)),
        "labels": LABELS,
        "confusion": confusion_matrix(truth, pred, labels=LABELS).tolist(),
        "per_class": classification_report(
            truth, pred, labels=LABELS, output_dict=True, zero_division=0
        ),
    }


def daily_sentiment_index(scored: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        scored.groupby("date")
        .agg(score=("score", "mean"), n_headlines=("score", "size"))
        .reset_index()
        .sort_values("date")
    )
    grouped["label"] = [label_from_score(score) for score in grouped["score"]]
    return grouped
