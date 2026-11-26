"""Markdown rendering for pipeline metrics."""

from __future__ import annotations

from typing import Any


def _pct(value: Any, digits: int = 1) -> str:
    try:
        return f"{float(value) * 100:.{digits}f}%"
    except (TypeError, ValueError):
        return "n/a"


def _num(value: Any, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "n/a"


def render_report(metrics: dict[str, Any]) -> str:
    data = metrics.get("data", {})
    lines: list[str] = [
        "# NEPSE Lens — run report",
        "",
        f"Generated: `{metrics.get('generated_at', '')}`",
        "",
        "## Data",
        "",
        f"- Symbols: **{data.get('symbols', 'n/a')}**",
        f"- Sessions: **{data.get('sessions', 'n/a')}** ({data.get('start', '?')} → {data.get('end', '?')})",
        f"- Headlines: **{data.get('headlines', 'n/a')}**",
        "",
        "## Next-session direction (walk-forward)",
        "",
        "| Model | Accuracy | ROC AUC | Brier | Predictions |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]

    for name, model in metrics.get("direction", {}).get("models", {}).items():
        lines.append(
            f"| `{name}` | {_pct(model.get('accuracy_mean'))} ± {_pct(model.get('accuracy_std'))} "
            f"| {_num(model.get('roc_auc_mean'))} | {_num(model.get('brier_mean'))} "
            f"| {model.get('n_predictions', 0):,} |"
        )

    lines += [
        "",
        "> Accuracy near the base rate means no edge. On the synthetic fixture the "
        "generator injects a small AR(1) component, so a modest lift over `naive_prior` "
        "only proves the harness works. Re-run on real NEPSE data before claiming anything.",
        "",
        "## Anomaly detection",
        "",
        "| Detector | Avg precision | ROC AUC | Precision@k | Recall@k | Lift@k |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, result in metrics.get("anomaly", {}).items():
        lines.append(
            f"| `{name}` | {_num(result.get('average_precision'))} | {_num(result.get('roc_auc'))} "
            f"| {_pct(result.get('precision_at_k'))} | {_pct(result.get('recall_at_k'))} "
            f"| {_num(result.get('lift_at_k'))}x |"
        )

    lines += [
        "",
        "## Headline sentiment",
        "",
        "| Dataset | Model | Accuracy | Macro F1 | n |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for dataset, result in metrics.get("sentiment", {}).items():
        lines.append(
            f"| {dataset} | `{result.get('model', 'lexicon')}` | {_pct(result.get('accuracy'))} "
            f"| {_num(result.get('macro_f1'))} | {result.get('n', 0):,} |"
        )

    lines += ["", "## Notes", ""]
    for note in metrics.get("notes", []):
        lines.append(f"- {note}")
    lines.append("")
    return "\n".join(lines)
