# NEPSE Lens — run report

Generated: `2026-11-26T15:12:21+00:00`

## Data

- Symbols: **59**
- Sessions: **760** (2023-12-28 → 2026-11-25)
- Headlines: **3,491**

## Next-session direction (walk-forward)

| Model | Accuracy | ROC AUC | Brier | Predictions |
| --- | ---: | ---: | ---: | ---: |
| `naive_prior` | 51.8% ± 3.1% | 0.500 | 0.250 | 33,570 |
| `logistic` | 52.6% ± 2.2% | 0.532 | 0.250 | 33,570 |
| `hist_gbm` | 52.9% ± 1.8% | 0.529 | 0.257 | 33,570 |

> Accuracy near the base rate means no edge. On the synthetic fixture the generator injects a small AR(1) component, so a modest lift over `naive_prior` only proves the harness works. Re-run on real NEPSE data before claiming anything.

Fitted detectors train on sessions on or before **2025-06-11**; the metrics below use later sessions only.

## Anomaly detection

| Detector | Avg precision | ROC AUC | Precision@k | Recall@k | Lift@k |
| --- | ---: | ---: | ---: | ---: | ---: |
| `robust_zscore` | 0.553 | 0.998 | 55.7% | 55.7% | 204.859x |
| `isolation_forest` | 0.996 | 1.000 | 96.7% | 96.7% | 355.490x |

## Headline sentiment

| Dataset | Model | Accuracy | Macro F1 | n |
| --- | --- | ---: | ---: | ---: |
| synthetic templates | `lexicon` | 95.2% | 0.953 | 3,491 |

## Notes

- Prices are a deterministic NEPSE-shaped fixture unless live ingestion was used.
- Direction accuracy near the base rate is the honest result on synthetic data.
- Anomaly metrics are out-of-sample: fitted detectors never see the evaluation half.
- Anomaly labels are injected events, so precision@k measures recovery, not alpha.
- The lexicon is a baseline; FinBERT is the production path (pip install -e '.[nlp]').
