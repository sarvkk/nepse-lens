# NEPSE Lens

**Market intelligence for the Nepal Stock Exchange.** Ingestion, point-in-time features, anomaly detection, walk-forward direction models, financial-news sentiment, a FastAPI service, and a Streamlit dashboard.

[![CI](https://github.com/sarvkk/nepse-lens/actions/workflows/ci.yml/badge.svg)](https://github.com/sarvkk/nepse-lens/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%20%7C%203.12-3776ab)
![License](https://img.shields.io/badge/license-MIT-17140f)

![NEPSE Lens dashboard](docs/preview.png)

---

## Why this exists

NEPSE is one of the least tooled major exchanges in South Asia. There is no clean public API, no open daily dataset, and most retail analysis happens in forum threads and screenshots. Meanwhile the interesting problems are exactly the ones a market-intelligence product needs: messy ingestion, point-in-time feature engineering, anomaly triage, and turning financial text into a signal you can actually plot.

NEPSE Lens is a small, end-to-end answer to that. It is deliberately a **pipeline**, not a notebook: every stage is a module with tests, the evaluation is walk-forward, and the output is an API plus a dashboard rather than a single chart.

---

## What it does

| Stage | Implementation |
| --- | --- |
| **Ingest** | Live NEPSE client (`Authorization: Salter` token, retries, raw JSON archive) or a bundled deterministic fixture snapshot. |
| **Store** | DuckDB single-file warehouse. Idempotent upserts; no server to run. |
| **Features** | Returns, rolling volatility, volume/turnover z-scores, MA gap, RSI, intrabar range, gap, market breadth — all point-in-time. |
| **Anomaly detection** | Robust z-score, Isolation Forest, and an optional LSTM autoencoder, scored with precision@k and lift over a labelled event set. |
| **Direction** | Next-session up/down with a naive prior, logistic regression, and histogram gradient boosting, evaluated on an expanding walk-forward split. |
| **NLP** | Finance-tuned lexicon baseline plus an optional FinBERT path; evaluated on templated headlines and on Financial PhraseBank. |
| **Serve** | FastAPI (`/ohlcv`, `/anomalies`, `/sentiment`, `/predictions`, `/market/summary`) and a Streamlit dashboard. |
| **Ship** | Docker, docker-compose, GitHub Actions (ruff + pytest + full pipeline), model card, reproducible `make` targets. |

---

## Results

All numbers below are produced by `make train` on the bundled fixture (59 symbols, 760 sessions) plus Financial PhraseBank.

**Next-session direction — expanding walk-forward, 5 folds, no shuffling**

| Model | Accuracy | ROC AUC | Brier |
| --- | ---: | ---: | ---: |
| `naive_prior` | 51.8% ± 3.1% | 0.500 | 0.250 |
| `logistic` | 52.6% ± 2.2% | 0.532 | 0.250 |
| `hist_gbm` | **52.9% ± 1.8%** | 0.529 | 0.257 |

**Anomaly detection — 118 labelled events**

| Detector | Precision@k | Recall@k | Lift@k | Avg precision |
| --- | ---: | ---: | ---: | ---: |
| `robust_zscore` | 53.4% | 53.4% | 203× | 0.528 |
| `isolation_forest` | 96.6% | 96.6% | 367× | 0.995 |

**Headline sentiment**

| Dataset | Model | Accuracy | Macro F1 | n |
| --- | --- | ---: | ---: | ---: |
| Templated NEPSE headlines | finance lexicon | 95.4% | 0.955 | 3,422 |
| Financial PhraseBank (all-agree) | finance lexicon | 73.1% | 0.602 | 2,264 |

> **Read the caveats before quoting these.** The price fixture is synthetic, and the generator injects a small AR(1) component so the walk-forward harness can be validated end to end. A 1.1-point lift over the naive prior proves the plumbing works; it is **not** evidence of a tradeable edge. The anomaly labels are injected extreme events, so detectors look better than they will on real data. The honest headline is the PhraseBank result: a no-training lexicon reaches 73% on real financial sentences, and FinBERT is the path to ~90%. See [MODEL_CARD.md](MODEL_CARD.md).

---

## Quickstart

```bash
git clone https://github.com/sarvkk/nepse-lens.git
cd nepse-lens

python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

make train     # generate fixture → features → models → reports
make dash      # http://localhost:8501
make api       # http://localhost:8000/docs
```

`make train` writes `reports/metrics.json` and `reports/report.md`. The database lands in `data/nepse.duckdb` and is regenerable.

### Optional extras

```bash
pip install -e ".[torch]"   # LSTM autoencoder
pip install -e ".[nlp]"     # FinBERT (downloads ~440 MB)
python scripts/train.py --torch --phrasebank data/external/FinancialPhraseBank-v1.0/Sentences_AllAgree.txt
```

### Docker

```bash
docker compose up --build
# dashboard → http://localhost:8501
# api       → http://localhost:8000/docs
```

---

## Data: live vs fixture

NEPSE's Angular frontend mints a short-lived `Salter` bearer token for every JSON call. There is no stable public token endpoint, so live mode asks you to supply one:

```bash
cp .env.example .env
# paste the Authorization header value from DevTools into NEPSE_TOKEN
python scripts/fetch_data.py --source live
```

Live mode archives every raw payload under `data/raw/` before normalising, so a schema change never destroys the original response. Because a single snapshot is not enough for features, a real deployment would run this on a schedule and backfill history.

CI and the demo use `--source fixture` instead: a deterministic, NEPSE-shaped snapshot with 59 tickers, ten sectors, fat-tailed returns, volume that co-moves with absolute return, and labelled anomaly events. It is checked in under `data/fixtures/` so the repository is reproducible without credentials.

---

## Evaluation discipline

The two things that make market ML look better than it is are leakage and random K-fold. This repo blocks both:

- `tests/test_features.py::test_no_future_leakage` multiplies the last five sessions by 3× and asserts that every feature on or before the cutoff is byte-identical.
- `tests/test_features.py::test_target_matches_next_close` checks the label is exactly `close[t+1] > close[t]`.
- `walk_forward_splits` is expanding-window and strictly ordered; the test asserts each test block starts after the previous one ends.
- Sentiment is evaluated on human labels (PhraseBank), not on the templates the lexicon was tuned against.

---

## Architecture

```
 NEPSE API ──┐
             ├──► ingest ──► DuckDB ──► features ──┬──► anomaly detectors ──┐
 fixture CSV ┘                                      ├──► direction models ───┤
                                                    └──► headline sentiment ─┤
                                                                             ▼
                                                        reports/ + FastAPI + Streamlit
```

```
src/nepse_lens/
├── config.py            # env-overridable paths and settings
├── store.py             # DuckDB schema + idempotent upserts
├── ingest/
│   ├── client.py        # live NEPSE client (token, retries, raw archive)
│   └── fixtures.py      # deterministic NEPSE-shaped snapshot
├── features.py          # point-in-time feature engineering
├── models/
│   ├── anomaly.py       # robust z, Isolation Forest, LSTM-AE
│   ├── direction.py     # walk-forward evaluation
│   └── torch_models.py  # optional LSTM-AE + TCN
├── nlp/
│   ├── lexicon.py       # finance vocabulary, negators, intensifiers
│   ├── sentiment.py     # scoring + FinBERT
│   └── phrasebank.py    # benchmark loader
├── pipeline.py          # orchestration
├── report.py            # markdown metrics report
├── api.py               # FastAPI
└── dashboard.py         # Streamlit
```

---

## Design decisions

**DuckDB over Postgres.** One file, analytical SQL, no container to babysit. The whole store is 40 MB and rebuilds in seconds.

**Fixture-first.** A portfolio project that only runs with a private token is not reproducible. The fixture is the default; live mode is opt-in and documented.

**Optional heavy dependencies.** PyTorch and transformers are extras. The core pipeline and CI install in under a minute, which keeps the feedback loop honest.

**Honest metrics over impressive ones.** The direction model is barely better than the base rate, and the README says so. A reviewer who sees a 92% accuracy claim on daily equity direction should assume leakage; showing the walk-forward harness and the base rate is the stronger signal.

**Anomaly labels are injected, and labelled as such.** Precision@k measures whether the detector recovers known events, not whether those events predict returns.

---

## What I'd do next

- **Real history.** Scheduled ingestion with a multi-year backfill, corporate-action adjustment, and a data-quality report (missing sessions, zero-volume days, splits).
- **Better labels.** Replace injected events with hand-audited real events (circuit breakers, block deals, earnings surprises) and measure detector agreement with analysts.
- **FinBERT fine-tune.** Fine-tune `ProsusAI/finbert` on PhraseBank + NEPSE headlines; target ~88–90% macro F1, and calibrate the neutral class instead of collapsing it.
- **Cross-sectional model.** Predict relative rank within sector rather than absolute direction; that is closer to how a market-visualization product would surface "what moved and why."
- **Deployment.** Model registry, scheduled retrain, drift monitoring on feature distributions, and a proper dashboard auth layer.

---

## License

MIT for the code. Financial PhraseBank is **not** redistributed; `scripts/fetch_phrasebank.py` downloads it under CC BY-NC-SA 3.0 (Malo et al., 2014). Fixture data is synthetic.
