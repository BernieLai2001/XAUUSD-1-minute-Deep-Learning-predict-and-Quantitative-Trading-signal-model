# Short-Term Gold Futures Trade Outcome Prediction from Minute-Level Market Data

中文版：[README_zh.md](README_zh.md)

IE7615 Deep Learning, Fall 2026. Kaiyuan Lai (lai.kaiy@northeastern.edu).

## Problem and intended user

**Research question.** Using only minute-level OHLCV and MBP-1 data and the technical indicators computed from
them, can deep learning estimate the probability and duration of short-term trend persistence in XAU/USD accurately
enough to achieve a positive expected return per trade after transaction costs?

**Target product and users.** The target product is a trading-signal tool: it receives 1-minute bars in real time
through a market-data API and, at the close of each bar, outputs the probabilities of the possible trade outcomes and
a suggested action (long, short or no trade). It is a research prototype for retail and institutional traders of
gold-related products, not a system for executing live trades. The project focuses on a single deep and highly
liquid market rather than several assets, because a model for one market is easier to validate rigorously within one
term. Gold trades almost continuously on weekdays through spot and derivatives markets, and its price is hard for any
single participant to manipulate.

**What the model predicts.** The model takes the last L = 60 minutes of COMEX gold futures (GC) market data, 33 features per minute, and predicts
for each of six output heads (long / short × holding periods of 15, 30 and 60 minutes) whether a trade entered at the
next minute's open would first reach its take-profit barrier, first reach its stop-loss barrier, or reach neither
before the holding period ends (triple-barrier labels, judged on real bid/ask prices).

## Data

| | |
|---|---|
| Source | Databento, CME Globex, dataset `GLBX.MDP3`, continuous front-month contract `GC.v.0` |
| Schemas | MBP-1 (top of book and trades; used for everything) and OHLCV-1m (only to check the bars rebuilt from MBP-1) |
| Time range | 2025-09-29 to 2026-09-29 |
| Download date | 2026-10-24 (MBP-1 batch job `GLBX-20261025-PT9C6P3YKJ`) |
| Size | MBP-1: 13 monthly files, 10.1 GB compressed (42.7 GB uncompressed); OHLCV-1m: 6.2 MB |
| Cost | about US$71.60 (MBP-1) and US$1.28 (OHLCV-1m) on a Databento usage-based account |
| Licence | CME market data under the CME Information License Agreement: internal personal use only, **no redistribution** |

Because the licence forbids redistribution, **no data is included in this repository**, and the processed arrays
used for training are kept in a *private* Kaggle dataset. To reproduce the results you need your own Databento
account:

1. Create an account at <https://databento.com> and an API key.
2. Save the key in `databento_key.txt` in the project root (this file is listed in `.gitignore`; never commit it).
3. Run the two download scripts in step 1 below.

## Installation

Two Python environments are used, because the data-processing steps need `pyarrow` and `databento`, and the
training steps use TensorFlow with the Apple Metal plug-in:

| Environment | Python | Used for | Key packages |
|---|---|---|---|
| processing | 3.13.9 | steps 1–4 and all figures | databento 0.88.0, pandas 2.2.3, numpy 2.1.3, pyarrow 19.0.0, scipy 1.15.3, matplotlib 3.10.0, zstandard 0.24.0 |
| training (local) | 3.10.20 | steps 5–6 | tensorflow 2.16.2, tensorflow-metal 1.2.0, keras 3.12.1, numpy 1.26.4 |
| training (Kaggle) | Kaggle default image | formal training and timing | TensorFlow 2.20.0, Tesla T4 |

```bash
pip install -r requirements-processing.txt
```

```bash
pip install -r requirements-train.txt
```

Then, in the training environment, install the project's own package `gold_dl` (in `src/`) in editable mode, so
scripts and notebooks can `import gold_dl`:

```bash
pip install -e .
```

`gold_dl.config` reads `config.yaml` and resolves every path relative to the project root;
`gold_dl.data_pipeline` holds the tf.data pipeline (`make_dataset`) and the masked loss used by every split.
On Kaggle, `kaggle_timing.py` carries its own copy of the pipeline so that only one file has to be uploaded.

## Platform

- **Development and smoke tests:** Apple M4 Pro, 24 GB, TensorFlow with the Metal GPU.
- **Training (M4 onward):** Kaggle Notebooks, GPU T4 × 2 (one GPU is used), 30 GPU hours per week.
  Upload `model_data/model_arrays.npz` (55 MB) as a **private** Kaggle dataset and add it to the notebook; the
  scripts find it under `/kaggle/input` automatically. Training checkpoints are written to `/kaggle/working`,
  because the session disk is wiped when the session ends.

## Reproducing the results

Run from the project root, in this order. Every script finds the project root from its own location, so the
clone can live in any folder. Random seeds are fixed at 42 wherever randomness affects a result.

**1. Download (processing environment)**

```bash
python scripts/databento_gc_mbp1.py
```

```bash
python scripts/databento_gc_ohlcv1m.py
```

**2. Rebuild 1-minute data from MBP-1**

```bash
python scripts/process_gc_mbp1.py
```

```bash
python scripts/fix_gc_delta.py
```

```bash
python scripts/build_gc_quote_1min.py
```

```bash
python scripts/build_gc_book_1min.py
```

```bash
python scripts/build_gc_depth_1min.py
```

```bash
python scripts/build_gc_ofi_1min.py
```

```bash
python scripts/build_gc_spread_tw_1min.py
```

```bash
python scripts/build_gc_trades_1min.py
```

**3. Features, labels and the split**

```bash
python scripts/build_features.py
```

```bash
python scripts/make_labels.py
```

```bash
python scripts/save_default_labels.py
```

```bash
python scripts/make_labels_with_preroll_check.py
```

`make_labels.py` builds the labels for all 12 barrier combinations and assigns the time-based split
(train / validation / test, with 60-minute gaps between them); `save_default_labels.py` selects the default
combination (k_SL = 2, 1:1). `make_labels_with_preroll_check.py` rebuilds the labels without excluding the
3 days before each roll, only for the before/after comparison in figure 6(c).

**4. Long-tail treatment, standardisation and export (statistics from the training set only)**

```bash
python scripts/transform_tail_features.py
```

```bash
python scripts/standardize_features.py
```

```bash
python scripts/export_model_arrays.py
```

**5. Input pipeline checks and smoke test (training environment)**

```bash
python scripts/section6_pipeline_checks.py
```

The same checks, step by step and without writing any files, are in `notebooks/01_smoke_test.ipynb` (kernel:
the training environment). To run it from top to bottom without opening Jupyter:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/01_smoke_test.ipynb
```

**6. Timing on Kaggle** (in a Kaggle notebook cell, after adding the dataset)

```bash
python kaggle_timing.py
```

```bash
python kaggle_timing.py --L 120
```

### Where each result in the report comes from

| Report content | Script | Output |
|---|---|---|
| Data quality counts | `section2_data_quality.py` | `数据说明/第2节_数据质量统计.txt` |
| Label audit (60 samples, 360 labels) | `audit_labels_sample.py` | `数据说明/标签审核/` |
| Spread and volatility by hour and weekday | `plot_time_profiles.py` | `数据说明/` figures and CSV files |
| Split table | `split_table.py`, `check_split_order.py` | printed |
| EDA figures 1–8 | `plot_eda_figures.py` | `数据说明/EDA图/`, `eda_stats.txt` |
| Autocorrelation (figure 7) | `plot_return_acf.py` | `数据说明/EDA图/`, `acf_stats.txt` |
| Pipeline batch, smoke test, local timing | `section6_pipeline_checks.py`, `time_cnn_bilstm.py` | `数据说明/第6节/` |
| Kaggle timing | `kaggle_timing.py` | `kaggle_timing.json`; copied to `数据说明/第6节/kaggle_timing_T4.csv` |
| Univariate feature checks, monthly class shares | `section7_checks.py`, `check_monthly_sl_drift.py` | `数据说明/第7节_特征单变量检查.csv`, printed |

### Split test

The split is by time: every minute belongs to exactly one set, the sets are in time order, and each boundary
leaves a gap of 60 minutes (the longest holding period). The test checks this on the label table and on the
samples in `model_arrays.npz` (processing environment, after step 4):

```bash
pytest tests/test_split.py
```

## Repository structure

```
project/
├── README.md
├── LICENSE                       MIT, code only
├── config.yaml                   all settings: paths, split, labels, lookback, batch size, seed
├── requirements-processing.txt   pinned packages for steps 1–4 and figures
├── requirements-train.txt        pinned packages for steps 5–6
├── pyproject.toml                makes src/gold_dl installable (pip install -e .)
├── src/gold_dl/        reusable package: config.py (paths and settings), data_pipeline.py (tf.data pipeline, loss)
├── scripts/            processing, labelling, check and figure scripts
├── notebooks/          01_smoke_test.ipynb: batch shapes, types, label values, window check, 300-step smoke test
├── tests/              test_split.py: no minute in two sets, time order, 60-minute gaps
├── 数据说明/            report notes, figures and tables (Chinese and English figures)
├── GC_mbp1/            raw MBP-1 files            (not committed)
├── GC_processed/       1-minute data rebuilt from MBP-1 (not committed)
├── features/           feature tables             (not committed)
├── labels/             label tables               (not committed)
└── model_data/         model_arrays.npz           (not committed)
```

<!-- TODO: only gold_dl and tests/ read config.yaml so far; the scripts still keep their own constants. -->

## Licence

Code: see `LICENSE`. Data: CME market data via Databento, not redistributable (see above).
