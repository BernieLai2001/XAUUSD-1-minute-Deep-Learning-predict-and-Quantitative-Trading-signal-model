# Code index: the 11 steps of report Table 1

"Main scripts" produce the files the next step uses and run in step order; "check scripts" only verify or report
statistics and produce no pipeline files.

**How to find the code**: no line numbers (they change whenever the code changes). Each entry gives a function name,
a `# --- section ---` heading or a variable name; search for it in the editor. Script paths are relative to `scripts/`;
`gold_dl/` means `src/gold_dl/`. Main scripts are in the numbered folders `1_download/` … `5_export/`, checks quoted in
the report are in `report_checks/`, other checks are in the local `explore/` folder (not committed). All parameters come
from `config.yaml`; the setting is given in brackets.

| Step | What it does | Main script (where) | Check scripts |
|---|---|---|---|
| 1 | Read raw data; verify files; check duplicates and invalid records | `1_download/databento_gc_mbp1.py` (MBP-1), `1_download/databento_gc_ohlcv1m.py` (OHLCV-1m; date range `data.start`, `data.end`); sections `1. file integrity`, `2. duplicates`, `3. invalid bars` of `report_checks/section2_data_quality.py` | `explore/inspect_mbp1_sample.py`, `explore/inspect_ohlcv1m_sample.py`, `explore/check_crossed_book.py`, `explore/check_one_side_empty.py` |
| 2 | Aggregate MBP-1 to 1-minute bars, order flow and book statistics; compare with official OHLCV | `2_aggregate/process_gc_mbp1.py` (trades → buy/sell volume, delta), `fix_gc_delta.py`, `build_gc_trades_1min.py`, `build_gc_quote_1min.py` (bid/ask OHLC), `build_gc_book_1min.py`, `build_gc_spread_tw_1min.py`, `build_gc_depth_1min.py`, `build_gc_ofi_1min.py`; chunked reading and the valid-book test are `chunks()` and `valid_book()` in `gold_dl/mbp1.py` | `explore/compare_ohlcv_vs_mbp1.py`, `explore/validate_gc_orderflow.py`, `explore/check_side_n.py` |
| 3 | Time zones; trading sessions; detect rolls and back-adjust | Time zones: `utc_to_bj()` (UTC → Beijing, called by the `2_aggregate/` scripts), `bj_to_et()`, `trade_date()` in `gold_dl/timeutil.py`; sessions: section `# --- 时间 ---` of `3_features/build_features.py`; back-adjustment: variable `factor` in the same file (after `# ================= 读入 GC 数据 =================`) | `explore/sanity_gc_features.py`, `explore/check_atr_floor_roll.py`, `explore/inspect_roll_day_poc.py` |
| 4 | Fill intraday missing minutes (`cleaning.max_fill_gap_min`) | variables `fill_end`, `is_filled` in `3_features/build_features.py` | `explore/check_missing_minutes.py` |
| 5 | ATR (with floor) and all features | `3_features/build_features.py`: ATR from `atr()` in `gold_dl/indicators.py` (period `labels.atr_period`), floor is variable `floor_adj`; feature groups in sections `# --- 价格变化 ---` (price change), `趋势` (trend), `趋势强度` (trend strength), `动量` (momentum), `波动率` (volatility), `订单流` (order flow), `流动性` (liquidity), `时间` (time), `成交量` (volume) | `explore/check_new_features.py`, `explore/validate_features.py`, `explore/check_feature_leakage.py` (cut-off test) |
| 6 | Triple-barrier labels from bid/ask; mark missing labels | `build_labels()` in `gold_dl/labels.py` (called by `4_labels/make_labels.py`): entry prices `entry_long`, `entry_short`; barrier hits `windows()`, `first_hit()`; distances `sl_dist()`; grid `labels.k_sl_grid`, `rr_grid`, `holding_min` | `report_checks/audit_labels_sample.py`, `docs_tools/list_audit_flags.py`, `4_labels/make_labels_with_preroll_check.py` (comparison version without the pre-roll exclusion) |
| 7 | Remove warm-up, pre-roll days and broken windows | Warm-up: comment `# 预热期` in `3_features/build_features.py`; window breaks `seg_id`: variables `brk`, `sched` in the same file; pre-roll (`cleaning.pre_roll_days`): variables `pre_roll`, `sample_ok` in `build_labels()`; final filter: variable `base` in `5_export/export_model_arrays.py` | `explore/check_row_diff.py` |
| 8 | Time-ordered train / validation / test split with gaps | variable `split` in `build_labels()` (`split.train_frac`, `val_frac`, `gap_min`) | `tests/test_split.py`, `report_checks/check_split_order.py`, `report_checks/split_table.py` |
| 9 | Fit parameters on the training set only | Label parameters: `stats`, `combo` in `build_labels()` + `4_labels/save_default_labels.py` (default `labels.default_k_sl`, `default_rr`); extreme-spread threshold: `med`, `thr` in `build_labels()` (`cleaning.wide_spread_mult`), and `wide_threshold()` in `gold_dl/labels.py` for the figures; OFI time-of-day factors: variables `FAC_PATH`, `fit_rows` in `3_features/build_features.py`; long-tail transform and clipping: `3_features/transform_tail_features.py` (`features.log_then_clip`, `clip_only`); z-score: `3_features/standardize_features.py` | `explore/spread_threshold_check.py`, `explore/summarize_wide_spread.py`, `explore/inspect_ofi_depth_adj.py` |
| 10 | Exploratory analysis on the training set only | `figures/plot_eda_figures.py` (section 4 figures), `figures/plot_return_acf.py`, `figures/plot_time_profiles.py`, `figures/feature_corr_before_after.py`, `report_checks/section7_checks.py` (single-horizon figures use `report.focus_holding_min`) | `report_checks/check_monthly_sl_drift.py` |
| 11 | Export arrays; training input pipeline | `5_export/export_model_arrays.py`, `gold_dl/data_pipeline.py` (`pipeline.lookback`, `batch_size`), `gold_dl/config.py` | `report_checks/section6_pipeline_checks.py`, `notebooks/01_smoke_test.ipynb`, `timing/time_cnn_bilstm.py`, `timing/kaggle_timing.py` |

## Actual run order

Table 1 groups the work by logic and one script can cover several steps, so the run order differs from the step
numbers; the folder numbers give the run order:

1. `1_download/` → `2_aggregate/`: downloads → `process_gc_mbp1.py` → `fix_gc_delta.py` → each `build_gc_*_1min.py` (steps 1–2)
2. `3_features/build_features.py` (steps 3, 4, 5, warm-up and seg_id of 7, OFI factors of 9)
3. `4_labels/make_labels.py` → `save_default_labels.py` (steps 6, pre-roll part of 7, 8, label parameters and extreme spread of 9)
4. `3_features/transform_tail_features.py` → `standardize_features.py` (long tail and z-score of step 9; they need the split from step 3)
5. `figures/`, `report_checks/` (step 10 and the statistics in the report)
6. `5_export/export_model_arrays.py` → read by `gold_dl.data_pipeline` during training (step 11)

Each step saves the settings it ran with as `run_config_<step>.yaml` in its output folder. Column names carry no
parameter values: the ATR column is `raw_atr` and the pre-roll flag is `pre_roll` (`raw_atr14` and `pre_roll_3d` in the
other M3 notes are the names before the change).
