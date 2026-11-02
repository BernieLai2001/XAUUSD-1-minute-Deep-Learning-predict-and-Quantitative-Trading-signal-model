# Long-tail feature treatment

English version of `数据说明/长尾特征处理.md`.

Script: `scripts/transform_tail_features.py`. Thresholds are computed from training-set samples only and applied unchanged to the validation and test sets.
Samples = minutes with trades that are not within the 3 days before a roll (`pre_roll_3d = 0`); filled minutes (`is_filled = 1`) are transformed in the same way but are not used to compute thresholds.
Input `features/features_1min.parquet` (not modified), output `features/features_1min_tailfix.parquet` (same column names; only the 8 columns in the tables below change); thresholds and statistics are in `features/tail_transform_params.csv`.
The subsequent standardisation (z-score) is in `scripts/standardize_features.py` (output `features/features_1min_scaled.parquet`), which likewise uses only the mean and standard deviation of training samples.
The "Cause" column was verified from the times at which the most extreme 0.1% of training samples (232 per feature) occur; times are ET.

All statistics in the tables use training samples (232,608). Log-transformed features (spread_atr, relvol_60, atr_rel, vol_ratio_5_60) are in log units after treatment, with values converted back to original units in parentheses. The range after winsorising is exactly the 1% and 99% percentiles before treatment; for every feature, 1% of training samples sit at the lower bound and 1% at the upper bound. Neither the log transform nor winsorising changes the ordering, so the median (converted back to original units) is unchanged.
Filled minutes have volume 0, so relvol_60 = 0; before taking the log a tiny floor (10⁻¹²) is applied, and the value is then clipped to the lower threshold.

**Compared with before the 3 pre-roll days were excluded:** the kurtosis of spread_atr fell from 2,680 to 496 and its 99th percentile from 4.25 to 0.55, showing that its original long tail came mainly from the days just before rolls; thresholds of the other features changed little.

## Severe long tail (kurtosis > 100)

| Feature | Cause | Treatment | Kurtosis (before → after) | Skewness (before → after) | Range (before → after) | Mean / std after treatment |
|---|---|---|---|---|---|---|
| relvol_60 | Sudden volume surge after the 18:00 ET reopen (about 15% within 10 minutes of the open; most at 18–20 ET) and US economic data releases at 08:30; the largest values are in the first minute after a roll: new-contract volume divided by the average volume of the old contract's last 60 minutes (including filled zero-volume minutes), up to 472 | Log, then winsorise at 1%/99% | 35,088 → 0.02 | 131 → −0.02 | 0.0069 – 472 → −2.20 – 1.64 (0.110 – 5.14) | −0.25 / 0.75 |
| spread_atr | 44% on 2026-03-26 (3.5 days before a roll, just outside the excluded window, when the old contract had already become thin); about 13% within 10 minutes of the 18:00 ET reopen; most of the rest on holidays (Thanksgiving 2025-11-27, Presidents' Day 2026-02-16) | Log, then winsorise at 1%/99% | 496 → −0.19 | 12.0 → 0.09 | 0.030 – 9.51 → −2.76 – −0.60 (0.063 – 0.548) | −1.74 / 0.44 |
| ofi_depth_adj | 37% concentrated in 2026-02-01 to 02-02 (violent market after the gold price crash); about 21% within 10 minutes of the 18:00 ET reopen | Winsorise at 1%/99% | 239 → 1.8 | −0.84 → −0.03 | −160 – 142 → −6.31 – 6.24 | 0.05 / 1.94 |
| d_open | 34% are gaps within 10 minutes of the 18:00 ET reopen (46% in the 18 ET hour in total); the rest scattered | Winsorise at 1%/99% | 176 → 0.9 | 1.33 → 0.01 | −8.41 – 8.84 → −0.438 – 0.440 | 0.00 / 0.15 |

## Moderate long tail (kurtosis 10–35)

| Feature | Cause | Treatment | Kurtosis (before → after) | Skewness (before → after) | Range (before → after) | Mean / std after treatment |
|---|---|---|---|---|---|---|
| atr_rel | 71% of the right tail concentrated in the violent market of 2026-01-30 to 02-02; all positive | Log, then winsorise at 1%/99% | 27.8 → 0.2 | 3.92 → 0.51 | 0.000066 – 0.0090 → −8.69 – −5.95 (0.00017 – 0.0026) | −7.56 / 0.55 |
| vol_ratio_5_60 | 62% within 10 minutes of the reopen: the opening gap suddenly enlarges ATR5 before ATR60 catches up; all positive | Log, then winsorise at 1%/99% | 20.7 → 0.1 | 2.86 → 0.31 | 0.140 – 8.31 → −0.76 – 0.84 (0.466 – 2.31) | −0.05 / 0.31 |
| d_high | One-sided long tail; 36% in the 18 ET hour (reopen), next the 08:30 data releases | Winsorise at 1%/99% | 11.4 → 0.9 | 1.74 → 0.97 | −5.13 – 11.2 → −0.284 – 2.01 | 0.48 / 0.46 |
| d_low | Same as above, opposite direction; 19% in the 18 ET hour | Winsorise at 1%/99% | 11.0 → 1.2 | −1.81 → −1.08 | −11.8 – 6.06 → −2.15 – 0.284 | −0.49 / 0.48 |

The high kurtosis of relvol_60 before treatment comes mainly from a few extreme values in the first minute after a roll. These samples have look-back windows that cross the roll and are dropped anyway under the sampling rules; after the log transform and winsorising the kurtosis is only 0.02.

## Share of samples winsorised

In the training set it is 2% for every feature (by definition). Validation and test sets (samples only):

| Feature | Validation set | Test set |
|---|---|---|
| spread_atr | 2.5% | 1.9% |
| relvol_60 | 3.2% | 2.5% |
| atr_rel | 1.4% | 1.7% |
| vol_ratio_5_60 | 2.0% | 1.9% |
| ofi_depth_adj | 0.7% | 0.7% |
| d_open | 4.6% | 2.4% |
| d_high | 3.5% | 2.3% |
| d_low | 3.0% | 1.9% |

## Notes

- After excluding the 3 pre-roll days, the kurtosis of spread_atr after treatment fell from 7.6 to −0.19, and values no longer pile up at the upper bound.
- Winsorising removes the size information of d_open, d_high and d_low at reopen gaps (only the direction and "above threshold" remain). If needed, a separate 0/1 flag for "first minute after the reopen" could be added.
- The share of d_open winsorised in the validation set is about 2.3 times that in the training set, showing more extreme values of this feature in the validation period; this is worth watching in evaluation.
- 44% of the extreme values of spread_atr fall 3.5 days before a roll, just outside the 3-day exclusion window. For a more thorough treatment later, the exclusion window could be widened to 4 days.
