---
title: Feature list (33 features)
---

English version of `数据说明/特征清单.md`.

All features use only the COMEX gold front-month futures GC (Databento GC.v.0); spot prices are not used. The time axis is the GC 1-minute bars with trades (Beijing time, timestamp = start of the minute), plus the occasional missing minutes within a session: when adjacent traded bars of the same contract are 2–30 minutes apart, every minute in between is filled, with price equal to the previous minute's close (flat bar), volume-type fields set to 0, and order-book fields carried forward from the previous minute when there is no record. Filled minutes are used only to complete look-back windows, not as samples. Each feature uses only data from minute t and earlier.

ATR14 is the simple average of the true range over the last 14 bars, consistent with the label definition. Rolls are handled by proportional back-adjustment: only prices after the roll are adjusted, relative changes within a contract are identical to the raw prices, and data before a roll contain no future roll gap. For every feature divided by ATR, ATR has a floor: no lower than GC's minimum price increment of $0.10, to avoid ATR = 0 in quiet markets.

There are also 10 non-feature columns: `raw_open`, `raw_high`, `raw_low`, `raw_close` (raw GC trade prices, not back-adjusted), `raw_atr14` (ATR14, USD) and `instrument_id` (current contract), used to compute labels; `flag_degraded_day` marks dates Databento later flagged as degraded data quality, used only to filter samples, not as a model input; `is_filled` (1 = filled minute) and `seg_id` (window segment number, incremented only at contract changes and at long unscheduled gaps; the daily halt, weekends and holidays are not breaks; a look-back window is usable only if it is constant within the window) are used only to organise samples; `raw_ofi_depth` (OFI / depth before the time-of-day adjustment) is used only for diagnostics. The flag for the 3 days before a roll, `pre_roll_3d`, is in the label file.

`range_pos_60` ((close − 60-minute low) / (60-minute high − low)) has been removed: it is highly correlated with `ma_dist_60` (training set r = 0.93), and both answer "where is the price relative to the last hour"; `ma_dist_60` is kept so that the multi-scale set ma_dist 15 / 60 / 240 stays complete.

# 1. Price change (4)

| Feature | Calculation |
|---|---|
| `d_open` | (this minute's open − previous minute's close) / ATR14 |
| `d_high` | (this minute's high − previous minute's close) / ATR14 |
| `d_low` | (this minute's low − previous minute's close) / ATR14 |
| `d_close` | (this minute's close − previous minute's close) / ATR14 |

# 2. Trend (3)

| Feature | Calculation |
|---|---|
| `ma_dist_15` | (close − 15-minute moving average) / ATR14 |
| `ma_dist_60` | (close − 60-minute moving average) / ATR14 |
| `ma_dist_240` | (close − 240-minute moving average) / ATR14 |

# 3. Trend strength (2)

| Feature | Calculation |
|---|---|
| `er_15` | Efficiency ratio: |15-minute net change| / sum of absolute per-minute changes over 15 minutes; 0 when the price did not move at all |
| `er_60` | Efficiency ratio: |60-minute net change| / sum of absolute per-minute changes over 60 minutes; 0 when the price did not move at all |

# 4. Momentum (4)

| Feature | Calculation |
|---|---|
| `ret_5` | Price change over the past 5 minutes / ATR14 |
| `ret_15` | Price change over the past 15 minutes / ATR14 |
| `ret_60` | Price change over the past 60 minutes / ATR14 |
| `rsi_14` | RSI14 = 100 × average gain / (average gain + average loss), Wilder smoothing (α = 1/14); 50 when both average gain and average loss are 0 |

# 5. Volatility (2)

| Feature | Calculation |
|---|---|
| `atr_rel` | ATR14 / close |
| `vol_ratio_5_60` | ATR5 / ATR60; above 1 means volatility is expanding |

# 6. Order flow (5)

| Feature | Calculation |
|---|---|
| `delta_ratio_1` | Over the past 1 minute: (buyer-initiated volume − seller-initiated volume) / total volume |
| `delta_ratio_5` | Over the past 5 minutes: (buyer-initiated volume − seller-initiated volume) / total volume |
| `delta_ratio_15` | Over the past 15 minutes: (buyer-initiated volume − seller-initiated volume) / total volume |
| `book_imb` | At the end of minute t: (best bid size − best ask size) / (sum of the two), −1 to 1; filled minutes without an order-book record carry forward the previous minute; other minutes without a valid order book are 0 |
| `ofi_depth_adj` | Order-flow imbalance OFI / average top-of-book size in minute t, then divided by the training-set typical value for its time bucket (median of the absolute OFI/depth); time buckets = each of minutes 1–5 after the open as its own bucket, the rest in 30-minute buckets in ET; OFI is the cumulative change in best bid / best ask prices and sizes within the minute (contracts; Cont-Kukanov-Stoikov definition); average size = (best bid size + best ask size) / 2, time-weighted |

# 7. Liquidity (1)

| Feature | Calculation |
|---|---|
| `spread_atr` | Time-weighted average bid-ask spread in minute t (USD) / ATR14; each spread is weighted by how long it lasted within the minute; when the minute has no valid order book, the previous minute's spread is carried forward |

# 8. Time (11)

| Feature | Calculation |
|---|---|
| `tod_sin` | Sine encoding of the minute of the day in ET |
| `tod_cos` | Cosine encoding of the minute of the day in ET |
| `dow_mon` | Weekday (by CME trading day): mon |
| `dow_tue` | Weekday (by CME trading day): tue |
| `dow_wed` | Weekday (by CME trading day): wed |
| `dow_thu` | Weekday (by CME trading day): thu |
| `dow_fri` | Weekday (by CME trading day): fri |
| `sess_london` | London session only (London time 08:00–16:30) |
| `sess_ny` | New York session only (ET 08:00–17:00) |
| `sess_overlap` | London–New York overlap |
| `sess_asia` | Other hours (mainly the Asian session) |

# 9. Volume (1)

| Feature | Calculation |
|---|---|
| `relvol_60` | Volume in minute t / average volume of the previous 60 bars |

# Highly correlated feature pairs (training samples, |r| > 0.9)

| Feature A | Feature B | Correlation |
|---|---|---|
| None | — | — |

After removing range_pos_60, no feature pair in the training samples has |r| > 0.9.

# Count summary

| Category | Count |
|---|---|
| 1. Price change | 4 |
| 2. Trend | 3 |
| 3. Trend strength | 2 |
| 4. Momentum | 4 |
| 5. Volatility | 2 |
| 6. Order flow | 5 |
| 7. Liquidity | 1 |
| 8. Time | 11 |
| 9. Volume | 1 |
| **Total** | **33** |
