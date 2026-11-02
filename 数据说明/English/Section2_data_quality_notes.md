# Section 2 notes: data quality (GC futures data only)

English version of `数据说明/第2节_数据质量笔记.md`.

Scope: Databento GC front-month futures 1-minute bars (OHLCV-1m), MBP-1 (top-of-book quotes + individual trades), and the files derived from MBP-1. The discontinued spot data (Twelve Data), Dukascopy and the futures-spot basis are not covered here.
Statistics scripts: `scripts/section2_data_quality.py` (results in `数据说明/第2节_数据质量统计.txt`), `scripts/audit_labels_sample.py`.

## 1. Corrupted files

- **Count**: 0. The SHA-256 checksums of all 15 files in the MBP-1 batch job (13 monthly data files + `metadata.json` + `condition.json`) match `manifest.json`; all 350,611 records of the OHLCV-1m file were read in completely.
- **Cause / handling**: no handling needed.
- **Why we check this way**: the checksums confirm that the downloaded content is byte-for-byte identical to the files on the Databento server; a complete read confirms that compression and encoding did not truncate anything.

## 2. Duplicates

- **Count**: 0. OHLCV-1m has no duplicate timestamps; MBP-1 is split by month, and no minute appears in two monthly files; the processed 1-minute data and the feature table have no duplicate timestamps.
- **Cause / handling**: as a safeguard, the processing script sorts by time and keeps the last record for any duplicate minute; this was never actually triggered.

## 3. Inconsistent formats / invalid data

| Issue | Count | Cause | Handling | Why |
|---|---|---|---|---|
| Time zone | All records | Raw timestamps are UTC (nanoseconds) | Converted to Beijing time for the index; time features and per-session statistics use ET | CME trading sessions are fixed in US time and the US has daylight saving time; only grouping by ET guarantees that one group is the same market session |
| Size field is an unsigned integer | All trades | `size` is uint32, so the subtraction (aggressive buys − aggressive sells) overflows when negative | On 2026-10-25 changed to int64 and recomputed Delta | Otherwise minutes with negative Delta become huge positive numbers |
| Invalid bars (high < low, open/close outside high-low, price ≤ 0, volume ≤ 0) | 0 / 350,611 | — | No handling needed | — |
| Flat bars (open, high, low and close equal) | 2,408 bars (0.69%) | Only one trade price in that minute, mostly in quiet periods | Kept | They are real trades, not errors |
| Crossed bid/ask per minute (best ask ≤ best bid at the close) | 0 / 353,022 | — | — | Spread, depth and OFI are computed only from valid quotes (best ask > best bid, size > 0 on both sides); the few crossed records in the raw event stream are excluded at this step |
| Minutes without trades (missing minutes) | 9,689 minutes, 2.70% of normal CME trading minutes | GC bars are built from trades, so no trade means no record; most are holiday early closes (2025-12-25, 2026-04-03, 2026-01-01, 2025-11-28) and thin trading in the old contract just before a roll (2026-05-28, 07-30) | The 2,956 intraday missing minutes where adjacent traded bars of the same contract are 2–30 minutes apart are filled as flat bars: price = previous minute's close, volume-type features = 0; book-type features are computed as usual if there are MBP-1 records, otherwise carried forward from the previous minute, OFI = 0 (`is_filled = 1`). Filled minutes are used only to complete look-back windows, not as samples. Gaps longer than 30 minutes (holiday early closes, daily breaks, weekends, etc.) are not filled and are treated as break points | The absence of trades is itself real information: the trade price did not change and nobody traded; MBP-1 is event-based, so no record means the book did not change, and the fill uses only past information. `缺失分钟检查.txt`: 0% of these minutes are on degraded-quality dates, 99.8% are within the 3 days before a roll, runs of consecutive missing records have a median of 1 minute and a maximum of 25 minutes; this is not a data outage |
| Price gaps at rolls | 5, gaps of +30 to +65 USD | At a roll the continuous contract switches from the old to the new contract, and the two have a price difference | Prices proportionally back-adjusted | Removes the false gaps; only prices after the roll are adjusted, so prices before the roll do not contain future information in advance |
| Wider spreads just before a roll | 6,906 extreme-spread minutes (≥ 20 ticks), 79% concentrated in the 1–2 days before 8 rolls | Traders have moved to the new contract, so liquidity in the old contract deteriorates | No labels in the 3 days before a roll (minute t is ≤ 72 hours before the next roll, `pre_roll_3d = 1`); other extreme spreads are kept and marked with `wide_spread`, and M4 decides whether to skip opening trades | In these days the old contract does not represent the contract that would be traded live (live trading would use the new contract); before exclusion, 79.6% of extreme-spread minutes were stop-losses (43.5% for normal minutes), which would become a shortcut and distort the winsorization thresholds and standardization parameters. After exclusion, extreme spreads are only 0.29% of training-set samples, with a stop-loss share of 46.5%, close to normal minutes. This rule is an extension of "no label if the holding period crosses a roll" |

**Cross-check**: 1-minute bars aggregated from the individual MBP-1 trades were compared minute by minute with the official OHLCV-1m: 0 differences in open, high, low, close and volume across 350,611 minutes; total volume 42,074,743 contracts.

### Roll details (instrument_id changes)

| Time (Beijing time) | Old contract → new contract | Old contract close | New contract open | Gap |
|---|---|---|---|---|
| 2025-11-27 08:00 | 39231 → 42001025 | 4162.0 | 4196.5 | +34.5 |
| 2026-01-30 08:00 | 42001025 → 42000890 | 5418.4 | 5467.5 | +49.1 |
| 2026-03-30 08:00 | 42000890 → 19181 | 4446.9 | 4477.0 | +30.1 |
| 2026-05-29 08:00 | 19181 → 42011464 | 4491.8 | 4525.5 | +33.7 |
| 2026-07-31 08:00 | 42011464 → 41271 | 4099.9 | 4164.8 | +64.9 |

5 rolls and 6 contracts in total, all at 08:00 Beijing time (00:00 UTC).

### Exclusions in the 3 days before a roll (`pre_roll_3d = 1`)

Rule: if minute t is less than 72 hours before the next roll (the first minute of the new contract), it is not used as a sample and gets no label. What is excluded is samples, not data: these minutes stay in the feature table but get no labels. Any contract change inside a sample window counts as a break point, so normal samples after a roll have windows that do not cross the roll and do not include these minutes.

| Roll time (Beijing time) | Minutes excluded | Data set | Note |
|---|---|---|---|
| 2025-11-27 08:00 | 3,647 | Training set | |
| 2026-01-30 08:00 | 3,652 | Training set | |
| 2026-03-30 08:00 | 1,208 | Training set | The roll is on a Monday; most of the preceding 72 hours is a weekend with no trading |
| 2026-05-29 08:00 | 3,286 | Training set | |
| 2026-07-31 08:00 | 3,185 | Validation set | |
| **Total** | **14,978 (4.29% of all minutes with trades)** | Training set 11,793 (4.83%), validation set 3,185 (6.09%), test set 0 | Affects 17 of 258 trading days |

There are only 5 rolls in the data period, not 6 per year. GC.v.0 selects the front-month contract by volume; the October contract has low volume and was probably skipped, rolling directly from the August to the December contract. When the data end (2026-09-29) the next roll has not yet happened, so the test set contains no pre-roll days.

Relation between filling and exclusion: of the 2,956 filled minutes, 2,674 (90.5%) are within the 3 days before a roll; the remaining 282 are spread over other days and are kept as window content.

### Trades with side = N (aggressor side unknown)

- **Count**: 855,785 trades, 2.92% of all 29,326,766 trades; 2.82% of volume. Volume with a known aggressor side is 97.18%.
- **Cause**: the exchange did not give an aggressor side. Only 0.6% of this volume occurs in the first minute after a reopen, so it is not caused by the opening auction. Databento's explanation is that these trades mainly come from matches without a single aggressor, such as leg fills of spread contracts; I did not verify this trade by trade.
- **Handling**: included in total volume, not in aggressive buy or aggressive sell volume.
- **Why**: total volume must match the official bars; forcing trades with unknown aggressor direction to one side would bias Delta.

## 4. Samples removed or not used in training

| Item | Count | Cause | Handling |
|---|---|---|---|
| Warm-up period | 1,380 rows | First trading day (2025-09-29), long-window indicators cannot be computed yet or are unstable | Removed; feature table 352,187 rows = 349,231 minutes with trades + 2,956 filled minutes (filled minutes are not samples; the label file has 349,231 rows) |
| Data set boundaries | 120 rows | 60 minutes before each of the two boundaries, whose labels would use prices from the next segment | Not in any data set |
| Empty label: no quote update at entry | 5 rows (excluding the 3 days before a roll; 737 rows before exclusion, almost all just before a roll) | No new quote at minute t+1, so the entry price is unreliable | Label set to empty |
| Empty label: break during the holding period (no quotes for more than 10 consecutive minutes) | H15 979 / H30 4,663 / H60 12,032 (excluding the 3 days before a roll) | Holding period crosses the CME daily break or a weekend | Label set to empty |
| Empty label: holding period crosses a roll | 0 (before exclusion H15 3 / H30 8 / H60 18, all already within the 3 days before a roll) | The front-month contract changes during the holding period, so prices before and after are not comparable | Label set to empty |
| Total empty labels (including the 3 days before a roll) | H15 16,017 (4.59%) / H30 19,686 (5.64%) / H60 27,025 (7.74%) | The items above overlap; the total is deduplicated by row; excluding the 3 days before a roll it is H15 1,039 (0.31%) / H30 4,708 (1.41%) / H60 12,047 (3.60%), and the remaining few dozen rows are boundary cases such as the end of the data | Samples kept; during training only the output heads with empty labels are masked (Section 5) |
| Degraded-quality days | 3,419 rows (all in the training set, about 1.4%) | The 8 UTC dates marked degraded in Databento `condition.json`: 2025-11-28, 2026-01-31, 03-15, 03-16, 03-21, 04-10, 05-24, 08-29. Of these, 01-31, 03-21 and 08-29 are Saturdays with no data; 03-15 and 05-24 are Sundays with only the 2 hours after the evening open (120 rows each). Databento does not describe the specific problem; the charts look normal and these days have no missing minutes | **Decided to keep**, marked with `flag_degraded_day`. No specific problem was found, and removing them would only reduce the data. M4 retrains once on a training set without these days as a sensitivity check |
| The 3 days before a roll | 14,978 rows (training set 4.83%, validation set 6.09%, test set 0) | The old contract is about to be abandoned; its liquidity and spread do not represent the contract that would be traded live | Label set to empty |
| Extreme spread | 6,906 rows (outside the 3 days before a roll only 0.29% of training-set samples) | See above | Labels in the 3 days before a roll are already empty; the rest are kept and marked |

## 5. Labels

- **Definition**: default setting k_SL = 2, reward-to-risk ratio 1:1; distance = 2 × ATR14 × √(H/15). Longs enter at the best ask and are judged against the best bid touching the lines; shorts the reverse. 0 = take-profit, 1 = stop-loss, 2 = timeout.
- **Distribution (training-set samples, excluding the 3 days before a roll)**: take-profit 33.3%–38.3%, stop-loss 39.2%–44.7%, timeout 22.0%–22.5%. After excluding the 3 days before a roll, the default setting was re-selected from training-set statistics, and the result is still k_SL = 2, 1:1 (the smallest class share across the 6 heads is 22.0%, the highest of all settings).
- **Candidate settings for M4**: the pass criterion is that the smallest class share across the 6 output heads is ≥ 15%. This criterion uses training-set labels only, is fixed in M3, and does not look at validation results. The take-profit distance ÷ fixed cost is at least 45 times in all 12 settings, so it filters out none of them; the share criterion is what actually matters. The shares of the 12 settings have a gap between 9.9% and 15.9%, so any threshold between 10% and 15% gives the same result; with a threshold of 20% only the default setting would remain and M4 would have no candidates to compare. The 5 settings that pass (k_SL, reward-to-risk ratio): (1.5, 1.5:1) 15.9%, (1.5, 2:1) 16.6%, (2, 1:1) 22.0%, (2, 1.5:1) 18.4%, (3, 1:1) 19.7%. M4 compares the net expected return after costs of these 5 settings on the validation set. Data: `labels/label_combo_summary.csv`.
- **Both lines touched in the same minute**: 0.02%–0.07%, recorded as stop-loss (conservative, tends to underestimate strategy performance).
- **Why stop-losses outnumber take-profits**: entry is at the best ask and exit is judged on the best bid, so the spread (median 4 ticks) pushes the take-profit line farther away and brings the stop-loss line closer.

## 6. Metadata (not model inputs)

| Field | Use |
|---|---|
| `datetime` (Beijing time) | Aligns features and labels, and splits the training/validation/test sets in time order; time features are generated from it after conversion to ET; whether they are inputs, see Section 5 |
| `symbol` (GC.v.0) | Identical for all rows, used only to identify the data |
| `instrument_id` | Identifies rolls: back-adjustment, checking whether labels cross a roll |
| `flag_degraded_day` | Marks degraded-quality days, can be used for exclusion |
| `is_filled` | 1 = filled intraday missing minute, used only to complete look-back windows, not as a sample |
| `seg_id` | Window segment number: incremented by 1 only at contract changes and at unscheduled long gaps; a look-back window is usable only if it is the same throughout |
| `pre_roll_3d` | 1 = minute t is within the 3 days before a roll, no label generated |
| `split` | Data set split; empty for the 120 rows removed at the boundaries |
| `wide_spread`, `spread_ticks`, `entry_ask`, `entry_bid` | Entry spread and entry prices, for M4 to decide whether to open a trade |
| `raw_open/high/low/close`, `raw_atr14`, `raw_ofi_depth` | Raw prices, ATR (used for labels), value before the OFI adjustment (for diagnostics) |

## 7. Label audit (random sample)

- **Method**: redone on the current data after filling and excluding the 3 days before a roll. From training-set samples not within the 3 days before a roll and with all 6 labels non-empty, 60 were drawn at random (fixed random seed 42), 360 labels in total. Each label was re-derived with a separately written minute-by-minute loop, and the best bid / best ask high-low for the 15 minutes after entry, with the take-profit and stop-loss lines, was plotted for manual inspection. All times are ET. Figures: `数据说明/标签审核/中文/标签审核_图_1.png` to `_5.png`, English version `English/Label_audit_page_1.png` to `_5.png`; details: `标签审核/标签审核_抽样60.csv`.
- **Result**: the 360 re-derived labels have **0 mismatches** with the label file.
- **Checks for suspicious cases**:
  - Both lines touched in the same minute: 0 samples;
  - Entry spread too wide: 0;
  - Minutes with carried-forward old quotes within the 60-minute holding period: 0;
  - Within 1 day before a roll: 0 (the 3 days before a roll are already excluded);
  - Degraded-quality day: 1, 2026-03-15 Sunday 22:10 (ET), entry spread 9 ticks. The chart looks normal, both sides are timeouts, and the label is correct.
- **Sample coverage**: the 60 samples run from 2025-09-30 to 2026-06-11 and cover ET hours 1–16 and 18–23 (hour 0 was not drawn; 17 is the daily break). Label distribution: H15 long take-profit 16 / stop-loss 25 / timeout 19, short 19 / 21 / 20.
- **Differences from the old version**: the old audit was done before filling and before excluding the 3 days before a roll, and used Beijing time; of the 3 suspicious samples it drew, 1 (2025-11-25 20:37 ET, one day before a roll, spread 26 ticks) no longer gets a label. The old figures and details were moved to `数据说明/_old_audit_bj/`.
- **Limitations**: the audit checks whether the labeling logic was executed correctly, and its input is the same per-minute bid/ask data; the order of events within a minute was not re-checked against the individual trade data.
