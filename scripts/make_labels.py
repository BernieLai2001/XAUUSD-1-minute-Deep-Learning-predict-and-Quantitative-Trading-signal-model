# 三重障碍标签（按 GC 真实买卖价判断碰线）
# 入场：t+1 分钟开盘时的盘口；做多按卖一价入场，看买一价碰线；做空按买一价入场，看卖一价碰线
# 距离：止损 = k_SL × ATR_t × √(H/15)，止盈 = 盈亏比 × 止损距离；ATR_t 是 GC 期货最近 14 根 1 分钟 K 线 TR 的简单平均（美元）
# 编码：0 = 止盈，1 = 止损，2 = 到期都没碰到；同一分钟两条线都碰到记为止损；无法判断的为空
# 以下情况标签为空：t+1 分钟没有报价更新、持有期内有超过 10 分钟没有报价（如 CME 每日休市）、持有期跨合约换月、
#   第 t 分钟在换月前 3 天内（旧合约快被弃用，流动性和点差都不代表实盘会交易的合约；pre_roll_3d = 1）
# 样本只取有成交的分钟；特征文件里补出的分钟（is_filled = 1）只用来补全回看窗口，不生成标签
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from pathlib import Path
P = str(Path(__file__).resolve().parent.parent)

OUT = f"{P}/labels"
K_SL = [1, 1.5, 2, 3]
RR = [1, 1.5, 2]
HS = [15, 30, 60]
TICK = 0.1
TRAIN_FRAC, VAL_FRAC = 0.7, 0.15   # 按时间顺序：前 70% 训练，接着 15% 验证，最后 15% 测试
GAP = max(HS)                      # 每个分界处删掉 60 分钟，避免前一段的标签用到后一段的价格
PRE_ROLL = pd.Timedelta(days=3)   # 换月前 3 天不生成标签
FFILL_LIMIT = 10      # 没有报价更新的分钟，最多沿用前 10 分钟的盘口
WIDE_MULT = 5         # 极端点差阈值 = 训练集点差中位数的 5 倍
# 固定成本（每手 100 盎司，单边）：交易所费用 1.65 + NFA 0.02 + 佣金 0.85 美元（非会员费率、IBKR 最低档，需按实际券商核对）
COST_PER_SIDE = 1.65 + 0.02 + 0.85
COST_RT_PER_OZ = 2 * COST_PER_SIDE / 100

import os
os.makedirs(OUT, exist_ok=True)

feat = pd.read_parquet(f"{P}/features/features_1min.parquet", columns=["raw_atr14", "instrument_id", "is_filled"])
feat = feat[feat["is_filled"] == 0]
# 换月前 3 天：换月时刻 = 新合约的第一分钟；合约换月日程事先公布，只用于筛选样本，不作为特征
roll_t = feat.index[(feat["instrument_id"] != feat["instrument_id"].shift()).to_numpy() & (np.arange(len(feat)) > 0)]
nxt = np.searchsorted(roll_t.values, feat.index.values, side="right")
next_roll = pd.DatetimeIndex(np.where(nxt < len(roll_t), roll_t.values[np.minimum(nxt, len(roll_t) - 1)], np.datetime64("NaT")))
pre_roll = np.asarray((next_roll - feat.index) <= PRE_ROLL)
q = pd.read_parquet(f"{P}/GC_processed/GC_quote_1min.parquet")

grid = pd.date_range(q.index.min(), q.index.max(), freq="min")
q = q.reindex(grid)
has_update = q["n"].notna().to_numpy()
for side in ["bid", "ask"]:
    last = q[f"{side}_close"].ffill(limit=FFILL_LIMIT)
    for c in ["open", "high", "low", "close"]:
        q[f"{side}_{c}"] = q[f"{side}_{c}"].fillna(last)
inst = q["instrument_id"].ffill(limit=FFILL_LIMIT).to_numpy()

# 每个特征行在分钟网格里的位置
pos = grid.get_indexer(feat.index)
ok_pos = (pos >= 0) & (pos + 1 + max(HS) <= len(grid))
pos_safe = np.where(ok_pos, pos, 0)
e = pos_safe + 1                                  # 入场分钟
N = len(feat)
atr = feat["raw_atr14"].to_numpy()

entry_long = q["ask_open"].to_numpy()[e]
entry_short = q["bid_open"].to_numpy()[e]
spread_ticks = (entry_long - entry_short) / TICK
entry_ok = ok_pos & has_update[e]
sample_ok = entry_ok & ~pre_roll


def windows(col, H):
    a = q[col].to_numpy()
    a = np.concatenate([a, np.full(max(HS), np.nan)])
    return sliding_window_view(a, H)[e]           # 第 i 行 = 入场分钟起的 H 分钟


inst_pad = np.concatenate([inst, np.full(max(HS), np.nan)])


def first_hit(hit):
    any_ = hit.any(axis=1)
    return np.where(any_, hit.argmax(axis=1), 10_000)


n_train, n_val = int(N * TRAIN_FRAC), int(N * VAL_FRAC)
t_val, t_test = feat.index[n_train], feat.index[n_train + n_val]
gap = pd.Timedelta(minutes=GAP)
split = np.full(N, "", dtype=object)
split[feat.index < t_val - gap] = "train"
split[(feat.index >= t_val) & (feat.index < t_test - gap)] = "val"
split[feat.index >= t_test] = "test"
is_train = split == "train"
labels = {}
rows = []
for H in HS:
    w = {c: windows(c, H) for c in ["bid_high", "bid_low", "ask_high", "ask_low"]}
    iw = sliding_window_view(inst_pad, H)[e]
    win_ok = sample_ok & np.isfinite(w["bid_low"]).all(axis=1) & (iw == iw[:, :1]).all(axis=1)
    for k in K_SL:
        d_sl = k * atr * np.sqrt(H / 15)  # k_SL 和盈亏比在 15 分钟持有期上定义，30、60 分钟按 √(H/15) 放大
        for rr in RR:
            d_tp = rr * d_sl
            for side in ["long", "short"]:
                if side == "long":
                    tp = first_hit(w["bid_high"] >= (entry_long + d_tp)[:, None])
                    sl = first_hit(w["bid_low"] <= (entry_long - d_sl)[:, None])
                else:
                    tp = first_hit(w["ask_low"] <= (entry_short - d_tp)[:, None])
                    sl = first_hit(w["ask_high"] >= (entry_short + d_sl)[:, None])
                lab = np.where(sl <= tp, np.where(sl < 10_000, 1, 2), 0).astype("float32")
                lab[~win_ok] = np.nan
                name = f"{side}_H{H}_sl{k:g}_rr{rr:g}"
                labels[name] = lab
                m = is_train & win_ok
                lt = lab[m]
                rows.append({
                    "side": side, "H": H, "k_SL": k, "rr": rr, "k_TP": k * rr,
                    "n_train": int(m.sum()),
                    "tp": (lt == 0).mean(), "sl": (lt == 1).mean(), "timeout": (lt == 2).mean(),
                    "same_bar_both": ((tp == sl) & (sl < 10_000))[m].mean(),
                    "tp_dist_median_usd": np.median(d_tp[m]),
                    "tp_over_fixed_cost": np.median(d_tp[m]) / COST_RT_PER_OZ,
                    "tp_over_spread": np.median(d_tp[m] / (spread_ticks[m] * TICK)),
                })
    print("H", H, "done", flush=True)

stats = pd.DataFrame(rows)
stats.to_csv(f"{OUT}/label_grid_stats.csv", index=False, float_format="%.4f")

# 每组 (k_SL, 盈亏比) 在 6 个输出头上的均衡程度：取每个头三类中最小的比例，再在 6 个头里取最小
ent = lambda r: -sum(p * np.log(p) for p in [r.tp, r.sl, r.timeout] if p > 0) / np.log(3)
stats["entropy"] = stats.apply(ent, axis=1)
combo = stats.groupby(["k_SL", "rr"]).agg(
    min_class_share=("tp", lambda s: stats.loc[s.index, ["tp", "sl", "timeout"]].min(axis=1).min()),
    mean_entropy=("entropy", "mean"),
    min_tp_over_fixed_cost=("tp_over_fixed_cost", "min"),
    min_tp_over_spread=("tp_over_spread", "min"),
).reset_index()
combo.to_csv(f"{OUT}/label_combo_summary.csv", index=False, float_format="%.4f")

# 极端点差统计（入场分钟的点差）
med = np.nanmedian(spread_ticks[is_train & sample_ok])
thr = WIDE_MULT * med
wide = entry_ok & (spread_ticks > thr)   # 换月前 3 天的标签已为空，这里仍标出，便于统计
idx = feat.index
et = idx.tz_localize("Asia/Shanghai").tz_convert("America/New_York")
sp = pd.DataFrame({"wide": wide[entry_ok], "hour_bj": idx.hour[entry_ok], "hour_et": et.hour[entry_ok]})
by_hour = sp.groupby("hour_et").agg(minutes=("wide", "size"), wide=("wide", "sum"))
by_hour["wide_share"] = by_hour["wide"] / by_hour["minutes"]
by_hour["hour_bj"] = (by_hour.index + 12) % 24  # 仅作参考，夏令时为 +12，冬令时为 +13
by_hour.to_csv(f"{OUT}/wide_spread_by_hour_et.csv", float_format="%.4f")
by_day = pd.Series(wide, index=idx).groupby(idx.date).sum().sort_values(ascending=False)
by_day[by_day > 0].to_csv(f"{OUT}/wide_spread_by_day.csv", header=["wide_minutes"])

out = pd.DataFrame(labels, index=feat.index).astype("Int8")
out["entry_ask"] = np.where(entry_ok, entry_long, np.nan)
out["entry_bid"] = np.where(entry_ok, entry_short, np.nan)
out["spread_ticks"] = np.where(entry_ok, spread_ticks, np.nan)
out["wide_spread"] = wide.astype("int8")
out["pre_roll_3d"] = pre_roll.astype("int8")
out["split"] = pd.Categorical(split)   # 空字符串 = 分界处删除的 60 分钟
out.to_parquet(f"{OUT}/labels_all.parquet")

print(f"rows {N}, split counts", pd.Series(split).value_counts().to_dict(), "val starts", t_val, "test starts", t_test)
print(f"entry ok {entry_ok.mean():.4f}, pre-roll 3d rows {pre_roll.sum()} ({pre_roll.mean():.4f}), by split",
      pd.Series(pre_roll).groupby(split).mean().round(4).to_dict())
print(f"fixed cost round trip {COST_RT_PER_OZ:.4f} USD/oz")
print(f"spread median (train) {med:.1f} ticks, threshold {thr:.1f} ticks, wide minutes {wide.sum()} ({wide.sum()/entry_ok.sum():.4%})")
print(combo.to_string())
