# 三重障碍标签（按 GC 真实买卖价判断碰线）
# 入场：t+1 分钟开盘时的盘口；做多按卖一价入场，看买一价碰线；做空按买一价入场，看卖一价碰线
# 距离：止损 = k_SL × ATR_t × √(H/H0)，止盈 = 盈亏比 × 止损距离；H0 = labels.base_holding_min（15 分钟），
#   ATR_t 是 GC 期货最近 labels.atr_period（14）根 1 分钟 K 线 TR 的简单平均（美元）
# 编码：0 = 止盈，1 = 止损，2 = 到期都没碰到；同一分钟两条线都碰到记为止损；无法判断的为空
# 以下情况标签为空：t+1 分钟没有报价更新、持有期内有超过 10 分钟没有报价（如 CME 每日休市）、持有期跨合约换月、
#   第 t 分钟在换月前 pre_roll_days 天内（旧合约快被弃用，流动性和点差都不代表实盘会交易的合约；pre_roll = 1）
# 样本只取有成交的分钟；特征文件里补出的分钟（is_filled = 1）只用来补全回看窗口，不生成标签
# 调用：scripts/make_labels.py（正式版，剔除换月前 pre_roll_days 天）、scripts/make_labels_with_preroll_check.py（对照版，不剔除）
# 本文件也提供各脚本共用的标签参数和小工具（HEADS、label_col、sl_dist、wide_threshold 等），画图和检查脚本从这里取，不再各写一份
import os

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from gold_dl.config import CFG, path, save_run_config
from gold_dl.timeutil import bj_to_et

# 参数都来自 config.yaml
K_SL = CFG["labels"]["k_sl_grid"]
RR = CFG["labels"]["rr_grid"]
HS = CFG["labels"]["holding_min"]
TICK = CFG["data"]["tick"]
TRAIN_FRAC, VAL_FRAC = CFG["split"]["train_frac"], CFG["split"]["val_frac"]   # 按时间顺序：训练、验证，其余为测试
GAP = CFG["split"]["gap_min"]      # 每个分界处删掉的分钟数，避免前一段的标签用到后一段的价格
FFILL_LIMIT = CFG["cleaning"]["quote_ffill_limit_min"]   # 没有报价更新的分钟，最多沿用前几分钟的盘口
WIDE_MULT = CFG["cleaning"]["wide_spread_mult"]          # 极端点差阈值 = 训练集点差中位数的倍数
# 固定成本（每手 100 盎司，单边）：交易所费用 1.65 + NFA 0.02 + 佣金 0.85 美元（非会员费率、IBKR 最低档，需按实际券商核对）
COST_PER_SIDE = CFG["labels"]["cost_per_side_usd"]
COST_RT_PER_OZ = 2 * COST_PER_SIDE / 100
BASE_H = CFG["labels"]["base_holding_min"]    # k_SL 和盈亏比在这个持有期上定义，其他持有期的止损按 √(H / BASE_H) 放大
K_DEFAULT, RR_DEFAULT = CFG["labels"]["default_k_sl"], CFG["labels"]["default_rr"]   # 默认组合（labels_default.parquet）
ATR_N = CFG["labels"]["atr_period"]           # ATR 周期，只用于文字说明（列名 raw_atr 不带周期）
PRE_ROLL_DAYS = CFG["cleaning"]["pre_roll_days"]
FOCUS_H = CFG["report"]["focus_holding_min"]  # 单个持有期的图和检查用哪个持有期
SIDES = ["long", "short"]
HEADS = [f"{s}_H{h}" for s in SIDES for h in HS]   # 6 个输出头，顺序与模型输出一致
assert GAP >= max(HS), "split.gap_min 不能小于最长持有期，否则分界前的标签会用到下一段的价格"
assert FOCUS_H in HS, "report.focus_holding_min 必须是 labels.holding_min 中的一个"
assert K_DEFAULT in K_SL and RR_DEFAULT in RR, "默认组合必须在 k_sl_grid × rr_grid 里"


def label_col(side, H, k, rr):
    """labels_all.parquet 里某个组合的列名，例如 long_H30_sl2_rr1"""
    return f"{side}_H{H}_sl{k:g}_rr{rr:g}"


def sl_dist(atr, H, k):
    """止损距离（美元）= k × ATR × √(H / BASE_H)；止盈距离 = 盈亏比 × 止损距离"""
    return k * atr * np.sqrt(H / BASE_H)


def wide_threshold(spread_ticks, split, pre_roll):
    """极端点差阈值（跳）= cleaning.wide_spread_mult × 训练集样本（不在换月前）入场点差的中位数，与标签文件的 wide_spread 一致"""
    m = (np.asarray(split).astype(str) == "train") & (np.asarray(pre_roll) == 0)
    return WIDE_MULT * np.nanmedian(np.asarray(spread_ticks, dtype=float)[m])
assert abs(CFG["split"]["train_frac"] + CFG["split"]["val_frac"] + CFG["split"]["test_frac"] - 1) < 1e-9


def build_labels(out, pre_roll_days=None):
    """生成全部 (方向, 持有期, k_SL, 盈亏比) 组合的标签和统计表，写到 out 目录
    pre_roll_days：第 t 分钟距下一次换月不超过这么多天时不生成标签；默认取 config.yaml 的
    cleaning.pre_roll_days，0 = 不剔除（对照版）"""
    if pre_roll_days is None:
        pre_roll_days = CFG["cleaning"]["pre_roll_days"]
    OUT = out
    PRE_ROLL = pd.Timedelta(days=pre_roll_days)
    os.makedirs(OUT, exist_ok=True)

    feat = pd.read_parquet(path("features", "features_1min.parquet"), columns=["raw_atr", "instrument_id", "is_filled"])
    feat = feat[feat["is_filled"] == 0]
    # 换月前 pre_roll_days 天：换月时刻 = 新合约的第一分钟；合约换月日程事先公布，只用于筛选样本，不作为特征
    roll_t = feat.index[(feat["instrument_id"] != feat["instrument_id"].shift()).to_numpy() & (np.arange(len(feat)) > 0)]
    nxt = np.searchsorted(roll_t.values, feat.index.values, side="right")
    next_roll = pd.DatetimeIndex(np.where(nxt < len(roll_t), roll_t.values[np.minimum(nxt, len(roll_t) - 1)], np.datetime64("NaT")))
    pre_roll = np.asarray((next_roll - feat.index) <= PRE_ROLL)
    q = pd.read_parquet(path("processed", "GC_quote_1min.parquet"))

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
    atr = feat["raw_atr"].to_numpy()

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
            d_sl = sl_dist(atr, H, k)
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
                    name = label_col(side, H, k, rr)
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
    wide = entry_ok & (spread_ticks > thr)   # 换月前几天的标签已为空，这里仍标出，便于统计
    idx = feat.index
    et = bj_to_et(idx)
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
    out["pre_roll"] = pre_roll.astype("int8")
    out["split"] = pd.Categorical(split)   # 空字符串 = 分界处删除的 60 分钟
    out.to_parquet(f"{OUT}/labels_all.parquet")
    save_run_config(OUT, "make_labels")

    print(f"rows {N}, split counts", pd.Series(split).value_counts().to_dict(), "val starts", t_val, "test starts", t_test)
    print(f"entry ok {entry_ok.mean():.4f}, pre-roll {pre_roll_days}d rows {pre_roll.sum()} ({pre_roll.mean():.4f}), by split",
          pd.Series(pre_roll).groupby(split).mean().round(4).to_dict())
    print(f"fixed cost round trip {COST_RT_PER_OZ:.4f} USD/oz")
    print(f"spread median (train) {med:.1f} ticks, threshold {thr:.1f} ticks, wide minutes {wide.sum()} ({wide.sum()/entry_ok.sum():.4%})")
    print(combo.to_string())
