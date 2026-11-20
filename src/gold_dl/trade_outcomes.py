# 默认组合（labels.default_k_sl、default_rr）下每个样本"如果在这一分钟开仓"的交易结果，供策略层回测用
# 入场、碰线规则与 labels.py 完全相同：t+1 分钟开盘入场，做多按卖一价入场、看买一价碰线，做空相反；同一分钟两条线都碰到记为止损
# 每个样本、每个头输出：
#   outcome：0 = 止盈、1 = 止损、2 = 到期、−1 = 标签为空（与 labels_default.parquet 逐个核对，必须完全一致）
#   exit_min：平仓所在分钟距第 t 分钟的分钟数（1 … H）；碰线 = 碰线那一分钟，到期 = 持有期最后一分钟（按该分钟收盘价平仓）
#   pnl_usd：每盎司毛盈亏（美元，已含点差，未扣固定成本）；止盈 = +止盈距离，止损 = −止损距离（按线上价格成交，不计滑点），
#            到期：做多 = 最后一分钟买一收盘价 − 入场卖一价，做空 = 入场买一价 − 最后一分钟卖一收盘价
#   r_unit：1R = 止损距离（美元/盎司）= k_SL × ATR × √(H / H0)
# 输出：model_data/trade_outcomes.npz，行与 model_arrays.npz 的 X_all 对齐（按 time_ns），补出的分钟和无标签的行为 −1 / NaN
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from gold_dl.config import CFG, path, save_run_config
from gold_dl.labels import FFILL_LIMIT, HEADS, HS, K_DEFAULT, RR_DEFAULT, sl_dist


def build_trade_outcomes():
    feat = pd.read_parquet(path("features", "features_1min.parquet"), columns=["raw_atr", "is_filled"])
    lab = pd.read_parquet(path("labels", "labels_default.parquet"))
    q = pd.read_parquet(path("processed", "GC_quote_1min.parquet"))
    grid = pd.date_range(q.index.min(), q.index.max(), freq="min")
    q = q.reindex(grid)
    for side in ["bid", "ask"]:   # 与 labels.py 相同：没有报价更新的分钟最多沿用前 FFILL_LIMIT 分钟的收盘价
        last = q[f"{side}_close"].ffill(limit=FFILL_LIMIT)
        for c in ["open", "high", "low", "close"]:
            q[f"{side}_{c}"] = q[f"{side}_{c}"].fillna(last)

    rows = lab.index                       # 有成交的分钟（标签文件的行）
    pos = grid.get_indexer(rows)
    ok = (pos >= 0) & (pos + 1 + max(HS) <= len(grid))
    e = np.where(ok, pos, 0) + 1           # 入场分钟在网格里的位置
    atr = feat["raw_atr"].reindex(rows).to_numpy()
    entry_long, entry_short = lab["entry_ask"].to_numpy(), lab["entry_bid"].to_numpy()
    pad = lambda col: np.concatenate([q[col].to_numpy(), np.full(max(HS), np.nan)])

    n = len(rows)
    out = {k: np.full((n, len(HEADS)), v, dtype=t) for k, v, t in
           [("outcome", -1, np.int8), ("exit_min", -1, np.int16), ("pnl_usd", np.nan, np.float32), ("r_unit", np.nan, np.float32)]}
    for H in HS:
        w = {c: sliding_window_view(pad(c), H)[e] for c in ["bid_high", "bid_low", "ask_high", "ask_low", "bid_close", "ask_close"]}
        d_sl = sl_dist(atr, H, K_DEFAULT)
        d_tp = RR_DEFAULT * d_sl
        for side in ["long", "short"]:
            j = HEADS.index(f"{side}_H{H}")
            if side == "long":
                tp_hit, sl_hit = w["bid_high"] >= (entry_long + d_tp)[:, None], w["bid_low"] <= (entry_long - d_sl)[:, None]
                timeout_pnl = w["bid_close"][:, -1] - entry_long
            else:
                tp_hit, sl_hit = w["ask_low"] <= (entry_short - d_tp)[:, None], w["ask_high"] >= (entry_short + d_sl)[:, None]
                timeout_pnl = entry_short - w["ask_close"][:, -1]
            tp = np.where(tp_hit.any(axis=1), tp_hit.argmax(axis=1), 10_000)
            sl = np.where(sl_hit.any(axis=1), sl_hit.argmax(axis=1), 10_000)
            res = np.where(sl <= tp, np.where(sl < 10_000, 1, 2), 0)
            exit_off = np.where(res == 0, tp, np.where(res == 1, sl, H - 1)) + 1   # 入场分钟 = t + 1
            pnl = np.where(res == 0, d_tp, np.where(res == 1, -d_sl, timeout_pnl))
            valid = lab[HEADS[j]].notna().to_numpy()
            out["outcome"][valid, j] = res[valid]
            out["exit_min"][valid, j] = exit_off[valid]
            out["pnl_usd"][valid, j] = pnl[valid]
            out["r_unit"][valid, j] = d_sl[valid]

    # 核对：结局与标签文件完全一致
    y = lab[HEADS].astype("float").fillna(-1).to_numpy().astype(np.int8)
    mismatch = int((out["outcome"] != y).sum())
    assert mismatch == 0, f"{mismatch} 个结局与 labels_default.parquet 不一致"
    assert np.isfinite(out["pnl_usd"][y >= 0]).all(), "有标签的样本盈亏不能为空"

    # 对齐到 model_arrays.npz 的全部行（含补出的分钟）
    t_all = np.load(path("model_arrays"))["time_ns"]
    idx = pd.Index(t_all).get_indexer(rows.asi8)
    assert (idx >= 0).all()
    full = {k: np.full((len(t_all), len(HEADS)), -1 if v.dtype.kind == "i" else np.nan, dtype=v.dtype) for k, v in out.items()}
    for k, v in out.items():
        full[k][idx] = v
    np.savez(path("model_data", "trade_outcomes.npz"), time_ns=t_all, head_names=np.array(HEADS), **full)
    save_run_config(path("model_data"), "make_trade_outcomes")
    print("trade outcomes saved; labels matched:", int((y >= 0).sum()), "head-samples")
    return full
