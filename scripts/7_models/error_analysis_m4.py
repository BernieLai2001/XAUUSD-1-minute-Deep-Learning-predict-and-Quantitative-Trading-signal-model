# M4 错误分析（验证集后一半 val2）：混淆矩阵、错误分类和数量、深度模型的代表性错误样本、wide_spread 和数据质量下降日的检查
#   python scripts/7_models/error_analysis_m4.py      （处理环境，需要 pyarrow；先运行 run_baseline.py 和 summarize_m4.py）
# 预测：每个模型 3 个种子的校准后概率取平均，再取概率最大的一类作为"预测类别"
# 表（results/m4/tables/）：
#   confusion_matrices.csv：每个模型、每个头的 3 × 3 混淆矩阵（行 = 真实，列 = 预测，样本数）
#   class_scores.csv：每个模型、每个头、每一类的精确率、召回率、F1
#   error_types.csv：选定深度基准（单层 LSTM + dropout 0.2）每个头的错误分类和数量
#   error_causes.csv：错误分类的可能原因的检验（波动、过去 60 分钟走势、美东小时）
#   shortcut_wide_degraded.csv：去掉"窗口里有极端点差"的样本前后的主要指标；数据质量下降日在 val2 中的样本数
# 图（results/m4/figures/{中文,English}/）：混淆矩阵、代表性错误样本
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gold_dl import evaluation as ev
from gold_dl.config import CFG, path
from gold_dl.plotting import BASE_STYLE, BLUE, FONT, INK, INK2, ORANGE
from gold_dl.timeutil import bj_to_et

R = path("results", "m4")
TAB, FIG = R / "tables", R / "figures"
plt.rcParams.update(BASE_STYLE)
SELECTED = ["logreg", "lgbm", "conv1d_do02", "lstm_do02"]        # 选定的基准（深度基准选加 dropout 的版本）
ALL = [m for m in ["naive", "logreg", "lgbm", "conv1d", "lstm", "conv1d_do02", "lstm_do02"] if (R / m).is_dir()]
DEEP = "lstm_do02"                                               # 错误分类和代表性样本用的深度基准（深度基准里最好的）
CLS = ["tp", "sl", "timeout"]

# ---------- 读预测 ----------
P = {}
for m in ALL:
    ps = [dict(np.load(p, allow_pickle=False)) for p in sorted((R / m).glob("seed*/predictions.npz"))]
    P[m] = np.mean([p["probs"] for p in ps], axis=0)
z = dict(np.load(sorted((R / ALL[0]).glob("seed*/predictions.npz"))[0], allow_pickle=False))
y, t_ns, h2 = z["y"], z["time_ns"], ~z["half1"]
HEADS = list(z["head_names"])
arr = np.load(path("model_arrays"), allow_pickle=False)
p_naive = ev.naive_probs(arr["y_train"])
rows_all = pd.Index(arr["time_ns"]).get_indexer(t_ns)          # 每个验证样本在 model_arrays / trade_outcomes 中的行号
assert (rows_all >= 0).all()
yv, tv = y[h2], pd.DatetimeIndex(t_ns[h2])


def primary(probs, yy):
    imp = []
    for j in range(6):
        m = yy[:, j] >= 0
        imp.append(1 - ev.head_logloss(probs[m, j], yy[m, j]) / ev.head_logloss(np.broadcast_to(p_naive[j], (m.sum(), 3)), yy[m, j]))
    return float(np.mean(imp))


# ---------- 1. 混淆矩阵和每类得分 ----------
cm_rows, sc_rows = [], []
for m in ALL:
    pred = P[m][h2].argmax(-1)
    for j, h in enumerate(HEADS):
        ok = yv[:, j] >= 0
        cm = np.zeros((3, 3), int)
        np.add.at(cm, (yv[ok, j], pred[ok, j]), 1)
        for a in range(3):
            cm_rows.append({"model": m, "head": h, "true": CLS[a], **{f"pred_{CLS[b]}": int(cm[a, b]) for b in range(3)}})
            prec = cm[a, a] / cm[:, a].sum() if cm[:, a].sum() else np.nan
            rec = cm[a, a] / cm[a].sum()
            f1 = 2 * prec * rec / (prec + rec) if cm[:, a].sum() and (prec + rec) else 0.0
            sc_rows.append({"model": m, "head": h, "class": CLS[a], "n_true": int(cm[a].sum()), "n_pred": int(cm[:, a].sum()),
                            "precision": prec, "recall": rec, "f1": f1})
pd.DataFrame(cm_rows).to_csv(TAB / "confusion_matrices.csv", index=False)
scores = pd.DataFrame(sc_rows)
scores.to_csv(TAB / "class_scores.csv", index=False, float_format="%.4f")

# ---------- 2. 错误分类（选定的深度基准） ----------
feat = pd.read_parquet(path("features", "features_1min.parquet"), columns=["raw_close", "raw_atr", "atr_rel", "flag_degraded_day"])
lab = pd.read_parquet(path("labels", "labels_default.parquet"), columns=["wide_spread", "entry_ask", "entry_bid"])
past60 = (feat["raw_close"] / feat["raw_close"].shift(59) - 1).reindex(tv).to_numpy()   # 过去 60 分钟的涨跌（窗口内）
atr_rel = feat["atr_rel"].reindex(tv).to_numpy()
et_hour = bj_to_et(tv).hour.to_numpy()
pd_ = P[DEEP][h2]
pred = pd_.argmax(-1)
conf = pd_.max(-1)
TYPES = {"A": ("pred tp, true sl", 0, 1), "B": ("pred sl, true tp", 1, 0),
         "C1": ("pred tp, true timeout", 0, 2), "C2": ("pred sl, true timeout", 1, 2),
         "D1": ("pred timeout, true tp", 2, 0), "D2": ("pred timeout, true sl", 2, 1)}
et_rows = []
for j, h in enumerate(HEADS):
    ok = yv[:, j] >= 0
    n_err = int((ok & (pred[:, j] != yv[:, j])).sum())
    row = {"head": h, "n": int(ok.sum()), "n_errors": n_err, "error_rate": n_err / ok.sum()}
    for k, (_, pc, tc) in TYPES.items():
        row[k] = int((ok & (pred[:, j] == pc) & (yv[:, j] == tc)).sum())
    hi = ok & (conf[:, j] >= np.quantile(conf[ok, j], 0.9))   # 最有把握的 10%
    row["top10_conf_error_rate"] = float((pred[hi, j] != yv[hi, j]).mean())
    row["true_timeout_share"] = float((yv[ok, j] == 2).mean())
    row["pred_timeout_share"] = float((pred[ok, j] == 2).mean())
    et_rows.append(row)
et = pd.DataFrame(et_rows)
tot = et[["n", "n_errors"] + list(TYPES)].sum()
et = pd.concat([et, pd.DataFrame([{"head": "all", **tot.to_dict(), "error_rate": tot.n_errors / tot.n}])], ignore_index=True)
et.to_csv(TAB / "error_types.csv", index=False, float_format="%.4f")

# 可能原因的检验：方向错误（A、B）是否集中在"顺着过去 60 分钟走势下注、随后反转"；是否集中在高波动、某些小时
cause = []
for j, h in enumerate(HEADS):
    ok = (yv[:, j] >= 0) & np.isfinite(past60)
    long_ = h.startswith("long")
    with_trend = (past60 > 0) if long_ else (past60 < 0)          # 预测止盈 = 押这个方向；与过去 60 分钟同向就是"顺势"
    A = ok & (pred[:, j] == 0) & (yv[:, j] == 1)
    predtp = ok & (pred[:, j] == 0)
    hi_vol = atr_rel >= np.nanquantile(atr_rel[ok], 0.8)
    err = ok & (pred[:, j] != yv[:, j])
    cause.append({"head": h,
                  "pred_tp_share_with_trend": with_trend[predtp].mean(),           # 预测止盈的样本中，顺势的比例
                  "all_share_with_trend": with_trend[ok].mean(),                    # 所有样本中顺势的比例
                  "A_share_with_trend": with_trend[A].mean(),                       # 把止损当止盈的样本中，顺势的比例
                  "tp_precision_with_trend": (yv[predtp & with_trend, j] == 0).mean(),
                  "tp_precision_against_trend": (yv[predtp & ~with_trend, j] == 0).mean(),
                  "error_rate_high_vol": err[hi_vol & ok].sum() / (hi_vol & ok).sum(),
                  "error_rate_other": err[~hi_vol & ok].sum() / (~hi_vol & ok).sum(),
                  "timeout_true_high_vol": (yv[hi_vol & ok, j] == 2).mean(), "timeout_true_other": (yv[~hi_vol & ok, j] == 2).mean()})
pd.DataFrame(cause).to_csv(TAB / "error_causes.csv", index=False, float_format="%.4f")
err_any = np.array([(yv[:, j] >= 0) & (pred[:, j] != yv[:, j]) for j in range(6)]).T
lab_ok = (yv >= 0)
by_h = pd.DataFrame({"et_hour": et_hour, "err": err_any.sum(1), "n": lab_ok.sum(1)}).groupby("et_hour").sum()
by_h["error_rate"] = by_h.err / by_h.n
by_h.to_csv(TAB / "error_rate_by_et_hour_lstm_do02.csv", float_format="%.4f")

# ---------- 3. 捷径检查：wide_spread、数据质量下降日 ----------
wide = lab["wide_spread"].reindex(feat.index).fillna(0).astype(int)
wide_win = wide.rolling(CFG["pipeline"]["lookback"], min_periods=1).max().reindex(tv).to_numpy().astype(bool)   # 60 分钟窗口里有极端点差
wide_entry = lab["wide_spread"].reindex(tv).fillna(0).to_numpy().astype(bool)
deg = feat["flag_degraded_day"].reindex(tv).fillna(0).to_numpy().astype(bool)
sc = []
for m in [x for x in ALL if x != "naive"]:
    pv = P[m][h2]
    sc.append({"model": m, "all": primary(pv, yv), "without_wide_window": primary(pv[~wide_win], yv[~wide_win]),
               "only_wide_window": primary(pv[wide_win], yv[wide_win])})
scd = pd.DataFrame(sc)
scd["n_all"], scd["n_wide_window"], scd["n_wide_entry"], scd["n_degraded_day_val2"] = len(yv), int(wide_win.sum()), int(wide_entry.sum()), int(deg.sum())
deg_all = feat["flag_degraded_day"] == 1
scd["degraded_days_total"] = len(set(feat.index[deg_all].date))
scd.to_csv(TAB / "shortcut_wide_degraded.csv", index=False, float_format="%.4f")

# ---------- 4. 图 ----------
q = pd.read_parquet(path("processed", "GC_quote_1min.parquet"), columns=["bid_high", "bid_low", "bid_close", "ask_close"])
q = q.reindex(pd.date_range(q.index.min(), q.index.max(), freq="min"))
TO = np.load(path("model_data", "trade_outcomes.npz"), allow_pickle=False)
HF = CFG["report"]["focus_holding_min"]
jf = HEADS.index(f"long_H{HF}")
TX = {"zh": {"dir": "中文", "cm": "混淆矩阵（验证集后一半，每行按真实类别归一化，格中为召回率和样本数）", "true": "真实", "pred": "预测",
             "cls": ["止盈", "止损", "到期"], "f_cm": "混淆矩阵_选定基准.png", "f_ex": "代表性错误样本_lstm_do02.png",
             "ex": f"单层 LSTM + dropout 0.2 的代表性错误样本（做多 {HF} 分钟，验证集后一半）", "x": "相对入场的分钟", "y": "价格（美元/盎司）",
             "mid": "中间价", "band": "持有期内买价高低", "tpl": "止盈线", "sll": "止损线", "ent": "入场价（卖价）",
             "t": {"A": "A 把止损当止盈（最有把握的一例）", "B": "B 把止盈当止损（最有把握的一例）", "C1": "C 漏掉到期（预测止盈）", "ok": "对照：预测止盈且正确"},
             "lab": "预测 止盈 {:.2f} / 止损 {:.2f} / 到期 {:.2f}，真实：{}，第 {} 分钟平仓",
             "names": {"logreg": "逻辑回归", "lgbm": "LightGBM", "conv1d_do02": "Conv1D + dropout 0.2", "lstm_do02": "单层 LSTM + dropout 0.2"},
             "head": {"long_H15": "做多 15", "long_H30": "做多 30", "long_H60": "做多 60", "short_H15": "做空 15", "short_H30": "做空 30", "short_H60": "做空 60"}},
      "en": {"dir": "English", "cm": "Confusion matrices (validation second half; rows normalised by true class: recall and count)",
             "true": "True", "pred": "Predicted", "cls": ["TP", "SL", "Timeout"], "f_cm": "confusion_matrices_selected.png",
             "f_ex": "error_examples_lstm_do02.png", "ex": f"LSTM + dropout 0.2: representative errors (long {HF} min, validation second half)",
             "x": "Minutes from entry", "y": "Price (USD/oz)", "mid": "Mid price", "band": "Bid high-low while holding",
             "tpl": "Take-profit", "sll": "Stop-loss", "ent": "Entry (ask)",
             "t": {"A": "A: SL predicted as TP (most confident)", "B": "B: TP predicted as SL (most confident)",
                   "C1": "C: missed timeout (predicted TP)", "ok": "Contrast: TP predicted, correct"},
             "lab": "pred TP {:.2f} / SL {:.2f} / TO {:.2f}; true: {}; exit at minute {}",
             "names": {"logreg": "Logistic regression", "lgbm": "LightGBM", "conv1d_do02": "Conv1D + dropout 0.2", "lstm_do02": "LSTM + dropout 0.2"},
             "head": {h: h.replace("_H", " ").replace("long", "Long").replace("short", "Short") for h in HEADS}}}

# 代表性样本：每类取模型最有把握的一个（同一交易日只取一个，避免重复）
ok = yv[:, jf] >= 0
pick = {}
for k, (pc, tc) in {"A": (0, 1), "B": (1, 0), "C1": (0, 2), "ok": (0, 0)}.items():
    cand = np.where(ok & (pred[:, jf] == pc) & (yv[:, jf] == tc))[0]
    cand = cand[np.argsort(-pd_[cand, jf, pc])]
    used = {tv[i].date() for i in pick.values()}
    pick[k] = next(i for i in cand if tv[i].date() not in used)
ex_rows = []
for k, i in pick.items():
    r = rows_all[h2][i]
    ex_rows.append({"type": k, "time_bj": str(tv[i]), "p_tp": pd_[i, jf, 0], "p_sl": pd_[i, jf, 1], "p_to": pd_[i, jf, 2],
                    "true": CLS[yv[i, jf]], "exit_min": int(TO["exit_min"][r, jf]), "past60_change": past60[i], "atr_rel": atr_rel[i]})
pd.DataFrame(ex_rows).to_csv(TAB / "error_examples_lstm_do02.csv", index=False, float_format="%.4f")

for lang, tx in TX.items():
    out = FIG / tx["dir"]
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams["font.family"] = FONT[lang]
    fig, axes = plt.subplots(len(SELECTED), 6, figsize=(15, 2.5 * len(SELECTED) + 0.6))
    cmd = pd.DataFrame(cm_rows)
    for a_, m in enumerate(SELECTED):
        for j, h in enumerate(HEADS):
            ax = axes[a_, j]
            c = cmd[(cmd.model == m) & (cmd["head"] == h)][[f"pred_{x}" for x in CLS]].to_numpy()
            rn = c / c.sum(1, keepdims=True)
            ax.imshow(rn, cmap="Blues", vmin=0, vmax=1)
            for u in range(3):
                for v in range(3):
                    ax.text(v, u, f"{rn[u, v]:.2f}\n{c[u, v]:,}", ha="center", va="center", fontsize=6.5,
                            color="white" if rn[u, v] > 0.55 else INK)
            ax.set_xticks(range(3)); ax.set_yticks(range(3)); ax.grid(False)
            ax.set_xticklabels(tx["cls"], fontsize=7); ax.set_yticklabels(tx["cls"] if j == 0 else [], fontsize=7)
            if a_ == 0:
                ax.set_title(tx["head"][h], fontsize=9, color=INK)
            if j == 0:
                ax.set_ylabel(f"{tx['names'][m]}\n{tx['true']}", fontsize=8)
            if a_ == len(SELECTED) - 1:
                ax.set_xlabel(tx["pred"], fontsize=8)
    fig.suptitle(tx["cm"], x=0.01, ha="left", color=INK, fontsize=11)
    fig.tight_layout(); fig.savefig(out / tx["f_cm"], dpi=160); plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5))
    for ax, (k, i) in zip(axes.ravel(), pick.items()):
        r = rows_all[h2][i]
        t0 = tv[i]
        e = t0 + pd.Timedelta(minutes=1)                      # 入场分钟 = 窗口最后一分钟的下一分钟（按买价卖价的开盘价）
        w = q.loc[t0 - pd.Timedelta(minutes=59): e + pd.Timedelta(minutes=HF - 1)]
        x = (w.index - e) / pd.Timedelta(minutes=1)
        ax.plot(x, (w.bid_close + w.ask_close) / 2, color=INK2, lw=1.1, label=tx["mid"])
        hold = x >= 0
        ax.fill_between(x[hold], w.bid_low[hold], w.bid_high[hold], color=BLUE, alpha=0.18, label=tx["band"])
        ent = lab.entry_ask.loc[t0]
        d = TO["r_unit"][r, jf]
        ax.axhline(ent, color=INK, lw=0.8, ls=":", label=tx["ent"])
        ax.axhline(ent + d, color="#2f9e6b", lw=1.1, ls="--", label=tx["tpl"])
        ax.axhline(ent - d, color="#d1495b", lw=1.1, ls="--", label=tx["sll"])
        ax.axvline(0, color=INK2, lw=0.6)
        ax.axvline(TO["exit_min"][r, jf] - 1, color=ORANGE, lw=0.8)
        ax.set_title(f"{tx['t'][k]}\n{t0:%Y-%m-%d %H:%M}（北京）" if lang == "zh" else f"{tx['t'][k]}\n{t0:%Y-%m-%d %H:%M} Beijing",
                     loc="left", fontsize=9, color=INK)
        ax.text(0.01, 0.02, tx["lab"].format(*pd_[i, jf], tx["cls"][yv[i, jf]], int(TO["exit_min"][r, jf])),
                transform=ax.transAxes, fontsize=8, color=INK)
        ax.set_xlabel(tx["x"]); ax.set_ylabel(tx["y"])
    axes[0, 0].legend(fontsize=7, loc="upper left")
    fig.suptitle(tx["ex"], x=0.01, ha="left", color=INK, fontsize=11)
    fig.tight_layout(); fig.savefig(out / tx["f_ex"], dpi=160); plt.close(fig)

print(et.to_string())
print(pd.DataFrame(cause).round(3).to_string())
print(scd.round(4).to_string())
print(pd.DataFrame(ex_rows).round(3).to_string())
