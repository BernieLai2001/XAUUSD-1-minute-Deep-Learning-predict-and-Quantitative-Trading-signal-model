# 汇总 M4 结果：读 results/m4/ 下每个模型、每个种子的 metrics.json 和 predictions.npz，输出报告用的表和图（中英文各一套）
#   python scripts/7_models/summarize_m4.py
# 表（results/m4/tables/）：
#   main_results.csv：所有模型在 val2 上的主要指标和次要指标（均值 ± 标准差，来自各个种子）
#   per_head_primary.csv：每个头的主要指标（相对朴素基准的 log-loss 改进）
#   per_head_class_f1.csv：每个头、每一类的 F1
#   strategy_val2.csv：策略层全部指标；theta_selection_val1.csv：θ 的选择过程
#   deep_vs_lgbm.csv：深度基准与 LightGBM 的主要指标差距 vs 种子波动
#   search_logreg.csv、search_lgbm.csv：选设置的每次尝试（val1）
#   by_et_hour.csv、by_week.csv：错误分析，按入场美东小时、按周分组的主要指标（val2，所有种子的预测取平均后计算）
# 图（results/m4/figures/{中文,English}/）：深度基准训练曲线、止盈类可靠性图、按美东小时的主要指标、6 个头 × 周的热力图
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gold_dl import evaluation as ev
from gold_dl.config import path
from gold_dl.plotting import BASE_STYLE, BLUE, FONT, INK, INK2, ORANGE
from gold_dl.timeutil import bj_to_et, trade_date

R = path("results", "m4")
TAB, FIG = R / "tables", R / "figures"
TAB.mkdir(parents=True, exist_ok=True)
plt.rcParams.update(BASE_STYLE)
MODELS = [m for m in ["naive", "logreg", "lgbm", "conv1d", "lstm", "conv1d_do02", "lstm_do02"] if (R / m).is_dir()]
NAME = {"zh": {"naive": "朴素模型", "logreg": "逻辑回归", "lgbm": "LightGBM", "conv1d": "Conv1D", "lstm": "单层 LSTM",
               "conv1d_do02": "Conv1D + dropout 0.2", "lstm_do02": "单层 LSTM + dropout 0.2"},
        "en": {"naive": "Naive", "logreg": "Logistic regression", "lgbm": "LightGBM", "conv1d": "Conv1D", "lstm": "LSTM",
               "conv1d_do02": "Conv1D + dropout 0.2", "lstm_do02": "LSTM + dropout 0.2"}}
HEAD_ZH = {"long_H15": "做多 15", "long_H30": "做多 30", "long_H60": "做多 60",
           "short_H15": "做空 15", "short_H30": "做空 30", "short_H60": "做空 60"}
COLORS = {"naive": "#9a9890", "logreg": "#7b5ea7", "lgbm": "#2f9e6b", "conv1d": BLUE, "lstm": ORANGE,
          "conv1d_do02": "#7fb0e8", "lstm_do02": "#f2a27f"}

runs = {m: [json.load(open(p)) for p in sorted((R / m).glob("seed*/metrics.json"))] for m in MODELS}
preds = {m: [dict(np.load(p, allow_pickle=False)) for p in sorted((R / m).glob("seed*/predictions.npz"))] for m in MODELS}
p_naive = None
y0 = preds[MODELS[0]][0]
y, t, h1 = y0["y"], y0["time_ns"], y0["half1"]
h2 = ~h1
HEADS = list(y0["head_names"])
arr = np.load(path("model_arrays"), allow_pickle=False)
p_naive = ev.naive_probs(arr["y_train"])


def ms(vals, fmt="{:.4f}"):
    vals = [v for v in vals if v is not None and np.isfinite(v)]
    if not vals:
        return "—"
    return fmt.format(np.mean(vals)) if len(vals) == 1 else f"{fmt.format(np.mean(vals))} ± {fmt.format(np.std(vals, ddof=1))}"


def train_time(m):
    ex = runs[m][0]["extra"]
    if m == "naive":
        return "no training"
    if "epoch_seconds_mean" in ex:
        return (f"{ms([r['extra']['epoch_seconds_mean'] for r in runs[m]], '{:.1f}')} s/epoch, "
                f"{ms([r['extra']['epochs_run'] for r in runs[m]], '{:.0f}')} epochs, local Metal")
    return f"{ms([r['extra']['train_seconds'] / 60 for r in runs[m]], '{:.1f}')} min, CPU (12 threads)"


# ---------- 主结果表 ----------
rows = []
for m in MODELS:
    c = [r["model_layer_val2_calibrated"]["summary"] for r in runs[m]]
    raw = [r["model_layer_val2_raw"]["summary"] for r in runs[m]]
    s = [r["strategy_val2"] for r in runs[m]]
    rows.append({"model": m, "n_runs": len(runs[m]),
                 "primary_calibrated": ms([x["improvement"] for x in c]), "primary_raw": ms([x["improvement"] for x in raw]),
                 "macro_f1": ms([x["macro_f1"] for x in c], "{:.3f}"), "brier": ms([x["brier"] for x in c]),
                 "auc_tp": ms([x["auc_tp"] for x in c], "{:.3f}"),
                 "mean_net_r_per_trade": ms([x["mean_net_r"] for x in s]), "trades_per_day": ms([x["trades_per_day"] for x in s], "{:.1f}"),
                 "training_time": train_time(m)})
main = pd.DataFrame(rows)
main.to_csv(TAB / "main_results.csv", index=False)

# ---------- 每个头 ----------
ph = pd.DataFrame({m: {h: ms([r["model_layer_val2_calibrated"]["per_head"][h]["improvement"] for r in runs[m]]) for h in HEADS}
                   for m in MODELS}).T
ph.to_csv(TAB / "per_head_primary.csv")
f1rows = []
for m in MODELS:
    for h in HEADS:
        d = {"model": m, "head": h}
        for k in ["f1_tp", "f1_sl", "f1_timeout", "recall_timeout", "logloss", "naive_logloss"]:
            d[k] = ms([r["model_layer_val2_calibrated"]["per_head"][h][k] for r in runs[m]])
        f1rows.append(d)
pd.DataFrame(f1rows).to_csv(TAB / "per_head_class_f1.csv", index=False)

# ---------- 策略层 ----------
srows, trows = [], []
for m in MODELS:
    for r in runs[m]:
        srows.append({"model": m, "seed": r["seed"], "theta": r["theta"], "theta_meets_n_min": r["theta_meets_n_min"],
                      "n_min": r["strategy_train_stats"]["n_min"], **r["strategy_val2"]})
        trows += [{"model": m, "seed": r["seed"], **x} for x in r["theta_selection_val1"]]
pd.DataFrame(srows).to_csv(TAB / "strategy_val2.csv", index=False)
pd.DataFrame(trows).to_csv(TAB / "theta_selection_val1.csv", index=False)

# ---------- 深度基准 vs LightGBM ----------
prim = {m: [r["model_layer_val2_calibrated"]["summary"]["improvement"] for r in runs[m]] for m in MODELS}
if "lgbm" in prim:
    cmp = []
    for m in [x for x in MODELS if x.startswith(("conv1d", "lstm"))]:
        sd = max(np.std(prim[m], ddof=1) if len(prim[m]) > 1 else 0, np.std(prim["lgbm"], ddof=1) if len(prim["lgbm"]) > 1 else 0)
        gap = np.mean(prim[m]) - np.mean(prim["lgbm"])
        cmp.append({"comparison": f"{m} - lgbm", "gap": gap, "larger_seed_sd": sd, "gap_exceeds_sd": abs(gap) > sd})
    pd.DataFrame(cmp).to_csv(TAB / "deep_vs_lgbm.csv", index=False)

# ---------- 选设置的过程 ----------
log = [json.loads(l) for l in open(R / "runs.jsonl", encoding="utf-8")]
for m in ["logreg", "lgbm"]:
    rr = [{"stage": x["stage"], "input": x.get("input"), **x.get("params", {}), "seed": x.get("seed"),
           "val1_primary": x.get("val1_primary"), "time_s": x.get("time_s"),
           **({"all_converged": x["all_converged"]} if "all_converged" in x else {}),
           **({"trees": x["best_iterations"]} if "best_iterations" in x else {})} for x in log if x["model"] == m]
    if rr:
        pd.DataFrame(rr).to_csv(TAB / f"search_{m}.csv", index=False)


# ---------- 错误分析：按美东小时、按周（val2，各种子的概率取平均） ----------
def mean_probs(m):
    return np.mean([p["probs"] for p in preds[m]], axis=0)


def primary_on(mask, probs):
    vals = []
    for j in range(len(HEADS)):
        mm = mask & (y[:, j] >= 0)
        if mm.sum() < 30:
            vals.append(np.nan)
            continue
        ll = ev.head_logloss(probs[mm, j], y[mm, j])
        ll0 = ev.head_logloss(np.broadcast_to(p_naive[j], (mm.sum(), 3)), y[mm, j])
        vals.append(1 - ll / ll0)
    return vals


et = bj_to_et(pd.DatetimeIndex(t))
hours = [h for h in range(24) if h != 17]
week = trade_date(pd.DatetimeIndex(t)).to_period("W-FRI").start_time   # CME 交易周：周日晚开盘到周五
weeks = sorted(week[h2].unique())
hour_rows, week_rows = [], []
for m in MODELS:
    if m == "naive":
        continue
    P = mean_probs(m)
    for hr in hours:
        v = primary_on(h2 & (et.hour == hr), P)
        hour_rows.append({"model": m, "et_hour": hr, "n": int((h2 & (et.hour == hr)).sum()), "primary": np.nanmean(v)})
    for wk in weeks:
        v = primary_on(h2 & (week == wk), P)
        week_rows.append({"model": m, "week_start": str(pd.Timestamp(wk).date()), **dict(zip(HEADS, v))})
by_hour = pd.DataFrame(hour_rows)
by_week = pd.DataFrame(week_rows)
by_hour.to_csv(TAB / "by_et_hour.csv", index=False)
by_week.to_csv(TAB / "by_week.csv", index=False)

# ---------- 图 ----------
TXT = {"zh": {"dir": "中文", "epoch": "epoch", "loss": "屏蔽空标签后的交叉熵", "train": "训练集", "val": "验证集前一半",
              "curve": "{}：训练曲线（3 个种子）", "rel": "止盈类可靠性图（验证集后一半，校准后）", "pred": "预测的止盈概率",
              "act": "实际止盈比例", "diag": "完全校准", "hour": "按入场美东小时的主要指标（验证集后一半）",
              "hx": "美东时间（小时）", "hy": "6 个头 log-loss 相对朴素基准的平均改进", "heat": "{}：每个头、每周的主要指标（验证集后一半）",
              "wk": "交易周（起始日）", "f_curve": "训练曲线_{}.png", "f_rel": "可靠性图_止盈类.png", "f_hour": "按美东小时_主要指标.png",
              "f_heat": "热力图_头x周_{}.png", "head": HEAD_ZH},
       "en": {"dir": "English", "epoch": "Epoch", "loss": "Masked cross-entropy", "train": "Training set", "val": "Validation, first half",
              "curve": "{}: training curves (3 seeds)", "rel": "Reliability of the take-profit class (validation second half, calibrated)",
              "pred": "Predicted take-profit probability", "act": "Observed take-profit share", "diag": "Perfect calibration",
              "hour": "Primary metric by entry ET hour (validation second half)", "hx": "ET hour",
              "hy": "Mean log-loss improvement over naive", "heat": "{}: primary metric per head and week (validation second half)",
              "wk": "Trading week (start)", "f_curve": "training_curves_{}.png", "f_rel": "reliability_take_profit.png",
              "f_hour": "primary_by_ET_hour.png", "f_heat": "heatmap_head_x_week_{}.png",
              "head": {h: h.replace("_H", " ").replace("long", "Long").replace("short", "Short") for h in HEADS}}}
for lang, tx in TXT.items():
    out = FIG / tx["dir"]
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams["font.family"] = FONT[lang]
    nm = NAME[lang]
    for m in [x for x in MODELS if x.startswith(("conv1d", "lstm"))]:
        fig, ax = plt.subplots(figsize=(7, 4))
        for k, p in enumerate(sorted((R / m).glob("seed*/history.json"))):
            h = json.load(open(p))
            ep = np.arange(1, len(h["loss"]) + 1)
            ax.plot(ep, h["loss"], color=BLUE, lw=1.4, alpha=0.85, label=tx["train"] if k == 0 else None)
            ax.plot(ep, h["val_loss"], color=ORANGE, lw=1.4, alpha=0.85, label=tx["val"] if k == 0 else None)
            b = int(np.argmin(h["val_loss"]))
            ax.scatter([b + 1], [h["val_loss"][b]], color=ORANGE, s=18, zorder=3)
        ax.set_xlabel(tx["epoch"]); ax.set_ylabel(tx["loss"]); ax.legend()
        ax.set_title(tx["curve"].format(nm[m]), loc="left", color=INK)
        fig.tight_layout(); fig.savefig(out / tx["f_curve"].format(m), dpi=170); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5.2))
    ax.plot([0, 0.8], [0, 0.8], color=INK2, ls="--", lw=1, label=tx["diag"])
    for m in [x for x in MODELS if x != "naive"]:
        rel = ev.reliability(mean_probs(m)[h2], y[h2], cls=0, bins=10)
        rel = rel[rel["n"] >= 200]
        ax.plot(rel["pred"], rel["actual"], marker="o", ms=3.5, lw=1.4, color=COLORS[m], label=nm[m])
    ax.axhline(np.mean(y[h2][y[h2] >= 0] == 0), color="#c3c2b7", lw=0.8)
    ax.set_xlabel(tx["pred"]); ax.set_ylabel(tx["act"]); ax.legend(fontsize=8)
    ax.set_title(tx["rel"], loc="left", color=INK, fontsize=10)
    fig.tight_layout(); fig.savefig(out / tx["f_rel"], dpi=170); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4.2))
    x = np.arange(len(hours))
    for m in [x_ for x_ in MODELS if x_ != "naive"]:
        d = by_hour[by_hour.model == m]
        ax.plot(x, d["primary"], marker="o", ms=3, lw=1.3, color=COLORS[m], label=nm[m])
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(hours)
    ax.set_xlabel(tx["hx"]); ax.set_ylabel(tx["hy"]); ax.legend(ncol=3, fontsize=8)
    ax.set_title(tx["hour"], loc="left", color=INK)
    fig.tight_layout(); fig.savefig(out / tx["f_hour"], dpi=170); plt.close(fig)

    for m in [x_ for x_ in MODELS if x_ != "naive"]:
        d = by_week[by_week.model == m].set_index("week_start")[HEADS].T
        fig, ax = plt.subplots(figsize=(1.3 * len(d.columns) + 3, 3.6))
        lim = np.nanmax(np.abs(d.to_numpy())) or 0.01
        im = ax.imshow(d.to_numpy(), cmap="RdBu", vmin=-lim, vmax=lim, aspect="auto")
        ax.set_yticks(range(len(HEADS))); ax.set_yticklabels([tx["head"][h] for h in HEADS])
        ax.set_xticks(range(len(d.columns))); ax.set_xticklabels(d.columns, rotation=0, fontsize=8)
        for i in range(d.shape[0]):
            for j in range(d.shape[1]):
                v = d.iat[i, j]
                ax.text(j, i, "—" if np.isnan(v) else f"{v:+.3f}", ha="center", va="center", fontsize=7.5, color=INK)
        ax.grid(False); ax.set_xlabel(tx["wk"])
        fig.colorbar(im, ax=ax, fraction=0.03)
        ax.set_title(tx["heat"].format(nm[m]), loc="left", color=INK, fontsize=10)
        fig.tight_layout(); fig.savefig(out / tx["f_heat"].format(m), dpi=170); plt.close(fig)

pd.set_option("display.width", 220)
print(main.to_string(index=False))
print(ph.to_string())
print(pd.DataFrame(srows).groupby("model")[["theta", "n_trades", "mean_net_r", "win_rate", "profit_factor", "max_drawdown_r", "cum_net_r"]].mean().round(4))
