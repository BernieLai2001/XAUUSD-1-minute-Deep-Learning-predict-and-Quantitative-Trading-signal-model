# 按时间统计 GC 点差和波动率（仅训练集），输出报告第 2 节用的图和表
# 点差：每分钟平均点差（GC_book_1min 的 spread_sum / spread_n，ticks，1 tick = config.yaml 的 data.tick 美元）；
#   极端点差占比用标签里的 wide_spread（入场点差 > wide_spread_mult × 训练集中位数）
# 波动率：1 分钟 K 线振幅（最高价 − 最低价，美元，未复权）；另附 ATR（美元，周期见 labels.atr_period）
# 时间一律换成美东时间（CME 交易时段按美东时间固定，夏令时自动处理）；17:00–18:00 ET 每日休市
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gold_dl.config import path
from gold_dl.timeutil import bj_to_et
from gold_dl.plotting import BLUE, CORE_STYLE, FONT, INK, INK2, ORANGE
from gold_dl.labels import ATR_N, WIDE_MULT, wide_threshold

OUT = path("notes")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.family": FONT["zh"]} | CORE_STYLE)

f = pd.read_parquet(path("features", "features_1min.parquet"),
                    columns=["raw_high", "raw_low", "raw_atr"])
book = pd.read_parquet(path("processed", "GC_book_1min.parquet"), columns=["spread_sum", "spread_n"])
f["gc_spread_mean"] = (book["spread_sum"] / book["spread_n"].replace(0, np.nan)).reindex(f.index).ffill()
lab = pd.read_parquet(path("labels", "labels_all.parquet"), columns=["wide_spread", "spread_ticks", "split", "pre_roll"])
WIDE_TICKS = wide_threshold(lab["spread_ticks"], lab["split"], lab["pre_roll"])
d = f.join(lab)
d = d[d["split"] == "train"]
et = bj_to_et(d.index)
d = pd.DataFrame({
    "spread": d["gc_spread_mean"].to_numpy(),
    "range": (d["raw_high"] - d["raw_low"]).to_numpy(),
    "atr": d["raw_atr"].to_numpy(),
    "wide": d["wide_spread"].to_numpy(),
    "entry_ok": d["spread_ticks"].notna().to_numpy(),
    "hour": et.hour, "wd": et.dayofweek,   # 0 = 周一 … 6 = 周日
})
WD_NAME = {6: "周日", 0: "周一", 1: "周二", 2: "周三", 3: "周四", 4: "周五"}
WD_ORDER = [6, 0, 1, 2, 3, 4]   # CME 交易周从周日 18:00 ET 开始
HOURS = [h for h in range(24) if h != 17]
q25 = lambda s: s.quantile(0.25)
q75 = lambda s: s.quantile(0.75)


def profile(key):
    g = d.groupby(key)
    t = pd.DataFrame({
        "minutes": g.size(),
        "spread_median": g["spread"].median(), "spread_p25": g["spread"].agg(q25), "spread_p75": g["spread"].agg(q75),
        "range_median": g["range"].median(), "range_p25": g["range"].agg(q25), "range_p75": g["range"].agg(q75),
        "atr_median": g["atr"].median(),
        "wide_share": d[d["entry_ok"]].groupby(key)["wide"].mean(),
    })
    return t


by_hour = profile("hour").reindex(HOURS)
by_wd = profile("wd").reindex(WD_ORDER)
by_hour.to_csv(f"{OUT}/按美东小时_点差与波动率_训练集.csv", float_format="%.4f")
by_wd.rename(index=WD_NAME).to_csv(f"{OUT}/按星期_点差与波动率_训练集.csv", float_format="%.4f")
cnt = d.groupby(["wd", "hour"]).size().unstack()
MIN_N = 100   # 分钟数太少的格子不画（如周日 18 点前、周五 17 点后）
heat_spread = d.groupby(["wd", "hour"])["spread"].median().unstack().where(cnt >= MIN_N)
heat_range = d.groupby(["wd", "hour"])["range"].median().unstack().where(cnt >= MIN_N)
heat_spread.reindex(index=WD_ORDER, columns=range(24)).rename(index=WD_NAME).to_csv(
    f"{OUT}/热力图_点差中位数_星期x小时_训练集.csv", float_format="%.3f")
heat_range.reindex(index=WD_ORDER, columns=range(24)).rename(index=WD_NAME).to_csv(
    f"{OUT}/热力图_振幅中位数_星期x小时_训练集.csv", float_format="%.3f")
to_et = bj_to_et
t_start, t_end = to_et(f.index[0]), to_et(lab.index[lab["split"] == "train"][-1])
x = np.arange(len(HOURS))
xw = np.arange(len(WD_ORDER))

# 中英文各一套：中文图在 数据说明/ 下（文件名不变），英文图在 数据说明/English/ 下
T = {
    "zh": {"dir": OUT, "font": FONT["zh"],
           "span": f"训练集，美东时间 {t_start:%Y-%m-%d %H:%M} 至 {t_end:%Y-%m-%d %H:%M}",
           "xhour": "美东时间（小时，17 点为每日休市）", "iqr": "25%–75% 区间", "med": "中位数",
           "spread": "每分钟平均点差（ticks）", "wide": "极端点差占比（%）",
           "wide_note": f"极端点差：入场点差 ≥ 训练集中位数 × {WIDE_MULT:g}（{WIDE_TICKS:g} ticks）",
           "t3": "GC 点差按美东小时分布（{}）", "f3": "图3_点差按美东小时.png",
           "r_iqr": "1 分钟振幅 25%–75% 区间", "r_med": "1 分钟振幅中位数", "atr": f"ATR{ATR_N} 中位数", "usd": "美元/盎司",
           "t4": "GC 波动率按美东小时分布（{}）", "f4": "图4_波动率按美东小时.png",
           "range": "1 分钟振幅（美元）", "s_spread": "点差", "s_range": "波动率", "wd": WD_NAME,
           "t5": "GC 点差与波动率按星期分布（{}）", "f5": "图5_点差与波动率按星期.png"},
    "en": {"dir": f"{OUT}/English", "font": FONT["en"],
           "span": f"training set, ET {t_start:%Y-%m-%d %H:%M} to {t_end:%Y-%m-%d %H:%M}",
           "xhour": "ET hour (17:00 is the daily maintenance break)", "iqr": "25%–75% range", "med": "Median",
           "spread": "Mean spread per minute (ticks)", "wide": "Extreme-spread share (%)",
           "wide_note": f"Extreme spread: entry spread ≥ {WIDE_MULT:g} × training median ({WIDE_TICKS:g} ticks)",
           "t3": "GC spread by ET hour ({})", "f3": "Fig3_spread_by_ET_hour.png",
           "r_iqr": "1-minute range, 25%–75%", "r_med": "1-minute range, median", "atr": f"ATR{ATR_N}, median", "usd": "USD per ounce",
           "t4": "GC volatility by ET hour ({})", "f4": "Fig4_volatility_by_ET_hour.png",
           "range": "1-minute range (USD)", "s_spread": "Spread", "s_range": "Volatility",
           "wd": {6: "Sun", 0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri"},
           "t5": "GC spread and volatility by weekday ({})", "f5": "Fig5_spread_volatility_by_weekday.png"},
}

for lang, tx in T.items():
    os.makedirs(tx["dir"], exist_ok=True)
    plt.rcParams["font.family"] = tx["font"]

    def hour_axis(ax):
        ax.set_xticks(x)
        ax.set_xticklabels([f"{h}" for h in HOURS])
        ax.set_xlabel(tx["xhour"])

    # ---------- 图 3：点差按小时 ----------
    fig, axes = plt.subplots(2, 1, figsize=(12, 7.5), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    ax = axes[0]
    ax.fill_between(x, by_hour["spread_p25"], by_hour["spread_p75"], color=BLUE, alpha=0.18, lw=0, label=tx["iqr"])
    ax.plot(x, by_hour["spread_median"], color=BLUE, lw=2, marker="o", ms=4, label=tx["med"])
    ax.set_ylabel(tx["spread"])
    ax.legend(loc="upper left", frameon=False, labelcolor=INK)
    ax.set_title(tx["t3"].format(tx["span"]), loc="left", color=INK, fontsize=13)
    ax = axes[1]
    ax.bar(x, by_hour["wide_share"] * 100, color=ORANGE, width=0.7)
    ax.set_ylabel(tx["wide"])
    ax.text(0.01, 0.95, tx["wide_note"], transform=ax.transAxes, va="top", color=INK2, fontsize=9)
    hour_axis(ax)
    fig.tight_layout()
    fig.savefig(f"{tx['dir']}/{tx['f3']}", dpi=160)
    plt.close(fig)

    # ---------- 图 4：波动率按小时 ----------
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.fill_between(x, by_hour["range_p25"], by_hour["range_p75"], color=BLUE, alpha=0.18, lw=0, label=tx["r_iqr"])
    ax.plot(x, by_hour["range_median"], color=BLUE, lw=2, marker="o", ms=4, label=tx["r_med"])
    ax.plot(x, by_hour["atr_median"], color=ORANGE, lw=2, marker="s", ms=4, label=tx["atr"])
    ax.set_ylabel(tx["usd"])
    ax.legend(loc="upper left", frameon=False, labelcolor=INK)
    ax.set_title(tx["t4"].format(tx["span"]), loc="left", color=INK, fontsize=13)
    hour_axis(ax)
    fig.tight_layout()
    fig.savefig(f"{tx['dir']}/{tx['f4']}", dpi=160)
    plt.close(fig)

    # ---------- 图 5：按星期 ----------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, col, lo, hi, ylab, ttl in [
            (axes[0], "spread_median", "spread_p25", "spread_p75", tx["spread"], tx["s_spread"]),
            (axes[1], "range_median", "range_p25", "range_p75", tx["range"], tx["s_range"])]:
        ax.bar(xw, by_wd[col], color=BLUE, width=0.6, label=tx["med"])
        ax.errorbar(xw, by_wd[col], yerr=[by_wd[col] - by_wd[lo], by_wd[hi] - by_wd[col]],
                    fmt="none", ecolor=INK2, capsize=4, lw=1, label=tx["iqr"])
        ax.set_xticks(xw)
        ax.set_xticklabels([tx["wd"][w] for w in WD_ORDER])
        ax.set_ylabel(ylab)
        ax.set_title(ttl, loc="left", color=INK, fontsize=12)
    axes[0].legend(loc="upper right", frameon=False, labelcolor=INK)
    fig.suptitle(tx["t5"].format(tx["span"]), x=0.01, ha="left", color=INK, fontsize=13)
    fig.tight_layout()
    fig.savefig(f"{tx['dir']}/{tx['f5']}", dpi=160)
    plt.close(fig)

# 星期 × 美东小时的热力图已移到第 4 节 EDA 图 8（plot_eda_figures.py），这里不再画

print(by_hour.round(3).to_string())
print(by_wd.rename(index=WD_NAME).round(3).to_string())
