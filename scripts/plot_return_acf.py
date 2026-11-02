# 第 4 节补充图 7：相对变化率与相对变化率绝对值的自相关（滞后 1–240 分钟），作为第 5 节选择回看窗口 L 的证据（中英文各一张）
# 只用训练集样本：有成交的分钟，且不在换月前 3 天。相对变化率 r = ln(本分钟收盘价 ÷ 上一分钟收盘价)，用未复权收盘价；只取同一合约、相隔正好 1 分钟的两根 K 线，跨休市、缺失分钟、换月的变化记为空
# 自相关对每个滞后 k 只用两端都非空的样本对计算
# |r| 三种口径：原始；去掉日内时段规律和当天波动水平（只看日内的延续）；除以上一分钟的 ATR14（接近模型实际看到的输入）
# 输出：数据说明/EDA图/中文/图7_相对变化率自相关.png、English/Fig7_relative_change_autocorrelation.png、数据说明/EDA图/acf_stats.txt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pathlib import Path
P = str(Path(__file__).resolve().parent.parent)
OUT = f"{P}/数据说明/EDA图"
MAX_LAG = 240
MARK_LAGS = [30, 60, 120]
REPORT_LAGS = [1, 2, 5, 10, 15, 30, 45, 60, 90, 120, 180, 240]
BASE_STYLE = {
    "axes.unicode_minus": False,
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": "#52514e",
    "xtick.color": "#52514e", "ytick.color": "#52514e",
    "axes.grid": True, "axes.axisbelow": True, "grid.color": "#e8e7e2", "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 11, "axes.labelsize": 9.5, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "legend.fontsize": 8.5, "legend.frameon": False,
}
FONT = {"zh": ["Hiragino Sans GB", "Arial Unicode MS"], "en": ["Arial", "Helvetica"]}
COL = {"raw": "#2a78d6", "intraday": "#eb6834", "atr": "#52514e"}

T = {
    "zh": {
        "file": "中文/图7_相对变化率自相关.png",
        "title": "1 分钟相对变化率的自相关（训练集，滞后 1–240 分钟）",
        "a": "(a) 相对变化率 r = ln(本分钟收盘价 ÷ 上一分钟收盘价)（涨跌方向）", "b": "(b) 相对变化率绝对值 |r|（波动大小）",
        "x": "滞后 k（分钟）", "y": "自相关系数",
        "band": "95% 显著性范围（±1.96/√N）",
        "raw": "原始 |r|",
        "intraday": "去掉日内时段规律和当天波动水平后的 |r|（只看日内的延续）",
        "atr": "|r| ÷ 上一分钟 ATR14（模型输入的口径）",
    },
    "en": {
        "file": "English/Fig7_relative_change_autocorrelation.png",
        "title": "Autocorrelation of 1-minute relative changes (training set, lags 1–240 min)",
        "a": "(a) Relative change r = ln(close ÷ previous close) (direction)", "b": "(b) Absolute relative change |r| (volatility)",
        "x": "Lag k (minutes)", "y": "Autocorrelation",
        "band": "95% significance band (±1.96/√N)",
        "raw": "Raw |r|",
        "intraday": "|r| without time-of-day pattern and daily level (intraday memory only)",
        "atr": "|r| ÷ previous-minute ATR14 (as the model inputs see it)",
    },
}


def acf(s, max_lag=MAX_LAG):
    v = s.to_numpy(dtype=float)
    out = []
    for k in range(1, max_lag + 1):
        x, y = v[:-k], v[k:]
        m = np.isfinite(x) & np.isfinite(y)
        out.append(np.corrcoef(x[m], y[m])[0, 1])
    return pd.Series(out, index=range(1, max_lag + 1))


f = pd.read_parquet(f"{P}/features/features_1min.parquet", columns=["raw_close", "raw_atr14", "instrument_id", "is_filled"])
lab = pd.read_parquet(f"{P}/labels/labels_default.parquet", columns=["split", "pre_roll_3d"]).reindex(f.index)
f = f[(lab["split"].astype(str) == "train") & (f["is_filled"] == 0) & (lab["pre_roll_3d"] == 0)]
# 补成连续的分钟网格，缺失分钟为空，这样滞后 k 就是真实的 k 分钟
g = f.reindex(pd.date_range(f.index[0], f.index[-1], freq="min"))
r = np.log(g["raw_close"]).diff()
r[g["instrument_id"] != g["instrument_id"].shift()] = np.nan
a = r.abs()

et = g.index.tz_localize("Asia/Shanghai").tz_convert("America/New_York")
minute_of_day = et.hour * 60 + et.minute
trading_day = (et - pd.Timedelta(hours=18)).date  # CME 交易日，美东 18:00 开始
# 日内时段规律：按美东每分钟求 |r| 的均值，再做 15 分钟循环平滑
tod = a.groupby(minute_of_day).mean().reindex(range(1440))
tod = pd.Series(np.r_[tod.values[-7:], tod.values, tod.values[:7]]).rolling(15, center=True, min_periods=1).mean().values[7:-7]
a_tod = a / tod[minute_of_day]
a_intraday = a_tod / a_tod.groupby(trading_day).transform("mean")
a_atr = a / (g["raw_atr14"] / g["raw_close"]).shift(1)

n = int(r.notna().sum())
band = 1.96 / np.sqrt(n)
res = {"r": acf(r), "raw": acf(a), "intraday": acf(a_intraday), "atr": acf(a_atr)}

lines = [f"训练集，有效 1 分钟相对变化率 N = {n:,}；95% 显著性范围 ±{band:.4f}", ""]
lines.append("滞后 k: " + "  ".join(f"{k:>6}" for k in REPORT_LAGS))
for name, s in res.items():
    lines.append(f"{name:>8}: " + "  ".join(f"{s[k]:6.3f}" for k in REPORT_LAGS))
lines.append("")
for name in ["raw", "intraday", "atr"]:
    s = res[name]
    for frac in [0.5, 0.25]:
        below = s[s < frac * s[1]]
        lines.append(f"{name}: 首次降到滞后 1 的 {frac:.0%} 以下的滞后 = {below.index[0] if len(below) else '> 240'}")
    lines.append(f"{name}: 首次进入显著性范围的滞后 = {s[s.abs() < band].index[0] if (s.abs() < band).any() else '> 240'}")
lines.append(f"r: |自相关| 最大值 {res['r'].abs().max():.4f}（滞后 {res['r'].abs().idxmax()}）；超出显著性范围的滞后个数 {(res['r'].abs() > band).sum()} / {MAX_LAG}")
open(f"{OUT}/acf_stats.txt", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))

for lang, t in T.items():
    plt.rcParams.update(BASE_STYLE)
    plt.rcParams["font.sans-serif"] = FONT[lang]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for ax in axes:
        for k in MARK_LAGS:
            ax.axvline(k, color="#9a9890", ls=":", lw=0.9)
        ax.axhline(0, color="#52514e", lw=0.6)
        ax.set_xlim(0, MAX_LAG)
        ax.set_xticks([0, 30, 60, 90, 120, 180, 240])
        ax.set_xlabel(t["x"])
        ax.set_ylabel(t["y"])
    ax = axes[0]
    ax.axhspan(-band, band, color="#e6eef8", label=t["band"])
    ax.bar(res["r"].index, res["r"].values, width=0.8, color=COL["raw"])
    lim = max(0.03, res["r"].abs().max() * 1.3)
    ax.set_ylim(-lim, lim)
    ax.set_title(t["a"], loc="left")
    ax.legend(loc="upper right")
    ax = axes[1]
    ax.axhspan(-band, band, color="#e6eef8")
    for name in ["raw", "intraday", "atr"]:
        ax.plot(res[name].index, res[name].values, color=COL[name], lw=1.6, label=t[name])
    ax.set_ylim(-0.05, 0.42)
    ax.set_title(t["b"], loc="left")
    ax.legend(loc="upper right")
    fig.suptitle(t["title"], x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(f"{OUT}/{t['file']}", dpi=200)
    plt.close(fig)
print("saved figure 7 (zh, en)")
