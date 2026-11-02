# 第 4 节探索性分析图（中英文各一套）：目标分布、输入分布、组结构、元数据与目标、每类示例、捷径检查、点差与波动率热力图（图 8）
# 图中时间一律用美东时间（数据索引是北京时间，只在统计和画图时转换）
# 样本 = 有成交的分钟（补出的分钟不算），且不在换月前 3 天（pre_roll_3d = 0）
# 除图 1、图 6 的跨数据集对比外，所有统计只用训练集样本；标签用默认组合（k_SL = 2，盈亏比 1:1），0 = 止盈，1 = 止损，2 = 超时
# 输出：数据说明/EDA图/中文/图*.png、数据说明/EDA图/English/Fig*.png，统计数字存 数据说明/EDA图/eda_stats.txt
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pathlib import Path
P = str(Path(__file__).resolve().parent.parent)
OUT = f"{P}/数据说明/EDA图"
SEED, K = 42, 2.0
HEADS = [f"{s}_H{h}" for s in ["long", "short"] for h in [15, 30, 60]]
SPLITS = ["train", "val", "test"]
TP, SL, TO = 0, 1, 2
C3 = {TP: "#2a78d6", SL: "#eb6834", TO: "#9a9890"}
SPLIT_BG = {"train": "#ffffff", "val": "#f1efe8", "test": "#e6eef8"}
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

T = {
    "zh": {
        "dir": "中文", "sep": "　", "cls": {TP: "止盈", SL: "止损", TO: "超时"},
        "head": lambda s, h: f"{'做多' if s == 'long' else '做空'} {h} 分钟",
        "split": {"train": "训练集", "val": "验证集", "test": "测试集"}, "corr": "相关系数",
        "f1": ("图1_目标分布.png", "6 个输出头的标签分布：训练集、验证集、测试集", "输出头（方向 × 持有期）", "占比（%）"),
        "f2": ("图2_输入特征分布.png", "训练集中 4 个关键输入特征的分布（虚线 = 1% 与 99% 分位数）", "样本数（对数刻度）"),
        "f2x": ["收盘价变化 d_close（本分钟 − 上一分钟收盘价，ATR14 倍数）", "盘口深度订单流不平衡 ofi_depth_adj（标准化后）",
                "点差 / ATR14（spread_atr）", "相对成交量 relvol_60（本分钟 / 过去 60 分钟均值）"],
        "f2note": lambda k, out: f"峰度 {k:.1f}\n图外样本 {out:.2%}",
        "f3": ("图3_组结构_交易日与标签相关性.png", "组结构：按交易日分组，相邻样本的标签高度相关"),
        "f3a": ("(a) 每个交易日的样本数（训练集）", "交易日（CME，美东 18:00 开盘）", "样本数（分钟）"),
        "f3b": ("(b) 相隔 k 分钟的两个样本标签相同的比例", "间隔 k（分钟，对数刻度）", "标签相同的比例（%）",
                "随机配对的基准"),
        "f4": ("图4_元数据与目标_时段和月份.png", "元数据与目标的关系（训练集，做多 30 分钟）"),
        "f4a": ("(a) 按美东时间小时", "美东时间（小时，17 点为每日休市）", "占比（%）"),
        "f4b": ("(b) 按月份", "月份", "占比（%）"),
        "f5": ("图5_每类标签随机示例.png", "每类标签随机抽 4 个训练样本（做多 30 分钟）：入场后每分钟买一价的高低区间（子图时间为美东时间）"),
        "f5lab": ("入场价（卖一）", "止盈线", "止损线", "首次碰线"),
        "f5x": "入场后分钟", "f5y": "价格（美元）",
        "f6": ("图6_捷径检查.png", "捷径检查（做多/做空 30 分钟）"),
        "f6a": ("(a) 每周做多止盈率 − 做空止盈率（柱）与 GC 收盘价（线）", "周", "止盈率差（百分点）", "GC 收盘价（美元）"),
        "f6b": ("(b) 超时比例随相对成交量分组的变化（训练集）", "relvol_60 十分位（1 = 最清淡）", "超时比例（%）"),
        "f6c": ("(c) 极端点差（≥ 20 跳）与正常分钟的标签分布（训练集）", "", "占比（%）", ("正常点差", "极端点差"),
                {"before": "剔除换月前 3 天之前", "after": "剔除换月前 3 天之后（现在的样本）"}, "占样本"),
        "f8": ("图8_点差与波动率热力图_星期x美东小时.png", "点差与波动率：星期 × 美东小时（训练集样本，美东时间 {}）"),
        "f8a": ("(a) 每分钟平均点差的中位数", "跳（1 跳 = 0.1 美元）"),
        "f8b": ("(b) 1 分钟振幅（最高价 − 最低价）的中位数", "美元/盎司"),
        "f8x": "美东时间（小时）；周日一行 = 周日 18:00 开盘后的时段；空白 = 每日休市（17 点）或样本少于 100 分钟",
        "wd": ["周一", "周二", "周三", "周四", "周五", "周六", "周日"],
        "span": lambda a, b: f"{a:%Y-%m-%d %H:%M} 至 {b:%Y-%m-%d %H:%M}",
        "month_fmt": lambda p: f"{p.year % 100:02d}年{p.month}月",
    },
    "en": {
        "dir": "English", "sep": "  ", "cls": {TP: "Take-profit", SL: "Stop-loss", TO: "Timeout"},
        "head": lambda s, h: f"{'Long' if s == 'long' else 'Short'} {h} min",
        "split": {"train": "Train", "val": "Validation", "test": "Test"}, "corr": "corr.",
        "f1": ("Fig1_target_distribution.png", "Label distribution of the 6 output heads: training, validation and test sets",
               "Output head (direction × horizon)", "Share (%)"),
        "f2": ("Fig2_input_distributions.png",
               "Distributions of 4 key input features (training set; dashed = 1st and 99th percentiles)",
               "Count (log scale)"),
        "f2x": ["Close change d_close (close − previous close, in ATR14 units)", "Depth order-flow imbalance ofi_depth_adj (normalized)",
                "Spread / ATR14 (spread_atr)", "Relative volume relvol_60 (this minute / 60-min mean)"],
        "f2note": lambda k, out: f"kurtosis {k:.1f}\noutside plot {out:.2%}",
        "f3": ("Fig3_group_structure.png", "Group structure: samples group by trading day; nearby labels are highly correlated"),
        "f3a": ("(a) Samples per trading day (training set)", "Trading day (CME, opens 18:00 ET)", "Samples (minutes)"),
        "f3b": ("(b) Share of sample pairs k minutes apart with the same label", "Lag k (minutes, log scale)",
                "Same label (%)", "Random-pair baseline"),
        "f4": ("Fig4_metadata_vs_target.png", "Metadata vs. target (training set, long 30 min)"),
        "f4a": ("(a) By US Eastern hour", "US Eastern time (hour; 17 = daily halt)", "Share (%)"),
        "f4b": ("(b) By month", "Month", "Share (%)"),
        "f5": ("Fig5_random_examples_per_class.png",
               "4 random training samples per class (long 30 min): per-minute bid high-low after entry (panel times in US Eastern)"),
        "f5lab": ("Entry (ask)", "Take-profit line", "Stop-loss line", "First touch"),
        "f5x": "Minutes after entry", "f5y": "Price (USD)",
        "f6": ("Fig6_shortcut_checks.png", "Shortcut checks (long/short 30 min)"),
        "f6a": ("(a) Weekly long TP rate − short TP rate (bars) and GC close (line)", "Week",
                "TP-rate difference (pp)", "GC close (USD)"),
        "f6b": ("(b) Timeout share by relative-volume decile (training set)", "relvol_60 decile (1 = quietest)",
                "Timeout share (%)"),
        "f6c": ("(c) Label mix: extreme-spread (≥ 20 ticks) vs. normal minutes (training set)",
                "", "Share (%)", ("Normal spread", "Extreme spread"), {"before": "Before excluding the 3 pre-roll days", "after": "After excluding them (current samples)"}, "share of samples"),
        "f8": ("Fig8_spread_volatility_heatmap_weekday_x_ET_hour.png",
               "Spread and volatility by weekday × US Eastern hour (training samples, {} ET)"),
        "f8a": ("(a) Median of the per-minute average spread", "Ticks (1 tick = $0.10)"),
        "f8b": ("(b) Median 1-minute range (high − low)", "USD/oz"),
        "f8x": "US Eastern time (hour); Sunday row = session after the Sunday 18:00 open; blank = daily halt (17:00) or fewer than 100 minutes",
        "wd": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
        "span": lambda a, b: f"{a:%Y-%m-%d %H:%M} to {b:%Y-%m-%d %H:%M}",
        "month_fmt": lambda p: p.strftime("%b %y"),
    },
}

# ---------- 数据 ----------
lab = pd.read_parquet(f"{P}/labels/labels_default.parquet")
feat = pd.read_parquet(f"{P}/features/features_1min.parquet",
                       columns=["d_close", "ofi_depth_adj", "spread_atr", "relvol_60", "raw_close", "raw_high", "raw_low", "raw_atr14",
                                "instrument_id"])
book = pd.read_parquet(f"{P}/GC_processed/GC_book_1min.parquet", columns=["spread_sum", "spread_n"])
feat["spread_mean"] = (book["spread_sum"] / book["spread_n"].replace(0, np.nan)).reindex(feat.index).ffill()
d = lab.join(feat)          # 标签文件只含有成交的分钟，补出的分钟自然不在内
d = d[d["pre_roll_3d"] == 0]
d["split"] = d["split"].astype(str)
et = d.index.tz_localize("Asia/Shanghai").tz_convert("America/New_York").tz_localize(None)
d["et_hour"] = et.hour
d["et_wd"] = et.dayofweek          # 0 = 周一 … 6 = 周日
d["et_time"] = et
d["tday"] = (et + pd.Timedelta(hours=6)).normalize()   # CME 交易日：美东 18:00 开始算下一天
tr = d[d["split"] == "train"]
stats = []


def share(s):
    return s.dropna().astype(int).value_counts(normalize=True).reindex([TP, SL, TO]).fillna(0) * 100


# 图 1：每个输出头的标签占比，训练 / 验证 / 测试集分开
s1 = {sp: pd.DataFrame({h: share(d.loc[d["split"] == sp, h]) for h in HEADS}) for sp in SPLITS}
for sp in SPLITS:
    stats.append(f"图1 标签占比(%) {sp}\n" + s1[sp].round(1).to_string())

# 图 2：特征分布
F2 = ["d_close", "ofi_depth_adj", "spread_atr", "relvol_60"]
f2 = {}
for c in F2:
    s = tr[c].dropna()
    lo, hi = s.quantile([0.0005, 0.9995])
    f2[c] = dict(s=s, q01=s.quantile(0.01), q99=s.quantile(0.99), kurt=s.kurt(),
                 lo=lo if c in ("d_close", "ofi_depth_adj") else 0, hi=hi)
    f2[c]["out"] = ((s < f2[c]["lo"]) | (s > hi)).mean()
    stats.append(f"图2 {c}: 1%={f2[c]['q01']:.3f} 50%={s.median():.3f} 99%={f2[c]['q99']:.3f} "
                 f"峰度={f2[c]['kurt']:.1f} 偏度={s.skew():.2f}")

# 图 3：组结构
per_day = tr.groupby("tday").size()
LAGS = [1, 2, 3, 5, 10, 15, 20, 30, 45, 60, 90, 120, 240, 480, 1440]
agree = {}
for h in ["long_H15", "long_H30", "long_H60"]:
    a = tr[h].astype("float")
    same = []
    for k in LAGS:
        b = a.shift(-k)
        ok = a.notna() & b.notna()
        same.append((a[ok] == b[ok]).mean() * 100)
    p = a.dropna().value_counts(normalize=True)
    agree[h] = (same, (p ** 2).sum() * 100)
stats.append(f"图3 交易日数 {len(per_day)}，每日样本数中位数 {per_day.median():.0f}，最少 {per_day.min()}")
stats.append("图3 标签相同比例(%) " + "; ".join(
    f"{h}: " + ", ".join(f"k={k}:{v:.1f}" for k, v in zip(LAGS, agree[h][0])) + f" 基准 {agree[h][1]:.1f}"
    for h in agree))

# 图 4：时段、月份
y30 = "long_H30"
by_hour = tr.groupby("et_hour")[y30].apply(share).unstack().reindex([h for h in range(24) if h != 17])
mon = tr.index.to_period("M")
cnt_m = tr.groupby(mon)[y30].count()
by_mon = tr.groupby(mon)[y30].apply(share).unstack()[cnt_m >= 5000]   # 去掉只有几天的月份（2025-09）
stats.append("图4 按美东小时(%)\n" + by_hour.round(1).to_string())
stats.append("图4 按月份(%)\n" + by_mon.round(1).to_string())

# 图 5：每类随机 4 个样本，入场后 30 分钟买一价
q = pd.read_parquet(f"{P}/GC_processed/GC_quote_1min.parquet")
q = q.reindex(pd.date_range(q.index.min(), q.index.max(), freq="min"))
for side in ["bid", "ask"]:
    last = q[f"{side}_close"].ffill(limit=10)
    for c in ["open", "high", "low", "close"]:
        q[f"{side}_{c}"] = q[f"{side}_{c}"].fillna(last)
pool = tr[tr[y30].notna() & (tr["wide_spread"] == 0)]
ex = {c: pool[pool[y30] == c].sample(4, random_state=SEED) for c in [TP, SL, TO]}


def first_touch(win, entry, dist):
    for i, (_, r) in enumerate(win.iterrows()):
        if r.bid_low <= entry - dist:
            return i, SL, entry - dist
        if r.bid_high >= entry + dist:
            return i, TP, entry + dist
    return None, TO, None


ex_rows = []
for c, smp in ex.items():
    for t, r in smp.iterrows():
        e = t + pd.Timedelta(minutes=1)
        win = q.loc[e:e + pd.Timedelta(minutes=29)]
        dist = K * r.raw_atr14 * np.sqrt(30 / 15)
        i, y, yp = first_touch(win, r.entry_ask, dist)
        ex_rows.append((c, t, win, r.entry_ask, dist, i, yp))
        assert y == c, f"重新判断与标签不一致：{t}"
stats.append("图5 示例（美东时间）：" + "; ".join(f"{T['zh']['cls'][c]} {tr.at[t, 'et_time']:%Y-%m-%d %H:%M}" for c, t, *_ in ex_rows)
             + "（12 个均与重新判断一致）")

# 图 6：捷径
# 周按美东时间划分（周日 18:00 开盘到周五收盘算一周）；跨数据集分界的周拆开，分别归入各自的数据集
wk = d[d["split"].isin(["train", "val", "test"])].copy()
wk["week"] = wk["et_time"].dt.to_period("W-FRI")


def week_change(x):
    # 当周金价涨跌幅：只用当周最后一个合约的价格，避免换月跳空
    x = x[x["instrument_id"] == x["instrument_id"].iloc[-1]]
    return (x["raw_close"].iloc[-1] / x["raw_close"].iloc[0] - 1) * 100


g = wk.groupby(["split", "week"])
weekly = pd.DataFrame({
    "diff": (g["long_H30"].apply(lambda s: (s.dropna() == TP).mean())
             - g["short_H30"].apply(lambda s: (s.dropna() == TP).mean())) * 100,
    "close": g["raw_close"].last(),
    "chg": g.apply(week_change),
    "n": g["long_H30"].count(),
}).reset_index(level="split").sort_index()
weekly = weekly[weekly["n"] > 3000]   # 只用完整的周：去掉首尾和数据集分界处不满一周的部分
sp_corr = weekly.groupby("split").apply(lambda x: x["diff"].corr(x["chg"]))
sp_weeks = weekly.groupby("split").size()
tr_rv = tr[tr[y30].notna()].copy()
tr_rv["dec"] = pd.qcut(tr_rv["relvol_60"].rank(method="first"), 10, labels=range(1, 11))
to_by_dec = tr_rv.groupby("dec", observed=True)[y30].apply(lambda s: (s == TO).mean() * 100)
# 极端点差（入场点差 ≥ 20 跳，与 wide_spread 标记一致：阈值 = 5 × 训练集中位数，浮点误差使 20 跳本身也被计入）
# 剔除换月前 3 天之前 / 之后。剔除前用对照版标签（make_labels_with_preroll_check.py，换月前 3 天也生成标签）
WIDE_TICKS = 20
pre = pd.read_parquet(f"{P}/labels/check_with_preroll/labels_all.parquet", columns=["long_H30_sl2_rr1", "spread_ticks", "split"])
pre = pre[pre["split"] == "train"]
wide_cmp = {
    "before": pre.groupby(pre["spread_ticks"].round() >= WIDE_TICKS)["long_H30_sl2_rr1"].apply(share).unstack(),
    "after": tr.groupby(tr["spread_ticks"].round() >= WIDE_TICKS)[y30].apply(share).unstack(),
}
wide_frac = {k: (x["spread_ticks"].round() >= WIDE_TICKS).mean() * 100 for k, x in [("before", pre), ("after", tr)]}
stats.append("图6 每周做多止盈率 − 做空止盈率 与当周金价涨跌幅的相关系数：" + sp_corr.round(2).to_dict().__repr__()
             + "；周数：" + sp_weeks.to_dict().__repr__())
stats.append("图6 每周差值范围 " + f"{weekly['diff'].min():.1f} 到 {weekly['diff'].max():.1f}")
stats.append("图6 超时比例按 relvol_60 十分位(%)：" + to_by_dec.round(1).to_dict().__repr__())
for k in ["before", "after"]:
    stats.append(f"图6 按极端点差（≥ {WIDE_TICKS} 跳）(%)，{'剔除换月前 3 天之前' if k == 'before' else '剔除之后'}\n"
                 + wide_cmp[k].round(1).to_string() + f"\n极端点差样本占训练集 {wide_frac[k]:.2f}%")

# 图 8：点差、1 分钟振幅按星期 × 美东小时（训练集样本）；格子少于 100 分钟不画
WD_ORDER = [6, 0, 1, 2, 3, 4]   # CME 交易周从周日 18:00 ET 开始
tr8 = tr.assign(rng=tr["raw_high"] - tr["raw_low"])
g8 = tr8.groupby(["et_wd", "et_hour"])
cnt8 = g8.size().unstack()
heat = {k: g8[c].median().unstack().where(cnt8 >= 100).reindex(index=WD_ORDER, columns=range(24))
        for k, c in [("spread", "spread_mean"), ("range", "rng")]}
span8 = (tr8["et_time"].min(), tr8["et_time"].max())
stats.append(f"图8 训练集样本 {len(tr8)} 分钟，美东 {span8[0]} 至 {span8[1]}")
stats.append("图8 点差中位数（跳）\n" + heat["spread"].round(2).to_string())
stats.append("图8 1 分钟振幅中位数（美元）\n" + heat["range"].round(2).to_string())


# ---------- 画图 ----------
def save(fig, lang, name):
    os.makedirs(f"{OUT}/{T[lang]['dir']}", exist_ok=True)
    fig.savefig(f"{OUT}/{T[lang]['dir']}/{name}", dpi=160, bbox_inches="tight")
    plt.close(fig)


def class_legend(ax, L, **kw):
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=C3[c]) for c in [TP, SL, TO]],
              labels=[L["cls"][c] for c in [TP, SL, TO]], **kw)


for lang, L in T.items():
    plt.rcParams.update(BASE_STYLE | {"font.family": FONT[lang]})

    # 图 1
    fig, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=True)
    x = np.arange(len(HEADS))
    w = 0.27
    for ax, sp in zip(axes, SPLITS):
        for j, c in enumerate([TP, SL, TO]):
            b = ax.bar(x + (j - 1) * w, s1[sp].loc[c], w, color=C3[c], label=L["cls"][c])
            ax.bar_label(b, fmt="%.1f", fontsize=7.5, padding=2, color="#52514e")
        ax.set_xticks(x, [L["head"](*h.split("_H")) for h in HEADS])
        ax.set_ylabel(L["f1"][3])
        ax.set_title(L["split"][sp], loc="left")
        ax.set_ylim(0, 58)
    axes[-1].set_xlabel(L["f1"][2])
    axes[0].legend(ncol=3, loc="upper right")
    fig.suptitle(L["f1"][1], x=0.01, ha="left", fontsize=11.5)
    fig.tight_layout()
    save(fig, lang, L["f1"][0])

    # 图 2
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, c, xl in zip(axes.flat, F2, L["f2x"]):
        v = f2[c]
        s = v["s"][(v["s"] >= v["lo"]) & (v["s"] <= v["hi"])]
        ax.hist(s, bins=120, color="#2a78d6", alpha=0.85)
        ax.set_yscale("log")
        for qv in (v["q01"], v["q99"]):
            ax.axvline(qv, color="#eb6834", ls="--", lw=1)
        ax.text(0.98, 0.95, L["f2note"](v["kurt"], v["out"]), transform=ax.transAxes, ha="right", va="top",
                fontsize=8.5, color="#52514e")
        ax.set_xlabel(xl)
        ax.set_ylabel(L["f2"][2])
    fig.suptitle(L["f2"][1], x=0.01, ha="left", fontsize=11.5)
    fig.tight_layout()
    save(fig, lang, L["f2"][0])

    # 图 3
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.4), gridspec_kw={"width_ratios": [1.3, 1]})
    a.bar(per_day.index, per_day.values, width=0.9, color="#2a78d6")
    a.set_title(L["f3a"][0], loc="left")
    a.set_xlabel(L["f3a"][1])
    a.set_ylabel(L["f3a"][2])
    for h, col in zip(agree, ["#7fb0ea", "#2a78d6", "#0b3d7a"]):
        hh = int(h.split("H")[1])
        b.plot(LAGS, agree[h][0], marker="o", ms=3.5, color=col, label=L["head"]("long", hh))
        b.axvline(hh, color=col, ls=":", lw=0.9)
    b.axhline(np.mean([v[1] for v in agree.values()]), color="#9a9890", ls="--", lw=1, label=L["f3b"][3])
    b.set_xscale("log")
    b.set_xticks([1, 5, 15, 30, 60, 240, 1440], ["1", "5", "15", "30", "60", "240", "1440"])
    b.set_title(L["f3b"][0], loc="left")
    b.set_xlabel(L["f3b"][1])
    b.set_ylabel(L["f3b"][2])
    b.legend()
    fig.suptitle(L["f3"][1], x=0.01, ha="left", fontsize=11.5)
    fig.tight_layout()
    save(fig, lang, L["f3"][0])

    # 图 4
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.4), gridspec_kw={"width_ratios": [1.4, 1]})
    xh = np.arange(len(by_hour))
    for c in [TP, SL, TO]:
        a.plot(xh, by_hour[c], marker="o", ms=3.5, color=C3[c], label=L["cls"][c])
    a.set_xticks(xh, by_hour.index)
    a.set_title(L["f4a"][0], loc="left")
    a.set_xlabel(L["f4a"][1])
    a.set_ylabel(L["f4a"][2])
    xm = np.arange(len(by_mon))
    for c in [TP, SL, TO]:
        b.plot(xm, by_mon[c], marker="o", ms=3.5, color=C3[c], label=L["cls"][c])
    b.set_xticks(xm, [L["month_fmt"](p) for p in by_mon.index], rotation=45, ha="right")
    b.set_title(L["f4b"][0], loc="left")
    b.set_xlabel(L["f4b"][1])
    b.set_ylabel(L["f4b"][2])
    for ax in (a, b):
        ax.set_ylim(10, 55)
    a.legend(ncol=3, loc="upper left")
    fig.suptitle(L["f4"][1], x=0.01, ha="left", fontsize=11.5)
    fig.tight_layout()
    save(fig, lang, L["f4"][0])

    # 图 5
    fig, axes = plt.subplots(3, 4, figsize=(15, 9.5), sharex=True)
    for ax, (c, t, win, entry, dist, i, yp) in zip(axes.flat, ex_rows):
        xx = np.arange(len(win))
        ax.vlines(xx, win.bid_low, win.bid_high, color="#52514e", lw=2.2)
        ax.axhline(entry, color="#0b0b0b", lw=1)
        ax.axhline(entry + dist, color=C3[TP], lw=1.2, ls="--")
        ax.axhline(entry - dist, color=C3[SL], lw=1.2, ls="--")
        if i is not None:
            ax.plot(i, yp, marker="*", ms=13, color=C3[c], mec="#0b0b0b", mew=0.6, zorder=5)
        ax.set_title(f"{L['cls'][c]}{L['sep']}{tr.at[t, 'et_time']:%Y-%m-%d %H:%M}", fontsize=9.5, color=C3[c] if c != TO else "#52514e")
        ax.tick_params(labelsize=7.5)
    for ax in axes[-1]:
        ax.set_xlabel(L["f5x"])
    for ax in axes[:, 0]:
        ax.set_ylabel(L["f5y"])
    hl = [plt.Line2D([], [], color="#0b0b0b", lw=1), plt.Line2D([], [], color=C3[TP], ls="--"),
          plt.Line2D([], [], color=C3[SL], ls="--"),
          plt.Line2D([], [], marker="*", ms=11, color="#9a9890", ls="", mec="#0b0b0b", mew=0.6)]
    fig.legend(hl, L["f5lab"], ncol=4, loc="upper right", bbox_to_anchor=(0.995, 0.995))
    fig.suptitle(L["f5"][1], x=0.01, y=0.99, ha="left", fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    save(fig, lang, L["f5"][0])

    # 图 6
    fig = plt.figure(figsize=(13, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1])
    a = fig.add_subplot(gs[0, :])
    xs = weekly.index.to_timestamp()
    for sp in ["train", "val", "test"]:
        m = weekly["split"] == sp
        a.axvspan(xs[m].min() - pd.Timedelta(days=3.5), xs[m].max() + pd.Timedelta(days=3.5),
                  color=SPLIT_BG[sp], zorder=0)
        a.text(xs[m].min() + (xs[m].max() - xs[m].min()) / 2, 0.03,
               f"{L['split'][sp]}  {L['corr']} {sp_corr[sp]:+.2f}", transform=a.get_xaxis_transform(),
               ha="center", va="bottom", fontsize=9, color="#52514e")
    a.bar(xs, weekly["diff"], width=5, color=np.where(weekly["diff"] >= 0, C3[TP], C3[SL]), zorder=2)
    a.axhline(0, color="#52514e", lw=0.8)
    a.set_ylabel(L["f6a"][2])
    a.set_xlabel(L["f6a"][1])
    a.set_title(L["f6a"][0], loc="left")
    a2 = a.twinx()
    a2.plot(xs, weekly["close"], color="#0b0b0b", lw=1.3, zorder=3)
    a2.set_ylabel(L["f6a"][3])
    a2.grid(False)
    a2.spines["right"].set_visible(True)
    b = fig.add_subplot(gs[1, 0])
    b.bar(to_by_dec.index.astype(int), to_by_dec.values, color=C3[TO])
    b.set_xticks(range(1, 11))
    b.set_title(L["f6b"][0], loc="left")
    b.set_xlabel(L["f6b"][1])
    b.set_ylabel(L["f6b"][2])
    c_ = fig.add_subplot(gs[1, 1])
    # 从上到下：剔除前 极端 / 正常，剔除后 极端 / 正常
    rows = [("before", True), ("before", False), ("after", True), ("after", False)]
    ys = np.array([4.0, 3.0, 1.4, 0.4])
    left = np.zeros(len(rows))
    for c in [TP, SL, TO]:
        vals = np.array([wide_cmp[k].loc[w, c] for k, w in rows])
        c_.barh(ys, vals, left=left, height=0.8, color=C3[c], label=L["cls"][c])
        for yy, lv, vv in zip(ys, left, vals):
            c_.text(lv + vv / 2, yy, f"{vv:.0f}", ha="center", va="center", color="white", fontsize=9)
        left += vals
    c_.axhline(2.2, color="#c3c2b7", lw=0.8)
    c_.grid(False)
    c_.set_yticks(ys, [L["f6c"][3][1 if w else 0] + (f"\n{L['f6c'][5]} {wide_frac[k]:.2f}%" if w else "") for k, w in rows])
    for yy, k in [(4.6, "before"), (2.0, "after")]:
        c_.text(0, yy, L["f6c"][4][k], ha="left", va="center", fontsize=9.5, fontweight="bold", color="#52514e")
    c_.set_ylim(-0.1, 4.9)
    c_.set_xlim(0, 100)
    c_.set_xlabel(L["f6c"][2])
    c_.set_title(L["f6c"][0], loc="left")
    c_.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    fig.suptitle(L["f6"][1], x=0.01, ha="left", fontsize=11.5)
    fig.tight_layout()
    save(fig, lang, L["f6"][0])

for lang, L in T.items():
    plt.rcParams.update(BASE_STYLE | {"font.family": FONT[lang]})
    # 图 8
    fig, axes = plt.subplots(2, 1, figsize=(13, 7.5))
    for ax, h, cmap, (ttl, cb), fmt in [(axes[0], heat["spread"], "Blues", L["f8a"], "{:.1f}"),
                                        (axes[1], heat["range"], "Oranges", L["f8b"], "{:.2f}")]:
        im = ax.imshow(h.to_numpy(), aspect="auto", cmap=cmap)
        ax.set_yticks(range(len(WD_ORDER)), [L["wd"][w] for w in WD_ORDER])
        ax.set_xticks(range(24), range(24))
        ax.grid(False)
        vmax = np.nanmax(h.to_numpy())
        for i in range(h.shape[0]):
            for j in range(24):
                v = h.iat[i, j]
                if np.isfinite(v):
                    ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=7,
                            color="white" if v > 0.6 * vmax else "#0b0b0b")
        fig.colorbar(im, ax=ax, label=cb, pad=0.01)
        ax.set_title(ttl, loc="left")
    axes[1].set_xlabel(L["f8x"])
    fig.suptitle(L["f8"][1].format(L["span"](*span8)), x=0.01, ha="left", fontsize=11.5)
    fig.tight_layout()
    save(fig, lang, L["f8"][0])

open(f"{OUT}/eda_stats.txt", "w").write("\n\n".join(stats) + "\n")
print("\n\n".join(stats))
print("saved to", OUT)
