# 标签审核：随机抽 60 个训练样本（固定种子），用独立的逐分钟循环重新判断默认组合（k_SL = 2，1:1）的 6 个标签，
# 与标签文件对比；同时标出可疑情况，并画出每个样本入场后 15 分钟的买一/卖一高低价和止盈止损线
# 时间一律用美东时间（数据索引是北京时间，只在输出时转换）
# 输出：数据说明/标签审核/标签审核_抽样60.csv；图：中文/标签审核_图_1–5.png，English/Label_audit_page_1–5.png
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path
P = str(Path(__file__).resolve().parent.parent)
OUT = f"{P}/数据说明/标签审核"
N_SAMPLE, SEED, K, FFILL = 60, 42, 2.0, 10
lab = pd.read_parquet(f"{P}/labels/labels_default.parquet")
atr = pd.read_parquet(f"{P}/features/features_1min.parquet", columns=["raw_atr14", "flag_degraded_day", "instrument_id"])
heads = [f"{s}_H{h}" for s in ["long", "short"] for h in [15, 30, 60]]
pool = lab[(lab["split"] == "train") & (lab["pre_roll_3d"] == 0) & lab[heads].notna().all(axis=1)]
smp = pool.sample(N_SAMPLE, random_state=SEED).sort_index()

q = pd.read_parquet(f"{P}/GC_processed/GC_quote_1min.parquet")
q = q.reindex(pd.date_range(q.index.min(), q.index.max(), freq="min"))
upd = q["n"].notna()
for side in ["bid", "ask"]:
    last = q[f"{side}_close"].ffill(limit=FFILL)
    for c in ["open", "high", "low", "close"]:
        q[f"{side}_{c}"] = q[f"{side}_{c}"].fillna(last)

rolls = atr.index[atr["instrument_id"].ne(atr["instrument_id"].shift()) & atr["instrument_id"].shift().notna()]


def to_et(t):
    return t.tz_localize("Asia/Shanghai").tz_convert("America/New_York").tz_localize(None)


def judge(win, entry, d, side):
    # 逐分钟往后看：做多看买一，做空看卖一；同一分钟两条线都碰到记为止损
    for _, r in win.iterrows():
        if side == "long":
            tp, sl = r.bid_high >= entry + d, r.bid_low <= entry - d
        else:
            tp, sl = r.ask_low <= entry - d, r.ask_high >= entry + d
        if sl:
            return 1, tp
        if tp:
            return 0, False
    return 2, False


rows = []
for t, s in smp.iterrows():
    e = t + pd.Timedelta(minutes=1)
    a = atr.at[t, "raw_atr14"]
    rec = {"datetime": t, "entry_ask": s.entry_ask, "entry_bid": s.entry_bid, "spread_ticks": s.spread_ticks, "atr14": a}
    both = False
    for h in [15, 30, 60]:
        win = q.loc[e:e + pd.Timedelta(minutes=h - 1)]
        d = K * a * np.sqrt(h / 15)
        for side, entry in [("long", s.entry_ask), ("short", s.entry_bid)]:
            y, b = judge(win, entry, d, side)
            both |= b
            rec[f"{side}_H{h}"] = int(s[f"{side}_H{h}"])
            rec[f"{side}_H{h}_recheck"] = y
    w60 = slice(e, e + pd.Timedelta(minutes=59))
    rec["mismatch"] = sum(rec[f"{c}"] != rec[f"{c}_recheck"] for c in heads)
    rec["flag_both_hit_same_minute"] = both
    rec["flag_wide_spread"] = bool(s.wide_spread)
    rec["flag_stale_quote_minutes_in_60"] = int((~upd.loc[w60]).sum())
    rec["flag_degraded_day"] = bool(atr.at[t, "flag_degraded_day"])
    rec["flag_within_1day_before_roll"] = bool(((rolls - t) > pd.Timedelta(0)).any() and ((rolls - t)[(rolls - t) > pd.Timedelta(0)].min() <= pd.Timedelta(days=1)))
    rec["datetime_et"] = to_et(t)
    rows.append(rec)

r = pd.DataFrame(rows).set_index("datetime")
r.set_index("datetime_et").rename_axis("datetime_et").to_csv(f"{OUT}/标签审核_抽样60.csv", encoding="utf-8-sig")
print("samples", len(r), "| labels checked", len(r) * 6, "| mismatches", int(r["mismatch"].sum()))
print("label distribution in sample:", {h: r[h].value_counts().sort_index().to_dict() for h in heads})
for c in [c for c in r.columns if c.startswith("flag_")]:
    v = r[c]
    print(f"{c}: {int((v > 0).sum())} samples" + (f" (max {v.max()})" if v.dtype != bool else ""))
print("ET hour of samples:", r["datetime_et"].dt.hour.value_counts().sort_index().to_dict())

# 画图：每个样本入场后 15 分钟，买一（蓝）/卖一（橙）每分钟高低区间，做多止盈止损线（实线，基于卖一入场）、做空（虚线，基于买一入场）
import os
plt.rcParams.update({"font.family": ["Hiragino Sans GB", "Arial Unicode MS"], "axes.unicode_minus": False})
TXT = {
    "zh": dict(dir="中文", file="标签审核_图_{}.png", name={0: "止盈", 1: "止损", 2: "超时"},
               title="{}（美东）  多:{} 空:{}", xlabel="入场后第几分钟",
               sup="标签审核（第 {}/5 页）：入场后 15 分钟，时间为美东时间。蓝 = 买一高低，橙 = 卖一高低；"
                   "蓝实线 = 做多止盈/止损（卖一入场 ± 2×ATR），橙虚线 = 做空止盈/止损（买一入场 ∓ 2×ATR）"),
    "en": dict(dir="English", file="Label_audit_page_{}.png", name={0: "TP", 1: "SL", 2: "Timeout"},
               title="{} ET  Long: {}  Short: {}", xlabel="Minutes after entry",
               sup="Label audit (page {}/5): 15 minutes after entry, times in US Eastern. Blue = bid high-low, orange = ask high-low; "
                   "blue solid = long TP/SL (ask entry ± 2×ATR), orange dashed = short TP/SL (bid entry ∓ 2×ATR)"),
}
WD_ZH = "一二三四五六日"
for lang, T in TXT.items():
    os.makedirs(f"{OUT}/{T['dir']}", exist_ok=True)
    for page in range(5):
        fig, axes = plt.subplots(3, 4, figsize=(16, 10.5))
        for ax, (t, rec) in zip(axes.flat, list(r.iterrows())[page * 12:(page + 1) * 12]):
            e = t + pd.Timedelta(minutes=1)
            win = q.loc[e:e + pd.Timedelta(minutes=14)]
            x = np.arange(len(win))
            ax.vlines(x - 0.12, win.bid_low, win.bid_high, color="#2a78d6", lw=3)
            ax.vlines(x + 0.12, win.ask_low, win.ask_high, color="#eb6834", lw=3)
            d = K * rec.atr14
            for y in [rec.entry_ask + d, rec.entry_ask - d]:
                ax.axhline(y, color="#2a78d6", ls="-", lw=1)
            for y in [rec.entry_bid - d, rec.entry_bid + d]:
                ax.axhline(y, color="#eb6834", ls="--", lw=1)
            et = rec.datetime_et
            stamp = f"{et:%Y-%m-%d} 周{WD_ZH[et.weekday()]} {et:%H:%M}" if lang == "zh" else f"{et:%Y-%m-%d %a %H:%M}"
            ax.set_title(T["title"].format(stamp, T["name"][rec.long_H15], T["name"][rec.short_H15]), fontsize=9)
            ax.set_xlabel(T["xlabel"], fontsize=7)
            ax.tick_params(labelsize=7)
        fig.suptitle(T["sup"].format(page + 1), fontsize=10)
        fig.tight_layout()
        fig.savefig(f"{OUT}/{T['dir']}/{T['file'].format(page + 1)}", dpi=110)
        plt.close(fig)
