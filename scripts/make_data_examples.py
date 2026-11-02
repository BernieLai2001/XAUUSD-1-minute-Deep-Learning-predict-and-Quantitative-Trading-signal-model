# 生成数据说明文档用的示例图：一天的现货/期货价格 + 成交量 + Delta；10 秒逐笔成交片段
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

from pathlib import Path
P = str(Path(__file__).resolve().parent.parent)
OUT = f"{P}/数据说明"
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({
    "font.family": ["Hiragino Sans GB", "Arial Unicode MS"],
    "axes.unicode_minus": False,
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": "#52514e",
    "xtick.color": "#52514e", "ytick.color": "#52514e",
    "axes.grid": True, "grid.color": "#e8e7e2", "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
})
BLUE, ORANGE, INK, INK2 = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e"

DAY = "2026-09-15"
m = pd.read_csv(f"{P}/GC_processed/XAUUSD_spot_with_GC_orderflow.csv", index_col=0, parse_dates=True).loc[DAY]
T0, T1 = pd.Timestamp("2026-09-15 20:30:00"), pd.Timestamp("2026-09-15 20:30:10")
t = pd.read_parquet(f"{P}/GC_processed/trades/glbx-mdp3-20260901-20260928.parquet")
s = t.loc[(t.index >= T0) & (t.index < T1)].copy()
s.to_csv(f"{OUT}/示例_10秒逐笔成交_{T0:%Y%m%d_%H%M%S}.csv")

# 中英文各一套：中文图在 数据说明/ 下（文件名不变），英文图在 数据说明/English/ 下
T = {
    "zh": {"dir": OUT, "font": ["Hiragino Sans GB", "Arial Unicode MS"],
           "spot": "XAU/USD 现货（Twelve Data）", "gc": "GC 主力期货（Databento）", "price": "价格（美元/盎司）",
           "t1": f"{DAY}（北京时间）现货与期货价格、GC 每分钟成交量与 Delta",
           "vol": "GC 成交量（手/分钟）", "buy": "主动买占优（Delta > 0）", "sell": "主动卖占优（Delta < 0）",
           "delta": "Delta（手/分钟）", "bj": "北京时间", "f1": "图1_一天的价格_成交量_Delta.png",
           "ask": "卖一价（Ask）", "bid": "买一价（Bid）", "tb": "主动买成交", "ta": "主动卖成交",
           "t2": (f"GC 逐笔成交 10 秒片段：{T0:%Y-%m-%d %H:%M:%S} – {T1:%H:%M:%S}（北京时间，美国 8:30 ET）"
                  f"\n共 {len(s)} 笔、{int(s['size'].sum())} 手；点的大小代表成交手数"),
           "f2": "图2_10秒逐笔成交片段.png"},
    "en": {"dir": f"{OUT}/English", "font": ["Arial", "Helvetica"],
           "spot": "XAU/USD spot (Twelve Data)", "gc": "GC front-month futures (Databento)", "price": "Price (USD per ounce)",
           "t1": f"{DAY} (Beijing time): spot and futures prices, GC volume and delta per minute",
           "vol": "GC volume (lots per minute)", "buy": "Buyer-initiated dominates (delta > 0)",
           "sell": "Seller-initiated dominates (delta < 0)", "delta": "Delta (lots per minute)", "bj": "Beijing time",
           "f1": "Fig1_one_day_price_volume_delta.png",
           "ask": "Best ask", "bid": "Best bid", "tb": "Buyer-initiated trade", "ta": "Seller-initiated trade",
           "t2": (f"GC trades in a 10-second window: {T0:%Y-%m-%d %H:%M:%S} – {T1:%H:%M:%S} (Beijing time, 8:30 ET)"
                  f"\n{len(s)} trades, {int(s['size'].sum())} lots; marker size shows the number of lots"),
           "f2": "Fig2_10_second_trades.png"},
}

for lang, tx in T.items():
    os.makedirs(tx["dir"], exist_ok=True)
    plt.rcParams["font.family"] = tx["font"]

    # ---------- 图 1：一天的数据 ----------
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True, gridspec_kw={"height_ratios": [2.2, 1, 1]})
    ax = axes[0]
    ax.plot(m.index, m["close"], color=BLUE, lw=1.4, label=tx["spot"])
    ax.plot(m.index, m["gc_close"], color=ORANGE, lw=1.4, label=tx["gc"])
    ax.set_ylabel(tx["price"])
    ax.legend(loc="lower left", frameon=False, labelcolor=INK)
    ax.set_title(tx["t1"], loc="left", color=INK, fontsize=13)
    ax = axes[1]
    ax.bar(m.index, m["gc_volume"], width=1 / 1440, color=BLUE, linewidth=0)
    ax.set_ylabel(tx["vol"])
    ax = axes[2]
    d = m["gc_delta"]
    ax.bar(d.index, d.where(d > 0, 0), width=1 / 1440, color=BLUE, linewidth=0, label=tx["buy"])
    ax.bar(d.index, d.where(d < 0, 0), width=1 / 1440, color=ORANGE, linewidth=0, label=tx["sell"])
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_ylabel(tx["delta"])
    ax.legend(loc="lower left", frameon=False, labelcolor=INK, ncol=2)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    ax.set_xlabel(tx["bj"])
    fig.tight_layout()
    fig.savefig(f"{tx['dir']}/{tx['f1']}", dpi=160)
    plt.close(fig)

    # ---------- 图 2：10 秒逐笔成交片段 ----------
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.step(s.index, s["ask_px_00"], where="post", color=INK2, lw=1, ls="--", label=tx["ask"])
    ax.step(s.index, s["bid_px_00"], where="post", color=INK2, lw=1, ls=":", label=tx["bid"])
    for side, color, marker, label in [("B", BLUE, "^", tx["tb"]), ("A", ORANGE, "v", tx["ta"])]:
        x = s[s["side"] == side]
        ax.scatter(x.index, x["price"], s=40 + x["size"] * 40, color=color, marker=marker,
                   edgecolors="#fcfcfb", linewidths=1, label=label, zorder=3)
    ax.set_title(tx["t2"], loc="left", color=INK, fontsize=12)
    ax.set_ylabel(tx["price"])
    ax.ticklabel_format(axis="y", useOffset=False, style="plain")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    ax.legend(loc="upper center", frameon=False, labelcolor=INK, ncol=4, bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    fig.savefig(f"{tx['dir']}/{tx['f2']}", dpi=160)
    plt.close(fig)

print(len(m), "minutes;", len(s), "trades in 10s;", int(s["size"].sum()), "lots")
print(s[["price", "size", "side", "bid_px_00", "ask_px_00", "bid_sz_00", "ask_sz_00"]].head(8).to_string())
