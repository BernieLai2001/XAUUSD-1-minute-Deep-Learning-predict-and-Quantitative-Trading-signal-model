# 第 2 节数据质量统计：损坏文件、重复、格式/无效 K 线、缺失分钟、删除的样本、换月、side = N、元数据
# 结果打印到屏幕，并存为 数据说明/第2节_数据质量统计.txt
import hashlib, io, json
import numpy as np
import pandas as pd
import databento as db
from gold_dl.config import path
from gold_dl.timeutil import bj_to_et
from gold_dl.labels import FFILL_LIMIT, HEADS, HS


SRC = path("mbp1_batch")
buf = io.StringIO()


def out(*a):
    print(*a)
    print(*a, file=buf)


# ---------- 1. 损坏文件：SHA-256 与 manifest 对比；各文件能否完整读入 ----------
out("== 1. file integrity")
man = json.load(open(f"{SRC}/manifest.json"))
bad = 0
for f in man["files"]:
    h = hashlib.sha256()
    with open(f"{SRC}/{f['filename']}", "rb") as fh:
        for b in iter(lambda: fh.read(1 << 24), b""):
            h.update(b)
    ok = "sha256:" + h.hexdigest() == f["hash"]
    bad += not ok
out(f"MBP-1 job files checked: {len(man['files'])}, hash mismatches: {bad}")
d3 = db.DBNStore.from_file(path("raw_ohlcv1m")).to_df()
out(f"D3 OHLCV-1m dbn records read: {len(d3)}")
spot = pd.read_csv(path("spot_xauusd"), index_col="datetime", parse_dates=True)
out(f"D1 spot csv rows read: {len(spot)}, unparseable/NaN price cells: {int(spot.isna().sum().sum())}")

# ---------- 2. 重复 ----------
out("\n== 2. duplicates")
out(f"D1 spot duplicate timestamps: {int(spot.index.duplicated().sum())}")
out(f"D3 duplicate timestamps: {int(d3.index.duplicated().sum())}")
minute_files = {}
import glob, os
for p in sorted(glob.glob(str(path("processed", "trades", "*.parquet")))):
    t = pd.read_parquet(p, columns=["size", "side"])
    minute_files[os.path.basename(p)] = t
mins = pd.concat([pd.Series(t.index.floor("min").unique()) for t in minute_files.values()])
out(f"GC minutes appearing in more than one monthly file (merged at processing): {int(mins.duplicated().sum())}")
of = pd.read_csv(path("processed", "GC_1min_orderflow.csv"), index_col="datetime", parse_dates=True)
feat = pd.read_parquet(path("features", "features_1min.parquet"),
                       columns=["raw_open", "raw_high", "raw_low", "raw_close", "instrument_id", "flag_degraded_day"])
out(f"duplicate timestamps after processing: orderflow {int(of.index.duplicated().sum())}, features {int(feat.index.duplicated().sum())}")

# ---------- 3. 格式 / 无效 K 线 ----------
out("\n== 3. invalid bars")
def bar_check(name, o, h, l, c, v=None):
    r = {"high<low": int((h < l).sum()),
         "open/close outside [low, high]": int(((o > h) | (o < l) | (c > h) | (c < l)).sum()),
         "price <= 0": int(((o <= 0) | (h <= 0) | (l <= 0) | (c <= 0)).sum()),
         "flat bar (O=H=L=C)": int(((o == h) & (h == l) & (l == c)).sum())}
    if v is not None:
        r["volume <= 0"] = int((v <= 0).sum())
    out(f"{name}: rows {len(o)}, {r}")
bar_check("D1 spot", spot.open, spot.high, spot.low, spot.close)
bar_check("GC 1-min (from MBP-1)", of.open, of.high, of.low, of.close, of.volume)
q = pd.read_parquet(path("processed", "GC_quote_1min.parquet"))
out(f"GC quote 1-min: rows {len(q)}, bid_high<bid_low {int((q.bid_high < q.bid_low).sum())}, "
    f"ask_high<ask_low {int((q.ask_high < q.ask_low).sum())}, close crossed (ask<=bid) {int((q.ask_close <= q.bid_close).sum())}")

# ---------- 4. 缺失分钟：相对 CME 正常交易分钟（周日 18:00 到周五 17:00 美东，每日 17:00–18:00 休市） ----------
out("\n== 4. missing minutes")
idx = feat.index
grid = pd.date_range(idx.min(), idx.max(), freq="min")
et = bj_to_et(grid)
mod = et.hour * 60 + et.minute
wd = et.dayofweek
open_ = (mod < 17 * 60) | (mod >= 18 * 60)
open_ &= ~((wd == 5) | ((wd == 4) & (mod >= 17 * 60)) | ((wd == 6) & (mod < 18 * 60)))
exp = grid[open_]
missing = exp.difference(idx)
out(f"expected CME trading minutes {len(exp)}, present {len(idx.intersection(exp))}, missing {len(missing)} ({len(missing) / len(exp):.2%})")
md = pd.Series(1, index=missing).groupby(missing.date).size().sort_values(ascending=False)
out("days with most missing minutes (holiday early closes etc.):", md.head(6).to_dict())
gap = idx.to_series().diff().dt.total_seconds().div(60).iloc[1:]
out(f"gaps between bars: 1 min {(gap == 1).mean():.2%}, 2-5 {gap.between(2, 5).mean():.2%}, 6-30 {gap.between(6, 30).mean():.2%}, >30 {(gap > 30).mean():.2%}")

# ---------- 5. 删除的样本 ----------
out("\n== 5. removed samples")
out(f"GC 1-min bars {len(of)}, feature rows {len(feat)}, warm-up removed {len(of) - len(feat)}")
lab = pd.read_parquet(path("labels", "labels_default.parquet"))
out(f"split counts: {lab['split'].astype(str).value_counts().to_dict()}")
heads = HEADS
# 标签为空的原因：入场分钟无报价更新 / 持有期跨换月 / 持有期内连续超过 FFILL_LIMIT 分钟无报价（如休市）/ 数据末尾
qq = q.reindex(pd.date_range(q.index.min(), q.index.max(), freq="min"))
has = qq["n"].notna().to_numpy()
inst_q = qq["instrument_id"].ffill(limit=FFILL_LIMIT).to_numpy()
pos = qq.index.get_indexer(lab.index)
entry_ok = lab["entry_ask"].notna().to_numpy()
for h in HS:
    na = lab[f"long_H{h}"].isna().to_numpy() | lab[f"short_H{h}"].isna().to_numpy()
    GAP_KEY = f"quote gap > {FFILL_LIMIT} min (halt)"
    reasons = {"no quote at entry": 0, "crosses roll": 0, GAP_KEY: 0, "end of data": 0}
    for i in np.where(na)[0]:
        if not entry_ok[i]:
            reasons["no quote at entry"] += 1
            continue
        e = pos[i] + 1
        if e + h > len(qq):
            reasons["end of data"] += 1
            continue
        w = inst_q[e:e + h]
        if np.isnan(w).any():
            reasons[GAP_KEY] += 1
        elif (w != w[0]).any():
            reasons["crosses roll"] += 1
        else:
            reasons[GAP_KEY] += 1
    tr = (lab["split"] == "train").to_numpy()
    out(f"H{h}: label NaN {int(na.sum())} ({na.mean():.2%}), in train {int((na & tr).sum())}; reasons {reasons}")
out(f"degraded-day rows (flag kept, NOT removed): {int(feat['flag_degraded_day'].sum())}; "
    f"dates {sorted(set(feat.index[feat['flag_degraded_day'] == 1].strftime('%Y-%m-%d')))}")
out(f"wide-spread minutes (kept, flagged): {int(lab['wide_spread'].sum())}")

# ---------- 6. 换月 ----------
out("\n== 6. rolls")
inst = feat["instrument_id"]
chg = inst.ne(inst.shift()) & inst.shift().notna()
out(f"instrument_id changes in feature period: {int(chg.sum())}; contracts: {inst.unique().tolist()}")
for t in feat.index[chg]:
    i = feat.index.get_loc(t)
    pc, nc = feat["raw_close"].iloc[i - 1], feat["raw_open"].iloc[i]
    out(f"  {t} (Beijing): {inst.iloc[i - 1]} -> {inst.iloc[i]}, old close {pc}, new open {nc}, gap {nc - pc:+.1f}")

# ---------- 7. side = N ----------
out("\n== 7. side = N trades")
allt = pd.concat(minute_files.values())
n_trades, n_vol = int((allt["side"] == "N").sum()), int(allt.loc[allt["side"] == "N", "size"].sum())
out(f"trades {len(allt)}, side=N trades {n_trades} ({n_trades / len(allt):.2%}), side=N volume {n_vol} ({n_vol / allt['size'].sum():.2%})")
ts = of.index.to_series()
seg = (ts.diff() > pd.Timedelta(minutes=30)).cumsum()
first_min = set(ts[ts.groupby(seg).transform("first") == ts].index)
nm = allt[allt["side"] == "N"].index.floor("min")
in_open = pd.Series(nm).isin(first_min)
nvol = allt.loc[allt["side"] == "N", "size"].to_numpy()
out(f"side=N volume falling in the first minute after a reopen: {nvol[in_open.to_numpy()].sum() / nvol.sum():.1%} "
    f"(first minutes are {len(first_min)} of {len(of)} minutes)")
out(f"total volume {int(of.volume.sum())}, buy {int(of.buy_volume.sum())}, sell {int(of.sell_volume.sum())}, "
    f"share with side known {(of.buy_volume.sum() + of.sell_volume.sum()) / of.volume.sum():.2%}")

open(path("notes", "第2节_数据质量统计.txt"), "w").write(buf.getvalue())
