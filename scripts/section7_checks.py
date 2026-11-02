# 第 7 节核对：各特征按训练集十分位分组后，标签比例的变化幅度（单变量、只用训练集样本）
#   方向信号 = 每组 (做多止盈率 − 做空止盈率)，看最高组与最低组之差；超时幅度 = 每组超时比例的最大 − 最小
#   噪声参照：把标签随机打乱 20 次，同样计算，取 95% 分位数
# 另统计：做多/做空的标签分布、每月类别比例、各持有期超时比例
import numpy as np
import pandas as pd

from pathlib import Path
P = str(Path(__file__).resolve().parent.parent)
f = pd.read_parquet(f"{P}/features/features_1min_tailfix.parquet")
l = pd.read_parquet(f"{P}/labels/labels_default.parquet").reindex(f.index)
m = (l["split"].astype(str) == "train") & (f["is_filled"] == 0) & (l["pre_roll_3d"] == 0)
NON = ["raw_open", "raw_high", "raw_low", "raw_close", "raw_atr14", "instrument_id", "flag_degraded_day",
       "raw_ofi_depth", "is_filled", "seg_id"]
feats = [c for c in f.columns if c not in NON and not c.startswith(("dow_", "sess_"))]
H = 30
yl, ys = l.loc[m, f"long_H{H}"], l.loc[m, f"short_H{H}"]
ok = yl.notna() & ys.notna()
yl, ys = yl[ok].astype(int).to_numpy(), ys[ok].astype(int).to_numpy()
rng = np.random.default_rng(42)


def spreads(q, yl, ys):
    d = pd.DataFrame({"q": q, "dl": (yl == 0).astype(float) - (ys == 0), "to": (yl == 2).astype(float)})
    g = d.groupby("q").mean()
    return (g["dl"].iloc[-1] - g["dl"].iloc[0]) * 100, (g["to"].max() - g["to"].min()) * 100


rows = []
for c in feats:
    x = f.loc[m, c][ok]
    q = pd.qcut(x.rank(method="first"), 10, labels=False).to_numpy()
    dsig, to = spreads(q, yl, ys)
    null = [spreads(q, *(lambda p: (yl[p], ys[p]))(rng.permutation(len(yl)))) for _ in range(20)]
    rows.append({"feature": c, "direction_top_minus_bottom_pp": dsig, "timeout_range_pp": to,
                 "null95_direction": np.quantile([abs(a) for a, _ in null], 0.95), "null95_timeout": np.quantile([b for _, b in null], 0.95)})
r = pd.DataFrame(rows).set_index("feature").round(2)
r = r.reindex(r["direction_top_minus_bottom_pp"].abs().sort_values(ascending=False).index)
pd.set_option("display.width", 200)
print(f"H{H}，训练集样本 {ok.sum():,}")
print(r.to_string())
r.to_csv(f"{P}/数据说明/第7节_特征单变量检查.csv", encoding="utf-8-sig")
