# 按训练集统计选出 M3 默认组合，导出 6 个输出头的标签
import pandas as pd
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
P = str(ROOT / "labels")
MIN_COST_RATIO = 10  # 止盈距离至少为固定成本的 10 倍
# 按训练集统计选默认组合：止盈距离明显大于固定成本的组合里，6 个输出头中最小类别占比最大的那组
combo = pd.read_csv(f"{P}/label_combo_summary.csv")
ok = combo[combo.min_tp_over_fixed_cost >= MIN_COST_RATIO]
best = ok.sort_values(["min_class_share", "mean_entropy"], ascending=False).iloc[0]
K, RR = best.k_SL, best.rr
print("default combo: k_SL", K, "rr", RR)
l = pd.read_parquet(f"{P}/labels_all.parquet")
cols = [f"{s}_H{h}_sl{K:g}_rr{RR:g}" for s in ["long", "short"] for h in [15, 30, 60]]
d = l[cols + ["entry_ask", "entry_bid", "spread_ticks", "wide_spread", "pre_roll_3d", "split"]]
d = d.rename(columns={c: c.rsplit("_sl", 1)[0] for c in cols})
d.to_parquet(f"{P}/labels_default.parquet")
print(d.columns.tolist(), len(d))
print(d[d.split == "train"].filter(regex="^(long|short)").apply(lambda s: s.value_counts(normalize=True)).round(3))
