# 导出默认组合（config.yaml 的 labels.default_k_sl / default_rr）的 6 个输出头的标签
# 同时按训练集统计重新跑一遍选组合的规则，结果与 config 不同时给出提示（不改变导出的组合）
import pandas as pd
from gold_dl.config import CFG, path, save_run_config
from gold_dl.labels import HEADS, HS, SIDES, label_col
P = str(path("labels"))
MIN_COST_RATIO = 10  # 止盈距离至少为固定成本的 10 倍
# 按训练集统计选默认组合：止盈距离明显大于固定成本的组合里，6 个输出头中最小类别占比最大的那组
combo = pd.read_csv(f"{P}/label_combo_summary.csv")
ok = combo[combo.min_tp_over_fixed_cost >= MIN_COST_RATIO]
best = ok.sort_values(["min_class_share", "mean_entropy"], ascending=False).iloc[0]
K, RR = CFG["labels"]["default_k_sl"], CFG["labels"]["default_rr"]
print("default combo (config.yaml): k_SL", K, "rr", RR)
if (best.k_SL, best.rr) != (K, RR):
    print(f"注意：按训练集规则选出的组合是 k_SL {best.k_SL:g}、rr {best.rr:g}，与 config.yaml 不同")
l = pd.read_parquet(f"{P}/labels_all.parquet")
cols = [label_col(s, h, K, RR) for s in SIDES for h in HS]   # 顺序与 HEADS 相同
d = l[cols + ["entry_ask", "entry_bid", "spread_ticks", "wide_spread", "pre_roll", "split"]]
d = d.rename(columns=dict(zip(cols, HEADS)))
d.to_parquet(f"{P}/labels_default.parquet")
print(d.columns.tolist(), len(d))
print(d[d.split == "train"].filter(regex="^(long|short)").apply(lambda s: s.value_counts(normalize=True)).round(3))
save_run_config(path("labels"), "save_default_labels")
