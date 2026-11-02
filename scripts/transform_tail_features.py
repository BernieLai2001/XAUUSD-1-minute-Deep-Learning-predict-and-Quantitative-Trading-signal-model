# 长尾特征处理：只用训练集算阈值，再原样用到验证集和测试集
# 拟合阈值和统计只用训练集样本行：有成交的分钟（is_filled = 0），且不在换月前 3 天（pre_roll_3d = 0）
# 补出的分钟可能出现 0（如成交量为 0 时 relvol_60 = 0），取对数前先设一个极小的下限，再被缩尾到下限阈值，不会出现负无穷
#   不用 ln(1 + x)：relvol_60 大多小于 1，ln(1 + x) 在这个范围里接近 x，处理后偏度仍有 1.05（ln(x) 为 −0.02）
#   全为正的特征（spread_atr、relvol_60、atr_rel、vol_ratio_5_60）：先取自然对数，再按训练集 1% / 99% 分位数缩尾
#   有正有负的特征（ofi_depth_adj、d_open、d_high、d_low）：按训练集 1% / 99% 分位数缩尾
# 列名不变，其余特征原样保留；标准化留到建模阶段（同样只用训练集的均值和标准差）
# 输入：features/features_1min.parquet（不修改）
# 输出：features/features_1min_tailfix.parquet，features/tail_transform_params.csv（阈值与处理前后统计）
import numpy as np
import pandas as pd

from pathlib import Path
P = str(Path(__file__).resolve().parent.parent)

LOWER_Q, UPPER_Q = 0.01, 0.99
LOG_THEN_CLIP = ["spread_atr", "relvol_60", "atr_rel", "vol_ratio_5_60"]
CLIP_ONLY = ["ofi_depth_adj", "d_open", "d_high", "d_low"]

f = pd.read_parquet(f"{P}/features/features_1min.parquet")
lab = pd.read_parquet(f"{P}/labels/labels_default.parquet", columns=["split", "pre_roll_3d"]).reindex(f.index)
split = lab["split"].astype(str)
sample = (f["is_filled"] == 0) & (lab["pre_roll_3d"] == 0)
train = (split == "train") & sample
LOG_FLOOR = 1e-12
out = f.copy()
rows = []
for c in LOG_THEN_CLIP + CLIP_ONLY:
    x = f[c]
    if c in LOG_THEN_CLIP:
        assert (x[f["is_filled"] == 0].dropna() > 0).all(), f"{c} 在有成交的分钟里有非正数，不能取对数"
        x = np.log(x.clip(lower=LOG_FLOOR))
    lo, hi = x[train].quantile([LOWER_Q, UPPER_Q])
    out[c] = x.clip(lo, hi)
    rows.append({
        "feature": c,
        "method": "log + winsorize 1%/99%" if c in LOG_THEN_CLIP else "winsorize 1%/99%",
        "lower": lo, "upper": hi,
        "lower_original_units": np.exp(lo) if c in LOG_THEN_CLIP else lo,
        "upper_original_units": np.exp(hi) if c in LOG_THEN_CLIP else hi,
        "kurt_before_train": f.loc[train, c].kurt(), "kurt_after_train": out.loc[train, c].kurt(),
        **{f"clipped_share_{s}": ((x < lo) | (x > hi))[(split == s) & sample].mean() for s in ["train", "val", "test"]},
        "skew_before_train": f.loc[train, c].skew(), "skew_after_train": out.loc[train, c].skew(),
        "mean_after_train": out.loc[train, c].mean(), "std_after_train": out.loc[train, c].std(),
        "min_before_train": f.loc[train, c].min(), "max_before_train": f.loc[train, c].max(),
    })

params = pd.DataFrame(rows).set_index("feature")
params.to_csv(f"{P}/features/tail_transform_params.csv", float_format="%.6g")
out.to_parquet(f"{P}/features/features_1min_tailfix.parquet")
pd.set_option("display.width", 200)
print(params.round(4).to_string())
print("saved features_1min_tailfix.parquet", out.shape)
