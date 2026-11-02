# 标准化：在长尾处理后的特征上做 z-score，(x − 均值) / 标准差，均值和标准差只用训练集计算，再原样用到验证集和测试集
# 拟合只用训练集样本行：有成交的分钟（is_filled = 0），且不在换月前 3 天（pre_roll_3d = 0）；补出的分钟同样做变换
#   22 个连续特征做标准化；9 个 0/1 哑变量（星期、交易时段）和 tod_sin、tod_cos 保持原样
#   列名不变，raw_*、instrument_id、flag_degraded_day 等非输入列原样保留
# 输入：features/features_1min_tailfix.parquet（不修改）
# 输出：features/features_1min_scaled.parquet，features/scaler_params.csv（训练集均值、标准差，以及标准化后各数据集的均值、标准差）
import numpy as np
import pandas as pd

from pathlib import Path
P = str(Path(__file__).resolve().parent.parent)

DUMMIES = ["dow_mon", "dow_tue", "dow_wed", "dow_thu", "dow_fri", "sess_london", "sess_ny", "sess_overlap", "sess_asia"]
KEEP_AS_IS = DUMMIES + ["tod_sin", "tod_cos"]
CONTINUOUS = [
    "d_open", "d_high", "d_low", "d_close",
    "ma_dist_15", "ma_dist_60", "ma_dist_240", "er_15", "er_60",
    "ret_5", "ret_15", "ret_60", "rsi_14", "atr_rel", "vol_ratio_5_60",
    "delta_ratio_1", "delta_ratio_5", "delta_ratio_15", "book_imb", "ofi_depth_adj",
    "spread_atr", "relvol_60",
]
assert len(CONTINUOUS) == 22 and len(KEEP_AS_IS) == 11

f = pd.read_parquet(f"{P}/features/features_1min_tailfix.parquet")
lab = pd.read_parquet(f"{P}/labels/labels_default.parquet", columns=["split", "pre_roll_3d"]).reindex(f.index)
split = lab["split"].astype(str)
sample = (f["is_filled"] == 0) & (lab["pre_roll_3d"] == 0)
train = (split == "train") & sample

mean = f.loc[train, CONTINUOUS].mean()
std = f.loc[train, CONTINUOUS].std(ddof=0)
assert (std > 0).all(), std[std <= 0]
out = f.copy()
out[CONTINUOUS] = ((f[CONTINUOUS] - mean) / std).astype("float32")

params = pd.DataFrame({"train_mean": mean, "train_std": std})
for s in ["train", "val", "test"]:
    params[f"{s}_mean_after"] = out.loc[(split == s) & sample, CONTINUOUS].mean()
    params[f"{s}_std_after"] = out.loc[(split == s) & sample, CONTINUOUS].std(ddof=0)
params.index.name = "feature"
params.to_csv(f"{P}/features/scaler_params.csv", float_format="%.6g")
out.to_parquet(f"{P}/features/features_1min_scaled.parquet")

pd.set_option("display.width", 200)
print(params.round(3).to_string())
print("未标准化的列（训练集取值范围）：")
print(out.loc[train, KEEP_AS_IS].agg(["min", "max"]).round(3).to_string())
print("saved features_1min_scaled.parquet", out.shape)
