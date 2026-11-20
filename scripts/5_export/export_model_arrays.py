# 把模型输入导出为 numpy 数组（tf 环境没有 pyarrow，读不了 parquet）
# 输入：features/features_1min_scaled.parquet（已补值、长尾处理、标准化）、labels/labels_default.parquet、features/scaler_params.csv
# 输出：model_data/model_arrays.npz
#   X_all：全部行（含补出的分钟）× 33 个特征，float32；样本的窗口从这里按行号切出来
#   seg_id：窗口段编号；end_{train,val,test}：每个样本窗口最后一行（第 t 分钟）在 X_all 里的行号
#   y_{split}：6 个输出头的标签，int8，0 = 止盈、1 = 止损、2 = 超时、−1 = 标签为空（训练时屏蔽）
#   feature_names、head_names、mean、std、log_feature：反标准化画图用
# 样本条件：有成交的分钟、不在换月前几天（pre_roll = 0）、回看窗口 L 行（pipeline.lookback）seg_id 相同、至少 1 个头有标签
import numpy as np
import pandas as pd
from gold_dl.config import CFG, path, save_run_config
from gold_dl.labels import HEADS


L = CFG["pipeline"]["lookback"]
NON_FEAT = ["raw_open", "raw_high", "raw_low", "raw_close", "raw_atr", "instrument_id", "flag_degraded_day",
            "raw_ofi_depth", "is_filled", "seg_id"]
LOG_FEATURES = CFG["features"]["log_then_clip"]

f = pd.read_parquet(path("features", "features_1min_scaled.parquet"))
feat_cols = [c for c in f.columns if c not in NON_FEAT]
assert len(feat_cols) == 33, len(feat_cols)
lab = pd.read_parquet(path("labels", "labels_default.parquet")).reindex(f.index)
scaler = pd.read_csv(path("features", "scaler_params.csv"), index_col="feature")

X = f[feat_cols].to_numpy(dtype=np.float32)
assert np.isfinite(X).all()
seg = f["seg_id"].to_numpy()
n = len(f)
win_ok = np.zeros(n, bool)
win_ok[L - 1:] = seg[L - 1:] == seg[:n - L + 1]
y = lab[HEADS].astype("float").fillna(-1).to_numpy().astype(np.int8)
base = (f["is_filled"] == 0).to_numpy() & (lab["pre_roll"] == 0).to_numpy() & win_ok & (y >= 0).any(axis=1)
split = lab["split"].astype(str).to_numpy()

out = {"X_all": X, "seg_id": seg.astype(np.int32), "feature_names": np.array(feat_cols),
       "head_names": np.array(HEADS), "time_ns": f.index.asi8,
       "mean": scaler.reindex(feat_cols)["train_mean"].fillna(0).to_numpy(np.float32),
       "std": scaler.reindex(feat_cols)["train_std"].fillna(1).to_numpy(np.float32),
       "log_feature": np.isin(feat_cols, LOG_FEATURES)}
for s in ["train", "val", "test"] + (["test_b"] if (split == "test_b").any() else []):
    end = np.where(base & (split == s))[0]
    out[f"end_{s}"] = end.astype(np.int64)
    out[f"y_{s}"] = y[end]
    if s == "test_b":   # 测试集 B 只导出，不打印任何统计
        print("test_b exported (not inspected)")
    else:
        print(s, "samples", len(end), "label NA share per head", (y[end] == -1).mean(axis=0).round(4))
np.savez(path("model_arrays"), **out)
print("X_all", X.shape, X.dtype, f"{X.nbytes / 1e6:.1f} MB")
save_run_config(path("model_arrays").parent, "export_model_arrays")
