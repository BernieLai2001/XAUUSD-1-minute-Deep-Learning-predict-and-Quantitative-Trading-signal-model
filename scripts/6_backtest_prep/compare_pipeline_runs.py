# 核对：另一个根目录（--other，例如加入测试集 B 后重跑的流水线）的输出，与本项目现有输出在"原有时间段"是否完全一致
#   python scripts/6_backtest_prep/compare_pipeline_runs.py --other ~/Desktop/gold_build_check
# 只比较测试集 B 开始前 gap_min 分钟之前的行（原有的训练、验证、测试集）和只用训练集拟合的参数；
# 测试集 B 的行一律不读取内容、不打印，只输出"一致 / 不一致"，以及原测试集末尾因留出隔离期而变化的行数
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from gold_dl.config import CFG, path

ap = argparse.ArgumentParser()
ap.add_argument("--other", required=True)
O = Path(ap.parse_args().other).expanduser()
cut = pd.Timestamp(CFG["split"]["test_b_start"]) - pd.Timedelta(minutes=CFG["split"]["gap_min"])
ok_all = True


def report(name, ok, note=""):
    global ok_all
    ok_all &= bool(ok)
    print(f"{'一致  ' if ok else '不一致'} {name} {note}")


def same_frame(a, b):
    if list(a.columns) != list(b.columns) or not a.index.equals(b.index):
        return False
    for c in a.columns:
        x, y = a[c], b[c]
        if x.dtype.kind in "fc" or y.dtype.kind in "fc":
            if not np.allclose(x.to_numpy(float), y.to_numpy(float), equal_nan=True, rtol=0, atol=0):
                return False
        elif not x.astype(str).equals(y.astype(str)):
            return False
    return True


# 1. 行数据：只取原有时间段
for sub, f in [("features", "features_1min.parquet"), ("features", "features_1min_tailfix.parquet"),
               ("features", "features_1min_scaled.parquet")]:
    a = pd.read_parquet(path(sub, f))
    b = pd.read_parquet(O / sub / f)
    b = b[b.index < a.index.max() + pd.Timedelta(minutes=1)]
    keep = a.index < cut   # 隔离期里的行不被任何样本使用；原下载最后一分钟在文件末尾被截断，加入新数据后会变完整
    report(f"{f}（{cut} 之前）", same_frame(a[keep], b[keep]))
    n_diff = sum(not same_frame(a[~keep].iloc[[i]], b[~keep].iloc[[i]]) for i in range((~keep).sum()))
    print(f"       {f}：隔离期内不同的行数 = {n_diff}")

for f in ["labels_all.parquet", "labels_default.parquet"]:
    a = pd.read_parquet(path("labels", f))
    b = pd.read_parquet(O / "labels" / f)
    b = b[b.index <= a.index.max()]
    keep = a.index < cut
    report(f"{f}（{cut} 之前）", same_frame(a[keep], b[keep]))
    changed = (a.loc[~keep, "split"].astype(str) != b.loc[~keep, "split"].astype(str)).sum()
    print(f"       {f}：原测试集最后 {CFG['split']['gap_min']} 分钟留作与测试集 B 的隔离期，划分改变的行数 = {changed}")

# 2. 只用训练集拟合的参数
for sub, f in [("features", "ofi_depth_seasonal_factors.csv"), ("features", "tail_transform_params.csv"),
               ("features", "scaler_params.csv"), ("labels", "label_grid_stats.csv"), ("labels", "label_combo_summary.csv")]:
    a, b = pd.read_csv(path(sub, f)), pd.read_csv(O / sub / f)
    cols = [c for c in a.columns if not c.startswith(("test_", "clipped_share_test"))]   # 原测试集的事后统计会因隔离期略变
    report(f, same_frame(a[cols], b[cols]))

# 3. 模型数组：训练集、验证集完全一致；原测试集只少了隔离期里的样本
A, Bz = np.load(path("model_arrays")), np.load(O / "model_data" / "model_arrays.npz")
n_old = len(A["time_ns"])
r_old = A["time_ns"] < cut.value
report(f"model_arrays: X_all（原有行，{cut} 之前）", np.array_equal(A["X_all"][r_old], Bz["X_all"][:n_old][r_old], equal_nan=True)
       and np.array_equal(A["time_ns"], Bz["time_ns"][:n_old]))
for s in ["train", "val"]:
    report(f"model_arrays: {s} 样本和标签", np.array_equal(A[f"end_{s}"], Bz[f"end_{s}"]) and np.array_equal(A[f"y_{s}"], Bz[f"y_{s}"]))
ta = A["time_ns"][A["end_test"]]
keep = ta < cut.value
report("model_arrays: test 样本和标签（隔离期之前）",
       np.array_equal(A["end_test"][keep], Bz["end_test"]) and np.array_equal(A["y_test"][keep], Bz["y_test"]))
print(f"       原测试集样本 {len(ta)} → {int(keep.sum())}（隔离期内的 {int((~keep).sum())} 个样本移出）")
print("       测试集 B 已导出：", "end_test_b" in Bz.files, "（内容未读取）")

# 4. 交易结果（原有行）
Ta, Tb = np.load(path("model_data", "trade_outcomes.npz")), np.load(O / "model_data" / "trade_outcomes.npz")
rows = np.concatenate([A["end_train"], A["end_val"], A["end_test"][keep]])
report("trade_outcomes（训练、验证、隔离期前的测试样本）",
       all(np.array_equal(Ta[k][rows], Tb[k][rows], equal_nan=True) for k in ["outcome", "exit_min", "pnl_usd", "r_unit"]))
print("\n全部通过" if ok_all else "\n有不一致，需要检查")
