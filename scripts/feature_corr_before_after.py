# 删除 range_pos_60 前后的特征相关系数表（训练集样本：有成交、不在换月前 3 天）
# range_pos_60 按删除前的定义重算：复权后 (收盘价 − 60 根 K 线最低价) / (60 根最高价 − 最低价)，区间为 0 时为 0.5
# 输出：数据说明/特征相关系数/相关系数_删除前_34.csv、相关系数_删除后_33.csv 和对应热力图，高相关特征对.txt
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pathlib import Path
P = str(Path(__file__).resolve().parent.parent)
OUT = f"{P}/数据说明/特征相关系数"
os.makedirs(OUT, exist_ok=True)
NON_FEAT = ["raw_open", "raw_high", "raw_low", "raw_close", "raw_atr14", "instrument_id", "flag_degraded_day",
            "raw_ofi_depth", "is_filled", "seg_id"]

f = pd.read_parquet(f"{P}/features/features_1min.parquet")
feat_cols = [c for c in f.columns if c not in NON_FEAT]
# 与 build_features.py 相同的比例向前复权
inst = f["instrument_id"]
factor = pd.Series(1.0, index=f.index)
for i in np.where(inst.ne(inst.shift()) & inst.shift().notna())[0]:
    factor.iloc[i:] *= f["raw_close"].iloc[i - 1] / f["raw_close"].iloc[i]
h, l, c = (f[f"raw_{k}"] * factor for k in ["high", "low", "close"])
hh, ll = h.rolling(60).max(), l.rolling(60).min()
rng = hh - ll
f["range_pos_60"] = ((c - ll) / rng.where(rng != 0)).fillna(0.5)

lab = pd.read_parquet(f"{P}/labels/labels_default.parquet", columns=["split", "pre_roll_3d"]).reindex(f.index)
rows = (lab["split"].astype(str) == "train") & (f["is_filled"] == 0) & (lab["pre_roll_3d"] == 0)
before_cols = feat_cols[:feat_cols.index("vol_ratio_5_60") + 1] + ["range_pos_60"] + feat_cols[feat_cols.index("vol_ratio_5_60") + 1:]
before = f.loc[rows, before_cols].corr()
after = before.drop(index="range_pos_60", columns="range_pos_60")
before.to_csv(f"{OUT}/相关系数_删除前_34.csv", float_format="%.3f", encoding="utf-8-sig")
after.to_csv(f"{OUT}/相关系数_删除后_33.csv", float_format="%.3f", encoding="utf-8-sig")


def pairs(m, thr):
    cols = m.columns
    out = [(a, b, m.loc[a, b]) for i, a in enumerate(cols) for b in cols[i + 1:] if abs(m.loc[a, b]) > thr]
    return sorted(out, key=lambda x: -abs(x[2]))


lines = [f"训练集样本 {int(rows.sum()):,} 行", ""]
for name, m in [("删除前（34 个特征）", before), ("删除后（33 个特征）", after)]:
    lines.append(f"{name}：|r| > 0.7 的特征对")
    lines += [f"  {a} ~ {b}: {r:.3f}" for a, b, r in pairs(m, 0.7)]
    off = m.where(~np.eye(len(m), dtype=bool)).abs().stack()
    lines.append(f"  |r| > 0.9 的对数 {len(pairs(m, 0.9))}；最大 |r| {off.max():.3f}")
    lines.append("")
lines.append("range_pos_60 与其他特征的相关系数（删除前，按绝对值排序，前 8 个）")
s = before["range_pos_60"].drop("range_pos_60")
lines += [f"  {k}: {v:.3f}" for k, v in s.reindex(s.abs().sort_values(ascending=False).index).head(8).items()]
open(f"{OUT}/高相关特征对.txt", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))

# 中英文各一套：中文图在 OUT 下（文件名不变），英文图在 OUT/English 下
T = {
    "zh": {"dir": OUT, "font": ["Hiragino Sans GB", "Arial Unicode MS"], "cbar": "相关系数",
           "file": {"before": "相关系数热力图_删除前_34.png", "after": "相关系数热力图_删除后_33.png"},
           "stage": {"before": "删除前，34", "after": "删除后，33"},
           "title": "特征相关系数（训练集样本，{} 个特征；只标出 |r| ≥ 0.5 的数值）"},
    "en": {"dir": f"{OUT}/English", "font": ["Arial", "Helvetica"], "cbar": "Correlation coefficient",
           "file": {"before": "Corr_heatmap_before_34.png", "after": "Corr_heatmap_after_33.png"},
           "stage": {"before": "before removal, 34", "after": "after removal, 33"},
           "title": "Feature correlation (training samples, {} features; only |r| ≥ 0.5 is labelled)"},
}
for lang, tx in T.items():
    os.makedirs(tx["dir"], exist_ok=True)
    plt.rcParams.update({"font.sans-serif": tx["font"], "axes.unicode_minus": False, "figure.facecolor": "#fcfcfb"})
    for key, m in [("before", before), ("after", after)]:
        n = len(m)
        fig, ax = plt.subplots(figsize=(13, 11.5))
        im = ax.imshow(m.values, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(n)); ax.set_xticklabels(m.columns, rotation=90, fontsize=8)
        ax.set_yticks(range(n)); ax.set_yticklabels(m.index, fontsize=8)
        for i in range(n):
            for j in range(n):
                v = m.values[i, j]
                if i != j and abs(v) >= 0.5:
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=5.5, color="white" if abs(v) > 0.75 else "#222")
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label=tx["cbar"])
        ax.set_title(tx["title"].format(tx["stage"][key]), loc="left", fontsize=11)
        fig.tight_layout()
        fig.savefig(f"{tx['dir']}/{tx['file'][key]}", dpi=180)
        plt.close(fig)
print("saved to", OUT)
