# 检查训练、验证、测试集是否按时间顺序排列、互不重叠
import pandas as pd
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent


d = pd.read_parquet(ROOT / "labels" / "labels_default.parquet", columns=["split"])
print("index sorted:", d.index.is_monotonic_increasing, "duplicates:", d.index.duplicated().sum())
for s in ["train", "", "val", "", "test"]:
    pass
g = d.reset_index().groupby("split", observed=True)["datetime"].agg(["min", "max", "size"])
print(g.loc[["train", "", "val", "test"]])
# 每段在原始顺序中是否连续（不被其他段打断）
runs = (d["split"] != d["split"].shift()).cumsum()
print(d.groupby(runs)["split"].first().tolist())
