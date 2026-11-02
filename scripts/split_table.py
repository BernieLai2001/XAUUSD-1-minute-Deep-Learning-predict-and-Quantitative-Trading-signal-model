# 训练集 / 隔离期 / 验证集 / 隔离期 / 测试集 的起止时间和行数（北京时间）
import pandas as pd
from pathlib import Path
P = str(Path(__file__).resolve().parent.parent)
s = pd.read_parquet(f"{P}/labels/labels_all.parquet", columns=["split"])["split"].astype(str)
block = (s != s.shift()).cumsum()
names = {"train": "Training", "val": "Validation", "test": "Test", "": "Gap (removed)"}
t = s.groupby(block).agg(set=lambda x: names[x.iloc[0]])
t["start"] = s.index.to_series().groupby(block.values).first().dt.strftime("%Y-%m-%d %H:%M").values
t["end"] = s.index.to_series().groupby(block.values).last().dt.strftime("%Y-%m-%d %H:%M").values
t["rows"] = s.groupby(block).size().map("{:,}".format).values
print(t.to_string(index=False))
