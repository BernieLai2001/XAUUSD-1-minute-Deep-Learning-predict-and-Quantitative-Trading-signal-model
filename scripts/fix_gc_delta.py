# 修复：size 为 uint32 时 buy - sell 为负会溢出。用 int64 重新计算 delta 和当日累计 delta
import pandas as pd
from pathlib import Path
D = str(Path(__file__).resolve().parent.parent / "GC_processed")
of = pd.read_csv(f"{D}/GC_1min_orderflow.csv", index_col=0, parse_dates=True)
of["delta"] = of["buy_volume"].astype("int64") - of["sell_volume"].astype("int64")
of["cum_delta_day"] = of.groupby(of.index.date)["delta"].cumsum()
of.to_csv(f"{D}/GC_1min_orderflow.csv")


print("delta range:", of["delta"].min(), "to", of["delta"].max())
print("negative-delta minutes:", int((of["delta"] < 0).sum()), "of", len(of))
print("check buy+sell<=volume:", bool((of["buy_volume"] + of["sell_volume"] <= of["volume"]).all()))
