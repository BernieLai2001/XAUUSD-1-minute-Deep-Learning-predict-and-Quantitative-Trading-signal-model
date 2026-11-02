# 从逐笔成交汇总：每分钟成交额（算精确 VWAP）、大单统计；每个交易日的 POC
# CME 交易日从芝加哥时间 17:00 开始，所以交易日 = (芝加哥时间 + 7 小时) 的日期
import glob
import pandas as pd
from pathlib import Path
D = str(Path(__file__).resolve().parent.parent / "GC_processed")
LARGE = 10  # 大单阈值（手）


def trade_date(idx_bj):
    ct = idx_bj.tz_localize("Asia/Shanghai").tz_convert("America/Chicago")
    return (ct + pd.Timedelta(hours=7)).tz_localize(None).normalize()


mins, poc_parts = [], []
for f in sorted(glob.glob(f"{D}/trades/*.parquet")):
    t = pd.read_parquet(f, columns=["price", "size", "side"])
    t["size"] = t["size"].astype("int64")
    big = t["size"] >= LARGE
    t["notional"] = t["price"] * t["size"]
    t["large_vol"] = t["size"].where(big, 0)
    t["large_buy"] = t["size"].where(big & (t["side"] == "B"), 0)
    t["large_sell"] = t["size"].where(big & (t["side"] == "A"), 0)
    g = t.groupby(t.index.floor("min"))
    mins.append(g[["notional", "large_vol", "large_buy", "large_sell"]].sum())
    t["tdate"] = trade_date(t.index)
    poc_parts.append(t.groupby(["tdate", "price"])["size"].sum())
    print(f.split("/")[-1], len(t), flush=True)

m = pd.concat(mins)
m = m.groupby(level=0).sum().sort_index()
m.index.name = "datetime"
m.to_parquet(f"{D}/GC_trades_extra_1min.parquet")

pv = pd.concat(poc_parts).groupby(level=[0, 1]).sum()
poc = pv.groupby(level=0).idxmax().map(lambda x: x[1]).rename("poc")
poc.index.name = "tdate"
poc.to_frame().to_parquet(f"{D}/GC_poc_daily.parquet")
print("minutes", len(m), "sessions", len(poc))
print(poc.tail())
