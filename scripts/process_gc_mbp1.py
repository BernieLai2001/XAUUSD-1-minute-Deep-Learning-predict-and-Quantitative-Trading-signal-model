# 处理 GC MBP-1：提取逐笔成交，汇总 1 分钟主动买卖量 / Delta
# side：B = 主动买（买方吃卖单），A = 主动卖，N = 无法判断（多为价差合约腿成交）
import glob, os
import databento as db
import pandas as pd
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
SRC = str(ROOT / "GC_mbp1" / "GLBX-20261025-PT9C6P3YKJ")
OUT = str(ROOT / "GC_processed")
os.makedirs(f"{OUT}/trades", exist_ok=True)

minute_parts = []
for path in sorted(glob.glob(f"{SRC}/*.mbp-1.dbn.zst")):
    name = os.path.basename(path).split(".")[0]
    store = db.DBNStore.from_file(path)
    trades = []
    for chunk in store.to_df(count=5_000_000):
        t = chunk[chunk["action"] == "T"]
        trades.append(t[["price", "size", "side", "bid_px_00", "ask_px_00",
                         "bid_sz_00", "ask_sz_00", "instrument_id"]])
    t = pd.concat(trades)
    t["size"] = t["size"].astype("int64")  # 原始为 uint32，相减会溢出
    t.index = t.index.tz_convert("Asia/Shanghai").tz_localize(None)
    t.index.name = "datetime"
    t.to_parquet(f"{OUT}/trades/{name}.parquet")

    t["buy"] = t["size"].where(t["side"] == "B", 0)
    t["sell"] = t["size"].where(t["side"] == "A", 0)
    g = t.groupby(t.index.floor("min"))
    m = pd.DataFrame({
        "open": g["price"].first(), "high": g["price"].max(),
        "low": g["price"].min(), "close": g["price"].last(),
        "volume": g["size"].sum(), "buy_volume": g["buy"].sum(),
        "sell_volume": g["sell"].sum(), "trades": g["size"].count(),
        "instrument_id": g["instrument_id"].last(),
    })
    m["delta"] = m["buy_volume"] - m["sell_volume"]
    minute_parts.append(m)
    print(f"{name}: {len(t):,} trades, {len(m):,} minutes", flush=True)

minutes = pd.concat(minute_parts).sort_index()
minutes = minutes[~minutes.index.duplicated(keep="last")]
minutes.index.name = "datetime"
minutes["cum_delta_day"] = minutes.groupby(minutes.index.date)["delta"].cumsum()
minutes.to_csv(f"{OUT}/GC_1min_orderflow.csv")
print("saved GC_1min_orderflow.csv", len(minutes), "rows", flush=True)
