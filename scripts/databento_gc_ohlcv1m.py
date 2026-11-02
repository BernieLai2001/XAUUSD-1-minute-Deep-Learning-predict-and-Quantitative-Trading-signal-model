# 下载 GC 主力合约一年 1 分钟 K 线（约 $1.28，从免费额度扣）
import databento as db
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent

key = open(ROOT / "databento_key.txt").read().strip()
client = db.Historical(key)

data = client.timeseries.get_range(
    dataset="GLBX.MDP3", schema="ohlcv-1m",
    symbols="GC.v.0", stype_in="continuous",
    start="2025-09-28", end="2026-09-29",
    path=str(ROOT / "GC_ohlcv1m_1year.dbn.zst"),
)
df = data.to_df()
df.index = df.index.tz_convert("Asia/Shanghai").tz_localize(None)
df.index.name = "datetime"
df = df[["symbol", "open", "high", "low", "close", "volume"]]
df.to_csv(str(ROOT / "GC_1min_1year.csv"))
print(len(df), "rows", df.index.min(), "->", df.index.max())
print(df.head(3))
print(df.tail(3))
print("contracts:", df["symbol"].value_counts().to_dict())
