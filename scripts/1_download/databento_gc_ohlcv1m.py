# 下载 GC 主力合约一年 1 分钟 K 线（约 $1.28，从免费额度扣）
import databento as db
from gold_dl.config import CFG, path
from gold_dl.timeutil import utc_to_bj

key = open(path("databento_key")).read().strip()
client = db.Historical(key)

data = client.timeseries.get_range(
    dataset=CFG["data"]["dataset"], schema="ohlcv-1m",
    symbols=CFG["data"]["symbol"], stype_in="continuous",
    start=CFG["data"]["start"], end=CFG["data"]["end"],
    path=str(path("raw_ohlcv1m")),
)
df = data.to_df()
df.index = utc_to_bj(df.index)
df.index.name = "datetime"
df = df[["symbol", "open", "high", "low", "close", "volume"]]
df.to_csv(str(path("ohlcv1m_csv")))
print(len(df), "rows", df.index.min(), "->", df.index.max())
print(df.head(3))
print(df.tail(3))
print("contracts:", df["symbol"].value_counts().to_dict())
