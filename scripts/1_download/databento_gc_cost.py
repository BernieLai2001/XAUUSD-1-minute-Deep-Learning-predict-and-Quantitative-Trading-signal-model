# 免费估价：GC 黄金期货一年的数据价格（不会产生费用）
import databento as db
from gold_dl.config import CFG, path as cfg_path

key = open(cfg_path("databento_key")).read().strip()
client = db.Historical(key)

START, END = CFG["data"]["start"], CFG["data"]["end"]  # UTC，与下载脚本相同的时间范围
symbol_sets = {
    "主力连续合约 GC.v.0": dict(symbols="GC.v.0", stype_in="continuous"),
    "全部 GC 合约 GC.FUT": dict(symbols="GC.FUT", stype_in="parent"),
}
for label, sym in symbol_sets.items():
    print(f"== {label}")
    for schema in ["ohlcv-1m", "trades", "tbbo", "mbp-1", "statistics"]:
        kw = dict(dataset=CFG["data"]["dataset"], schema=schema, start=START, end=END, **sym)
        cost = client.metadata.get_cost(**kw)
        size = client.metadata.get_billable_size(**kw)
        print(f"  {schema:10s} {size / 1e9:8.3f} GB   ${cost:8.2f}")
