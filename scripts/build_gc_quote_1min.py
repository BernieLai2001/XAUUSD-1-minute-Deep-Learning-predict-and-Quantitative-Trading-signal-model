# 从 MBP-1 原始文件汇总每分钟买一价、卖一价的开高低收（北京时间，时间戳 = 分钟开始）
# 用于三重障碍标签：做多按卖一价入场、按买一价判断碰线；做空相反
# 输出：~/Desktop/project/GC_processed/GC_quote_1min.parquet
import glob, os, time
import databento as db
import numpy as np
import pandas as pd
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent

SRC = str(ROOT / "GC_mbp1" / "GLBX-20261025-PT9C6P3YKJ")
OUT = str(ROOT / "GC_processed" / "GC_quote_1min.parquet")

AGG = {"bid_open": "first", "bid_high": "max", "bid_low": "min", "bid_close": "last",
       "ask_open": "first", "ask_high": "max", "ask_low": "min", "ask_close": "last",
       "n": "sum"}


def agg_chunk(c):
    bid, ask = c["bid_px_00"].to_numpy(), c["ask_px_00"].to_numpy()
    bsz = c["bid_sz_00"].to_numpy(dtype="int64")
    asz = c["ask_sz_00"].to_numpy(dtype="int64")
    valid = np.isfinite(bid) & np.isfinite(ask) & (ask > bid) & (bsz > 0) & (asz > 0)  # 排除交叉盘口和空盘口
    c = c[valid]
    minute = c.index.tz_convert("Asia/Shanghai").tz_localize(None).floor("min")
    f = pd.DataFrame({
        "minute": minute, "instrument_id": c["instrument_id"].to_numpy(),
        "bid_open": c["bid_px_00"].to_numpy(), "bid_high": c["bid_px_00"].to_numpy(),
        "bid_low": c["bid_px_00"].to_numpy(), "bid_close": c["bid_px_00"].to_numpy(),
        "ask_open": c["ask_px_00"].to_numpy(), "ask_high": c["ask_px_00"].to_numpy(),
        "ask_low": c["ask_px_00"].to_numpy(), "ask_close": c["ask_px_00"].to_numpy(),
        "n": 1,
    })
    return f.groupby(["minute", "instrument_id"], sort=False).agg(AGG)


parts = []
t0 = time.time()
for path in sorted(glob.glob(f"{SRC}/*.mbp-1.dbn.zst")):
    store = db.DBNStore.from_file(path)
    n = 0
    for chunk in store.to_df(count=5_000_000):
        parts.append(agg_chunk(chunk))
        n += len(chunk)
    print(f"{os.path.basename(path)}: {n:,} records, {time.time() - t0:.0f}s", flush=True)

q = pd.concat(parts).groupby(level=[0, 1], sort=True).agg(AGG).reset_index()
# 换月前后同一分钟可能有两个合约的报价，只保留更新次数多的那个
q = q.sort_values(["minute", "n"]).drop_duplicates("minute", keep="last")
q = q.set_index("minute").sort_index()
q.index.name = "datetime"
q.to_parquet(OUT)
print("saved", OUT, len(q), "minutes", flush=True)
print(q.head())
