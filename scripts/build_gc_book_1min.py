# 从 MBP-1 原始文件汇总每分钟盘口指标（北京时间，时间戳 = 分钟开始）
# 输出：~/Desktop/project/GC_processed/GC_book_1min.parquet
import glob, os, time
import databento as db
import numpy as np
import pandas as pd
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent

SRC = str(ROOT / "GC_mbp1" / "GLBX-20261025-PT9C6P3YKJ")
OUT = str(ROOT / "GC_processed" / "GC_book_1min.parquet")
TICK = 0.1  # GC 最小跳动 0.1 美元

SUM_COLS = ["updates", "spread_sum", "spread_n", "imb_sum", "imb_n",
            "buy_ask_sz_sum", "buy_n", "sell_bid_sz_sum", "sell_n"]
LAST_COLS = ["spread_last", "imb_last"]


def agg_chunk(c):
    c = c[["action", "side", "bid_px_00", "ask_px_00", "bid_sz_00", "ask_sz_00"]]
    minute = c.index.tz_convert("Asia/Shanghai").tz_localize(None).floor("min")
    bid, ask = c["bid_px_00"].to_numpy(), c["ask_px_00"].to_numpy()
    bsz = c["bid_sz_00"].to_numpy(dtype="int64")
    asz = c["ask_sz_00"].to_numpy(dtype="int64")
    act, side = c["action"].to_numpy(), c["side"].to_numpy()
    valid = np.isfinite(bid) & np.isfinite(ask) & (ask > bid) & (bsz + asz > 0)  # 排除竞价时的交叉盘口
    spread = np.where(valid, (ask - bid) / TICK, np.nan)
    imb = np.where(valid, (bsz - asz) / np.maximum(bsz + asz, 1), np.nan)
    is_buy = (act == "T") & (side == "B") & valid
    is_sell = (act == "T") & (side == "A") & valid
    f = pd.DataFrame({
        "updates": np.isin(act, ["A", "C", "M"]).astype("int64"),
        "spread_sum": np.nan_to_num(spread), "spread_n": valid.astype("int64"),
        "imb_sum": np.nan_to_num(imb), "imb_n": valid.astype("int64"),
        "buy_ask_sz_sum": np.where(is_buy, asz, 0), "buy_n": is_buy.astype("int64"),
        "sell_bid_sz_sum": np.where(is_sell, bsz, 0), "sell_n": is_sell.astype("int64"),
        "spread_last": spread, "imb_last": imb,
    }, index=minute)
    g = f.groupby(level=0)
    return pd.concat([g[SUM_COLS].sum(), g[LAST_COLS].last()], axis=1)


parts = []
t0 = time.time()
for path in sorted(glob.glob(f"{SRC}/*.mbp-1.dbn.zst")):
    store = db.DBNStore.from_file(path)
    n = 0
    for chunk in store.to_df(count=5_000_000):
        parts.append(agg_chunk(chunk))
        n += len(chunk)
    print(f"{os.path.basename(path)}: {n:,} records, {time.time() - t0:.0f}s", flush=True)

allp = pd.concat(parts)
g = allp.groupby(level=0)
book = pd.concat([g[SUM_COLS].sum(), g[LAST_COLS].last()], axis=1).sort_index()
book.index.name = "datetime"
book.to_parquet(OUT)
print("saved", OUT, len(book), "minutes", flush=True)
