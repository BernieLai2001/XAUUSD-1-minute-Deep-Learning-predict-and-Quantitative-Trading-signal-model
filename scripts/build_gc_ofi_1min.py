# 从 MBP-1 原始文件计算每分钟订单流不平衡 OFI（Cont, Kukanov & Stoikov 2014）
# 每条盘口记录 n 相对上一条记录（同一合约）的贡献：
#   e_n = [b_n ≥ b_{n−1}]·q^b_n − [b_n ≤ b_{n−1}]·q^b_{n−1} − [a_n ≤ a_{n−1}]·q^a_n + [a_n ≥ a_{n−1}]·q^a_{n−1}
#   b/a = 买一/卖一价，q^b/q^a = 买一/卖一挂单量（手）；正值 = 买方力量增强
# 每分钟 OFI = 这一分钟内所有 e_n 之和（北京时间，时间戳 = 分钟开始），按 (分钟, 合约) 分别汇总
# 只用有效盘口（卖一 > 买一，两边挂单量都 > 0），无效记录跳过，下一条有效记录与上一条有效记录比较
# 输出：~/Desktop/project/GC_processed/GC_ofi_1min.parquet
import glob, os, time
import databento as db
import numpy as np
import pandas as pd
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent

SRC = str(ROOT / "GC_mbp1" / "GLBX-20261025-PT9C6P3YKJ")
OUT = str(ROOT / "GC_processed" / "GC_ofi_1min.parquet")
COLS = ["instrument_id", "bid_px_00", "ask_px_00", "bid_sz_00", "ask_sz_00"]

carry = None   # 上一个数据块里每个合约的最后一条有效盘口
parts = []
t0 = time.time()
for path in sorted(glob.glob(f"{SRC}/*.mbp-1.dbn.zst")):
    store = db.DBNStore.from_file(path)
    n = 0
    for chunk in store.to_df(count=5_000_000):
        n += len(chunk)
        c = chunk[COLS].copy()
        c["bid_sz_00"] = c["bid_sz_00"].astype("int64")
        c["ask_sz_00"] = c["ask_sz_00"].astype("int64")
        valid = (np.isfinite(c["bid_px_00"]) & np.isfinite(c["ask_px_00"]) & (c["ask_px_00"] > c["bid_px_00"])
                 & (c["bid_sz_00"] > 0) & (c["ask_sz_00"] > 0))
        c = c[valid]
        c["minute"] = c.index.tz_convert("Asia/Shanghai").tz_localize(None).floor("min")
        c["is_carry"] = False
        if carry is not None:
            c = pd.concat([carry, c])
        prev = c.groupby("instrument_id")[["bid_px_00", "ask_px_00", "bid_sz_00", "ask_sz_00"]].shift(1)
        b, a, qb, qa = c["bid_px_00"], c["ask_px_00"], c["bid_sz_00"], c["ask_sz_00"]
        pb, pa, pqb, pqa = prev["bid_px_00"], prev["ask_px_00"], prev["bid_sz_00"], prev["ask_sz_00"]
        e = ((b >= pb) * qb - (b <= pb) * pqb - (a <= pa) * qa + (a >= pa) * pqa).where(pb.notna(), 0.0)
        c["ofi"] = e
        carry = c[~c["is_carry"]].groupby("instrument_id").tail(1).assign(is_carry=True)
        c = c[~c["is_carry"]]
        parts.append(c.groupby(["minute", "instrument_id"])["ofi"].sum())
    print(f"{os.path.basename(path)}: {n:,} records, {time.time() - t0:.0f}s", flush=True)

ofi = pd.concat(parts).groupby(level=[0, 1]).sum().rename("ofi").reset_index()
ofi = ofi.rename(columns={"minute": "datetime"})
ofi.to_parquet(OUT, index=False)
print("saved", OUT, len(ofi), "rows", flush=True)
print(ofi.describe())
