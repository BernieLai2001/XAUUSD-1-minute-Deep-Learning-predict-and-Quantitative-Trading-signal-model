# 从 MBP-1 原始文件计算每分钟按时间加权的平均点差（北京时间，时间戳 = 分钟开始）
# 每条有效盘口记录的点差，从它出现一直持续到同一合约的下一条有效记录，按持续时间加权
#   - 持续时间最多算 60 秒：超过 60 秒没有新报价（如每日休市、周末），之后的时间不再沿用旧点差
#   - 跨过整分钟的部分分到下一分钟：下一分钟开头、第一条新记录之前的点差就是上一分钟最后的点差
# 只用有效盘口（卖一 > 买一，两边挂单量都 > 0），按 (分钟, 合约) 分别汇总
# 输出：~/Desktop/project/GC_processed/GC_spread_tw_1min.parquet
#   spread_tw = Σ(点差 × 持续秒数) / Σ持续秒数（跳，1 跳 = 0.1 美元），covered_sec = 这一分钟被覆盖的秒数
import glob, os, time
import databento as db
import numpy as np
import pandas as pd
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent

SRC = str(ROOT / "GC_mbp1" / "GLBX-20261025-PT9C6P3YKJ")
OUT = str(ROOT / "GC_processed" / "GC_spread_tw_1min.parquet")
TICK = 0.1
NS_MIN = 60_000_000_000
BJ_OFFSET = 8 * 3600 * 1_000_000_000   # 北京时间没有夏令时，UTC 整分钟 + 8 小时就是北京时间整分钟


def agg(minute, inst, w, dur):
    return pd.DataFrame({"minute": minute, "inst": inst, "w": w, "dur": dur}).groupby(["minute", "inst"]).sum()


carry = None   # 每个合约还不知道持续到什么时候的最后一条记录
parts = []
t0 = time.time()
for path in sorted(glob.glob(f"{SRC}/*.mbp-1.dbn.zst")):
    store = db.DBNStore.from_file(path)
    n = 0
    for chunk in store.to_df(count=5_000_000):
        n += len(chunk)
        bid, ask = chunk["bid_px_00"].to_numpy(), chunk["ask_px_00"].to_numpy()
        bsz = chunk["bid_sz_00"].to_numpy(dtype="int64")
        asz = chunk["ask_sz_00"].to_numpy(dtype="int64")
        valid = np.isfinite(bid) & np.isfinite(ask) & (ask > bid) & (bsz > 0) & (asz > 0)
        c = pd.DataFrame({
            "inst": chunk["instrument_id"].to_numpy()[valid],
            "ts": chunk.index.asi8[valid],
            "sp": np.round((ask[valid] - bid[valid]) / TICK, 6),
        })
        if carry is not None:
            c = pd.concat([carry, c], ignore_index=True)
        nxt = c.groupby("inst")["ts"].shift(-1)
        done = nxt.notna().to_numpy()
        carry = c[~done]
        x = c[done]
        ts = x["ts"].to_numpy()
        end = np.minimum(nxt[done].to_numpy().astype("int64"), ts + NS_MIN)
        m0 = ts // NS_MIN * NS_MIN
        b = m0 + NS_MIN
        d1 = (np.minimum(end, b) - ts) / 1e9          # 落在本分钟的秒数
        d2 = np.maximum(end - b, 0) / 1e9             # 跨到下一分钟的秒数
        sp, inst = x["sp"].to_numpy(), x["inst"].to_numpy()
        parts.append(agg(m0, inst, sp * d1, d1))
        k = d2 > 0
        parts.append(agg(b[k], inst[k], sp[k] * d2[k], d2[k]))
    print(f"{os.path.basename(path)}: {n:,} records, {time.time() - t0:.0f}s", flush=True)

s = pd.concat(parts).groupby(level=[0, 1]).sum().reset_index()
s = s[s["dur"] > 0]
out = pd.DataFrame({
    "datetime": pd.to_datetime(s["minute"].to_numpy() + BJ_OFFSET),
    "instrument_id": s["inst"].to_numpy(),
    "spread_tw": (s["w"] / s["dur"]).to_numpy(),
    "covered_sec": s["dur"].to_numpy(),
})
out.to_parquet(OUT, index=False)
print("saved", OUT, len(out), "rows", flush=True)
print(out.describe())
