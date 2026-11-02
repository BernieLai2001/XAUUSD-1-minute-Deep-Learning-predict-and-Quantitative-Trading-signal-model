# 从 MBP-1 原始文件计算每分钟第一档平均挂单量（深度），用作 OFI 的分母（北京时间，时间戳 = 分钟开始）
# 深度 = (买一挂单量 + 卖一挂单量) / 2，单位手；与 Cont, Kukanov & Stoikov (2014) 的平均深度一致
#   depth_tw：按时间加权，每条记录的深度持续到同一合约的下一条有效记录（最多 60 秒，跨分钟的部分分到下一分钟），
#             与时间加权点差（build_gc_spread_tw_1min.py）的做法相同
#   depth_ev：按事件加权，这一分钟内所有有效记录的简单平均（对照用）
# 只用有效盘口（卖一 > 买一，两边挂单量都 > 0），按 (分钟, 合约) 分别汇总
# 输出：~/Desktop/project/GC_processed/GC_depth_1min.parquet
import glob, os, time
import databento as db
import numpy as np
import pandas as pd
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent

SRC = str(ROOT / "GC_mbp1" / "GLBX-20261025-PT9C6P3YKJ")
OUT = str(ROOT / "GC_processed" / "GC_depth_1min.parquet")
NS_MIN = 60_000_000_000
BJ_OFFSET = 8 * 3600 * 1_000_000_000   # 北京时间没有夏令时，UTC 整分钟 + 8 小时就是北京时间整分钟
KEY = 10 ** 9                          # 分组键 = 分钟序号 × 10⁹ + 合约号（合约号 < 10⁹）


def agg(minute_ns, inst, cols):
    key = (minute_ns // NS_MIN) * KEY + inst
    return pd.DataFrame(cols, index=key).groupby(level=0).sum()


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
            "inst": chunk["instrument_id"].to_numpy().astype("int64")[valid],
            "ts": chunk.index.asi8[valid],
            "dp": (bsz[valid] + asz[valid]) / 2.0,
        })
        # 按事件加权：只统计本数据块里新出现的记录
        m_ev = c["ts"].to_numpy() // NS_MIN * NS_MIN
        parts.append(agg(m_ev, c["inst"].to_numpy(), {"ev_sum": c["dp"].to_numpy(), "ev_n": np.ones(len(c)),
                                                       "w": 0.0, "dur": 0.0}))
        # 按时间加权
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
        dp, inst = x["dp"].to_numpy(), x["inst"].to_numpy()
        parts.append(agg(m0, inst, {"ev_sum": 0.0, "ev_n": 0.0, "w": dp * d1, "dur": d1}))
        k = d2 > 0
        parts.append(agg(b[k], inst[k], {"ev_sum": 0.0, "ev_n": 0.0, "w": dp[k] * d2[k], "dur": d2[k]}))
    print(f"{os.path.basename(path)}: {n:,} records, {time.time() - t0:.0f}s", flush=True)

s = pd.concat(parts).groupby(level=0).sum()
key = s.index.to_numpy()
out = pd.DataFrame({
    "datetime": pd.to_datetime((key // KEY) * NS_MIN + BJ_OFFSET),
    "instrument_id": key % KEY,
    "depth_tw": np.where(s["dur"] > 0, s["w"] / s["dur"].where(s["dur"] > 0, 1), np.nan),
    "depth_ev": np.where(s["ev_n"] > 0, s["ev_sum"] / s["ev_n"].where(s["ev_n"] > 0, 1), np.nan),
    "covered_sec": s["dur"].to_numpy(),
})
out.to_parquet(OUT, index=False)
print("saved", OUT, len(out), "rows", flush=True)
print(out.describe())
