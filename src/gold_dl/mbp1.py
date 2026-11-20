# 读取 Databento MBP-1 原始文件（每月一个 .dbn.zst），以及有效盘口的判断
import glob
import os
import time

import databento as db
import numpy as np

from gold_dl.config import path

CHUNK = 5_000_000   # 每次读入的记录数，一个月的文件约 1–2 亿条，分块读避免内存不够


def files(src=None):
    """MBP-1 下载目录（默认 config.yaml 的 mbp1_batch）下所有月度文件，按时间顺序"""
    return sorted(glob.glob(f"{src or path('mbp1_batch')}/*.mbp-1.dbn.zst"))


def file_chunks(fn, count=CHUNK):
    """逐块读一个 MBP-1 文件，每块是一个 DataFrame（索引为 UTC 时间）"""
    return db.DBNStore.from_file(fn).to_df(count=count)


def chunks(src=None, count=CHUNK):
    """按时间顺序逐块读所有 MBP-1 文件；每读完一个文件打印记录数和累计用时"""
    t0 = time.time()
    for fn in files(src):
        n = 0
        for chunk in file_chunks(fn, count):
            n += len(chunk)
            yield chunk
        print(f"{os.path.basename(fn)}: {n:,} records, {time.time() - t0:.0f}s", flush=True)


def valid_book(bid, ask, bsz, asz):
    """有效盘口：买一、卖一价都有值，卖一 > 买一（排除竞价时的交叉盘口），两边挂单量都 > 0"""
    return np.isfinite(bid) & np.isfinite(ask) & (ask > bid) & (bsz > 0) & (asz > 0)
