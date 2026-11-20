# 通用技术指标
import numpy as np
import pandas as pd


def atr(h, l, c, n):
    """ATR：真实波幅 TR_i = max(H_i − L_i, |H_i − C_{i−1}|, |L_i − C_{i−1}|)，ATR_t = 最近 n 根 TR 的简单平均"""
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def safe_div(a, b, fill=np.nan):
    """a / b，分母为 0 或结果为 ±inf 时记为空；给了 fill 就用 fill 填空"""
    b = b.replace(0, np.nan) if isinstance(b, pd.Series) else b
    return (a / b).replace([np.inf, -np.inf], np.nan).fillna(fill) if fill is not np.nan else (a / b).replace([np.inf, -np.inf], np.nan)
