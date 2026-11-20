# 时区转换。项目里所有处理后的时间都是北京时间、不带时区（Asia/Shanghai 没有夏令时）
import pandas as pd

BJ = "Asia/Shanghai"
ET = "America/New_York"
CT = "America/Chicago"


def utc_to_bj(idx):
    """Databento 的 UTC 时间 → 不带时区的北京时间"""
    return idx.tz_convert(BJ).tz_localize(None)


def bj_to(idx, tz):
    """不带时区的北京时间 → 指定时区（带时区）"""
    return idx.tz_localize(BJ).tz_convert(tz)


def bj_to_et(idx):
    """不带时区的北京时间 → 美东时间（带时区，自动处理夏令时）"""
    return bj_to(idx, ET)


def trade_date(idx):
    """CME 交易日：芝加哥时间 17:00 开盘算下一个交易日，即芝加哥时间 + 7 小时后取日期（不带时区）"""
    return (bj_to(idx, CT) + pd.Timedelta(hours=7)).tz_localize(None).normalize()
