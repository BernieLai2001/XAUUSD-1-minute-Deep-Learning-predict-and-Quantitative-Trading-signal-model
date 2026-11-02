# 计算全部特征，全部只用 COMEX 黄金主力期货 GC（GC.v.0），不用现货
# 时间轴：GC 有成交的 1 分钟 K 线（北京时间，时间戳 = 分钟开始），再补上盘中零星缺失的分钟（is_filled = 1）
#   补值：同一合约、相邻两根成交 K 线间隔 2–30 分钟时补齐中间每一分钟；价格 = 上一分钟收盘价（平线），成交量类 = 0；
#   盘口类：这一分钟 MBP-1 有记录就照常算，没有记录 = 盘口没变，沿用上一分钟，OFI = 0
#   超过 30 分钟的间隔（每日休市、周末、节假日）和换月处不补，作为断点；补出的分钟只用来补全回看窗口，不作为样本
#   seg_id：窗口段编号，遇到断点加 1；回看窗口内 seg_id 相同 = 窗口可用
#     断点 = 换合约，或间隔超过 30 分钟但不是按日程的休市；每日休市、周末、节假日（间隔里包含美东 17 点）不算断点，
#     窗口可以跨过它们，开盘初期的样本由美东时间特征告诉模型
# 所有特征只用第 t 分钟及之前的数据；第 t 分钟结束后可用，标签从 t+1 分钟开始
# 换月按比例向前复权：每次换月后，把之后所有价格乘以（旧合约最后一根收盘 / 新合约第一根收盘）
#   - 不用向后复权：向后复权会让换月前的价格提前包含未来换月的跳空
#   - 用比例而不是加减：同一合约内的对数收益与原始价格完全相同，只去掉换月那一下的跳空
import json, os
import numpy as np
import pandas as pd
from pathlib import Path
DESK = str(Path(__file__).resolve().parent.parent)

D = f"{DESK}/GC_processed"
OUT = f"{DESK}/features"
# 截断测试用：设置环境变量 CUTOFF 时，只用这个时间之前的数据，结果写到 OUT_DIR
CUTOFF = os.environ.get("CUTOFF")
if CUTOFF:
    OUT = os.environ["OUT_DIR"]
os.makedirs(OUT, exist_ok=True)

TICK = 0.1     # GC 最小价格变动（美元）
MAX_FILL_GAP = 30   # 只补连续 30 分钟以内的盘中缺失
PRE_ROLL = pd.Timedelta(days=3)   # 换月前 3 天不作为样本，拟合参数时也排除
FEATURES = []  # (列名, 类别, 说明)


def add(df, name, values, group, desc):
    df[name] = values
    FEATURES.append((name, group, desc))


def atr(h, l, c, n):
    # 与提案一致：真实波幅 TR_i = max(H_i − L_i, |H_i − C_{i−1}|, |L_i − C_{i−1}|)，ATR_t = 最近 n 根 TR 的简单平均
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def safe_div(a, b, fill=np.nan):
    b = b.replace(0, np.nan) if isinstance(b, pd.Series) else b
    return (a / b).replace([np.inf, -np.inf], np.nan).fillna(fill) if fill is not np.nan else (a / b).replace([np.inf, -np.inf], np.nan)


def trade_date(idx):
    ct = idx.tz_localize("Asia/Shanghai").tz_convert("America/Chicago")
    return pd.Series((ct + pd.Timedelta(hours=7)).tz_localize(None).normalize(), index=idx)


# ================= 读入 GC 数据 =================
of = pd.read_csv(f"{D}/GC_1min_orderflow.csv", index_col="datetime", parse_dates=True)
book = pd.read_parquet(f"{D}/GC_book_1min.parquet")
ofi = pd.read_parquet(f"{D}/GC_ofi_1min.parquet")
sptw = pd.read_parquet(f"{D}/GC_spread_tw_1min.parquet")
depth = pd.read_parquet(f"{D}/GC_depth_1min.parquet")
if CUTOFF:
    of, book = of[of.index < CUTOFF], book[book.index < CUTOFF]
    ofi, sptw, depth = (x[x["datetime"] < CUTOFF] for x in (ofi, sptw, depth))
of = of[of["volume"] > 0].sort_index()
print("GC bars:", len(of))

# 盘中缺失分钟补值
gap_min = of.index.to_series().diff().dt.total_seconds().div(60)
same_inst = of["instrument_id"].eq(of["instrument_id"].shift())
fill_end = of.index[(gap_min >= 2) & (gap_min <= MAX_FILL_GAP) & same_inst]
fill_idx = pd.DatetimeIndex(np.concatenate(
    [pd.date_range(t - pd.Timedelta(minutes=int(k) - 1), t - pd.Timedelta(minutes=1), freq="min").values
     for t, k in zip(fill_end, gap_min[fill_end])])) if len(fill_end) else pd.DatetimeIndex([])
of = of.reindex(of.index.union(fill_idx))
is_filled = of["volume"].isna()
of["close"] = of["close"].ffill()
for col in ["open", "high", "low"]:
    of[col] = of[col].fillna(of["close"])
for col in ["volume", "buy_volume", "sell_volume", "trades", "delta"]:
    of[col] = of[col].fillna(0)
of[["instrument_id", "cum_delta_day"]] = of[["instrument_id", "cum_delta_day"]].ffill()
of["instrument_id"] = of["instrument_id"].astype("int64")
idx = of.index
print("filled minutes:", int(is_filled.sum()))

# 换月按比例向前复权
inst = of["instrument_id"]
roll_pos = np.where(inst.ne(inst.shift()) & inst.shift().notna())[0]
factor = pd.Series(1.0, index=idx)
for i in roll_pos:
    factor.iloc[i:] *= of["close"].iloc[i - 1] / of["close"].iloc[i]
raw = of[["open", "high", "low", "close"]].copy()
o, h, l, c = (raw[col] * factor for col in ["open", "high", "low", "close"])
F = pd.DataFrame(index=idx)

atr14 = atr(h, l, c, 14)             # 复权价格单位
atr14_usd = atr14 / factor           # 换回当前合约的美元单位，用于点差特征和标签
# 凡是作分母的 ATR 都设下限 = GC 最小价格变动 0.10 美元（复权价格下乘以复权系数），避免冷清行情 ATR = 0
floor_adj = TICK * factor
atr14_f = np.maximum(atr14, floor_adj)
atr14_usd_f = np.maximum(atr14_usd, TICK)
atr5_f, atr60_f = np.maximum(atr(h, l, c, 5), floor_adj), np.maximum(atr(h, l, c, 60), floor_adj)

# --- 价格变化 ---
G = "价格变化"
for s, col, nm in [(o, "open", "开盘价"), (h, "high", "最高价"), (l, "low", "最低价"), (c, "close", "收盘价")]:
    add(F, f"d_{col}", (s - c.shift(1)) / atr14_f, G, f"(本分钟{nm} − 上一分钟收盘价) / ATR14")

# --- 趋势 ---
G = "趋势"
for n in [15, 60, 240]:
    add(F, f"ma_dist_{n}", (c - c.rolling(n).mean()) / atr14_f, G, f"(收盘价 − {n} 分钟均线) / ATR14")

# --- 趋势强度 ---
G = "趋势强度"
for n in [15, 60]:
    add(F, f"er_{n}", safe_div((c - c.shift(n)).abs(), c.diff().abs().rolling(n).sum(), 0), G,
        f"效率比率：|{n} 分钟净变动| / {n} 分钟内每分钟变动绝对值之和；价格完全没动时为 0")

# --- 动量 ---
G = "动量"
for k in [5, 15, 60]:
    add(F, f"ret_{k}", (c - c.shift(k)) / atr14_f, G, f"过去 {k} 分钟价格变动 / ATR14")
d = c.diff()
wilder = lambda x: x.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
gain, loss = wilder(d.clip(lower=0)), wilder((-d).clip(lower=0))
add(F, "rsi_14", safe_div(100 * gain, gain + loss, 50), G,
    "RSI14 = 100 × 平均上涨幅度 / (平均上涨幅度 + 平均下跌幅度)，Wilder 平滑（α = 1/14）；平均涨跌幅都为 0 时为 50")

# --- 波动率 ---
G = "波动率"
add(F, "atr_rel", atr14 / c, G, "ATR14 / 收盘价")
add(F, "vol_ratio_5_60", atr5_f / atr60_f, G, "ATR5 / ATR60，大于 1 表示波动在扩大")

# range_pos_60（60 分钟区间内的位置）已删除：与 ma_dist_60 高度相关（训练集 r = 0.93），
# 两者都在回答"价格相对于最近一小时处在什么位置"；保留 ma_dist_60，让 ma_dist 15 / 60 / 240 这组多尺度特征保持完整

# --- 订单流 ---
G = "订单流"
g = of.join(book.reindex(idx))
vol = g["volume"].astype(float)
delta = g["delta"].astype(float)
for n in [1, 5, 15]:
    add(F, f"delta_ratio_{n}", safe_div(delta.rolling(n).sum(), vol.rolling(n).sum(), 0), G,
        f"过去 {n} 分钟 (主动买量 − 主动卖量) / 总成交量")
imb = g["imb_last"].copy()
no_rec = is_filled & imb.isna()          # 补出的分钟没有盘口记录 = 盘口没变，沿用上一分钟
imb[no_rec] = imb.ffill()[no_rec]
add(F, "book_imb", imb.fillna(0), G,
    "第 t 分钟结束时 (买一挂单量 − 卖一挂单量) / (两者之和)，−1 到 1；补出的分钟没有盘口记录时沿用上一分钟；"
    "其余没有有效盘口的分钟为 0")
ofi_t = ofi.set_index(["datetime", "instrument_id"])["ofi"]
ofi_raw = pd.Series(ofi_t.reindex(pd.MultiIndex.from_arrays([idx, inst])).to_numpy(), index=idx).fillna(0)
dep_t = depth.set_index(["datetime", "instrument_id"])["depth_tw"]
dep = pd.Series(dep_t.reindex(pd.MultiIndex.from_arrays([idx, inst])).to_numpy(), index=idx).ffill()
ofi_depth = ofi_raw / dep
# 日内时段调整（思路同 Andersen & Bollerslev 1997）：除以所在时段 |OFI/深度| 的训练集中位数，在输出前完成
add(F, "ofi_depth_adj", ofi_depth, G,
    "订单流不平衡 OFI / 第 t 分钟第一档平均挂单量，再除以所在时段的训练集典型值（OFI/深度 绝对值的中位数）；"
    "时段 = 开盘后第 1–5 分钟各一组，其余按美东时间每 30 分钟一组；"
    "OFI 为这一分钟内买一卖一价格和挂单量变化的累计（手；Cont-Kukanov-Stoikov 定义），"
    "平均挂单量 = (买一量 + 卖一量) / 2 按时间加权")

# --- 流动性 ---
G = "流动性"
sptw_t = sptw.set_index(["datetime", "instrument_id"])["spread_tw"]
spread_usd = pd.Series(sptw_t.reindex(pd.MultiIndex.from_arrays([idx, inst])).to_numpy(), index=idx).ffill() * TICK
add(F, "spread_atr", spread_usd / atr14_usd_f, G,
    "第 t 分钟按时间加权的平均买卖点差（美元）/ ATR14；每个点差按它在这一分钟内持续的时间加权；"
    "当分钟没有有效盘口时沿用上一分钟点差")

# --- 时间 ---
G = "时间"
idx_tz = idx.tz_localize("Asia/Shanghai")
et = idx_tz.tz_convert("America/New_York")
ldn = idx_tz.tz_convert("Europe/London")
mod_et = pd.Series(et.hour * 60 + et.minute, index=idx)
add(F, "tod_sin", np.sin(2 * np.pi * mod_et / 1440), G, "美东时间一天中分钟数的正弦编码")
add(F, "tod_cos", np.cos(2 * np.pi * mod_et / 1440), G, "美东时间一天中分钟数的余弦编码")
tdate = trade_date(idx)
dow = tdate.dt.dayofweek
for i, nm in enumerate(["mon", "tue", "wed", "thu", "fri"]):
    add(F, f"dow_{nm}", (dow == i).astype(int), G, f"星期（按 CME 交易日）：{nm}")
ldn_m = ldn.hour * 60 + ldn.minute
ny_on = (mod_et >= 8 * 60) & (mod_et < 17 * 60)
ldn_on = pd.Series((ldn_m >= 8 * 60) & (ldn_m < 16 * 60 + 30), index=idx)
add(F, "sess_london", (ldn_on & ~ny_on).astype(int), G, "仅伦敦时段（伦敦时间 08:00–16:30）")
add(F, "sess_ny", (ny_on & ~ldn_on).astype(int), G, "仅纽约时段（美东时间 08:00–17:00）")
add(F, "sess_overlap", (ldn_on & ny_on).astype(int), G, "伦敦与纽约重叠时段")
add(F, "sess_asia", (~ldn_on & ~ny_on).astype(int), G, "其他时段（以亚洲盘为主）")

# --- 成交量 ---
G = "成交量"
add(F, "relvol_60", vol / vol.shift(1).rolling(60).mean(), G, "第 t 分钟成交量 / 之前 60 根 K 线的平均成交量")

# Databento 的数据质量标记是事后按整天给出的，不作为特征，只作为筛选用的列（flag_degraded_day）
cond = json.load(open(f"{DESK}/GC_mbp1/GLBX-20261025-PT9C6P3YKJ/condition.json"))
bad = {x["date"] for x in cond if x["condition"] != "available"}
flag_degraded = pd.Series(idx_tz.tz_convert("UTC").strftime("%Y-%m-%d"), index=idx).isin(bad).astype(int)

# ================= 合并、清理、输出 =================
out = pd.concat([raw.add_prefix("raw_"), pd.DataFrame({
    "raw_atr14": atr14_usd, "instrument_id": inst, "flag_degraded_day": flag_degraded,
    "is_filled": is_filled.astype("int8")}), F], axis=1)

# 预热期：丢弃第 1 个交易日（最长回看 240 根 K 线；Wilder 平滑的 RSI 起始值影响也基本消失）以及特征仍为空的行
all_tdates = sorted(set(tdate.values))
first_ok = all_tdates[1]
feat_cols = [n for n, _, _ in FEATURES]
keep = (tdate >= first_ok) & out[feat_cols].notna().all(axis=1)
out = out[keep]
out.index.name = "datetime"
ts_out = out.index.to_series()
gap_out = ts_out.diff()
et_halt = lambda a, b: (pd.date_range(a + pd.Timedelta(minutes=1), b - pd.Timedelta(minutes=1), freq="min")
                        .tz_localize("Asia/Shanghai").tz_convert("America/New_York").hour == 17).any()
sched = pd.Series(False, index=out.index)
for t in out.index[gap_out > pd.Timedelta(minutes=1)]:
    sched[t] = et_halt(ts_out.shift()[t], t)
brk = ((gap_out != pd.Timedelta(minutes=1)) & ~sched) | out["instrument_id"].ne(out["instrument_id"].shift())
print("window breaks:", int(brk.sum()), "scheduled halts crossed:", int(sched.sum()))
out["seg_id"] = brk.cumsum().astype("int32")

# 拟合参数用的行：有成交的分钟、不在换月前 3 天、训练集（与标签的切分一致：有成交分钟的前 70%，再去掉分界前 60 分钟）
trade_rows = out.index[out["is_filled"] == 0]
roll_t = out.index[(out["instrument_id"] != out["instrument_id"].shift()).to_numpy() & (np.arange(len(out)) > 0)]
nxt = np.searchsorted(roll_t.values, out.index.values, side="right")
next_roll = pd.DatetimeIndex(np.where(nxt < len(roll_t), roll_t.values[np.minimum(nxt, len(roll_t) - 1)], np.datetime64("NaT")))
pre_roll = np.asarray((next_roll - out.index) <= PRE_ROLL)   # 之后没有换月时为 NaT，比较结果为 False
train_end = trade_rows[int(len(trade_rows) * 0.7)] - pd.Timedelta(minutes=60)
fit_rows = (out.index < train_end) & (out["is_filled"] == 0).to_numpy() & ~np.asarray(pre_roll)

# OFI 日内时段调整：时段键 = 开盘后第 k 分钟（k = 0–4，开盘 = 与上一根 K 线间隔 > 30 分钟，即每日休市或周末后）；
# 其余分钟 = 100 + 美东时间所在的 30 分钟段。典型值只用训练集（与标签切分一致：前 70% 再去掉分界前 60 分钟）
ts = idx.to_series()
seg = (ts.diff() > pd.Timedelta(minutes=30)).cumsum()
mao = ((ts - ts.groupby(seg).transform("first")).dt.total_seconds() // 60).astype(int)
season_key = pd.Series(np.where(mao < 5, mao, 100 + mod_et // 30), index=idx).reindex(out.index)
FAC_PATH = f"{DESK}/features/ofi_depth_seasonal_factors.csv"
if CUTOFF:
    # 截断测试：典型值是在训练集上拟合的参数（同标准化），沿用完整数据算出的系数表
    fac = pd.read_csv(FAC_PATH, index_col="season_key")["median_abs_ofi_depth"]
else:
    tr = fit_rows
    a = out.loc[tr, "ofi_depth_adj"].abs()
    fac = a.groupby(season_key[tr]).median()
    cnt = a.groupby(season_key[tr]).size()
    fac.loc[-1], cnt.loc[-1] = a.median(), len(a)   # −1 = 全部训练集，训练集里没出现过的时段用它
    pd.DataFrame({"median_abs_ofi_depth": fac, "train_minutes": cnt}).rename_axis("season_key").to_csv(FAC_PATH)
out["raw_ofi_depth"] = out["ofi_depth_adj"]
out["ofi_depth_adj"] = out["ofi_depth_adj"] / season_key.map(fac).fillna(fac.loc[-1]).to_numpy()

out.to_parquet(f"{OUT}/features_1min.parquet")
if not CUTOFF:
    out.to_csv(f"{OUT}/features_1min.csv", float_format="%.6g")
pd.DataFrame(FEATURES, columns=["feature", "group", "definition"]).to_csv(
    f"{OUT}/feature_dictionary.csv", index=False, encoding="utf-8-sig")

print("rows", len(out), "features", len(feat_cols), "from", out.index.min(), "to", out.index.max())
print("NaN cells in features:", int(out[feat_cols].isna().sum().sum()))
# 相关系数只用拟合参数用的行（训练集、有成交、不在换月前 3 天）
corr = out.loc[fit_rows, feat_cols].corr()
corr.to_csv(f"{OUT}/feature_corr_train70.csv", encoding="utf-8-sig")
pairs = [(a, b, corr.loc[a, b]) for i, a in enumerate(feat_cols) for b in feat_cols[i + 1:]
         if abs(corr.loc[a, b]) > 0.9]
pairs = pd.DataFrame(pairs, columns=["feature_a", "feature_b", "corr"]).sort_values("corr", key=abs, ascending=False)
pairs.to_csv(f"{OUT}/high_corr_pairs.csv", index=False, encoding="utf-8-sig")
print("high-corr pairs:", len(pairs))
for _, r in pairs.iterrows():
    print(f"  {r.feature_a} ~ {r.feature_b}: {r['corr']:.3f}")
