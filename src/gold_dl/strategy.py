# M4 策略层：所有输出 6 × 3 概率的模型共用同一条"概率 → 开仓"规则和同一套回测
#   期望收益（以 R 为单位，1R = 这笔交易的止损距离）：
#     E[R] = p_止盈 × 盈亏比 − p_止损 × 1 + p_到期 × E[R | 到期] − 固定成本 ÷ 1R
#     E[R | 到期]：训练集里这个头所有"到期"样本的平均毛收益（R），每个头一个数，只用训练集
#     固定成本：往返 2 × (交易所和监管费 + 佣金) ÷ 每手盎司数（美元/盎司），点差已包含在入场和平仓价里
#   决策：每个样本分钟，在有标签的头里挑 E[R] 最高的一种交易；超过 θ 就开仓，否则不交易
#   持仓：同一时间只持有一笔；持仓期间的信号全部忽略，平仓那一分钟结束后才看新信号（第 t' ≥ 平仓分钟时才能再开仓）
#   θ：候选值在 val1 上选，要求交易笔数 ≥ 下限 n，在满足下限的候选里选每笔净收益最高的；没有候选满足下限时取笔数最多的 θ 并标记
#   交易次数下限 n = (2 × 单笔净盈亏标准差 ÷ δ)²，标准差用训练集"每分钟都开仓"（所有头、所有有标签样本）的净 R 计算
import numpy as np
import pandas as pd

from gold_dl.config import CFG, path
from gold_dl.timeutil import trade_date

S = CFG["strategy"]
RR = CFG["labels"]["default_rr"]
COST_RT_PER_OZ = 2 * (S["fee_per_side_usd"] + S["commission_per_side_usd"]) / S["contract_oz"]


def load_outcomes():
    return dict(np.load(path("model_data", "trade_outcomes.npz"), allow_pickle=False))


def sample_outcomes(out, rows):
    """取出某组样本（X_all 行号）的结局、平仓分钟、毛盈亏（美元/盎司）、1R（美元/盎司）"""
    return {k: out[k][rows] for k in ["outcome", "exit_min", "pnl_usd", "r_unit"]}


def train_stats(out, rows_train):
    """只用训练集：每个头 E[R | 到期]，以及"每分钟都开仓"时单笔净 R 的标准差和交易次数下限"""
    o = sample_outcomes(out, rows_train)
    gross_r = o["pnl_usd"] / o["r_unit"]
    net_r = gross_r - COST_RT_PER_OZ / o["r_unit"]
    e_timeout = np.array([np.nanmean(gross_r[o["outcome"][:, j] == 2, j]) for j in range(gross_r.shape[1])])
    sd = float(np.nanstd(net_r[o["outcome"] >= 0]))
    n_min = int(np.ceil((2 * sd / S["delta_r"]) ** 2))
    return {"e_timeout_r": e_timeout, "net_r_sd": sd, "n_min": n_min,
            "mean_net_r_every_minute": float(np.nanmean(net_r[o["outcome"] >= 0])),
            "cost_r_median": float(np.nanmedian((COST_RT_PER_OZ / o["r_unit"])[o["outcome"] >= 0]))}


def expected_r(probs, o, e_timeout):
    """每个样本、每个头的期望净收益（R）；没有标签的头为 −inf（不能选）"""
    er = probs[..., 0] * RR - probs[..., 1] + probs[..., 2] * e_timeout[None, :] - COST_RT_PER_OZ / o["r_unit"]
    return np.where(o["outcome"] >= 0, er, -np.inf)


def backtest(er, o, time_ns, theta):
    """按时间顺序逐分钟决策；返回每笔交易的 DataFrame"""
    order = np.argsort(time_ns)
    best = er.argmax(axis=1)
    best_er = er[np.arange(len(er)), best]
    busy_until = np.iinfo(np.int64).min
    trades = []
    minute = 60_000_000_000
    for i in order:
        if time_ns[i] < busy_until or not best_er[i] > theta:
            continue
        j = best[i]
        exit_ns = time_ns[i] + int(o["exit_min"][i, j]) * minute
        gross = float(o["pnl_usd"][i, j])
        r = float(o["r_unit"][i, j])
        trades.append({"time_ns": int(time_ns[i]), "head": int(j), "expected_r": float(best_er[i]),
                       "outcome": int(o["outcome"][i, j]), "gross_usd": gross,
                       "net_usd": gross - COST_RT_PER_OZ, "net_r": gross / r - COST_RT_PER_OZ / r, "exit_ns": exit_ns})
        busy_until = exit_ns
    return pd.DataFrame(trades)


def strategy_metrics(trades, n_days):
    if len(trades) == 0:
        return {"n_trades": 0, "trades_per_day": 0.0, "mean_net_r": float("nan"), "win_rate": float("nan"),
                "profit_factor": float("nan"), "max_drawdown_r": 0.0, "cum_net_r": 0.0, "cum_net_usd_oz": 0.0}
    r = trades["net_r"].to_numpy()
    cum = np.cumsum(r)
    gains, losses = r[r > 0].sum(), -r[r < 0].sum()
    return {"n_trades": int(len(r)), "trades_per_day": len(r) / n_days, "mean_net_r": float(r.mean()),
            "win_rate": float((r > 0).mean()), "profit_factor": float(gains / losses) if losses > 0 else float("inf"),
            "max_drawdown_r": float((np.maximum.accumulate(np.concatenate([[0], cum])) - np.concatenate([[0], cum])).max()),
            "cum_net_r": float(cum[-1]), "cum_net_usd_oz": float(trades["net_usd"].sum())}


def n_trading_days(time_ns):
    return int(trade_date(pd.DatetimeIndex(time_ns)).nunique())


def select_theta(er, o, time_ns, n_min):
    """在 val1 上选 θ：交易笔数 ≥ n_min 的候选里选每笔净收益最高的；返回 (θ, 每个候选的结果表, 是否满足下限)"""
    days = n_trading_days(time_ns)
    rows = [{"theta": th, **strategy_metrics(backtest(er, o, time_ns, th), days)} for th in S["theta_grid_r"]]
    tab = pd.DataFrame(rows)
    ok = tab[tab["n_trades"] >= n_min]
    if len(ok):
        return float(ok.sort_values("mean_net_r", ascending=False).iloc[0]["theta"]), tab, True
    return float(tab.sort_values("n_trades", ascending=False).iloc[0]["theta"]), tab, False
