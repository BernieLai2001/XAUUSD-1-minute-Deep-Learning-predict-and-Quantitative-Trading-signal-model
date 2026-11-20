# M4 每次运行的公共步骤：读数据、切验证集两半、记录每次尝试、对最终模型做校准 + 模型层评估 + 策略层评估并存盘
# 输出目录 results/m4/：
#   runs.jsonl：每次尝试一行（模型、阶段、输入方式、超参数、种子、每个头的 val1 log-loss、早停轮数、主要指标、耗时）
#   <模型>/seed<种子>/predictions.npz：验证集全部样本的时间、真实标签、原始 logits、校准后概率、属于哪一半、温度 T
#   <模型>/seed<种子>/metrics.json：val2 上的模型层指标（校准前、校准后）、θ 的选择过程（val1）、val2 上的策略层指标
#   <模型>/seed<种子>/trades_val2.csv：val2 上每一笔交易
import json
import os
import time

import numpy as np

from gold_dl import evaluation as ev
from gold_dl import strategy as st
from gold_dl.config import CFG, path
from gold_dl.labels import HEADS

OUT = path("results", "m4")


def load():
    """model_arrays + 验证集两半（加 end_val1 / y_val1 / end_val2 / y_val2，深度模型的流水线直接用）"""
    arr = dict(np.load(path("model_arrays"), allow_pickle=False))
    h1, h2, days = ev.val_halves(arr["time_ns"][arr["end_val"]])
    for k, m in [("val1", h1), ("val2", h2)]:
        arr[f"end_{k}"], arr[f"y_{k}"] = arr["end_val"][m], arr["y_val"][m]
    arr["val_half1"] = h1
    return arr, days


def primary(logits, y, p_naive):
    """主要指标：6 个头 log-loss 相对各自朴素基准的改进，再等权平均（用原始概率，选超参数用）"""
    _, s = ev.model_metrics(ev.softmax(logits), y, p_naive, HEADS)
    return s["improvement"]


def log_run(rec):
    os.makedirs(OUT, exist_ok=True)
    rec = {"logged_at": time.strftime("%Y-%m-%d %H:%M:%S"), **rec}
    with open(OUT / "runs.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False, default=float) + "\n")


def evaluate_final(model, seed, logits_val, arr, extra=None, calibrate=True):
    """最终模型：val1 上按头拟合温度 T、选 θ；val2 上报告模型层和策略层指标；全部存盘"""
    out_dir = OUT / model / f"seed{seed}"
    os.makedirs(out_dir, exist_ok=True)
    y, t = arr["y_val"], arr["time_ns"][arr["end_val"]]
    h1 = arr["val_half1"]
    h2 = ~h1
    p_naive = ev.naive_probs(arr["y_train"])
    T = ev.fit_temperature(logits_val[h1], y[h1]) if calibrate else np.ones(len(HEADS))
    probs = ev.calibrate(logits_val, T)
    raw_head, raw_sum = ev.model_metrics(ev.softmax(logits_val[h2]), y[h2], p_naive, HEADS)
    cal_head, cal_sum = ev.model_metrics(probs[h2], y[h2], p_naive, HEADS)

    outc = st.load_outcomes()
    stats = st.train_stats(outc, arr["end_train"])
    o = st.sample_outcomes(outc, arr["end_val"])
    er = st.expected_r(probs, o, stats["e_timeout_r"])
    sub = lambda d, m: {k: v[m] for k, v in d.items()}
    theta, theta_tab, ok = st.select_theta(er[h1], sub(o, h1), t[h1], stats["n_min"])
    trades = st.backtest(er[h2], sub(o, h2), t[h2], theta)
    strat = st.strategy_metrics(trades, st.n_trading_days(t[h2]))
    if len(trades):
        trades["head_name"] = [HEADS[j] for j in trades["head"]]
    trades.to_csv(out_dir / "trades_val2.csv", index=False)

    np.savez(out_dir / "predictions.npz", time_ns=t, y=y, logits=logits_val.astype(np.float32),
             probs=probs.astype(np.float32), half1=h1, temperature=T, head_names=np.array(HEADS))
    res = {"model": model, "seed": seed, "temperature": dict(zip(HEADS, T.round(4).tolist())),
           "model_layer_val2_calibrated": {"summary": cal_sum, "per_head": cal_head.to_dict(orient="index")},
           "model_layer_val2_raw": {"summary": raw_sum, "per_head": raw_head.to_dict(orient="index")},
           "strategy_train_stats": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in stats.items()},
           "theta_selection_val1": theta_tab.to_dict(orient="records"), "theta": theta, "theta_meets_n_min": ok,
           "strategy_val2": strat, "extra": extra or {}}
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1, default=float)
    print(f"[{model} seed {seed}] val2 primary (calibrated) {cal_sum['improvement']:.4f} | raw {raw_sum['improvement']:.4f} | "
          f"macro F1 {cal_sum['macro_f1']:.3f} | theta {theta} (n_min ok: {ok}) | val2 trades {strat['n_trades']}, "
          f"mean net R {strat['mean_net_r']:.4f}", flush=True)
    return res
