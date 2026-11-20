# M4 模型层评估：所有模型共用（朴素、逻辑回归、LightGBM、Conv1D、LSTM，以及之后的 CNN-BiLSTM）
#   ① 验证集按 CME 交易日分成前后两半（val1 / val2）：val1 用于选超参数、早停、决定是否加 dropout、概率校准、选开仓阈值 θ；
#      val2 只用于报告。按交易日切，分界处正好是每日休市，标签不会跨过休市，两半之间不会互相用到对方的价格
#   ② 朴素基准：每个头输出训练集里这个头的三类比例
#   ③ 概率校准：每个头单独做温度缩放 softmax(logits / T)，T 在 val1 上按 log-loss 最小拟合；模型的 logits 若是概率，先取对数
#   ④ 指标（每个头只用这个头有标签的样本）：log-loss、相对朴素基准的改进 1 − LL / LL_朴素（主要指标 = 6 个头等权平均）、
#      宏平均 F1 和每类 F1（预测 = 概率最大的类）、Brier score（三类概率与 one-hot 之差的平方和）、止盈类 AUC（止盈 vs 其他）
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.metrics import f1_score, roc_auc_score

from gold_dl.timeutil import trade_date

CLASSES = ["tp", "sl", "timeout"]   # 0 = 止盈、1 = 止损、2 = 到期
EPS = 1e-12


def val_halves(time_ns):
    """验证集样本按 CME 交易日分成前后两半，返回 (half1 布尔数组, half2 布尔数组, 每半的交易日列表)"""
    days = trade_date(pd.DatetimeIndex(time_ns))
    uniq = np.sort(days.unique())
    cut = uniq[(len(uniq) + 1) // 2]   # 交易日个数为奇数时，前一半多一天
    h1 = np.asarray(days < cut)
    return h1, ~h1, (uniq[uniq < cut], uniq[uniq >= cut])


def naive_probs(y_train):
    """朴素基准：每个头训练集里三类的比例，形状 [6, 3]"""
    return np.stack([np.bincount(y[y >= 0], minlength=3) / (y >= 0).sum() for y in y_train.T])


def softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def head_logloss(p, y):
    return float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], EPS, 1))))


def fit_temperature(logits, y):
    """每个头单独拟合温度 T（在 val1 上），返回长度 6 的数组；T > 1 表示原模型过于自信"""
    T = np.ones(logits.shape[1])
    for j in range(logits.shape[1]):
        m = y[:, j] >= 0
        z, yy = logits[m, j], y[m, j]
        res = minimize_scalar(lambda lt: head_logloss(softmax(z / np.exp(lt)), yy), bounds=(-3, 3), method="bounded")
        T[j] = np.exp(res.x)
    return T


def calibrate(logits, T):
    return softmax(logits / T[None, :, None])


def head_metrics(p, y, p_naive):
    """一个头的全部模型层指标；p [N, 3]，y [N]（已去掉空标签），p_naive [3]"""
    ll = head_logloss(p, y)
    ll0 = head_logloss(np.broadcast_to(p_naive, p.shape), y)
    pred = p.argmax(axis=1)
    onehot = np.eye(3)[y]
    f1 = f1_score(y, pred, labels=[0, 1, 2], average=None, zero_division=0)
    return {"n": int(len(y)), "logloss": ll, "naive_logloss": ll0, "improvement": 1 - ll / ll0,
            "macro_f1": float(f1.mean()), **{f"f1_{c}": float(v) for c, v in zip(CLASSES, f1)},
            "recall_timeout": float(((pred == 2) & (y == 2)).sum() / max((y == 2).sum(), 1)),
            "brier": float(((p - onehot) ** 2).sum(axis=1).mean()),
            "auc_tp": float(roc_auc_score(y == 0, p[:, 0])) if 0 < (y == 0).sum() < len(y) else float("nan")}


def model_metrics(probs, y, p_naive, heads):
    """6 个头分别计算，再等权平均；返回 (每个头的 DataFrame, 平均值 dict)"""
    rows = []
    for j, h in enumerate(heads):
        m = y[:, j] >= 0
        rows.append({"head": h, **head_metrics(probs[m, j], y[m, j], p_naive[j])})
    per_head = pd.DataFrame(rows).set_index("head")
    summary = per_head.drop(columns=["n"]).mean().to_dict()
    return per_head, summary


def reliability(probs, y, cls=0, bins=10):
    """可靠性图的数据：某一类的预测概率分 bins 组，每组的平均预测概率、实际比例、样本数（所有头合在一起）"""
    m = y >= 0
    p, hit = probs[..., cls][m], (y[m] == cls)
    edges = np.linspace(0, 1, bins + 1)
    k = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    return pd.DataFrame({"pred": [p[k == i].mean() if (k == i).any() else np.nan for i in range(bins)],
                         "actual": [hit[k == i].mean() if (k == i).any() else np.nan for i in range(bins)],
                         "n": [int((k == i).sum()) for i in range(bins)]})
