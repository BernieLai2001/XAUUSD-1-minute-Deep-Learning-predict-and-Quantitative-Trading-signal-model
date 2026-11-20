# 朴素基准：不看任何输入，每个样本都输出训练集里这个头的三类比例；没有参数要学，也没有随机性
import numpy as np

from gold_dl.evaluation import naive_probs


def naive_logits(y_train, n):
    return np.log(np.broadcast_to(naive_probs(y_train), (n, y_train.shape[1], 3)).copy())
