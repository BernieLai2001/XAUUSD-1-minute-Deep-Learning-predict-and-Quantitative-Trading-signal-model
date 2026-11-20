# LightGBM：6 个头各训练一个三分类模型（objective = multiclass），每个头只用这个头有标签的训练样本；不用类别权重
#   树的数量上限 2,000，按验证集前一半（val1）上这个头的 log-loss 早停（连续 50 棵没有改进就停），每个头记录自己停在第几棵
#   随机性只来自特征采样和样本采样（feature_fraction、bagging_fraction < 1 时），seed 控制
import time

import lightgbm as lgb
import numpy as np

MAX_TREES, EARLY_STOP = 2000, 50
SEARCH_SPACE = {"learning_rate": [0.03, 0.05, 0.1], "num_leaves": [15, 31, 63, 127], "max_depth": [-1, 4, 6, 8],
                "min_data_in_leaf": [100, 500, 2000], "feature_fraction": [0.5, 0.7, 1.0],
                "bagging_fraction": [0.5, 0.7, 1.0], "lambda_l2": [0, 1, 10]}
REFERENCE = {"learning_rate": 0.05, "num_leaves": 31, "max_depth": -1, "min_data_in_leaf": 500,
             "feature_fraction": 0.7, "bagging_fraction": 0.7, "lambda_l2": 0}


def sample_params(rng, n):
    """从 SEARCH_SPACE 随机抽 n 组不重复的设置"""
    seen, out = set(), []
    while len(out) < n:
        p = {k: v[rng.integers(len(v))] for k, v in SEARCH_SPACE.items()}
        key = tuple(p.values())
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def has_randomness(p):
    return p["feature_fraction"] < 1 or p["bagging_fraction"] < 1


def fit_heads(X, Y, Xv, Yv, params, seed, threads=12):
    models, info = [], []
    for j in range(Y.shape[1]):
        m, mv = Y[:, j] >= 0, Yv[:, j] >= 0
        t0 = time.time()
        p = {"objective": "multiclass", "num_class": 3, "verbose": -1, "seed": seed, "num_threads": threads,
             "bagging_freq": 1 if params["bagging_fraction"] < 1 else 0, **params}
        ds = lgb.Dataset(X[m], Y[m, j], free_raw_data=False)
        dv = lgb.Dataset(Xv[mv], Yv[mv, j], reference=ds)
        b = lgb.train(p, ds, MAX_TREES, valid_sets=[dv], callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)])
        info.append({"best_iteration": int(b.best_iteration), "val1_logloss": float(b.best_score["valid_0"]["multi_logloss"]),
                     "seconds": time.time() - t0})
        models.append(b)
    return models, info


def predict_logits(models, X):
    return np.log(np.clip(np.stack([b.predict(X, num_iteration=b.best_iteration) for b in models], axis=1), 1e-12, 1))
