# 多项逻辑回归：6 个头各训练一个三分类模型，每个头只用这个头有标签的训练样本；不用类别权重
#   L2 用 lbfgs 求解器，L1 用 saga（scikit-learn 里唯一支持 L1 + 多分类的求解器）
#   没有随机性：给定数据和设置，结果是固定的（saga 的随机打乱也固定 random_state）
import time
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

MAX_ITER = {"l2": 2000, "l1": 1000}


def fit_heads(X, Y, C, penalty):
    models, info = [], []
    for j in range(Y.shape[1]):
        m = Y[:, j] >= 0
        t0 = time.time()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", ConvergenceWarning)
            clf = LogisticRegression(C=C, penalty=penalty, solver="lbfgs" if penalty == "l2" else "saga",
                                     max_iter=MAX_ITER[penalty], tol=1e-4, random_state=0).fit(X[m], Y[m, j])
        info.append({"n_iter": int(clf.n_iter_[0]), "converged": not any(issubclass(x.category, ConvergenceWarning) for x in w),
                     "nonzero_coef": int((np.abs(clf.coef_) > 0).any(axis=0).sum()), "seconds": time.time() - t0})
        models.append(clf)
    return models, info


def predict_logits(models, X):
    return np.log(np.clip(np.stack([m.predict_proba(X) for m in models], axis=1), 1e-12, 1))
