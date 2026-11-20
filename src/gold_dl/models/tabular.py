# 逻辑回归和 LightGBM 只能接收一行数字：把第 t 分钟往前的回看窗口里挑出的几分钟拼成一行
#   current：只用第 t 分钟（33 列）
#   lags7：第 t、t−1、t−2、t−5、t−15、t−30、t−59 分钟（7 × 33 = 231 列），都在 60 分钟回看窗口内
# 样本的窗口在导出时已保证 60 行属于同一窗口段（seg_id 相同），所以这些滞后不会跨换月或数据中断
# 特征在 M3 已用训练集的均值和标准差做过 z-score，这里直接沿用，不再重新标准化
import numpy as np

LAG_SETS = {"current": [0], "lags7": [0, 1, 2, 5, 15, 30, 59]}


def make_tabular(X_all, rows, lags):
    return np.hstack([X_all[rows - k] for k in lags]).astype(np.float32)


def feature_names(names, lags):
    return [f"{n}_lag{k}" for k in lags for n in names]
