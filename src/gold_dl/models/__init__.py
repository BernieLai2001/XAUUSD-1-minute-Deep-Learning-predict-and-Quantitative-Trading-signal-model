# M4 的模型：朴素基准（naive）、多项逻辑回归（logreg）、LightGBM（lgbm）、简单深度基准 Conv1D / 单层 LSTM（deep）
# 所有模型都输出同一种东西：每个样本 6 个头 × 3 类的 logits（概率模型输出概率时取对数），之后的校准、指标和回测共用
