# 默认组合下每个样本"如果在这一分钟开仓"的平仓分钟和盈亏，输出到 model_data/trade_outcomes.npz（策略层回测用）
# 计算过程见 src/gold_dl/trade_outcomes.py；运行环境：处理数据的环境（需要读 parquet）
from gold_dl.trade_outcomes import build_trade_outcomes

build_trade_outcomes()
