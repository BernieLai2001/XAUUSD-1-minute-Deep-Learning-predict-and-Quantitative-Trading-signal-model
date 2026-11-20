# 三重障碍标签（正式版）：换月前 3 天（config.yaml 的 cleaning.pre_roll_days）不生成标签，输出到 labels/
# 计算过程见 src/gold_dl/labels.py
from gold_dl.config import path
from gold_dl.labels import build_labels

build_labels(path("labels"))   # 换月前天数取 config.yaml 的 cleaning.pre_roll_days
