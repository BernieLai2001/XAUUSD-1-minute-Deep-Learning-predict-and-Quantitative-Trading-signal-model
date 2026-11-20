# 三重障碍标签（对照版）：不剔除换月前 3 天，只用于图 6(c) 的剔除前对比，不用于训练
# 输出到 labels/check_with_preroll/；计算过程与正式版相同，见 src/gold_dl/labels.py
from gold_dl.config import path
from gold_dl.labels import build_labels

build_labels(path("labels", "check_with_preroll"), pre_roll_days=0)
