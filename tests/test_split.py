# 划分检查：训练、验证、测试集按时间先后排列、互不重叠，每个分界处至少隔开 gap_min 分钟（= 最长持有期）
# 本项目的"组"是时间：每一分钟只能属于一个数据集，任何一分钟都不能同时出现在两个数据集里
# 运行：pytest tests/test_split.py（需要先跑完 make_labels.py 和 export_model_arrays.py；数据不存在时跳过）
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
CFG = yaml.safe_load(open(ROOT / "config.yaml", encoding="utf-8"))
LABELS = ROOT / CFG["paths"]["labels"] / "labels_default.parquet"
ARRAYS = ROOT / CFG["paths"]["model_arrays"]
GAP = pd.Timedelta(minutes=CFG["split"]["gap_min"])
SPLITS = ["train", "val", "test"]


@pytest.fixture(scope="module")
def split():
    if not LABELS.exists():
        pytest.skip(f"{LABELS} 不存在，先运行 make_labels.py 和 save_default_labels.py")
    return pd.read_parquet(LABELS, columns=["split"])["split"].astype(str)


def test_each_minute_in_one_set(split):
    # 每一分钟只有一行，所以不会同时属于两个数据集
    assert not split.index.duplicated().any()
    assert set(split.unique()) <= set(SPLITS) | {""}   # "" = 分界处删掉的分钟


def test_sets_in_time_order_with_gap(split):
    t = {s: split.index[split == s] for s in SPLITS}
    for a, b in [("train", "val"), ("val", "test")]:
        assert len(t[a]) > 0 and len(t[b]) > 0
        # 后一个集合的第一分钟，至少比前一个集合的最后一分钟晚 gap_min + 1 分钟，中间正好空出 gap_min 分钟
        assert t[b].min() - t[a].max() > GAP, f"{a} 与 {b} 之间的间隔不足 {GAP}"


def test_samples_follow_the_split(split):
    # 模型实际用到的样本（model_arrays.npz 里的行号）也按同样的时间顺序分开、互不重叠
    if not ARRAYS.exists():
        pytest.skip(f"{ARRAYS} 不存在，先运行 export_model_arrays.py")
    arr = np.load(ARRAYS)
    t = {s: arr["time_ns"][arr[f"end_{s}"]] for s in SPLITS}
    ends = np.concatenate([arr[f"end_{s}"] for s in SPLITS])
    assert len(np.unique(ends)) == len(ends)
    assert t["train"].max() < t["val"].min() and t["val"].max() < t["test"].min()
