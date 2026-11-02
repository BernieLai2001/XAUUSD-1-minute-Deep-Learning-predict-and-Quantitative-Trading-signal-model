# tf.data 输入流水线：训练、验证、测试用完全相同的步骤，只有训练集打乱
#   ① 读入 model_arrays.npz（已补值、长尾处理、标准化，见 export_model_arrays.py），整张特征表常驻内存（46.5 MB）
#   ② 样本 = 窗口最后一行的行号 t 和 6 个头的标签；from_tensor_slices
#   ③ 只有训练集：shuffle（缓冲区 = 整个训练集，每个 epoch 重新打乱）
#   ④ batch
#   ⑤ map：按行号批量切窗口 X_all[t − L + 1 : t + 1] → [B, L, 33] float32
#   ⑥ prefetch
# 每个元素：(x [B, L, 33] float32, y [B, 6] int32（−1 = 标签为空）, t [B] int64（行号，只用于核对和画图，不进模型）)
import numpy as np
import tensorflow as tf

from pathlib import Path
P = str(Path(__file__).resolve().parent.parent)

L = 60
BATCH = 256
SEED = 42


def load_arrays(path=f"{P}/model_data/model_arrays.npz"):
    return dict(np.load(path, allow_pickle=False))


def make_dataset(arr, split, batch=BATCH, lookback=L, seed=SEED):
    X_all = tf.constant(arr["X_all"])
    end = arr[f"end_{split}"]
    y = arr[f"y_{split}"].astype(np.int32)
    offsets = tf.range(-lookback + 1, 1, dtype=tf.int64)
    ds = tf.data.Dataset.from_tensor_slices((end, y))
    if split == "train":
        ds = ds.shuffle(len(end), seed=seed, reshuffle_each_iteration=True)
    ds = ds.batch(batch)

    def to_window(t, yb):
        x = tf.gather(X_all, t[:, None] + offsets[None, :])
        return x, yb, t

    return ds.map(to_window, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)


def masked_sparse_ce(y_true, y_pred):
    # y_true [B, 6]，−1 = 标签为空；y_pred [B, 6, 3] 为 logits；按有标签的头取平均
    y_true = tf.cast(y_true, tf.int32)
    mask = tf.cast(y_true >= 0, tf.float32)
    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.maximum(y_true, 0), logits=y_pred)
    return tf.reduce_sum(ce * mask) / tf.maximum(tf.reduce_sum(mask), 1.0)
