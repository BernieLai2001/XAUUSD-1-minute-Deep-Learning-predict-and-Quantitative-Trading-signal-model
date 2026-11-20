# 简单深度基准：Conv1D 和单层 LSTM。输出头、损失、校准、训练流程与之后的主模型（CNN-BiLSTM）相同
#   Conv1D：Conv1D(64, 5, ReLU, 不补边) × 2 → 全局平均池化 → Dense(18) → Reshape(6, 3)，32,338 个参数
#   LSTM：LSTM(64，只返回最后一个时间步) → Dense(18) → Reshape(6, 3)，26,258 个参数
#   dropout > 0 时在 Dense 输出层前加一层 Dropout（不增加参数）；M4 默认不加，训练完看曲线再决定
# 训练：屏蔽空标签的交叉熵，Adam，批量 256，最多 max_epochs 个 epoch；按验证集前一半（val1）的损失早停，
#   耐心值 patience，停止后恢复 val1 损失最低那一轮的权重；不用类别权重
#   BackupAndRestore：训练中断后重新运行同一命令，从上次的 epoch 接着训练；TensorBoard：记录训练曲线
import os
import time

import keras
import numpy as np
import tensorflow as tf

from gold_dl.config import CFG
from gold_dl.data_pipeline import make_dataset, masked_sparse_ce

TR = CFG["training"]


def build(name, n_feat, lookback, n_heads=6, dropout=0.0):
    inp = [keras.Input((lookback, n_feat))]
    body = {"conv1d": [keras.layers.Conv1D(64, 5, activation="relu"), keras.layers.Conv1D(64, 5, activation="relu"),
                       keras.layers.GlobalAveragePooling1D()],
            "lstm": [keras.layers.LSTM(64)]}[name]
    drop = [keras.layers.Dropout(dropout)] if dropout > 0 else []
    head = [keras.layers.Dense(n_heads * 3, dtype="float32"), keras.layers.Reshape((n_heads, 3))]
    return keras.Sequential(inp + body[:] + drop + head, name=name)


def xy(ds):
    """流水线每个元素是 (x, y, t)；Keras 会把第三项当成样本权重，所以训练和预测时去掉行号 t"""
    return ds.map(lambda x, y, t: (x, y))


class EpochTimer(keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        self.t0 = time.time()

    def on_epoch_end(self, epoch, logs=None):
        logs["epoch_seconds"] = time.time() - self.t0


def train(name, arr, seed, out_dir, dropout=0.0):
    """训练一个种子；arr 需含 end_val1 / y_val1（验证集前一半）。返回 (模型, 每个 epoch 的记录, 停止信息)"""
    keras.utils.set_random_seed(seed)
    L = CFG["pipeline"]["lookback"]
    model = build(name, arr["X_all"].shape[1], L, dropout=dropout)
    model.compile(optimizer=keras.optimizers.Adam(TR["learning_rate"]), loss=masked_sparse_ce)
    ds_tr = xy(make_dataset(arr, "train", seed=seed))
    ds_v1 = xy(make_dataset(arr, "val1"))
    os.makedirs(out_dir, exist_ok=True)
    cb = [EpochTimer(),
          keras.callbacks.EarlyStopping(monitor="val_loss", patience=TR["patience"], restore_best_weights=True),
          keras.callbacks.BackupAndRestore(os.path.join(out_dir, "backup")),
          keras.callbacks.TensorBoard(os.path.join(out_dir, "tensorboard"))]
    hist = model.fit(ds_tr, validation_data=ds_v1, epochs=TR["max_epochs"], callbacks=cb, verbose=2)
    h = {k: [float(v) for v in vals] for k, vals in hist.history.items()}
    best = int(np.argmin(h["val_loss"]))
    info = {"epochs_run": len(h["val_loss"]), "best_epoch": best + 1, "best_val1_loss": h["val_loss"][best],
            "params": int(model.count_params()), "epoch_seconds_mean": float(np.mean(h["epoch_seconds"][1:] or h["epoch_seconds"]))}
    return model, h, info


def predict_logits(model, arr, split):
    return model.predict(xy(make_dataset(arr, split)), verbose=0)


def overfit_small_batch(name, arr, n=16, steps=500, seed=0):
    """过拟合一小批：从训练集里不同交易日各取一个、6 个头都有标签的样本，不加 dropout，反复训练 steps 步；
    代码没有错误时，损失应降到接近 0、96 个标签全部预测正确"""
    from gold_dl.timeutil import trade_date
    import pandas as pd
    keras.utils.set_random_seed(seed)
    end, y = arr["end_train"], arr["y_train"]
    full = np.where((y >= 0).all(axis=1))[0]
    days = trade_date(pd.DatetimeIndex(arr["time_ns"][end[full]]))
    rng = np.random.default_rng(seed)
    pick_days = rng.choice(np.unique(days), n, replace=False)
    idx = np.array([rng.choice(full[np.asarray(days == d)]) for d in pick_days])
    L = CFG["pipeline"]["lookback"]
    x = np.stack([arr["X_all"][t - L + 1:t + 1] for t in end[idx]])
    yy = y[idx].astype(np.int32)
    model = build(name, x.shape[2], L, dropout=0.0)
    model.compile(optimizer=keras.optimizers.Adam(TR["learning_rate"]), loss=masked_sparse_ce)
    losses = [float(model.train_on_batch(x, yy)) for _ in range(steps)]
    pred = model.predict(x, verbose=0).argmax(-1)
    return {"samples": n, "steps": steps, "final_loss": losses[-1], "n_correct": int((pred == yy).sum()),
            "n_labels": int(yy.size), "all_correct": bool((pred == yy).all()), "loss_curve": losses}
