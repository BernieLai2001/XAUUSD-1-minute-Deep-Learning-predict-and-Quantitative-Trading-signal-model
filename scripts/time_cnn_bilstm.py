# 补测 CNN-BiLSTM 完整遍历一次训练集的时间，和第 6 节 section6_pipeline_checks.py ④ 的计时方法相同（批量 256，L = 60，先预热 5 步）
# 输出：数据说明/第6节/cnn_bilstm_timing.json
import json, os, sys, time
import tensorflow as tf
import keras

from gold_dl.data_pipeline import BATCH, L, SEED, load_arrays, make_dataset, masked_sparse_ce

from pathlib import Path
P = str(Path(__file__).resolve().parent.parent)
keras.utils.set_random_seed(SEED)
arr = load_arrays()
nf, nh = len(arr["feature_names"]), len(arr["head_names"])
ds = {s: make_dataset(arr, s).map(lambda x, y, t: (x, y)) for s in ["train", "val"]}


def cnn_bilstm(units):
    return keras.Sequential([keras.Input((L, nf)),
                             keras.layers.Conv1D(64, 5, padding="same", activation="relu"),
                             keras.layers.Conv1D(64, 5, padding="same", activation="relu"),
                             keras.layers.Bidirectional(keras.layers.LSTM(units)),
                             keras.layers.Dense(nh * 3), keras.layers.Reshape((nh, 3))])


out = {"batch_size": BATCH, "lookback": L}
for name, model in [("lstm64", keras.Sequential([keras.Input((L, nf)), keras.layers.LSTM(64),
                                                 keras.layers.Dense(nh * 3), keras.layers.Reshape((nh, 3))])),
                    ("cnn_bilstm64", cnn_bilstm(64)), ("cnn_bilstm128", cnn_bilstm(128))]:
    model.compile(optimizer="adam", loss=masked_sparse_ce)
    model.fit(ds["train"].take(5), verbose=0)   # 预热：编译图
    t0 = time.perf_counter()
    model.fit(ds["train"], epochs=1, verbose=0)
    tr = time.perf_counter() - t0
    t0 = time.perf_counter()
    model.evaluate(ds["val"], verbose=0)
    va = time.perf_counter() - t0
    out[name] = {"params": int(model.count_params()), "train_s": tr, "val_s": va}
    print(name, out[name], flush=True)
json.dump(out, open(f"{P}/数据说明/第6节/cnn_bilstm_timing.json", "w"), indent=1)
