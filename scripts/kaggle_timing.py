# 在 Kaggle Notebooks 上测完整遍历一次训练集的时间（第 6 节 6.5 的数字要在实际训练的平台上重测）
# 计时方法和本机的 section6_pipeline_checks.py ④、time_cnn_bilstm.py 相同：批量 256，先预热 5 步编译图，再计一个 epoch 的训练和一遍验证
# 每个模型测两次：float32、混合精度（mixed_float16，课程要求开启）
# 用法：把 model_arrays.npz 上传成私有 Kaggle 数据集并加到 notebook，加速器选 GPU，然后在一个单元格里运行
#   !python kaggle_timing.py              # 完整测一遍，L = 60
#   !python kaggle_timing.py --L 120      # 换窗口长度
#   !python kaggle_timing.py --steps 20   # 只跑前 20 步，检查脚本能不能运行（不能用来估算）
# 输出：/kaggle/working/kaggle_timing.json，并打印表格和总时间估算
# 流水线和损失函数照抄 data_pipeline.py，这样只需要上传一个数据文件
import argparse, glob, json, os, platform, time
import numpy as np
import tensorflow as tf
import keras

ap = argparse.ArgumentParser()
ap.add_argument("--data", default=None, help="model_arrays.npz 的路径；不填就在 /kaggle/input 下自动找")
ap.add_argument("--L", type=int, default=60)
ap.add_argument("--batch", type=int, default=256)
ap.add_argument("--steps", type=int, default=0, help=">0 时只跑前这么多步，用于检查脚本")
ap.add_argument("--epochs", type=int, default=30, help="估算用：每次运行的 epoch 数")
ap.add_argument("--runs", type=int, default=40, help="估算用：运行次数")
ap.add_argument("--models", default="smoke_dense,conv1d,lstm64,cnn_bilstm64,cnn_bilstm128", help="要测的模型，逗号分隔")
ap.add_argument("--policies", default="float32,mixed_float16", help="要测的精度，逗号分隔")
args, _ = ap.parse_known_args()   # 直接粘贴进单元格运行时，忽略 Jupyter 自带的 -f 参数
SEED = 42
OUT = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."

path = args.data or (glob.glob("/kaggle/input/**/model_arrays.npz", recursive=True) or [None])[0]
assert path, "找不到 model_arrays.npz：请先把数据集加到 notebook，或用 --data 指定路径"
arr = dict(np.load(path, allow_pickle=False))
nf, nh = arr["X_all"].shape[1], arr["y_train"].shape[1]


def make_dataset(split, batch=args.batch, lookback=args.L, seed=SEED):
    X_all = tf.constant(arr["X_all"])
    end = arr[f"end_{split}"]
    y = arr[f"y_{split}"].astype(np.int32)
    # npz 里的样本按 L = 60 挑选；L 更长时，按 export_model_arrays.py 的同一规则（窗口内 seg_id 相同）重新筛
    keep = (end >= lookback - 1) & (arr["seg_id"][end] == arr["seg_id"][np.maximum(end - lookback + 1, 0)])
    end, y = end[keep], y[keep]
    stats.setdefault("samples", {})[split] = int(len(end))
    offsets = tf.range(-lookback + 1, 1, dtype=tf.int64)
    ds = tf.data.Dataset.from_tensor_slices((end, y))
    if split == "train":
        ds = ds.shuffle(len(end), seed=seed, reshuffle_each_iteration=True)
    ds = ds.batch(batch)

    def to_window(t, yb):
        return tf.gather(X_all, t[:, None] + offsets[None, :]), yb

    ds = ds.map(to_window, num_parallel_calls=tf.data.AUTOTUNE)
    if args.steps:
        ds = ds.take(args.steps)
    return ds.prefetch(tf.data.AUTOTUNE)


def masked_sparse_ce(y_true, y_pred):
    # y_true [B, 6]，−1 = 标签为空；y_pred [B, 6, 3] 为 logits；按有标签的头取平均
    y_true = tf.cast(y_true, tf.int32)
    y_pred = tf.cast(y_pred, tf.float32)
    mask = tf.cast(y_true >= 0, tf.float32)
    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.maximum(y_true, 0), logits=y_pred)
    return tf.reduce_sum(ce * mask) / tf.maximum(tf.reduce_sum(mask), 1.0)


def head():
    # 混合精度时输出层保持 float32，避免 softmax 数值溢出
    return [keras.layers.Dense(nh * 3, dtype="float32"), keras.layers.Reshape((nh, 3))]


def build(name):
    inp = [keras.Input((args.L, nf))]
    conv = [keras.layers.Conv1D(64, 5, padding="same", activation="relu"),
            keras.layers.Conv1D(64, 5, padding="same", activation="relu")]
    body = {"smoke_dense": [keras.layers.Flatten()],
            "conv1d": [keras.layers.Conv1D(64, 5, activation="relu"), keras.layers.Conv1D(64, 5, activation="relu"),
                       keras.layers.GlobalAveragePooling1D()],
            "lstm64": [keras.layers.LSTM(64)],
            "cnn_bilstm64": conv + [keras.layers.Bidirectional(keras.layers.LSTM(64))],
            "cnn_bilstm128": conv + [keras.layers.Bidirectional(keras.layers.LSTM(128))]}
    # M2 提案的完整结构：卷积层后接批归一化和 dropout，BiLSTM 之后接一个共享全连接层（取较宽的 64）
    m2 = [keras.layers.Conv1D(64, 5, padding="same"), keras.layers.BatchNormalization(), keras.layers.ReLU(),
          keras.layers.Dropout(0.2),
          keras.layers.Conv1D(64, 5, padding="same"), keras.layers.BatchNormalization(), keras.layers.ReLU(),
          keras.layers.Dropout(0.2)]
    for u in [64, 128]:   # 每次 build 只取其中一个，所以两者共用 m2 的层没有关系
        body[f"cnn_bilstm{u}_m2"] = m2 + [keras.layers.Bidirectional(keras.layers.LSTM(u)),
                                          keras.layers.Dense(64, activation="relu"), keras.layers.Dropout(0.2)]
    return keras.Sequential(inp + body[name] + head())


stats = {"platform": platform.node(), "tf": tf.__version__, "keras": keras.__version__,
         "gpus": [d.name for d in tf.config.list_physical_devices("GPU")],
         "gpu_detail": [tf.config.experimental.get_device_details(d).get("device_name", "")
                        for d in tf.config.list_physical_devices("GPU")],
         "lookback": args.L, "batch_size": args.batch, "steps_limit": args.steps, "results": []}
print("TF", tf.__version__, "| GPU:", stats["gpu_detail"] or "无 GPU，请在右侧 Settings 里选加速器", flush=True)
keras.utils.set_random_seed(SEED)
ds_tr, ds_va = make_dataset("train"), make_dataset("val")
print(f"L = {args.L}：训练集 {stats['samples']['train']:,} 个样本，验证集 {stats['samples']['val']:,} 个", flush=True)

t0 = time.perf_counter()
for _ in ds_tr:
    pass
stats["pipeline_only_s"] = time.perf_counter() - t0
print(f"只跑流水线：{stats['pipeline_only_s']:.2f} 秒", flush=True)

for policy in args.policies.split(","):
    keras.mixed_precision.set_global_policy(policy)
    for name in args.models.split(","):
        keras.backend.clear_session()
        model = build(name)
        model.compile(optimizer="adam", loss=masked_sparse_ce)
        model.fit(ds_tr.take(5), verbose=0)   # 预热：编译图
        t0 = time.perf_counter()
        model.fit(ds_tr, epochs=1, verbose=0)
        tr = time.perf_counter() - t0
        t0 = time.perf_counter()
        model.evaluate(ds_va, verbose=0)
        va = time.perf_counter() - t0
        r = {"model": name, "policy": policy, "params": int(model.count_params()), "train_s": tr, "val_s": va,
             "est_hours": (tr + va) * args.epochs * args.runs / 3600}
        stats["results"].append(r)
        print(f"{policy:14s} {name:14s} 参数 {r['params']:>8,}  训练 {tr:7.1f} 秒  验证 {va:6.1f} 秒  "
              f"{args.epochs} 个 epoch × {args.runs} 次 ≈ {r['est_hours']:5.1f} 小时", flush=True)
keras.mixed_precision.set_global_policy("float32")

json.dump(stats, open(f"{OUT}/kaggle_timing.json", "w"), ensure_ascii=False, indent=1)
print(f"\n已保存 {OUT}/kaggle_timing.json；每周配额 30 小时", flush=True)
if args.steps:
    print("注意：这次只跑了前", args.steps, "步，时间不能用来估算", flush=True)
