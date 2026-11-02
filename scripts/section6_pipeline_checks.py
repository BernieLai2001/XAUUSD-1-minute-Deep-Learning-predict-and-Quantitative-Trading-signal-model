# 第 6 节的四项检查：① 三个数据集用同一个 make_dataset；② 从流水线末端取一个批次反标准化画图，标签从批次读；
# ③ 冒烟测试：Flatten + 一个 Dense 输出层，训练 300 步，报告形状、类型、标签取值和 loss 曲线；④ 完整遍历训练集计时、内存估算
# 运行环境：/opt/anaconda3/envs/tf（TensorFlow 2.16 + Metal GPU）
# 输出：数据说明/第6节/ 下的图和 section6_stats.json、batch_check.csv
import json, os, sys, time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import keras

sys.path.insert(0, os.path.dirname(__file__))
from data_pipeline import BATCH, L, SEED, load_arrays, make_dataset, masked_sparse_ce

from pathlib import Path
P = str(Path(__file__).resolve().parent.parent)
OUT = f"{P}/数据说明/第6节"
os.makedirs(OUT, exist_ok=True)
SMOKE_STEPS = 300
tf.random.set_seed(SEED)
keras.utils.set_random_seed(SEED)
plt.rcParams.update({"font.sans-serif": ["Hiragino Sans GB", "Arial Unicode MS"], "axes.unicode_minus": False,
                     "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "axes.grid": True,
                     "grid.color": "#e8e7e2", "axes.spines.top": False, "axes.spines.right": False})
arr = load_arrays()
names = list(arr["feature_names"])
heads = list(arr["head_names"])
stats = {"devices": [d.name for d in tf.config.list_physical_devices()], "batch_size": BATCH, "lookback": L}
ds = {s: make_dataset(arr, s) for s in ["train", "val", "test"]}

# ---------- ① 批次形状、类型、标签取值 ----------
for s, d in ds.items():
    x, y, t = next(iter(d))
    stats[f"{s}_batch"] = {"x_shape": list(x.shape), "x_dtype": x.dtype.name, "y_shape": list(y.shape),
                           "y_dtype": y.dtype.name, "y_values": sorted(np.unique(y.numpy()).tolist()),
                           "n_samples": int(len(arr[f"end_{s}"])), "steps_per_epoch": int(np.ceil(len(arr[f"end_{s}"]) / BATCH))}
    # 核对：窗口就是 X_all 中以 t 结尾的 60 行，标签与数组一致
    tt = t.numpy()
    assert np.array_equal(x.numpy(), np.stack([arr["X_all"][i - L + 1:i + 1] for i in tt]))
    pos = np.searchsorted(arr[f"end_{s}"], tt)
    assert np.array_equal(y.numpy(), arr[f"y_{s}"][pos])

# ---------- ② 从训练流水线末端取一个批次画图（反标准化，标签从批次读） ----------
x, y, t = next(iter(ds["train"]))
x, y, t = x.numpy(), y.numpy(), t.numpy()
raw = x * arr["std"] + arr["mean"]
raw[:, :, arr["log_feature"]] = np.exp(raw[:, :, arr["log_feature"]])  # 取过对数的特征换回原单位（缩尾不可逆）
times = np.array(arr["time_ns"]).astype("datetime64[ns]")
np.savetxt(f"{OUT}/batch_check.csv", np.column_stack([t, y]), fmt="%d", delimiter=",",
           header="row," + ",".join(heads), comments="")
SHOW = ["d_close", "rsi_14", "relvol_60", "spread_atr"]
T = {
    "zh": {"dir": "中文", "f8": "图8_流水线批次示例.png", "f9": "图9_冒烟测试_loss.png", "font": ["Hiragino Sans GB", "Arial Unicode MS"],
           "unit": {"d_close": "ATR14 倍数", "rsi_14": "0–100", "relvol_60": "倍", "spread_atr": "点差 / ATR14"},
           "cls": {0: "止盈", 1: "止损", 2: "超时", -1: "空"}, "side": {"long": "做多", "short": "做空"}, "sep": "　",
           "tz": "（北京时间）", "x8": "相对第 t 分钟（分钟）",
           "t8": f"训练流水线末端的一个批次（共 {BATCH} 个样本，展示前 4 个）：反标准化后的 4 个特征，标签从批次读出",
           "step": "每一步的训练 loss", "run": "累计平均训练 loss", "uni": "均匀猜测 ln 3 = 1.099",
           "prior": "按类别比例猜测 = {:.3f}", "x9": "训练步数", "y9": "屏蔽空标签后的交叉熵",
           "t9": "冒烟测试：Flatten + Dense（{:,} 个参数），批次 {}，{} 步"},
    "en": {"dir": "English", "f8": "Fig8_pipeline_batch_example.png", "f9": "Fig9_smoke_test_loss.png", "font": ["Arial", "Helvetica"],
           "unit": {"d_close": "ATR14 units", "rsi_14": "0–100", "relvol_60": "ratio", "spread_atr": "spread / ATR14"},
           "cls": {0: "TP", 1: "SL", 2: "Timeout", -1: "missing"}, "side": {"long": "Long", "short": "Short"}, "sep": "  ",
           "tz": " (Beijing time)", "x8": "Minutes relative to t",
           "t8": f"One batch from the end of the training pipeline ({BATCH} samples, first 4 shown): 4 features after inverse standardization; labels read from the batch",
           "step": "Training loss per step", "run": "Running mean of training loss", "uni": "Uniform guess ln 3 = 1.099",
           "prior": "Class-prior guess = {:.3f}", "x9": "Training step", "y9": "Masked cross-entropy",
           "t9": "Smoke test: Flatten + Dense ({:,} parameters), batch {}, {} steps"},
}
for lang, tx in T.items():
    os.makedirs(f"{OUT}/{tx['dir']}", exist_ok=True)
    plt.rcParams["font.sans-serif"] = tx["font"]
    fig, axes = plt.subplots(len(SHOW), 4, figsize=(15, 9), sharex=True)
    for j in range(4):
        lab = "\n".join(f"{tx['side'][side]} " + tx["sep"].join(
            f"{h.split('_')[1]} {tx['cls'][int(v)]}" for h, v in zip(heads, y[j]) if h.startswith(side)) for side in ["long", "short"])
        axes[0, j].set_title(f"{str(times[t[j]])[:16].replace('T', ' ')}{tx['tz']}\n{lab}", fontsize=8.5, loc="left")
        for i, fn in enumerate(SHOW):
            k = names.index(fn)
            axes[i, j].plot(np.arange(-L + 1, 1), raw[j, :, k], color="#2a78d6", lw=1.2)
            if j == 0:
                axes[i, j].set_ylabel(f"{fn}\n({tx['unit'][fn]})", fontsize=9)
    for a in axes[-1]:
        a.set_xlabel(tx["x8"])
    fig.suptitle(tx["t8"], x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(f"{OUT}/{tx['dir']}/{tx['f8']}", dpi=170)
    plt.close(fig)

# ---------- ③ 冒烟测试：Flatten + 一个 Dense 输出层 ----------
def xy(d):
    return d.map(lambda x, y, t: (x, y))


smoke = keras.Sequential([keras.Input((L, len(names))), keras.layers.Flatten(),
                          keras.layers.Dense(len(heads) * 3), keras.layers.Reshape((len(heads), 3))])
smoke.compile(optimizer=keras.optimizers.Adam(1e-3), loss=masked_sparse_ce)
train_y = arr["y_train"]
prior = []
for k in range(len(heads)):
    v = train_y[:, k][train_y[:, k] >= 0]
    p = np.bincount(v, minlength=3) / len(v)
    prior.append(-(p * np.log(p)).sum())
stats["loss_reference"] = {"uniform_ln3": float(np.log(3)), "class_prior_entropy_mean": float(np.mean(prior))}
hist = []
cb = keras.callbacks.LambdaCallback(on_train_batch_end=lambda b, logs: hist.append(logs["loss"]))
val_before = smoke.evaluate(xy(ds["val"]).take(50), verbose=0)
smoke.fit(xy(ds["train"]).repeat(), steps_per_epoch=SMOKE_STEPS, epochs=1, callbacks=[cb], verbose=0)
val_after = smoke.evaluate(xy(ds["val"]).take(50), verbose=0)
stats["smoke"] = {"steps": SMOKE_STEPS, "params": int(smoke.count_params()), "train_loss_first10": float(np.mean(hist[:10])),
                  "train_loss_last10_running": float(np.mean(hist[-10:])), "val_loss_before": float(val_before),
                  "val_loss_after": float(val_after)}
# Keras 记录的是本 epoch 内的累计平均 loss，换算回每一步的 loss 再画
h = np.array(hist)
step_loss = np.r_[h[0], h[1:] * np.arange(2, len(h) + 1) - h[:-1] * np.arange(1, len(h))]
for lang, tx in T.items():
    plt.rcParams["font.sans-serif"] = tx["font"]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(np.arange(1, len(h) + 1), step_loss, color="#c3c2b7", lw=0.8, label=tx["step"])
    ax.plot(np.arange(1, len(h) + 1), h, color="#2a78d6", lw=1.8, label=tx["run"])
    ax.axhline(np.log(3), color="#eb6834", ls="--", lw=1, label=tx["uni"])
    ax.axhline(np.mean(prior), color="#52514e", ls=":", lw=1, label=tx["prior"].format(np.mean(prior)))
    ax.set_xlabel(tx["x9"])
    ax.set_ylabel(tx["y9"])
    ax.set_title(tx["t9"].format(smoke.count_params(), BATCH, SMOKE_STEPS), loc="left", fontsize=10.5)
    ax.legend(frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(f"{OUT}/{tx['dir']}/{tx['f9']}", dpi=170)
    plt.close(fig)

# ---------- ④ 计时：完整遍历一次训练集 ----------
t0 = time.perf_counter()
for _ in ds["train"]:
    pass
stats["time_pipeline_only_s"] = time.perf_counter() - t0
for name, model in [
    ("smoke_dense", smoke),
    ("lstm64", keras.Sequential([keras.Input((L, len(names))), keras.layers.LSTM(64),
                                 keras.layers.Dense(len(heads) * 3), keras.layers.Reshape((len(heads), 3))])),
    ("conv1d", keras.Sequential([keras.Input((L, len(names))), keras.layers.Conv1D(64, 5, activation="relu"),
                                 keras.layers.Conv1D(64, 5, activation="relu"), keras.layers.GlobalAveragePooling1D(),
                                 keras.layers.Dense(len(heads) * 3), keras.layers.Reshape((len(heads), 3))])),
]:
    if name != "smoke_dense":
        model.compile(optimizer="adam", loss=masked_sparse_ce)
        model.fit(xy(ds["train"]).take(5), verbose=0)   # 预热：编译图
    t0 = time.perf_counter()
    model.fit(xy(ds["train"]), epochs=1, verbose=0)
    tr = time.perf_counter() - t0
    t0 = time.perf_counter()
    model.evaluate(xy(ds["val"]), verbose=0)
    va = time.perf_counter() - t0
    stats[f"epoch_time_{name}"] = {"train_s": tr, "val_s": va, "params": int(model.count_params())}
    print(name, f"train {tr:.1f}s val {va:.1f}s", flush=True)

n_tr = len(arr["end_train"])
stats["memory"] = {"X_all_MB": arr["X_all"].nbytes / 1e6,
                   "train_windows_materialized_GB": n_tr * L * len(names) * 4 / 1e9,
                   "all_windows_materialized_GB": sum(len(arr[f"end_{s}"]) for s in ["train", "val", "test"]) * L * len(names) * 4 / 1e9}
json.dump(stats, open(f"{OUT}/section6_stats.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps(stats, ensure_ascii=False, indent=1))
