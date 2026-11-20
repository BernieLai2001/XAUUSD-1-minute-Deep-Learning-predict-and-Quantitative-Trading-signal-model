# 在 Kaggle（M5 要用的平台，T4 GPU、混合精度）上用现在的 M4 代码重跑两个简单深度基准：实测训练时间，并把检查点存进 Kaggle 输出
# Kaggle Notebook（GPU T4，打开 Internet，添加私有数据集：含 model_arrays.npz 和 trade_outcomes.npz）里依次运行：
#   !git clone --depth 1 https://github.com/BernieLai2001/XAUUSD-1-minute-Deep-Learning-predict-and-Quantitative-Trading-signal-model.git project
#   !pip install -q -e project --no-deps
#   !cd project && python scripts/7_models/kaggle_m4_deep.py
# 做的事：
#   1. 在 /kaggle/input 下找到 model_arrays.npz、trade_outcomes.npz，链接到 project/model_data/
#   2. 按顺序运行 run_baseline.py：conv1d、lstm（不加 dropout，作为对照）和 conv1d、lstm 加 dropout 0.2（选定的深度基准），各 3 个种子
#   3. 检查点（model.keras）留在 /kaggle/working/project/results/m4/<模型>/seed<种子>/，保存 Notebook 版本后成为 Kaggle 输出数据集
#   4. 把结果（不含检查点和 TensorBoard）打包成 /kaggle/working/m4_deep_results.zip，下载后解压到本地项目根目录即可
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for f in ["model_arrays.npz", "trade_outcomes.npz"]:
    hits = sorted(Path("/kaggle/input").rglob(f))
    if not hits:
        sys.exit(f"在 /kaggle/input 下找不到 {f}：请把它加进私有数据集")
    dst = ROOT / "model_data" / f
    dst.parent.mkdir(exist_ok=True)
    if not dst.exists():
        dst.symlink_to(hits[0])
    print("using", hits[0])

MODELS = ["conv1d_do02", "lstm_do02", "conv1d", "lstm"]   # 名字后缀 _do02 = 加 dropout 0.2
for m in MODELS:
    base, tag = m.split("_")[0], m[len(m.split("_")[0]):]
    cmd = [sys.executable, "scripts/7_models/run_baseline.py", "--model", base] + (["--dropout", "0.2", "--tag", tag] if tag else [])
    print(">>", " ".join(cmd), flush=True)
    with open(ROOT / f"kaggle_{m}.log", "w") as log:
        subprocess.run(cmd, cwd=ROOT, check=True, stdout=log, stderr=subprocess.STDOUT)
    print(open(ROOT / f"kaggle_{m}.log").read().splitlines()[-3:], flush=True)

out = Path("/kaggle/working/m4_deep_results.zip")
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for m in MODELS + [m.split("_")[0] for m in MODELS]:
        for p in (ROOT / "results" / "m4" / m).rglob("*"):
            if p.is_file() and "tensorboard" not in p.parts and "backup" not in p.parts and p.suffix != ".keras":
                z.write(p, p.relative_to(ROOT))
    z.write(ROOT / "results" / "m4" / "runs.jsonl", "results/m4/runs_kaggle.jsonl")
    for m in MODELS:
        z.write(ROOT / f"kaggle_{m}.log", f"results/m4/logs/kaggle_{m}.log")
print("结果已打包：", out, "；检查点在 /kaggle/working/project/results/m4/*/seed*/model.keras")
