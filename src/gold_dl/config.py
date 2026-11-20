# 读取项目根目录的 config.yaml，所有路径都相对项目根目录
# 脚本统一写 from gold_dl.config import path，不再用 Path(__file__).parent.parent 推算根目录，
# 所以脚本放在哪一层文件夹、从哪个目录运行都不影响找数据。
# 项目根目录 = 含 config.yaml 的目录，按以下顺序找：
#   ① 环境变量 GOLD_DL_ROOT（例如数据放在别的盘时）
#   ② 从本包所在位置往上找（pip install -e . 时就是仓库里的 src/gold_dl/）
#   ③ 从当前工作目录往上找
import os
from pathlib import Path

import yaml


def _find_root():
    env = os.environ.get("GOLD_DL_ROOT")
    if env:
        root = Path(env).expanduser().resolve()
        if not (root / "config.yaml").is_file():
            raise FileNotFoundError(f"GOLD_DL_ROOT={env} 下没有 config.yaml")
        return root
    for start in (Path(__file__).resolve().parent, Path.cwd().resolve()):
        for d in (start, *start.parents):
            if (d / "config.yaml").is_file():
                return d
    raise FileNotFoundError("找不到 config.yaml：请在项目目录内运行，或设置环境变量 GOLD_DL_ROOT")


ROOT = _find_root()
with open(ROOT / "config.yaml", encoding="utf-8") as fh:
    CFG = yaml.safe_load(fh)


def save_run_config(out_dir, step):
    """把这次运行用的 config.yaml 全部设置另存到输出目录（run_config_<step>.yaml），
    列名不带参数值（如 raw_atr、pre_roll），具体用了多少看这个文件"""
    out = Path(out_dir) / f"run_config_{step}.yaml"
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(f"# {step} 运行时的 config.yaml 设置（自动生成，不要手改）\n")
        yaml.safe_dump(CFG, fh, allow_unicode=True, sort_keys=False)
    return out


def path(key, *parts):
    """config.yaml 里 paths 下某一项的完整路径，可接子路径，例如 path("features", "features_1min.parquet")"""
    return ROOT.joinpath(CFG["paths"][key], *parts)
