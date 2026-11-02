# 读取项目根目录的 config.yaml，所有路径都相对项目根目录
# 包以可编辑方式安装（pip install -e .），本文件在 src/gold_dl/ 下，往上两层就是项目根目录
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
CFG = yaml.safe_load(open(ROOT / "config.yaml", encoding="utf-8"))


def path(key):
    """config.yaml 里 paths 下某一项的完整路径，例如 path("model_arrays")"""
    return ROOT / CFG["paths"][key]
