# 基于分钟级市场数据的黄金期货短期交易结果预测

English version: [README.md](README.md)

IE7615 Deep Learning，2026 年秋季。Kaiyuan Lai（lai.kaiy@northeastern.edu）。

## 问题与目标用户

**研究问题。** 仅使用分钟级 OHLCV 和 MBP-1 数据及由其计算的技术指标，深度学习能否足够准确地估计 XAU/USD 短期趋势延续的概率和持续时间，从而在扣除交易成本后实现每笔交易为正的期望收益？

**目标产品与用户。** 目标产品是一个交易信号生成工具：它通过行情 API 实时获取 1 分钟 K 线，在每根 K 线收盘时输出各种交易结局的概率，以及建议的操作（做多、做空或不交易）。它是面向交易黄金相关产品的散户和机构交易者的研究原型，不用于执行实盘交易。本项目聚焦于单一的、深度和流动性都很高的市场，而不是多个资产，因为专注于一个市场的模型更容易在一个学期内得到严格验证。黄金通过现货和衍生品在工作日几乎连续交易，单一参与者难以操纵其价格。

**模型具体预测什么。** 模型的输入是 COMEX 黄金期货（GC）最近 L = 60 分钟的市场数据，每分钟 33 个特征。模型有六个输出头（多 / 空 × 15、30、60 分钟三种持有期）。
对每个输出头，模型预测：在下一分钟开盘价入场的交易，在持有期结束前会先触及止盈障碍、先触及止损障碍，
还是两者都未触及（三重障碍标签，按真实的买卖报价判定）。

## 数据

| | |
|---|---|
| 来源 | Databento，CME Globex，数据集 `GLBX.MDP3`，连续近月合约 `GC.v.0` |
| 数据模式 | MBP-1（盘口最优档和成交；所有步骤都用它）和 OHLCV-1m（只用来核对由 MBP-1 重建的 K 线） |
| 时间范围 | 2025-09-29 至 2026-09-29 |
| 下载日期 | 2026-10-24（MBP-1 批量任务 `GLBX-20261025-PT9C6P3YKJ`） |
| 大小 | MBP-1：13 个月度文件，压缩后 10.1 GB（解压后 42.7 GB）；OHLCV-1m：6.2 MB |
| 费用 | 在 Databento 按量计费账户上约 US$71.60（MBP-1）和 US$1.28（OHLCV-1m） |
| 许可 | CME 市场数据，受 CME Information License Agreement 约束：仅限内部个人使用，**不得再分发** |

由于许可禁止再分发，**本仓库不包含任何数据**。训练用的处理后数组保存在一个*私有*的 Kaggle 数据集中。
要复现结果，你需要自己的 Databento 账户：

1. 在 <https://databento.com> 注册账户并创建 API key。
2. 把 key 保存到项目根目录下的 `databento_key.txt`（该文件已列入 `.gitignore`；切勿提交）。
3. 运行下面第 1 步中的两个下载脚本。

## 安装

本项目使用两个 Python 环境。原因是数据处理步骤需要 `pyarrow` 和 `databento`，
而训练步骤使用带 Apple Metal 插件的 TensorFlow：

| 环境 | Python | 用途 | 主要包 |
|---|---|---|---|
| processing | 3.13.9 | 第 1–4 步及所有图表 | databento 0.88.0, pandas 2.2.3, numpy 2.1.3, pyarrow 19.0.0, scipy 1.15.3, matplotlib 3.10.0, zstandard 0.24.0 |
| training（本地） | 3.10.20 | 第 5–6 步 | tensorflow 2.16.2, tensorflow-metal 1.2.0, keras 3.12.1, numpy 1.26.4 |
| training（Kaggle） | Kaggle 默认镜像 | 正式训练和计时 | TensorFlow 2.20.0, Tesla T4 |

```bash
pip install -r requirements-processing.txt
```

```bash
pip install -r requirements-train.txt
```

然后在训练环境中，以可编辑模式安装项目自带的包 `gold_dl`（位于 `src/`），
这样脚本和 notebook 就可以 `import gold_dl`：

```bash
pip install -e .
```

`gold_dl.config` 读取 `config.yaml`，并把所有路径解析为相对项目根目录的路径；
`gold_dl.data_pipeline` 包含 tf.data 数据管道（`make_dataset`）以及所有数据划分共用的掩码损失。
在 Kaggle 上，`kaggle_timing.py` 自带一份数据管道的副本，因此只需上传一个文件。

## 平台

- **开发和冒烟测试：** Apple M4 Pro，24 GB，TensorFlow 使用 Metal GPU。
- **训练（M4 起）：** Kaggle Notebooks，GPU T4 × 2（只使用一块 GPU），每周 30 GPU 小时。
  把 `model_data/model_arrays.npz`（55 MB）上传为**私有** Kaggle 数据集，并添加到 notebook 中；
  脚本会自动在 `/kaggle/input` 下找到它。训练检查点写入 `/kaggle/working`，
  因为会话结束时会话磁盘会被清空。

## 复现结果

在项目根目录下按以下顺序运行。每个脚本都根据自身位置找到项目根目录，因此克隆的仓库可以放在任何文件夹。
凡是随机性会影响结果的地方，随机种子都固定为 42。

**1. 下载（processing 环境）**

```bash
python scripts/databento_gc_mbp1.py
```

```bash
python scripts/databento_gc_ohlcv1m.py
```

**2. 由 MBP-1 重建 1 分钟数据**

```bash
python scripts/process_gc_mbp1.py
```

```bash
python scripts/fix_gc_delta.py
```

```bash
python scripts/build_gc_quote_1min.py
```

```bash
python scripts/build_gc_book_1min.py
```

```bash
python scripts/build_gc_depth_1min.py
```

```bash
python scripts/build_gc_ofi_1min.py
```

```bash
python scripts/build_gc_spread_tw_1min.py
```

```bash
python scripts/build_gc_trades_1min.py
```

**3. 特征、标签和数据划分**

```bash
python scripts/build_features.py
```

```bash
python scripts/make_labels.py
```

```bash
python scripts/save_default_labels.py
```

```bash
python scripts/make_labels_with_preroll_check.py
```

`make_labels.py` 为全部 12 种障碍组合生成标签，并按时间划分数据
（训练集 / 验证集 / 测试集，相邻两段之间留 60 分钟间隔）；`save_default_labels.py` 选出默认组合
（k_SL = 2，1:1）。`make_labels_with_preroll_check.py` 在不剔除每次换月前 3 天的情况下重新生成标签，
仅用于图 6(c) 中的前后对比。

**4. 长尾处理、标准化和导出（统计量只取自训练集）**

```bash
python scripts/transform_tail_features.py
```

```bash
python scripts/standardize_features.py
```

```bash
python scripts/export_model_arrays.py
```

**5. 输入管道检查和冒烟测试（training 环境）**

```bash
python scripts/section6_pipeline_checks.py
```

同样的检查也放在 `notebooks/01_smoke_test.ipynb` 中，逐步执行，且不写出任何文件（kernel：
training 环境）。不打开 Jupyter、从头到尾运行它：

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/01_smoke_test.ipynb
```

**6. 在 Kaggle 上计时**（添加数据集后，在 Kaggle notebook 单元格中运行）

```bash
python kaggle_timing.py
```

```bash
python kaggle_timing.py --L 120
```

### 报告中各项结果的来源

| 报告内容 | 脚本 | 输出 |
|---|---|---|
| 数据质量计数 | `section2_data_quality.py` | `数据说明/第2节_数据质量统计.txt` |
| 标签审核（60 个样本，360 个标签） | `audit_labels_sample.py` | `数据说明/标签审核/` |
| 按小时和星期几统计的点差与波动率 | `plot_time_profiles.py` | `数据说明/` 中的图和 CSV 文件 |
| 数据划分表 | `split_table.py`, `check_split_order.py` | 打印输出 |
| EDA 图 1–8 | `plot_eda_figures.py` | `数据说明/EDA图/`, `eda_stats.txt` |
| 自相关（图 7） | `plot_return_acf.py` | `数据说明/EDA图/`, `acf_stats.txt` |
| 管道批次、冒烟测试、本地计时 | `section6_pipeline_checks.py`, `time_cnn_bilstm.py` | `数据说明/第6节/` |
| Kaggle 计时 | `kaggle_timing.py` | `kaggle_timing.json`；复制到 `数据说明/第6节/kaggle_timing_T4.csv` |
| 单变量特征检查、每月类别占比 | `section7_checks.py`, `check_monthly_sl_drift.py` | `数据说明/第7节_特征单变量检查.csv`，打印输出 |

### 数据划分测试

数据按时间划分：每一分钟只属于一个集合，各集合按时间先后排列，每个边界处留出 60 分钟间隔（最长持有期）。
该测试在标签表和 `model_arrays.npz` 中的样本上检查这些条件（processing 环境，第 4 步之后）：

```bash
pytest tests/test_split.py
```

## 仓库结构

```
project/
├── README.md
├── LICENSE                       MIT，仅适用于代码
├── config.yaml                   全部设置：路径、数据划分、标签、回看窗口、批大小、随机种子
├── requirements-processing.txt   第 1–4 步及图表所用的固定版本包
├── requirements-train.txt        第 5–6 步所用的固定版本包
├── pyproject.toml                使 src/gold_dl 可安装（pip install -e .）
├── src/gold_dl/        可复用的包：config.py（路径和设置）、data_pipeline.py（tf.data 管道、损失）
├── scripts/            数据处理、打标签、检查和绘图脚本
├── notebooks/          01_smoke_test.ipynb：批次形状、类型、标签取值、窗口检查、300 步冒烟测试
├── tests/              test_split.py：没有分钟同时属于两个集合、时间顺序、60 分钟间隔
├── 数据说明/            报告笔记、图和表（中英文图）
├── GC_mbp1/            原始 MBP-1 文件              （未提交）
├── GC_processed/       由 MBP-1 重建的 1 分钟数据   （未提交）
├── features/           特征表                       （未提交）
├── labels/             标签表                       （未提交）
└── model_data/         model_arrays.npz             （未提交）
```

<!-- TODO：目前只有 gold_dl 和 tests/ 读取 config.yaml；各脚本仍使用各自的常量。 -->

## 许可

代码：见 `LICENSE`。数据：经 Databento 获取的 CME 市场数据，不得再分发（见上文）。
