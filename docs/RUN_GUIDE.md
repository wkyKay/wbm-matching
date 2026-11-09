# 五方法训练与统一 test 运行指南

本指南按 Wu、Yu、WBBS、Nakazawa、WMHD 分别给出正式运行命令、训练恢复方式和产物位置。
所有检索成绩均通过 `scripts/evaluate.py` 计算第一类指标：**查询与候选的完整缺陷标签组合相同才是正样本**。

**Wu、WBBS、WMHD 不需要训练；Yu、Nakazawa 需要先完成 CNN 训练。**
统一评价会自行提取指定集合的特征或编码，无需先执行旧方法入口的 `prepare`、`encode`、`valid` 或 `test`。

| 方法 | 是否训练 | 正式训练内容 | 统一评价读取的模型 |
| --- | --- | --- | --- |
| Wu | 否 | 手工特征与 NCC 匹配 | 无 checkpoint |
| Yu | 是 | 检测器 10 epoch，随后分类器 80 epoch | 主结果读取 `classifier/best.pt` |
| WBBS | 否 | 张量投票特征与 WBBS 匹配 | 无 checkpoint |
| Nakazawa | 是 | 分类 CNN 10 epoch | `checkpoint_last.pt`，固定最后一轮 |
| WMHD | 否 | 去噪、mountain 特征与方向性匹配 | 无 checkpoint |

## 1 运行目录与环境

以下命令统一从 `compared_methods/` 执行。本机进入方式如下；服务器上将 `cd` 后的路径换成实际仓库位置。

```bash
cd /Users/kayw/Documents/trae_projects/match-test/compared_methods
```

使用已经安装项目依赖的 Python 环境。本机已有 `.venv` 时执行：

```bash
source .venv/bin/activate
```

若使用新环境，在仓库根目录创建环境并安装依赖：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

GPU 服务器应使用与驱动匹配的 CUDA PyTorch 环境，再补齐 `requirements.txt` 中的依赖；
已有可用环境可直接使用。不要将本机 macOS 的 `.venv` 复制到 Linux。运行前检查当前解释器与 CUDA：

```bash
python -c "import sys, torch; print(sys.executable); print(torch.__version__, torch.version.cuda); print('CUDA available:', torch.cuda.is_available())"
```

已有 Conda 环境时先执行 `conda activate wbm`。非交互 SSH 会话通常需要先
`source ~/miniconda3/etc/profile.d/conda.sh`（按实际 Conda 安装位置调整）。
若遇到 `MKL_THREADING_LAYER=INTEL is incompatible with libgomp.so.1`，在启动 Python 前
设置 `export MKL_THREADING_LAYER=GNU`；这是当前会话的线程库兼容设置，无需重装 PyTorch。

Yu、Nakazawa 的正式命令以下均显式使用 `--device cuda`。没有 CUDA 时可改为 `--device cpu`，
训练和推断会在 CPU 执行。三个无训练方法使用 CPU，`--workers 4` 仅并行准备单图特征，
不表示逐查询的候选匹配也会自动并行。
完整评价可另加 `--rank-workers N`，将不同 query 分给 N 个进程；每个 query 仍匹配完整
冻结候选池，按原 query 顺序流式保存，最多只预提交 `2*N` 个 query。
例如空闲多核服务器可给 WBBS 设置 `--workers 4 --rank-workers 16`，WMHD 设置
`--workers 4 --rank-workers 48`；按服务器资源调整，不要在普通笔记本照搬进程数。
每个进程各持有一份特征，应同时检查内存，并用 `OMP_NUM_THREADS=1`、
`OPENBLAS_NUM_THREADS=1`、`MKL_NUM_THREADS=1` 避免 CPU 线程过量。

默认数据位置应保持为：

```text
工作区/
├── data/wm38k/Wafer_Map_Datasets.npz
└── compared_methods/
    ├── methods/
    ├── scripts/
    └── protocol/
        ├── deduplicated/split.csv
        └── retrieval_v1/
```

正常运行直接读取现有冻结协议，无需执行协议生成脚本。
`--config` 相对命令当前目录解析；配置中的相对路径以及 `--output-dir`、`--source-run` 相对
`compared_methods/` 解析，也支持绝对路径。

## 2 执行顺序与输出目录

有训练方法按 **train → 统一 valid → 固定设置 → 统一 test** 执行；无训练方法从统一 valid 开始。
训练使用 train，模型选择或参数确定只使用 valid；test 用于设置确定后的最终评价。

本指南采用以下独立目录，不修改各方法配置里的默认 `output_dir`：

| 方法 | 训练目录 | 统一 valid 目录 | 统一 test 目录 |
| --- | --- | --- | --- |
| Wu | 无 | `artifacts/evaluation_exact_v1/wu_valid` | `artifacts/evaluation_exact_v1/wu_test` |
| Yu | `artifacts/training_exact_v1/yu` | `artifacts/evaluation_exact_v1/yu_valid` | `artifacts/evaluation_exact_v1/yu_test` |
| WBBS | 无 | `artifacts/evaluation_exact_v1/wbbs_valid` | `artifacts/evaluation_exact_v1/wbbs_test` |
| Nakazawa | `artifacts/training_exact_v1/nakazawa` | `artifacts/evaluation_exact_v1/nakazawa_valid` | `artifacts/evaluation_exact_v1/nakazawa_test` |
| WMHD | 无 | `artifacts/evaluation_exact_v1/wmhd_valid` | `artifacts/evaluation_exact_v1/wmhd_test` |

首次训练使用新训练目录；已有训练按下文恢复，或直接复用已完成且签名匹配的来源运行。
**每次统一评价的输出目录必须尚不存在，已有空目录也会被拒绝；不要预先创建这些叶子目录。**
重复评价时换一个新目录，例如将 `wu_test` 改为 `wu_test_r2`。

若已有正式训练产物，将下文 `--source-run` 改为对应运行目录即可，不必重新训练。
评价必须使用与训练相同的科学配置、源码和数据协议；来源目录应保留完整运行文件，不只保留一个权重文件。

## 3 Wu

**不需要训练。** 统一入口提取 116 维手工特征，对完整候选池计算欧氏距离，按配置粗筛后用 NCC 重排。
正式配置的粗筛数量为 500，评价分母仍取完整固定候选库。

先运行正式 valid：

```bash
python -u scripts/evaluate.py \
  --config methods/wu/config.json \
  --split valid \
  --output-dir artifacts/evaluation_exact_v1/wu_valid \
  --workers 4
```

在 valid 上确定参数后，运行统一 test：

```bash
python -u scripts/evaluate.py \
  --config methods/wu/config.json \
  --split test \
  --output-dir artifacts/evaluation_exact_v1/wu_test \
  --workers 4
```

主要产物如下，通用文件含义见第 8 节。

| 产物 | 位置与内容 |
| --- | --- |
| valid 指标 | `artifacts/evaluation_exact_v1/wu_valid/valid/metrics.json` |
| test 指标 | `artifacts/evaluation_exact_v1/wu_test/test/metrics.json` |
| test 特征 | `wu_test/test/prepared/features.npz`，相对于 `artifacts/evaluation_exact_v1/`；包含 `sample_ids`、116 维 `features`、NCC 向量 `full` 与 `regions` |
| test 排名 | `wu_test/test/rankings.csv`；最多保存每 query 前 500 项，`coarse_distance` 记录粗筛距离 |
| 完成标记 | `artifacts/evaluation_exact_v1/wu_test/completed.json` |

Wu 没有训练权重。其输出不是整个 gallery 的完整 NCC 精排。方法细节见 [Wu README](../methods/wu/README.md)。

## 4 Yu

**需要训练。** 当前训练入口先运行检测器 10 epoch，再运行分类器 80 epoch；
正式 batch size 分别为 64 和 128。两个网络各自按最低 valid CE 保存 `best.pt`。

### 4.1 训练

```bash
python -u methods/yu/yu.py \
  --config methods/yu/config.json \
  --stage train \
  --device cuda \
  --output-dir artifacts/training_exact_v1/yu
```

中断后，使用同一配置、训练目录及原环境恢复。以下命令仅用于恢复，不是首次训练后的额外步骤：

```bash
python -u methods/yu/yu.py \
  --config methods/yu/config.json \
  --stage train \
  --device cuda \
  --output-dir artifacts/training_exact_v1/yu \
  --resume
```

恢复从各网络 `last.pt` 已保存的 epoch 和批次位置继续；检查点之后尚未保存的更新需要重做。
恢复要求 PyTorch 版本匹配。正式运行不添加 `--max-steps`，以免只完成部分训练。

训练目录 `artifacts/training_exact_v1/yu/` 中的关键文件：

| 文件 | 含义 |
| --- | --- |
| `classifier/best.pt` | 最低 valid CE 的分类器，统一主评价使用它 |
| `classifier/last.pt` | 最新分类器、优化器与恢复状态；用于恢复和校验训练是否完成 |
| `detector/best.pt`、`detector/last.pt` | 检测器的选定权重和恢复状态 |
| `classifier/history.json`、`detector/history.json` | 各网络逐次验证及训练进度记录 |
| `classifier/summary.json`、`detector/summary.json` | 完成轮数、步数、`training_complete`、选模规则等 |
| `run_manifest.json`、`resolved_config.json`、`environment.json` | 训练代码、配置、数据和环境身份 |
| `protocol/`、`yu_protocol.json` | 样本 ID、类别映射、协议与模型签名 |
| `preprocessing/per_sample.csv`、`preprocessing/summary.json` | 固定滤波处理对 train/valid 的影响记录 |

主评价 `global` 要求分类器完成全部正式训练；仅存在 `best.pt` 不代表训练已完成。
最佳 checkpoint 可以来自较早的 epoch，完成状态由最后训练状态核验。

### 4.2 统一 valid 和 test

先使用训练目录中的模型运行统一 valid：

```bash
python -u scripts/evaluate.py \
  --config methods/yu/config.json \
  --split valid \
  --variant global \
  --source-run artifacts/training_exact_v1/yu \
  --output-dir artifacts/evaluation_exact_v1/yu_valid \
  --device cuda
```

固定设置后运行统一 test，来源仍是同一个训练目录：

```bash
python -u scripts/evaluate.py \
  --config methods/yu/config.json \
  --split test \
  --variant global \
  --source-run artifacts/training_exact_v1/yu \
  --output-dir artifacts/evaluation_exact_v1/yu_test \
  --device cuda
```

`global` 是五方法主表中的 Yu 结果，使用分类器编码，不使用检测或预测类别门控。
每次统一评价都只在 train 上重新拟合 PCA256 和缩放，将编码器写入本次新评价目录；
valid/test 不参与拟合，也不向原训练目录写入编码器。

| 产物 | 位置与内容 |
| --- | --- |
| valid 指标 | `artifacts/evaluation_exact_v1/yu_valid/valid/metrics.json` |
| test 指标 | `artifacts/evaluation_exact_v1/yu_test/test/metrics.json` |
| test 编码 | `yu_test/test/prepared/features.npz`，相对于 `artifacts/evaluation_exact_v1/`；包含 `sample_ids`、256 维十值 `codes` 和 `predicted_classes` |
| PCA 编码器 | `yu_test/test/prepared/encoder.npz`；含 PCA 分量、均值、train 缩放上下界、`fit_ids` 与签名 |
| 完成标记 | `artifacts/evaluation_exact_v1/yu_test/completed.json` |

统一入口把变体名称记录在元数据中，不再在 `test/` 内创建旧格式的 `global/` 子目录。
训练命令本身不会产生上述检索指标或 `encoder.npz`。

### 4.3 可选的预测类别门控结果

`predicted_class` 是单独的辅助比较，需要分类器和检测器均完成训练；不要用它替换主表中的 `global` 后择优。
如果需要该结果，先用 `--split valid` 在独立目录验证该分支，再运行：

```bash
python -u scripts/evaluate.py \
  --config methods/yu/config.json \
  --split test \
  --variant predicted_class \
  --source-run artifacts/training_exact_v1/yu \
  --output-dir artifacts/evaluation_exact_v1/yu_predicted_class_test \
  --device cuda
```

辅助指标位于 `artifacts/evaluation_exact_v1/yu_predicted_class_test/test/metrics.json`。
其 `prepared/features.npz` 额外包含 `predicted_defect`；门控后的数量记录在 `query_diagnostics.csv`。
更多训练与预处理约定见 [Yu README](../methods/yu/README.md)。

## 5 WBBS

**不需要训练。** 统一入口完成张量投票、去噪和点描述提取，再计算查询与完整候选池的 WBBS 相似度。

正式 valid：

```bash
python -u scripts/evaluate.py \
  --config methods/wbbs/config.json \
  --split valid \
  --output-dir artifacts/evaluation_exact_v1/wbbs_valid \
  --workers 4
```

参数确定后的统一 test：

```bash
python -u scripts/evaluate.py \
  --config methods/wbbs/config.json \
  --split test \
  --output-dir artifacts/evaluation_exact_v1/wbbs_test \
  --workers 4
```

| 产物 | 位置与内容 |
| --- | --- |
| valid 指标 | `artifacts/evaluation_exact_v1/wbbs_valid/valid/metrics.json` |
| test 指标 | `artifacts/evaluation_exact_v1/wbbs_test/test/metrics.json` |
| test 特征 | `wbbs_test/test/prepared/features.npz`，相对于 `artifacts/evaluation_exact_v1/`；包含 `sample_ids`、`offsets`、`points`、三维 `descriptors` 和 `original_counts` |
| 特征记录 | `wbbs_test/test/prepared/prepared_metadata.json`；含参数、特征哈希、空点集数量和耗时 |
| 完成标记 | `artifacts/evaluation_exact_v1/wbbs_test/completed.json` |

WBBS 没有 checkpoint。每对图需要点集匹配，正式 test 耗时取决于点数，`--workers` 只加速特征准备。
用 `--rank-workers` 可并行不同 query 的全候选匹配。
论文参数与实现约定见 [WBBS README](../methods/wbbs/README.md)。

## 6 Nakazawa CNN

**需要训练。** 正式配置训练 10 epoch，有效 batch size 为 385，microbatch 为 8。
编码固定读取最后一轮权重；valid 分类指标用于监测，不用于挑选最低 loss checkpoint。

### 6.1 训练

```bash
python -u methods/nakazawa/nakazawa.py \
  --config methods/nakazawa/config.json \
  --stage train \
  --device cuda \
  --output-dir artifacts/training_exact_v1/nakazawa
```

中断后使用以下恢复命令，保持配置和设备类型一致：

```bash
python -u methods/nakazawa/nakazawa.py \
  --config methods/nakazawa/config.json \
  --stage train \
  --device cuda \
  --output-dir artifacts/training_exact_v1/nakazawa \
  --resume
```

恢复从最后一个已完成 epoch 继续，未完成 epoch 会重新训练。
已完成的同一训练运行会校验并复用。训练目录 `artifacts/training_exact_v1/nakazawa/` 的关键文件：

| 文件 | 含义 |
| --- | --- |
| `checkpoint_last.pt` | 最后完成 epoch 的模型、优化器、随机状态及签名；统一评价使用此权重 |
| `training_history.csv` | 每个 epoch 的训练与 valid 分类 loss、accuracy、样本数等 |
| `training_complete.json` | 训练完成记录、epoch 数、模型 SHA-256 和最后一轮统计 |
| `run_manifest.json`、`resolved_config.json`、`environment.json` | 代码、配置、数据和环境身份 |
| `protocol/`、`audit.json`、`prepared.json` | 原始 ID、固定池和训练准备记录 |

必须完成配置要求的全部 epoch，且完成记录与 checkpoint 哈希一致，才可进行正式评价。

### 6.2 统一 valid 和 test

```bash
python -u scripts/evaluate.py \
  --config methods/nakazawa/config.json \
  --split valid \
  --source-run artifacts/training_exact_v1/nakazawa \
  --output-dir artifacts/evaluation_exact_v1/nakazawa_valid \
  --device cuda
```

固定设置后运行统一 test：

```bash
python -u scripts/evaluate.py \
  --config methods/nakazawa/config.json \
  --split test \
  --source-run artifacts/training_exact_v1/nakazawa \
  --output-dir artifacts/evaluation_exact_v1/nakazawa_test \
  --device cuda
```

统一入口直接完成 sigmoid 特征提取和 256 位编码，不需要额外执行原入口的 `--stage encode`。

| 产物 | 位置与内容 |
| --- | --- |
| valid 指标 | `artifacts/evaluation_exact_v1/nakazawa_valid/valid/metrics.json` |
| test 指标 | `artifacts/evaluation_exact_v1/nakazawa_test/test/metrics.json` |
| test 特征与编码 | `nakazawa_test/test/prepared/features.npz`，相对于 `artifacts/evaluation_exact_v1/`；包含 `sample_ids`、256 维 `sigmoid_features` 和 256 位 `codes` |
| 模型与编码记录 | `nakazawa_test/test/prepared/prepared_metadata.json`；记录最后一轮模型哈希和编码信息 |
| 完成标记 | `artifacts/evaluation_exact_v1/nakazawa_test/completed.json` |

本流程不会生成原编码入口的 `valid_codes/*.npy`；统一编码使用上述 NPZ。
网络与输入适配见 [Nakazawa README](../methods/nakazawa/README.md)。

## 7 WMHD

**不需要训练。** 统一入口完成 ESRN 去噪和 mountain 特征提取，然后按 query → candidate 的方向计算距离。

正式 valid：

```bash
python -u scripts/evaluate.py \
  --config methods/wmhd/config.json \
  --split valid \
  --output-dir artifacts/evaluation_exact_v1/wmhd_valid \
  --workers 4
```

参数确定后的统一 test：

```bash
python -u scripts/evaluate.py \
  --config methods/wmhd/config.json \
  --split test \
  --output-dir artifacts/evaluation_exact_v1/wmhd_test \
  --workers 4
```

| 产物 | 位置与内容 |
| --- | --- |
| valid 指标 | `artifacts/evaluation_exact_v1/wmhd_valid/valid/metrics.json` |
| test 指标 | `artifacts/evaluation_exact_v1/wmhd_test/test/metrics.json` |
| test 特征 | `wmhd_test/test/prepared/features.npz`，相对于 `artifacts/evaluation_exact_v1/`；包含 `sample_ids`、`offsets`、`points`、九邻域 `neighborhoods` 和 `scalars` |
| 特征记录 | `wmhd_test/test/prepared/prepared_metadata.json`；含参数、空图数量和距离方向等 |
| 完成标记 | `artifacts/evaluation_exact_v1/wmhd_test/completed.json` |

`scalars` 的四列依次是坐标归一化所用的原图半径（像素单位）、beta、原始缺陷点数、真实 die 数。WMHD 没有 checkpoint。
完整 test 的点集匹配可能耗时较长；当前固定方向及参数的复现边界见 [WMHD README](../methods/wmhd/README.md)。
`--rank-workers` 可并行不同 query；不对候选抽样，也不改变 WMHD 的方向。

## 8 统一评价产物怎么读

以 `artifacts/evaluation_exact_v1/wu_test/` 为例，其他方法结构相同；valid 运行将内层 `test/` 换为 `valid/`。

```text
wu_test/
├── evaluation_manifest.json
├── resolved_config.json
├── environment.json
├── protocol/
├── raw_cache/
├── completed.json
└── test/
    ├── metrics.json
    ├── per_query.csv
    ├── per_class.csv
    ├── rankings.csv
    ├── query_diagnostics.csv
    ├── timing.json
    ├── method.json
    └── prepared/
        ├── features.npz
        └── prepared_metadata.json
```

| 文件 | 用途 |
| --- | --- |
| 根目录 `completed.json` | 成功完成标记，包含评价签名及产物哈希；先确认它存在，再读取成绩 |
| `test/metrics.json` | 总体 query 平均、完整组合类别宏平均、单／混合缺陷及标签数量分组指标 |
| `test/per_query.csv` | 每个 query 的完整候选数、相关样本数、返回数及 Hit/P/Recall/AP@K |
| `test/per_class.csv` | 每个完整组合类别的平均指标；逐类均值中的 AP 名为 mAP@K |
| `test/rankings.csv` | `query_id`、排名、`candidate_id`、分数及 Wu 的粗筛距离；仅保存前 `max(K)` 个返回结果 |
| `test/query_diagnostics.csv` | 每个 query 的输入候选数、最终评分数、粗筛距离计算数、可返回数量和排序耗时；各计数定义见下文 |
| `test/timing.json` | 准备与排序耗时；排序计时不含指标计算和 CSV 写出 |
| `test/method.json`、`test/prepared/prepared_metadata.json` | 方法参数、模型／特征来源、诊断及特征文件哈希 |
| `evaluation_manifest.json` | 本次方法、变体、K、并列规则、数据／代码／训练来源签名 |
| `protocol/` | 实际原始样本 ID、类别映射、去重和 query/gallery 审计 |
| `raw_cache/` | 本次独立原始数组缓存；不是方法特征，不应作为模型成绩解读 |

CNN 还会在 `test/` 下保存 `inference_environment.json`，记录推断种子、线程与 cuDNN 设置。
Yu 另有 `test/prepared/encoder.npz`。两个 CNN 的训练入口默认共用
`artifacts/training_exact_v1/_data/` 下按原始数据哈希区分的数组缓存；统一评价另建各自的 `raw_cache/`。

开启 query 并行后，`timing.json` 的 `ranking_total_seconds` 是各 query 工作进程耗时之和，
不是实际等待时间；`ranking_wall_seconds` 是排名阶段实际耗时，包含进程启动、指标计算、
CSV 写入与输出检查。执行进程数和队列上限记录在 `evaluation_manifest.json.execution`。

`query_diagnostics.csv` 中，`coarse_scored_count` 是 Wu 全库欧氏距离计算数，其他方法为 0；
`scored_count` 是最终相似度计算数，Wu 仅计粗筛入选后的 NCC 数量；`eligible_count` 是粗筛或门控后可返回数，
`returned` 是实际截断保存的条数。它们都不会改变 AP/Recall 的完整候选库分母。

所有特征文件的行身份由 `sample_ids` 指明，不能用原始 ID 直接当成 NPZ 内的行号。
WBBS、WMHD 的每图点数不同，通过 `offsets` 确定该图在拼接点数组中的范围。
排名中的分数统一为越大越相似：Yu/Nakazawa 保存负 Hamming 距离，WMHD 保存负距离，Wu/WBBS 保存相似度。
不同方法的原始分数值不作跨方法大小比较，比较上述 ground-truth 指标。

## 9 评价口径与重跑规则

五方法默认使用相同 K：**1、5、10、20、50、100、500**，输出 Hit@K、Precision@K、Recall@K、AP@K/mAP@K。
Precision 分母为 K；Recall 与截断 AP 的分母为完整固定候选池中的正样本数，不因粗筛或门控而缩小。
当前输出是截断 mAP@K，排名文件也只保存前 `max(K)`，不能称为完整全库 mAP 或完整排名。
指标公式、零相关 query 处理与统一哈希并列规则见 [EXACT_EVALUATION.md](EXACT_EVALUATION.md)。

| 集合 | query 数 | gallery 数 | 每 query 候选数 |
| --- | ---: | ---: | ---: |
| valid | 500 | 3,599 | 3,598 |
| test | 7,199 | 7,199 | 7,198 |

参数或设置改变时，先在 valid 上重新确定，再固定下来执行 test。需要调整 batch、microbatch 等科学配置时，
复制配置并使用新的训练目录，训练和后续评价均传入同一份配置。
评价时的 `--source-run` 只指定来源目录，不会放宽源码、配置、协议或 checkpoint 校验。
统一结果使用新的哈希并列规则，不能把旧 Wu/Yu 方法入口生成的成绩直接并入本次比较。

训练可按各方法的 `--resume` 恢复；**统一评价目前没有 `--resume`**。
评价失败后保留中间产物，可能生成 `failed.json`；即使存在局部 `metrics.json`，没有 `completed.json` 也不算完成。
重新评价须换新输出目录，不能删除原产物或修改 manifest 来绕过校验。

正式命令使用 `config.json`。`smoke.json` 仅用于小 valid 子集的流程检查，不能运行 test，也不能作为正式成绩。
当前实现验证范围见 [VALIDATION.md](VALIDATION.md)；运行本指南中的命令才会产生相应的新实验结果。

## 10 GPU 服务器上的五方法 smoke

以下命令在仓库根目录运行。先确认指定 GPU 空闲，再按实际编号调整 `CUDA_VISIBLE_DEVICES`。
仅 Yu、Nakazawa 进行 smoke 配置限定的少量训练；五个方法统一评价小 valid 池，不运行 test。
所有输出进入本次新目录，重复执行时自动生成新的目录名。

```bash
conda activate wbm
export CUDA_VISIBLE_DEVICES=0
export MKL_THREADING_LAYER=GNU
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
set -e
mkdir -p artifacts
smoke_run=$(mktemp -d "$PWD/artifacts/server_smoke_XXXXXXXX")
python -m unittest discover -s tests -v > "$smoke_run/tests.log" 2>&1
for method in yu nakazawa; do
  python -u "methods/$method/$method.py" \
    --config "methods/$method/smoke.json" --stage train --device cuda \
    --output-dir "$smoke_run/training/$method" \
    > "$smoke_run/${method}_train.log" 2>&1
done
for method in wu wbbs wmhd; do
  python -u scripts/evaluate.py \
    --config "methods/$method/smoke.json" --split valid --workers 2 \
    --output-dir "$smoke_run/evaluation/$method" \
    > "$smoke_run/${method}_evaluate.log" 2>&1
done
for method in yu nakazawa; do
  python -u scripts/evaluate.py \
    --config "methods/$method/smoke.json" --split valid --device cuda \
    --source-run "$smoke_run/training/$method" \
    --output-dir "$smoke_run/evaluation/$method" \
    > "$smoke_run/${method}_evaluate.log" 2>&1
done
printf 'Smoke outputs: %s\n' "$smoke_run"
```

成功时五个 `evaluation/<method>/completed.json` 都应存在，指标位于各自的
`evaluation/<method>/valid/metrics.json`。Yu 两个网络各更新 8 次；Nakazawa 仅训练
1 个 batch 并监测 1 个 valid batch。即使输出训练完成标记，也仅指 smoke 配置完成。
代码或配置签名变化后重新运行到新目录，不复用旧训练产物来绕过签名检查。
