# 晶圆检索方法在 MixedWM38 上的独立复现

本目录提供 Wu（2015）、Yu（2019）、WBBS（2023）、Nakazawa CNN（2018）和 WMHD（2020）的
独立实现。各方法共用冻结的数据划分和检索协议，训练与参数确定仅使用 train/valid。
五方法统一完整组合评价使用独立的 `scripts/evaluate.py`，显式选择 valid/test，输出 Hit@K、
Precision@K、Recall@K 和截断 mAP@K。运行命令、并列规则及来源模型要求见
[统一评价说明](docs/EXACT_EVALUATION.md)。原方法 CLI 和 Wu/Yu 的 `--stage test` 入口保留；
统一评价不训练模型、不改变训练签名，也不自动进入 test。
逐方法的训练、恢复、统一 valid/test 命令与产物位置见[五方法运行指南](docs/RUN_GUIDE.md)。
论文没有公开说明的细节已明确记录，不声称与作者未提供的源码逐行一致。

| 方法 | 主实现 | 运行与参数说明 |
| --- | --- | --- |
| Wu：116 维手工特征粗筛＋NCC 重排 | [methods/wu/wu.py](methods/wu/wu.py) | [Wu README](methods/wu/README.md) |
| Yu：检测/分类 CNN＋PCA256＋十值编码检索 | [methods/yu/yu.py](methods/yu/yu.py) | [Yu README](methods/yu/README.md) |
| WBBS：张量投票＋加权最佳伙伴相似度 | [methods/wbbs/wbbs.py](methods/wbbs/wbbs.py) | [WBBS README](methods/wbbs/README.md) |
| Nakazawa：CNN＋256 位二值编码＋Hamming | [methods/nakazawa/nakazawa.py](methods/nakazawa/nakazawa.py) | [Nakazawa README](methods/nakazawa/README.md) |
| WMHD：mountain 密度＋方向性加权 Hausdorff（W2-O1） | [methods/wmhd/wmhd.py](methods/wmhd/wmhd.py) | [WMHD README](methods/wmhd/README.md) |

详细论文依据、网络、复现约定及迁移差异见[单类缺陷对比方法](../对比方法说明/单类缺陷对比方法.md)。
三个新增方法对应[混合缺陷晶圆检索候选论文](../survey/混合缺陷晶圆检索候选论文.md)的 §1.1、§1.3、§1.5；
具体公式、原文已披露参数及实现约定以各方法 README 为准。Nakazawa 的架构和训练流程依据其自身论文，
不继承 Yu 的滤波、检测器、PCA、十值编码或最低 valid CE 选模流程。
实测环境和小规模验证见[验证记录](docs/VALIDATION.md)。

## 目录

```text
compared_methods/
├── methods/          # 每个方法的代码、配置与说明
│   ├── wu/           # wu.py、config.json、smoke.json、README.md
│   ├── yu/
│   ├── wbbs/
│   ├── nakazawa/
│   └── wmhd/
├── shared/           # 数据加载、去重、固定检索协议与评价
├── scripts/          # 协议生成与独立统一评价入口
├── protocol/         # 已冻结的划分与 query/gallery 清单
├── tests/            # 自动测试
├── docs/             # 验证记录
├── artifacts/        # 运行产物与缓存，不纳入版本控制
├── requirements.txt
└── README.md
```

新增方法放在 `methods/<方法名>/`，共用 `shared/` 和 `protocol/`。配置与方法代码放在一起，
配置中的数据、协议和输出路径仍以 `compared_methods/` 根目录为基准。
目录整理没有重新生成数据或固定协议。源码路径与导入变化会改变运行签名；已有产物保留，
需要运行整理后的代码时使用新的输出目录，不将旧缓存或 checkpoint 强行混入。

## Git 管理范围

`compared_methods/` 是独立 Git 仓库。版本控制保留方法源码、正式/smoke 配置、测试、
依赖清单、README、验证记录，以及运行必需的 `protocol/deduplicated/` 和 `protocol/retrieval_v1/`。
历史 `protocol/split.csv` 保留用于追溯，当前配置不再使用它。
`.gitattributes` 保留冻结划分的原始字节及换行符，避免不同平台检出后改变数据签名。

`.gitignore` 排除本地虚拟环境、数据集、数组缓存、模型权重、实验输出和日志，以及
本地环境变量文件与编辑器缓存；这些文件仍保留在本机。远程运行时按下文重建环境、
单独放置数据，训练产物在服务器上生成。`docs/VALIDATION.md` 中指向 `artifacts/` 的链接是
本地验证证据，克隆仓库后不会附带这些产物。

同级 `对比方法说明/` 和 `data/` 位于本仓库之外，不随本仓库提交；前者的详细说明链接
在完整工作区中可用，各方法 README 和运行代码已包含在仓库内。

## 环境和远程目录

从包含 `compared_methods/` 的外层目录运行以下命令。建议使用独立 Python 3.10–3.12 环境。
远程可沿用 `wbm-region-retrieval` 已正常运行的 PyTorch/CUDA 环境，再补齐本目录依赖。
若现有依赖已满足范围，无需升级 PyTorch；若是新环境，先安装与服务器驱动匹配的官方
CUDA PyTorch 包，再安装其余依赖。本目录不固定服务器 CUDA 版本。

```bash
python -m venv compared_methods/.venv
source compared_methods/.venv/bin/activate
python -m pip install -r compared_methods/requirements.txt

# 确认所用的 python 和 CUDA。GPU 正式训练应显式指定 --device cuda。
python -c "import sys, torch; print(sys.executable); print(torch.__version__, torch.version.cuda); print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

macOS 本地安装的是 CPU/Mac 版本 PyTorch，不能把本地 `.venv` 复制到 Linux 服务器。
远程同步代码、`compared_methods/protocol/deduplicated/`、`compared_methods/protocol/retrieval_v1/`
和数据即可，不依赖原工程的 Python 模块。
不需要下载预训练权重，也不依赖本机 Zotero、临时 PDF 或绝对路径。

默认目录关系为：

```text
工作区/
├── compared_methods/
└── data/wm38k/Wafer_Map_Datasets.npz
```

配置中的 `data_file`、`split_file`、`retrieval_protocol_dir`、`output_dir`（以及可选 `cache_dir`）以
`compared_methods/` 为基准，不随启动命令的 cwd 改变；绝对路径也支持。
`--config` 参数本身按命令当前目录解析。数据位置不同时复制配置并修改 `data_file`。
默认原始数组缓存位于输出目录的同级 `_data/`，按输入 SHA-256 分组，允许各方法共用。

## 数据与评价

- 先在完整原始数组上按像素值精确去重：SHA-256 筛选后逐像素确认，每组保留最小原始
  ID，同图标签冲突时报错。0/1/2/3 全部参与判定，不按旋转、翻转或视觉相似合并。
- 当前数据移除 1,024 条重复记录，保留 36,991 张唯一图，其中 35,991 张属于 37 类缺陷。
  以 seed=2026 按完整标签组合重新分层 70%/10%/20%，得到 train/valid/test=
  **25,193/3,599/7,199**。新划分为 `protocol/deduplicated/split.csv`；原始 NPZ 和旧划分均保留。
- Normal 也参与全局去重；当前 1,000 张均唯一，独立固定为 700 train、100 valid、200 holdout，
  只用于 Yu 检测器。配置数量若与唯一 Normal 数不符，会报错而不是静默缩放。
- 37 个分类标签由 train 的完整 multi-hot 组合建立，不使用八维 BCE。
- 训练和选模不使用 test；显式 test 阶段才提取 test 特征并评价。PCA 和编码缩放始终只在
  train 拟合，Normal holdout 不加入缺陷检索。读取原始 NPZ、计算图像哈希和核对划分
  属于数据审计。运行时重新核对去重代表及划分覆盖，拒绝含重复副本的旧协议。
- 正式 valid 在 37 种完整组合类别内分层固定抽取 **500 个 query**，按类别比例分配，
  每类至少 5 张（不足 5 张则全取），seed=2026；候选为完整 valid 集除自身，每个 **3,598 张**。
- 正式 test 的 **7,199 张全部轮流作为 query**，候选为完整 test 集除自身，每个 **7,198 张**。
  所有方法读取同一份已保存的 query/gallery 清单，不在方法运行时重新抽样。
- 五方法统一评价的相关性为完整缺陷组合相同；本版不包含方向覆盖率或 nDCG。
  查询可以互为候选，但不能检索自身。统一评价的 SHA-256 并列规则与旧方法入口的原始 ID
  并列规则不同，旧排名和指标不能直接混入新结果。
- 既有 Wu/Yu 的 P@K 分母为 K，少于 K 个返回时未返回位置按未命中；AP@K 的分母为原固定候选池中
  全部相关样本数。零相关 query 不计入 mAP/Recall，仍计入 Precision，并报告其数量。
  这个 AP 分母与 Yu 原文的 Top-500 内相关数不同，名称和口径已明确区分。

去重和划分由 `shared/data_protocol.py` 实现，显式生成入口为 `scripts/build_protocol.py`。
默认配置使用仓库内的新冻结清单，正常运行无需重新划分；重建方法及产物含义见
[协议说明](protocol/README.md)。`duplicates.csv` 记录每条被过滤记录对应的保留 ID，
`metadata.json` 记录数据哈希、规则、种子及数量。去重不改原数组内容，也不重编号样本。

检索协议由 `scripts/build_retrieval_protocol.py` 基于已有缺陷划分生成，默认保存到
`protocol/retrieval_v1/`。其中四个 CSV 固定 valid/test 的 query 与 gallery ID，元数据固定
排除自身的规则及文件哈希，不展开五千多万行 test 查询候选对。运行时校验该协议与原始数据、
缺陷划分的绑定关系。新增方法也应读取这份协议，不能自行另选 query 或候选。

旧原始划分、去重划分和已有运行产物均保留。候选库扩展后，旧检索指标不能直接并入新结果；
源码与协议签名也禁止混用旧运行目录。正式输出为 `artifacts/wu_retrieval_v1`、
`artifacts/yu_retrieval_v1`，smoke 输出为 `artifacts/wu_smoke_retrieval_v1`、
`artifacts/yu_smoke_retrieval_v1`。

## 原方法入口与统一评价

WBBS 和 WMHD 无需训练，`audit` 核对数据及协议，`prepare` 缓存 valid 图的单图特征。
Nakazawa 的 `train` 使用冻结 train 划分训练 CNN，`encode` 保存 valid 的 256 位二值码。
这三个方法自身 CLI 不输出检索指标，也没有 `test` stage；五方法的统一 valid/test 评价
通过 [scripts/evaluate.py](scripts/evaluate.py) 单独执行，见[完整命令](docs/EXACT_EVALUATION.md)。
正式默认 K 为 1、5、10、20、50、100、500；smoke 仅 valid，默认 K 为 1、5、10、20。
每次评价必须指定尚不存在的新输出目录，Yu/Nakazawa 以只读 `--source-run` 指向已完成训练的来源。

```bash
python compared_methods/methods/wbbs/wbbs.py --config compared_methods/methods/wbbs/config.json --stage prepare
python compared_methods/methods/wmhd/wmhd.py --config compared_methods/methods/wmhd/config.json --stage prepare
python -u compared_methods/methods/nakazawa/nakazawa.py --config compared_methods/methods/nakazawa/config.json --stage train --device cuda
python compared_methods/methods/nakazawa/nakazawa.py --config compared_methods/methods/nakazawa/config.json --stage encode --device cuda
```

先核对各方法 README 的论文依据、未披露参数、smoke 命令及恢复方式，再运行正式配置。
不调用上述命令也不影响已有冻结协议。统一入口调用各方法的单图编码和候选打分接口；
特征缓存均保留原始 `sample_id`，候选身份由公共冻结协议决定。统一产物位于新目录的
`valid/` 或 `test/` 下，根目录 `completed.json` 才表示完成；失败中间文件不能作为成绩。
本版不支持评价断点恢复，失败重跑需新目录。尚未完成正式 CNN 训练或完整 test 的记录仍然有效。

公共 `load_config` 增加 `require_ks=False`，允许这些仅产出特征／分数的方法不填写指标参数；
默认行为及 Wu/Yu 的指标配置校验保持原样。独立评价模块不加入旧训练签名；已有来源若因方法、
原共享源码、科学配置或数据协议变化而不匹配，仍会拒绝读取，不修改旧 manifest 绕过校验。

## 既有 Wu/Yu 入口的小规模验证

```bash
python -m unittest discover -s compared_methods/tests -v
python compared_methods/methods/wu/wu.py --config compared_methods/methods/wu/smoke.json --stage all --workers 1
python compared_methods/methods/yu/yu.py --config compared_methods/methods/yu/smoke.json --stage all --device cpu

# 目标 GPU 上使用独立目录做实际 CUDA 验证。
python compared_methods/methods/yu/yu.py --config compared_methods/methods/yu/smoke.json --stage all --device cuda --output-dir artifacts/yu_cuda_smoke_retrieval_v1

# 可选：核验 spawn/多进程 DataLoader 与串行预处理一致。
python compared_methods/tests/check_workers.py --workers 2
```

smoke 使用完整模型，但每个网络最多 8 次更新；没有完成正式 epoch，不能据其指标判断方法优劣。
PCA 始终保留 256 维，按该配置使用 370 张 train 图拟合。重复训练已有目录需 `--resume`，或者
指定新输出目录；若只需重新检索，可运行 `--stage valid`。
smoke 的 query 是正式固定 query 与小样本 valid 的交集，候选为小样本 valid 除自身；
它是派生的调试协议，不代表正式 500-query 评价，不能用于 `--stage test`。

## 既有 Wu/Yu 入口的正式 train/valid

```bash
# 无训练方法；CPU 并行提取特征并评价 valid。
python compared_methods/methods/wu/wu.py --config compared_methods/methods/wu/config.json --stage all --workers 4

# Yu：先审计，再训练两个 CNN；检测器 10 epoch，分类器 80 epoch。
python compared_methods/methods/yu/yu.py --config compared_methods/methods/yu/config.json --stage audit
python -u compared_methods/methods/yu/yu.py --config compared_methods/methods/yu/config.json --stage train --device cuda

# 中断后继续。
python -u compared_methods/methods/yu/yu.py --config compared_methods/methods/yu/config.json --stage train --device cuda --resume

# 冻结各网络最优 valid CE checkpoint，用 train 拟合 PCA 后进行 valid 检索。
python -u compared_methods/methods/yu/yu.py --config compared_methods/methods/yu/config.json --stage valid --device cuda

# 在 valid 上确定方法与模型后，显式进行最终 test；all 不会自动执行 test。
python compared_methods/methods/wu/wu.py --config compared_methods/methods/wu/config.json --stage test --workers 4
python -u compared_methods/methods/yu/yu.py --config compared_methods/methods/yu/config.json --stage test --device cuda
```

Yu 使用全 train 标签，FP32、无预训练、无自动混合精度；默认检测/分类 batch 为 64/128。
如果目标显存不足，复制配置并减少对应 batch，使用新输出目录并记录该适配；不在运行时
静默降低 batch。`num_workers`、`torch_threads` 和推断 batch 可以在配置中指定。
Wu 的 SciPy/scikit-image 特征处理主要在 CPU 上执行，不要求 GPU 支持。

## 产物和恢复

每个运行保存配置、源码/数据签名、类别映射、去重摘要和实际样本 ID，拒绝把不同科学设置混入同一目录。
更换方法源码、训练配置、数据或划分时应使用新输出目录。输出路径和执行设备可以变化。

Wu 分别保存 valid/test 的特征与 NCC 缓存、逐候选排名及评价；它只精排粗筛 Top-n，默认正式
n=500，不声称对整个 gallery 生成完整精排。Yu 保存两个网络的 best/last checkpoint、optimizer/RNG、
epoch/批次位置、训练日志、PCA 编码器，以及 global 和 predicted_class 两个独立检索报告。
正式 test 全量比较需要处理 51,818,402 个查询候选对；候选按需展开、排行逐 query 处理，
不把整张候选对表一次性放入内存。

`--max-steps` 是 Yu 每个网络的累计步数上限。中途停止会标记 `truncated_epoch`，后续恢复
继续尚未处理的批次。小规模验证不是性能实验；本地 CPU 成功不代表已在目标 CUDA 硬件实测。

52×52 上的 9×9 滤波会消除部分图像的缺陷像素。输入顺序按用户确认保持，逐图影响保存在
Yu 输出的 `preprocessing/` 中，不根据 valid 结果暗中更改算法。
