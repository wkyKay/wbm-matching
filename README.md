# MixedWM38 晶圆检索方法复现

本仓库独立实现并比较五种晶圆图检索方法：Wu、Yu、WBBS、Nakazawa CNN 和 WMHD。
五种方法共享同一份去重数据划分、固定 query/gallery 和统一评价入口，方法本身保留各自的
特征、模型和相似度定义。

当前统一评价实现的是**第一类任务：查询与候选的八维缺陷标签组合完全相同才算相关**。
它提供 Hit@K、Precision@K、Recall@K 和截断 mAP@K。论文中的方向性类别覆盖率、
Coverage-nDCG 和覆盖阈值指标尚未接入本仓库的统一评价入口；不要把当前 exact_v1 结果解释为
部分覆盖排序结果。

## 从哪里开始

| 需求 | 文档 |
| --- | --- |
| 了解五个方法、训练与 test 命令 | [五方法运行指南](docs/RUN_GUIDE.md) |
| 查看 exact_v1 公式、覆盖率设计、排序规则和产物 | [统一评价说明](docs/EXACT_EVALUATION.md) |
| 查看训练边界、参数与论文复现约定 | 各方法的 README：[`wu`](methods/wu/README.md)、[`yu`](methods/yu/README.md)、[`wbbs`](methods/wbbs/README.md)、[`nakazawa`](methods/nakazawa/README.md)、[`wmhd`](methods/wmhd/README.md) |
| 查看冻结数据与 query/gallery 协议 | [协议说明](protocol/README.md) |
| 查看实现验证和 smoke 范围 | [验证记录](docs/VALIDATION.md) |

需要逐方法运行时，先读运行指南；需要判断某个结果能否用于论文比较时，先读统一评价说明。

## 五种方法

| 方法 | 是否需要训练 | 检索表示／分数 | 实现与说明 |
| --- | --- | --- | --- |
| Wu (2015) | 否 | 116 维手工特征粗筛，再用 NCC 重排 | [代码](methods/wu/wu.py) · [说明](methods/wu/README.md) |
| Yu (2019) | 是 | 分类 CNN 特征、PCA、十值编码和 categorical Hamming | [代码](methods/yu/yu.py) · [说明](methods/yu/README.md) |
| WBBS (2023) | 否 | 张量投票、点描述子和加权最佳伙伴相似度 | [代码](methods/wbbs/wbbs.py) · [说明](methods/wbbs/README.md) |
| Nakazawa CNN (2018) | 是 | 256 维 sigmoid 特征二值化后用 Hamming 距离 | [代码](methods/nakazawa/nakazawa.py) · [说明](methods/nakazawa/README.md) |
| WMHD (2020) | 否 | mountain 密度和方向性加权 Hausdorff 距离 | [代码](methods/wmhd/wmhd.py) · [说明](methods/wmhd/README.md) |

三个无训练方法不需要 checkpoint。Yu 和 Nakazawa 必须先完成各自训练，统一评价只读其来源运行，
不会隐式训练或替换 checkpoint。论文没有公开的参数和流程细节在对应方法 README 中单独标为实现约定，
不宣称逐行恢复作者源码。

## 数据与固定协议

原始数据不随仓库提交。默认目录关系为：

```text
工作区/
├── compared_methods/
└── data/wm38k/Wafer_Map_Datasets.npz
```

运行时使用原始 NPZ 行号作为 `sample_id`，不会因去重重新编号。完整原图先按 0/1/2/3 像素精确去重，
每组保留最小原始 ID，再按 37 种非空 multi-hot 组合分层划分：

| 项目 | 数量／规则 |
| --- | --- |
| 原始记录 | 38,015 |
| 过滤的冗余记录 | 1,024 |
| 唯一图像 | 36,991，其中缺陷图 35,991、Normal 1,000 |
| 缺陷 train/valid/test | 25,193 / 3,599 / 7,199 |
| 标签顺序 | center、donut、edge-loc、edge-ring、loc、near-full、scratch、random |
| valid query/gallery | 500 / 3,599；每个 query 排除自身后 3,598 个候选 |
| test query/gallery | 7,199 / 7,199；每个 query 排除自身后 7,198 个候选 |

固定清单位于：

- `protocol/deduplicated/split.csv`
- `protocol/deduplicated/duplicates.csv`
- `protocol/retrieval_v1/valid_queries.csv`
- `protocol/retrieval_v1/valid_gallery.csv`
- `protocol/retrieval_v1/test_queries.csv`
- `protocol/retrieval_v1/test_gallery.csv`

方法不得自行重划数据、重抽 query、缩小正式 gallery 或使用 Normal 进入缺陷检索库。
协议加载时会校验源数据、划分和清单哈希；需要重建协议时使用 `scripts/build_protocol.py` 和
`scripts/build_retrieval_protocol.py`，输出到新目录，不覆盖冻结文件。

## 统一评价协议

统一入口是 [`scripts/evaluate.py`](scripts/evaluate.py)，从 `compared_methods/` 目录运行。
它先读取冻结协议，再准备目标 split 的特征，逐 query 对完整候选池评分并保存最多 `max(K)` 个结果。
当前正式 K 为 `1, 5, 10, 20, 50, 100, 500`；smoke 只使用 valid 派生小池。

对 query q 和排名第 r 的候选，令 `rel(r)` 表示完整八维标签组合是否相同，
令 `R` 为完整候选池中的相关候选数：

- `Hit@K`：前 K 项至少命中一个；
- `Precision@K`：前 K 项命中数除以 K；
- `Recall@K`：前 K 项命中数除以完整候选池中的 R；
- `AP@K`：前 K 项中各命中位置的 Precision 累加后除以 R；query 平均后为 `mAP@K`。

缺位按未命中处理。R 为零的 query 从 Recall/mAP 平均中排除，但仍参与 Hit/Precision。
同时报告所有 query 平均、完整组合类别宏平均、single/mixed 分组和按标签数量分组。

方法分数生成时不读取真实标签。方法产生分数后，统一入口使用固定的 SHA-256 规则处理同分；
Wu 还保留欧氏距离作为 NCC 重排的次排序键。Wu 的粗筛不会补排未进入 shortlist 的候选，
但 Recall/AP 的相关数分母仍来自完整候选池。

当前评价不包含 Coverage-nDCG、部分覆盖阈值或方向性覆盖率；这些属于下一阶段的独立评价设计。

## 运行入口的分工

方法 CLI 用于方法自身的审计、训练、编码或准备缓存；统一比较应使用 `scripts/evaluate.py`。

```text
scripts/evaluate.py
  └─ shared/evaluation.py
      ├─ shared/common.py              数据与冻结检索池
      ├─ shared/evaluation_adapters.py 五方法特征、分数和排序适配
      └─ shared/exact_metrics.py       exact_v1 指标与汇总
```

统一评价不会修改方法来源运行。每次评价必须指定新的输出目录；已有目录、checkpoint、缓存和排名
不得通过修改 manifest 强行混用。

## 环境与最小检查

建议使用 Python 3.10–3.12 和项目依赖。GPU 训练需要与驱动匹配的 PyTorch/CUDA；本机环境不能直接
复制到 Linux 服务器。

```bash
cd /path/to/match-test/compared_methods
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

已有 Conda 环境可直接激活，例如 `conda activate wbm`。五方法 smoke、正式训练、恢复和完整 test
命令见 [docs/RUN_GUIDE.md](docs/RUN_GUIDE.md)；不建议从根 README 复制单个方法的长命令，以免
遗漏来源 checkpoint、split 或输出目录要求。

## 产物与版本边界

运行产物通常写入 `artifacts/`，包括配置、环境、协议摘要、特征或模型、排名、逐 query 指标、
`metrics.json` 和完成标记。只有存在 `completed.json` 的统一评价目录才视为完成结果；失败目录中的
局部文件不作为 benchmark 成绩。

仓库不包含原始数据、模型权重、训练缓存或正式运行产物。源码、配置、协议和来源模型签名发生变化时，
使用新的运行目录；旧协议或旧候选池的结果不能与当前 exact_v1 结果混表。

## 目录结构

```text
compared_methods/
├── methods/          五种方法的代码、配置和方法 README
├── shared/           数据协议、检索池、统一 runner、适配器和指标
├── scripts/          协议生成与统一评价入口
├── protocol/         去重划分和固定 query/gallery 清单
├── tests/             回归测试与并行一致性检查
├── docs/              运行指南、评价说明和验证记录
├── artifacts/         本地运行产物（不纳入版本控制）
├── requirements.txt
└── README.md
```

新增方法放在 `methods/<method_name>/`，实现自己的特征／分数接口，并复用 `shared/` 的数据协议和
统一评价入口。方法 README 应说明论文依据、实现约定、限制、命令和产物；统一评价规则只在
`docs/EXACT_EVALUATION.md` 维护，避免各方法重复定义指标。
