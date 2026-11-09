# 五方法统一评价说明

统一入口为 `scripts/evaluate.py`，适用于 Wu、Yu、WBBS、Nakazawa 和 WMHD。
当前已实现第一类 exact_v1 评价；第二类方向性类别覆盖评价在本文件中记录设计，尚未接入代码。
方法生成排序时不读取真实标签，真实标签仅供协议审计和评价使用。
从训练到最终 test 的逐方法操作与产物位置见 [五方法运行指南](RUN_GUIDE.md)。

## 第一类评价：完整组合相同（exact_v1）

### 固定数据与 exact_v1 指标

继续读取既有 `protocol/deduplicated/split.csv` 与 `protocol/retrieval_v1/`，不重新划分、抽样或编号。
valid 为固定 500 query、3,599 gallery，每个 query 有 3,598 个候选；test 为全部 7,199 query、
7,199 gallery，每个 query 有 7,198 个候选。均排除自身，Normal 不加入检索。
smoke 只使用正式 query 与小 valid 子集的交集，只允许 `--split valid`，不能作为正式成绩。

正式默认 `K = 1, 5, 10, 20, 50, 100, 500`，smoke 默认为 `1, 5, 10, 20`。
可用 `--ks` 显式指定共同截断点；各方法必须保持一致，且 K 不能超过完整候选数。
令 `rel(r)` 为第 r 个返回结果是否与 query 完整组合相同，`R` 为**完整固定候选池**中的相关数：

| 逐 query 指标 | 定义 |
| --- | --- |
| Hit@K | 前 K 项至少命中一个则为 1，否则为 0 |
| Precision@K | 前 K 项命中数 / K |
| Recall@K | 前 K 项命中数 / R |
| AP@K | `Σ(r=1..K) Precision@r × rel(r) / R` |

AP@K 是截断 AP；对 query 平均后称为 mAP@K，不输出无截断限定的完整 mAP。
R 不因 Wu 粗筛或 Yu 门控而缩小。返回不足 K 或空返回时，缺失位置按未命中处理。
R=0 的 query 排除于 AP/Recall 平均，仍计入 Hit/Precision，同时报告数量和各指标有效计数。
Hit@K 不称为 Recall@K；它不区分前 K 项命中一个还是多个。

`metrics.json` 同时报告 query 平均和各实际出现的 query 组合类别等权的 class macro 平均，
并记录实际类数、各指标有效类数。另按单标签／混合标签、查询标签数量报告分组 query 平均。
`per_query.csv` 保存逐 query 的 AP@K 等指标，`per_class.csv` 保存逐组合类别的 mAP@K 等均值。
类别宏平均对应完整组合类别，不是八个基础缺陷类别的平均。

## 第二类评价设计：方向性类别覆盖（尚未接入代码）

论文的主任务是判断候选是否覆盖查询图中的缺陷类别。该任务与上面的 exact_v1 是两种不同的
相关性定义：exact_v1 要求完整八维标签组合相同；覆盖评价允许候选包含查询中没有的额外类别。
本节冻结评价设计，当前 `scripts/evaluate.py`、`shared/exact_metrics.py` 和五方法统一产物尚未实现
这些字段，也不能把本节当作已经完成的实验结果。

### 方向性覆盖率

令 `Lq` 和 `Lc` 分别为 query 与 candidate 的非空缺陷类别集合，定义：

```text
coverage(q, c) = |Lq ∩ Lc| / |Lq|
```

这是 query → candidate 的有方向定义。候选额外类别不降低覆盖率，因此：

| Query | Candidate | coverage | 解释 |
| --- | --- | ---: | --- |
| `{A,B}` | `{A,B}` | 1 | 完整覆盖，也是 exact 匹配 |
| `{A,B}` | `{A,B,C}` | 1 | 完整覆盖，但不是 exact 匹配 |
| `{A,B}` | `{A}` | 0.5 | 部分覆盖 |
| `{A,B}` | `{C}` | 0 | 不覆盖 |

反向查询的覆盖率一般不同，例如 `{A,B,C}` 查询 `{A,B}` 的覆盖率为 `2/3`。不使用 Jaccard
作为主相关性，因为 Jaccard 会惩罚候选额外类别，与本项目的覆盖目标不一致。

### 主排序指标：Coverage-nDCG 和 MeanCoverage

Coverage-nDCG 将上述 coverage 直接作为**线性分级增益**。对方法返回的第 `r` 个结果，令
`g_r=coverage(q,c_r)`：

```text
DCG@K  = Σ(r=1..K) g_r / log2(r+1)
IDCG@K = 对完整候选池中的 g 按降序排列后计算同一公式
Coverage-nDCG@K = DCG@K / IDCG@K
```

`IDCG` 必须由该 query 的完整固定 gallery 计算，而不是由方法返回的 Top-K 估计。完整覆盖候选
的增益为 1，部分覆盖候选的增益为 `0<g<1`，不覆盖候选的增益为 0。这样可以同时评价完整覆盖
是否排在部分覆盖之前，以及不同部分覆盖程度的相对顺序。若一个 query 的完整候选池全部为零覆盖、
导致 `IDCG=0`，该 query 的 Coverage-nDCG 记为未定义并从均值排除，同时报告有效 query 数。

另报告前 K 个结果的平均覆盖程度：

```text
MeanCoverage@K = (1/K) × Σ(r=1..K) g_r
```

返回不足 K 个结果时，缺位按零覆盖处理。MeanCoverage 反映前列结果实际覆盖了多少查询类别，
Coverage-nDCG 反映它们是否接近该 query 的理想覆盖排序；前者是直观辅助指标，后者是主要分级排序指标。

### 完整覆盖的二值指标

完整覆盖是覆盖率评价中的一个特例，而不是 exact 的同义词：

```text
rel_full(q,c) = 1[coverage(q,c)=1] = 1[Lq ⊆ Lc]
```

对 `rel_full` 沿用 Hit、Precision、Recall 和 AP/mAP：

| 指标 | 含义 |
| --- | --- |
| FullCoverage Hit@K | 前 K 项是否至少出现一个完整覆盖候选 |
| FullCoverage Precision@K | 前 K 项中完整覆盖候选的比例 |
| FullCoverage Recall@K | 前 K 项找回的完整覆盖候选数 / 完整 gallery 中的完整覆盖候选总数 |
| FullCoverage AP@K / mAP@K | 按完整覆盖二值相关性计算的截断 AP 及 query 平均 |

完整覆盖的 Recall/AP 分母记为 `R_full(q)`，必须从完整固定候选池计算，不能只统计已返回的结果，
也不能使用 exact 正样本数量。`R_full=0` 时，FullCoverage Recall/AP 从相应均值排除；
Hit/Precision 是否计入，沿用本节 exact 指标的零相关处理并报告有效数量。

### 部分覆盖阈值与分组

当前不把 `0<coverage<1` 单独当作二值正样本，因为这样会把更高等级的完整覆盖排除在外，
不符合“覆盖越多越好”的排序目标。若应用场景明确规定“至少覆盖一定比例才有用”，可在看 test
结果之前预先固定阈值 `τ`：

```text
rel_τ(q,c) = 1[coverage(q,c) ≥ τ]
```

再对 `rel_τ` 计算同一套 Hit/Precision/Recall/AP。`τ=0.5` 只是候选方案，不是当前已冻结的
正式阈值；对两标签 query 它表示至少覆盖一个类别，对三标签 query 则表示至少覆盖两个类别，
所以应同时按 query 标签数量报告。`coverage>0` 可作为“任意重合”的诊断，但不建议作为论文主指标。

Coverage-nDCG、MeanCoverage 和 FullCoverage 指标沿用当前固定 K、完整 gallery、自身排除、
query 平均、完整组合类别宏平均及 single/mixed 分组。单标签 query 的 coverage 只有 0 或 1，
不能检验部分覆盖等级；混合标签 query 是该任务的主要分析对象。

论文主评价组合暂定为：Coverage-nDCG@10/@20、MeanCoverage@10/@20、FullCoverage mAP@K 和
FullCoverage Recall@K；exact_v1 指标作为独立的完全匹配参考。具体主表 K、是否加入 `τ=0.5`
及其报告范围，须在正式 test 之前确定，不能根据 test 结果挑选。

## 排序与方法边界

所有方法保留自己的特征和距离；统一评价不训练模型，也不强制更换算法的粗筛或门控。

| 方法 | 统一评价中的行为 |
| --- | --- |
| Wu | 全部候选计算欧氏距离，保留配置中的粗筛数量（正式默认 500），仅对这些候选做 NCC 重排；NCC 同分先比较欧氏距离。不把未粗筛入选者补排，不称为全库 NCC 精排 |
| Yu `global` | **主比较结果**；全部候选按 categorical Hamming 距离排序，不使用检测器或预测类别门控；分类器使用原方法最低 valid CE checkpoint，PCA 与缩放只拟合 train |
| Yu `predicted_class` | 独立辅助结果；使用检测和预测组合类别门控，query 被检测为 Normal 可返回空；相关数分母仍来自完整候选池 |
| WBBS | 对完整候选池计算 WBBS，分数越大越相似；保留原方法点集匹配公式 |
| Nakazawa | 使用完成配置 epoch 的最后一轮 checkpoint，256 位 Hamming 距离排序；不按 valid loss 或检索指标重新选模型 |
| WMHD | 保留 query → candidate 的方向，按负距离排序；边界情形的 `-inf` 是合法最差分数，不删除该 query 或候选、不改为对称距离 |

剩余同分使用不读取标签的固定哈希排序：对 ASCII 字符串
`exact_v1:tie:{seed}:{query_id}:{candidate_id}`（十进制整数、无空格或换行）计算 SHA-256，
取前 8 字节为 big-endian uint64，升序排列，哈希碰撞时以原始 sample_id 升序兜底。
默认 tie seed 为 2026；Wu 在粗筛截断之前也使用该规则。
原方法入口的 sample_id 同分规则保留。新旧并列规则可能改变结果，**旧排名和旧指标不能直接混入 exact_v1 比较**。
这是一条确定的破并列规则，不是对所有并列排列取期望的 tie-aware 指标。

## 使用方式

以下命令从 `compared_methods/` 执行，先激活已安装本目录依赖的 Python 环境。
`--config` 相对当前命令目录；配置内路径、`--output-dir` 与 `--source-run` 的相对路径均相对
`compared_methods/`。每次 `--output-dir` 必须是**尚不存在**的新目录，包括已有空目录也拒绝使用。

三个无训练方法的 valid smoke：

```bash
python scripts/evaluate.py --config methods/wu/smoke.json --split valid --output-dir artifacts/exact_v1/wu_valid_smoke --workers 1
python scripts/evaluate.py --config methods/wbbs/smoke.json --split valid --output-dir artifacts/exact_v1/wbbs_valid_smoke --workers 1
python scripts/evaluate.py --config methods/wmhd/smoke.json --split valid --output-dir artifacts/exact_v1/wmhd_valid_smoke --workers 1
```

Yu 和 Nakazawa 先用各自现有训练入口完成训练，再执行统一评价。正式 valid/test 均拒绝未完成正式训练的来源；
Yu `global` 检查分类器完成状态，辅助门控分支另外检查检测器，选定的 best checkpoint 可以早于最后一轮。
Nakazawa 固定读取最后一轮。仅完成 smoke 训练的来源只能配合相同 smoke 配置评价 valid。

```bash
# 训练入口；已有训练可按方法 README 使用 --resume。不要为评价重复启动训练。
python methods/yu/yu.py --config methods/yu/config.json --stage train --device cuda
python methods/nakazawa/nakazawa.py --config methods/nakazawa/config.json --stage train --device cuda

# 完成相应训练后，先在 valid 上评价并确定设置。
python scripts/evaluate.py --config methods/yu/config.json --split valid --source-run artifacts/yu_retrieval_v1 --output-dir artifacts/exact_v1/yu_global_valid --device cuda
python scripts/evaluate.py --config methods/nakazawa/config.json --split valid --source-run artifacts/nakazawa_retrieval_v1 --output-dir artifacts/exact_v1/nakazawa_valid --device cuda
```

`--source-run` 仅适用于 Yu/Nakazawa，省略时使用配置中的 `output_dir`。
来源运行只读；新提取特征、Yu 的 train 拟合编码器和全部评价产物写入新的评价目录。
入口不会隐式训练，也不会从 valid 自动进入 test；每次必须显式指定 `--split`。

以下是**待执行的正式 test 命令**，不是已经完成实验的记录。须先完成正式训练（如适用），在 valid
上固定参数和模型，使用无 subset 的正式配置。WBBS/WMHD 完整 test 的逐对计算可能耗时较长。

```bash
python scripts/evaluate.py --config methods/wu/config.json --split test --output-dir artifacts/exact_v1/wu_test --workers 4
python scripts/evaluate.py --config methods/yu/config.json --split test --variant global --source-run artifacts/yu_retrieval_v1 --output-dir artifacts/exact_v1/yu_global_test --device cuda
python scripts/evaluate.py --config methods/wbbs/config.json --split test --output-dir artifacts/exact_v1/wbbs_test --workers 4
python scripts/evaluate.py --config methods/nakazawa/config.json --split test --source-run artifacts/nakazawa_retrieval_v1 --output-dir artifacts/exact_v1/nakazawa_test --device cuda
python scripts/evaluate.py --config methods/wmhd/config.json --split test --output-dir artifacts/exact_v1/wmhd_test --workers 4
```

Yu 的辅助结果单独运行、单独存放，例如：

```bash
python scripts/evaluate.py --config methods/yu/config.json --split test --variant predicted_class --source-run artifacts/yu_retrieval_v1 --output-dir artifacts/exact_v1/yu_predicted_class_test --device cuda
```

`--workers` 控制手工特征准备的进程数，不表示候选逐对排序自动并行。运行时间必须连同硬件、
worker 数及方法变体解释。上述命令不替代各方法 README 中的复现边界和参数说明。

## 签名、产物与失败处理

统一评价具有独立的 runner/metric 版本及代码、参数、数据、冻结协议、来源模型签名。
原方法 CLI、配置和训练签名机制不变；新增评价模块不加入旧训练签名。
已有训练来源的代码、科学配置、数据或协议与当前配置不匹配时会拒绝读取，不能修改 manifest 绕过检查。
正在变化的 checkpoint、源码或协议也不能发布为完成结果。已有原始数据、协议、训练和旧结果不会被覆盖。

设 `OUT` 为本次新评价目录，`SPLIT` 为 `valid` 或 `test`：

| 位置 | 内容 |
| --- | --- |
| `OUT/evaluation_manifest.json` | 评价版本和签名、来源签名、K、并列规则、池规模及方法变体 |
| `OUT/resolved_config.json`、`environment.json`、`protocol/` | 配置、环境及数据／固定 ID 审计；`raw_cache/` 为本次独立原始数组缓存 |
| `OUT/SPLIT/metrics.json` | 指标定义、query/class 均值和分组结果 |
| `OUT/SPLIT/per_query.csv`、`per_class.csv` | 逐 query 和逐组合类别统计 |
| `OUT/SPLIT/rankings.csv` | 逐 query 流式保存最多 `max(K)` 个返回结果，不是完整全库排名文件 |
| `OUT/SPLIT/query_diagnostics.csv` | 输入候选数、实际评分／粗筛／可返回数量及逐 query 排序耗时 |
| `OUT/SPLIT/method.json`、`prepared/` | 方法设置、诊断和本次特征／编码 |
| `OUT/SPLIT/inference_environment.json` | CNN 的随机种子、线程数和实际 cuDNN 确定性设置 |
| `OUT/SPLIT/timing.json` | 准备耗时、排序总耗时及均值／中位数／P95、总耗时；排序计时不含指标计算和 CSV 写出 |
| `OUT/completed.json` | 完成标记、评价签名及产物哈希 |

**只有存在 `completed.json` 的目录才是完成结果。** 目录建立后的失败保留 `failed.json` 和已有中间文件；
即使已有局部 `metrics.json`，也不能当作完成成绩。前置检查失败可能不会创建输出目录。
本版没有评价断点恢复；失败重跑需选择新目录，不覆盖或删除旧产物。

实际验证范围见 [VALIDATION.md](VALIDATION.md)。新增统一入口不代表已经完成正式 CNN 训练、
正式完整 test 或原论文数值复现；smoke 只验证实现链路。
