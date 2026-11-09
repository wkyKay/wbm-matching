# 验证记录

本文件随目录整理移入 `docs/`。下方历史命令保留执行时的旧路径；当前运行入口见
[总说明](../README.md)，方法位于 `methods/`，协议生成脚本位于 `scripts/`。

## 全候选 query 并行执行检查（2026-11-09）

正式全流程运行前新增 `--rank-workers`：只并行不同 query，各 query 的完整候选、
打分公式、并列规则与保存顺序不变。使用 spawn，每进程初始化一份只读特征，最多
预提交两倍进程数的 query，避免展开完整 test 配对到内存。未修改训练源码或配置。

- Linux 全套 **155 项测试通过**。新增覆盖五方法串行/spawn 排名与分数一致、粗筛边界
  并列、WMHD 空图负无穷、进程异常传播、有限队列及资源关闭、端到端评价一致与参数检查。
- Wu、WBBS、WMHD 真实数据 smoke 在相同单线程 BLAS 设置下，1 与 3 个 query 进程的
  `rankings.csv`、`per_query.csv`、`per_class.csv` 逐字节一致，总体/类别宏平均/分组指标一致。
- 对照旧的四线程 BLAS Wu smoke，排名 ID 不变，分数最大差约 `1.1e-6`；因此进程一致性
  检查固定同一 BLAS 设置，不将不同线程库归约顺序造成的浮点差异混入并行对照。

证据位于服务器 `artifacts/formal_exact_20261109_120738/` 的 `parallel_tests.log`、
`parallel_smoke/equivalence.json`、`parallel_smoke/` 与 `serial_smoke/`。
此节仅记录执行器验证，不表示该目录下正式训练与完整 test 已全部完成。

## Linux / CUDA 五方法 smoke（2026-11-09）

在现有 Conda `wbm` 环境验证：Python 3.10、NumPy 1.24.3、SciPy 1.15.3、
scikit-learn 1.7.2、scikit-image 0.25.2、PyTorch 2.4.0 / CUDA 12.4，单张 NVIDIA A30。
数据集 SHA-256 与冻结协议一致，未重建划分或检索清单。

首次回归暴露了两个兼容问题，初始记录保留于服务器
`artifacts/server_smoke_20261109_115121/`：

- scikit-learn 1.7.2 在未开启 whitening 时仍访问 `explained_variance_`。
  Yu 编码器现在保存和恢复真实 PCA 方差；旧档案缺失该字段时用空数组提供 NumPy namespace，
  未开启 whitening 的投影不会读取其数值。回归覆盖新档案往返、旧档案读取及再次保存。
- Conda 的 MKL Intel 线程层与已加载的 libgomp 冲突。本次进程设置
  `MKL_THREADING_LAYER=GNU`，未修改 Conda 环境安装或全局配置。

修复后全部重新运行到服务器 `artifacts/server_smoke_20261109_115338/`，未复用旧签名训练产物：

| 检查 | 结果 |
| --- | --- |
| 全套回归 | 150 项通过 |
| Wu / WBBS / WMHD | CPU、2 workers 准备特征，统一 valid smoke 通过 |
| Yu | CUDA 检测器与分类器各 8 次更新；train-only PCA256、编码及统一 valid smoke 通过 |
| Nakazawa | CUDA 训练 1 个 batch、监测 1 个 valid batch，最后轮权重编码及统一 valid smoke 通过 |
| 共同池 | 五方法均为相同 8 query、74 gallery，每 query 排除自身后 73 候选 |
| 输出 | 每方法 160 条 Top-20 排名；完成标记、产物哈希和源码哈希校验通过 |
| 独立复核 | 从标签与排名重新计算每 query 的 Hit/P/Recall/AP@1/5/10/20 及总体均值，与报告一致 |
| Yu 实际编码器 | 新保存文件成功重载并执行 1024 维输入到 256 维十值编码 |
| 输入保护 | 本轮运行前后全部已跟踪文件哈希一致，原始数据哈希一致 |

证据目录包含 `status.json`（逐步命令、退出码与耗时）、`tests.log`、各方法日志、
`environment.txt`、`runtime_environment.json`、`source_hashes.json`、`code.patch`、
`verify.py` 和 `verification.json`。`git_commit.txt` 记录基础提交，`code.patch` 记录本轮
验证时尚未提交的兼容性修复；精确执行源码由 `source_hashes.json` 确定。
训练产物位于 `training/{yu,nakazawa}/`，评价产物位于 `evaluation/<method>/`。
可重复命令见 [运行指南第 10 节](RUN_GUIDE.md#10-gpu-服务器上的五方法-smoke)。

以上均为流程 smoke，未运行正式 CNN 训练、完整 valid 或 test，不作为方法效果比较结果。

## 五方法第一类统一评价验证（2026-11-08）

用户本轮仅确定实现完整标签组合相同的第一类评价。新增独立 `scripts/evaluate.py`、
`shared/evaluation.py`、`shared/evaluation_adapters.py` 和 `shared/exact_metrics.py`，
五方法共用 Hit@K、Precision@K、Recall@K、AP@K/mAP@K；未加入覆盖率阈值、nDCG 或其他第二类指标。
公式、默认 K、命令、并列规则及产物说明见 [EXACT_EVALUATION.md](EXACT_EVALUATION.md)。

| 检查 | 结果 |
| --- | --- |
| 全套回归 | **150 项通过**：原 103 项，加指标 16、适配器 11、统一入口及来源隔离 20 项 |
| 指标语义 | 手算 Hit/P/Recall/AP；完整候选分母；短／空返回；零相关 query；类别宏平均及单／混合分组通过 |
| 五方法 CPU valid smoke | 均使用同一 8 个 query、74 张 gallery，每 query 73 个候选；逐 query 全覆盖、自身排除、排名与指标独立重算通过 |
| Yu smoke 来源 | 两个网络分别仅 8 次更新、未完成正式 epoch；统一评价使用 global，仅分类器特征和 train 拟合的 PCA／缩放 |
| Nakazawa smoke 来源 | 只读复用此前通过签名校验的 smoke 最后一轮权重；本轮未重新训练 |
| CNN 来源保护 | 正式未完成训练拒绝；Yu last 判断完成、best 保留原选模；Nakazawa final epoch／hash 校验；评价 K 不改变训练签名 |
| 并列及数值 | 固定 SHA-256 最终副键；Wu 粗筛边界使用同规则并保留 float32 与欧氏次键；WMHD 方向与合法 `-inf` 保留 |
| CPU/CUDA 设置 | CNN 统一入口继承原 `seed_everything` 及 cuDNN 确定性设置，实际标志保存；本轮只在 CPU 实测 |
| 运行隔离 | 旧运行目录、配置、协议内容与 mtime 不变的合成测试通过；已有输出拒绝覆盖；来源变化／失败不写完成标记 |
| 输入保护 | 原始 NPZ、全部冻结／历史协议、原共享实现与五方法源码／配置共 **34 个文件** SHA-256 与修改前一致 |
| 正式协议 | 五方法配置仍指向相同原始数据及冻结协议；test 清单 7,199 query／gallery，排除自身后每 query 7,198 候选 |
| 命令与文档 | 新入口从 `/tmp` 执行 `--help` 通过；文档链接与 `git diff --check` 通过 |

本次证据保存在新的
`artifacts/exact_validation_20261108_aulfj20r/`，旧运行与前一轮检查目录保留。
最终代码对应的五方法 smoke 输出在该目录的 `final/<method>/` 下，各自具备
`completed.json`，已核对产物哈希及当前评价源码哈希。

- [最终全套测试日志](../artifacts/exact_validation_20261108_aulfj20r/tests_final.log)
- [五方法身份、指标重算及保护文件检查](../artifacts/exact_validation_20261108_aulfj20r/integration.json)
- [Yu smoke 训练日志](../artifacts/exact_validation_20261108_aulfj20r/yu_training.log)
- 最终 smoke：[Wu](../artifacts/exact_validation_20261108_aulfj20r/final/wu/completed.json)、
  [Yu](../artifacts/exact_validation_20261108_aulfj20r/final/yu/completed.json)、
  [WBBS](../artifacts/exact_validation_20261108_aulfj20r/final/wbbs/completed.json)、
  [Nakazawa](../artifacts/exact_validation_20261108_aulfj20r/final/nakazawa/completed.json)、
  [WMHD](../artifacts/exact_validation_20261108_aulfj20r/final/wmhd/completed.json)。

**未运行正式 CNN 训练、正式完整 valid/test 评价或真实 test 特征计算。**
自动测试中的 test 使用小型合成数据；真实数据 smoke 只用于实现检查，不是方法优劣结论。
本版评价无断点恢复；已有输出即使为空也拒绝复用，失败后选择新目录。

## WBBS、Nakazawa CNN、WMHD 接入验证（2026-11-08）

本次新增三个独立方法目录，按原文公式实现并适配 MixedWM38。依用户要求，仅提供特征、
编码与打分接口；没有新增检索指标、相关性定义或统一 test 入口。CNN 训练及保存规则依据
Nakazawa 论文核查，原文未披露部分单列为实现约定，不继承 Yu 的选模流程。

| 检查 | 实际结果 |
| --- | --- |
| 全套自动测试 | **103 项通过**；含 WBBS 11 项、WMHD 15 项、Nakazawa 11 项及新增公共配置检查 |
| 原始数据与旧代码保留 | 原始 NPZ 及历史／冻结协议共 10 个文件，加上 Wu/Yu 原有 10 个方法文件，合计 **20 个文件** SHA-256 与修改前一致 |
| 五方法正式协议 | Wu、Yu、WBBS、WMHD、Nakazawa 的 train/valid/test 原始 ID 全部相同；25193/3599/7199 |
| valid 查询与候选 | 五方法都是同一 500 query、3599 gallery；逐 query 检查 3598 候选且排除自身 |
| WBBS 真实数据 smoke | 74 图特征；8 个固定 query 的交集，各 73 个完整候选；584 个分数有限且在 [0,1] |
| WMHD 真实数据 smoke | 74 图特征；本批 ESRN 后空图 0；8×73 个分数有限、候选完整、自身排除 |
| Nakazawa 真实数据 smoke | 原尺寸及完整网络；1 epoch 中仅 1 次更新、2 张 train、1 张 valid 监测；74 张 valid 编码为 256 维特征和 256 位二值码 |
| CNN 恢复检查 | 小模型 dropout＋Adam 在 epoch 边界中断后恢复，最终权重与不中断训练逐位相同；完整模型通过上述独立 smoke |
| 命令与文档 | 八个脚本异 cwd 的 `--help` 通过，十份方法配置保持根目录相对路径；六份 README 本地链接有效 |

证据：[全套测试](../artifacts/new_methods_validation_20261108_ecd779d2/tests.log)、
[五方法整合与保护检查](../artifacts/new_methods_validation_20261108_ecd779d2/integration.json)、
[WBBS 打分检查](../artifacts/validation/wbbs_ijsdxsr2/score_smoke.json)、
[WMHD 打分检查](../artifacts/validation/wmhd_final_20261108_oytaqy5m/score_api_check.json)、
[Nakazawa 训练记录](../artifacts/validation/nakazawa_final_20261108_bkaul1mh/training_complete.json)、
[Nakazawa 编码记录](../artifacts/validation/nakazawa_final_20261108_bkaul1mh/valid_codes/metadata.json)。

以上是 CPU 小规模实现检查，**未启动正式 10 epoch 训练、未计算检索评价指标、未提取或匹配 test
特征，也未验证目标 CUDA 硬件**。共享原始身份审计读取完整原始数组不属于 test 特征计算。
没有复现论文原始数据的数值成绩。尤其 WMHD 的 Cartesian–Polar 旋转对齐步骤在原文中未说明，
当前固定方向实现不能称为完整旋转预处理复现；各方法的其他未披露参数见对应 README。

新增方法使用 `load_config(..., require_ks=False)`，不放占位指标字段。旧方法默认校验保持，
但共享文件的源码哈希变化会触发原有签名保护；旧运行产物保留，重新执行须指定新目录。
所有 smoke、审计和证据写入新的独立目录，未覆盖旧运行。

## 目录整理验证（2026-11-08）

公共代码迁入 `shared/`，协议生成入口迁入 `scripts/`，Wu/Yu 及各自配置迁入 `methods/`。
配置内的相对路径仍以仓库根为基准，固定协议和历史运行产物保留原位置。

| 检查 | 实际结果 |
| --- | --- |
| 自动测试 | 65 项全部通过：原 62 项及 3 项目录回归 |
| 异目录启动 | 5 个直接脚本的 `--help` 通过；四份配置仍解析到原数据与固定协议 |
| 模块入口 | 两个生成脚本、Wu、Yu 的 `python -m ... --help` 全部通过 |
| 共享模块身份 | 方法、生成脚本与测试统一导入 `shared.*`，不加载旧顶级模块副本 |
| 真实协议读取 | 从 `/tmp` 准备两方法协议成功，valid/test 的全部 query 与候选仍一致 |
| 原始输入保留 | 原始 NPZ 及全部冻结 CSV/JSON 共 10 个文件的 SHA-256 未变化；四份方法配置逐字节一致 |
| 多进程路径 | Yu 的 2 个 spawn worker 成功读取 4 张 train 图，与串行预处理逐 tensor 一致 |

协议读取与文件核对见 [整合检查](../artifacts/layout_validation_20261108/integration.json)，
多进程结果见 [DataLoader 检查](../artifacts/validation/dataloader_workers_cb_7tgae/dataloader_workers.json)。
多进程检查首次被应用沙箱的共享内存限制拦截，经自动审批在沙箱外运行同一检查后通过。
本次没有启动真实数据训练或完整 test 模型评价。

源码位置与导入变化会改变运行签名；旧运行目录继续受到保护，不改写旧 manifest 来强行续跑。

## 固定检索协议建立时的验证（目录整理前）

验证日期：2026-11-08。固定检索协议建立时的 62 项自动测试全部通过，真实数据的全部
valid/test 候选身份检查通过。下方保留该阶段及更早阶段的原始记录。

在 `compared_methods/` 下执行 `.venv/bin/python build_retrieval_protocol.py`，生成新的
`protocol/retrieval_v1/`，没有重写 `protocol/deduplicated/` 或历史 `protocol/split.csv`。
固定文件为 valid/test 各自的 queries、gallery 四个 CSV，以及记录来源、哈希和规则的
`metadata.json`。候选规则为 gallery 排除当前 query，自身以外的候选全部保留。

| 检查 | 实际结果 |
| --- | --- |
| 自动测试 | 62 项通过：公共协议 20、去重划分 7、固定检索协议 10、Wu 12、Yu 13 |
| valid 分层 query | 恰为 500；37 类全部覆盖；Near-full 5、Random 11、其余 29 类各 14、6 类各 13 |
| valid 完整候选 | gallery 3,599；每 query 3,598；逐一核对 1,799,000 个候选身份 |
| test 全量检索协议 | query/gallery 均为 7,199；每 query 7,198；逐一核对 51,818,402 个候选身份 |
| 方法一致性 | Wu、Yu 所有 query 和所有候选身份的摘要完全一致 |
| 集合与自身隔离 | 无自身、无越界集合候选；六组数据合计 36,991 张唯一原图 |
| 少数类相关候选 | valid 每 query 最少 13 张相关图；test 最少 28 张 |
| smoke 派生池 | 两方法均为正式 query 与 74 张小 valid 的交集：8 个 query，每个 73 个候选；test 明确拒绝 smoke/subset |
| 输入保留 | 原 NPZ、两代已有 split、去重元数据与重复映射、原项目 query/candidate 文件共 8 个保护文件哈希全部不变 |
| 固定性与签名 | 检索 query 不随训练 seed 改变；运行签名包含固定清单及协议源码；文件篡改或旧运行目录混用被拒绝 |
| test 方法入口 | 合成／mock 测试确认显式 test 仅评价 test ID，Yu PCA 只拟合 train，`all` 不进入 test，valid/test 产物隔离 |

完整检查结果见 [integration.json](../artifacts/retrieval_protocol_validation_20261108/integration.json)。
该检查遍历的是候选身份，并未进行五千多万次模型相似度计算。测试中的小模型更新仅用于
恢复机制回归；未启动真实数据训练，也未声称已获得当前协议的模型性能。

候选以线性规模的固定 ID 清单保存，按 query 展开。Wu/Yu 排名逐 query 写出 Top `max(ks)`，
指标分母仍取完整 gallery 的相关样本数，避免在内存中积累完整 test 配对及全部排行。
新方法运行目录为 `*_retrieval_v1`，旧目录保留。Normal 仍按原来的运行时 seed 分配，
此次“检索种子独立”的检查不表示 Normal 分配已经独立于训练 seed。

## 去重划分验证历史（旧随机候选池）

执行命令（在 `compared_methods/` 下）：

```bash
.venv/bin/python build_protocol.py
.venv/bin/python -m unittest discover -s tests -v
```

| 检查 | 实际结果 |
| --- | --- |
| 自动测试 | 42 项通过：公共协议 15 项、去重与划分 7 项、Wu 10 项、Yu 10 项 |
| 全源精确去重 | 38,015 行中有 1,023 个重复组，移除 1,024 行，保留 36,991 张唯一原图 |
| 缺陷分层划分 | 35,991 张唯一缺陷图：train 25,193、valid 3,599、test 7,199 |
| Normal 划分 | 1,000 张均唯一，仍为 train 700、valid 100、holdout 200 |
| 全量集合隔离 | 上述六个集合合计 36,991 张，独立原始像素哈希检查的重复图数量为 0 |
| 两方法协议一致性 | Wu、Yu 的六组 ID 和验证候选池完全相同；500 个 query，各 1,000 个候选，无自身同图或重复候选图 |
| smoke 数据准备 | 两方法均为缺陷 train/valid 370/74、Normal train/valid 32/16；仅准备协议及候选池 |
| 旧协议防误用 | 运行入口拒绝含非代表重复 ID 的旧 manifest；代码签名拒绝复用旧运行目录 |
| 输入保留 | 原 NPZ、旧 `protocol/split.csv` 及 `wbm-region-retrieval` 的原划分文件哈希均未变化 |

去重以完整原图的 0/1/2/3 像素为准，SHA-256 定位候选后通过逐像素相等确认，
每组保留最小原始行 ID。标签冲突时明确报错。去重发生在划分之前，也覆盖 Normal。
新测试覆盖确定性、冲突标签、缺失代表、非代表 ID、Normal 重复、输入不改写、
禁止覆盖既有协议目录及新模块签名等边界。

生成的协议见 [split.csv](../protocol/deduplicated/split.csv)、
[重复记录映射](../protocol/deduplicated/duplicates.csv) 和 [元数据](../protocol/deduplicated/metadata.json)。
真实数据整合检查见 [integration.json](../artifacts/deduplication_validation_20261108/integration.json)。
新 split 的 SHA-256：

```text
2c1d259fec3e541adb9eed7d8042936f04c252d1ec8e4572add9f9ca3f2682d0
```

所有 test/holdout 原图仅参与身份与划分完整性检查，没有用于特征提取、训练或评价。
新配置指向独立的 `*_deduplicated` 输出目录；旧协议的 checkpoint、PCA、候选池和分数
不能直接沿用于新协议。尚未在新协议下重跑真实数据模型 smoke 或正式实验。

## 以下为旧协议历史验证

去重改动前，两种方法已在真实 MixedWM38 小批量数据上跑通，28 项自动测试通过。
以下隔离结论当时针对样本 ID；旧协议含跨集合同图，不能据此声称原始图像已隔离。
旧记录仅验证实现执行与恢复机制，没有开展完整训练、正式性能比较或 test 评价。

## 环境

| 项目 | 实测值 |
| --- | --- |
| 系统 | macOS 27.0.1，arm64 |
| Python | 3.12.14，独立 `compared_methods/.venv` |
| PyTorch | 2.15.0，CPU 执行；CUDA 不可用 |
| NumPy / SciPy | 2.5.4 / 1.18.1 |
| scikit-image / scikit-learn | 0.26.0 / 1.9.1 |
| Pillow | 13.0.0 |

环境由每个输出目录的 `environment.json` 记录。目标远程 CUDA 型号和软件版本尚未知，
没有声称已在目标 GPU 实测。代码支持显式 CUDA、分批推断、可调 batch/worker 和跨设备
checkpoint 加载，远程预检及 CUDA smoke 命令见[总说明](../README.md)。

## 已执行的检查

| 检查 | 实际结果 |
| --- | --- |
| 公共协议 8 项测试 | 通过；覆盖划分隔离、Normal 补充、固定候选池、签名、错误标签与评价分母 |
| Wu 10 项测试 | 通过；覆盖特征维数、几何/Radon/Hough、NCC、粗筛及确定性排序 |
| Yu 10 项测试 | 通过；覆盖完整网络、预处理顺序、增强、PCA256、十值 Hamming 和训练恢复 |
| Wu 真实数据 smoke | 74 张 valid → 74×116 特征；37 个 query，各 73 个候选；生成 2,701 行排名 |
| Wu CPU 并行 | workers=1 与 2 的特征/NCC 缓存逐值相同，排名 CSV/JSON 逐字节相同 |
| Yu 真实数据 smoke | 完整检测器和分类器各更新 8 次；保存 best/last、训练日志和验证结果 |
| Yu 特征与检索 | 用 370 张 train 图拟合 PCA256；global 与 predicted_class 均完成 37×73 检索 |
| Yu 编码器复用 | 再次运行 valid，`pca_cache_reused=true` |
| Yu 完整网络恢复 | 分别在第 4 步中断、恢复到第 8 步；两网络 last 的模型、optimizer、RNG 与连续 8 步逐值一致 |
| Yu 多进程 DataLoader | 2 个 spawn worker 读取 4 张 train 图，训练增强输出与串行逐 tensor 一致 |
| 实际产物隔离检查 | PCA 拟合 ID 恰为 train，Wu 特征 ID 恰为 valid；两方法查询/候选池相同，排名无自身或保留集样本 |

恢复检查比较的是 `last.pt` 的训练状态。中断实验在第 4 步额外进行了一次验证，因此
`best.pt` 的选择和验证 history 可以不同；没有声称它们也完全相同。这里的逐值一致限于
本次相同硬件/依赖的 CPU 实验，不承诺跨 CUDA 硬件逐位一致。

多进程 DataLoader 第一次在应用沙箱中被 `torch_shm_manager: Operation not permitted`
拦截；自动审批允许在沙箱外运行后，同一检查通过。没有为绕过限制改写预处理或禁用多进程。

## 样本范围

原冻结缺陷划分保持 25,910 train、3,700 valid、7,405 test。补充 Normal 为 700 train、
100 valid、200 holdout。`protocol/split.csv` 与原项目逐字节相同，SHA-256：

```text
e1785da387a70da63b65052975872da1377d15f690d4c37c5009840d08444cda
```

本次 smoke 每类只选 10 张缺陷 train、2 张缺陷 valid，总计 370/74；Normal 另选 32/16。
检测器实际 train/valid 为 402/90，分类器为 370/74。训练 batch=2，每个网络仅运行 8 个
optimizer step，`completed_epochs=0`、`truncated_epoch=true`、`training_complete=false`。
用于 PCA 的 370 张图均做关闭增强的特征提取；不要求它们都在这 8 个训练 step 中出现。

未对 7,405 张 test 或 200 张 Normal holdout 做预处理、特征提取、PCA、推断或评价。
原始 NPZ 解压缓存及划分完整性检查涉及原始数组；所有后续计算使用显式 train/valid ID。
未修改 `wbm-region-retrieval`，结束时该项目 Git 工作区仍干净。

## 9×9 滤波的实测影响

按用户确认保留“52×52 原图中值滤波 → 224×224 缩放”的顺序：

| 本次样本 | 数量 | 原始缺陷 die 数 | 滤波后缺陷 die 数 | 缺陷 die 保留率 | 缺陷被全部滤掉的图 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 缺陷 train | 370 | 260,332 | 116,390 | 44.71% | 22 |
| 缺陷 valid | 74 | 52,073 | 22,814 | 43.81% | 4 |

Normal 样本也可能含随机失效 die；本次 32/16 张 Normal 的原有噪声失效点均被滤掉。
以上只描述这批样本，不能外推为全数据统计。没有因审计结果改变滤波核或顺序。
逐图统计见 [Yu 预处理审计](../artifacts/yu_smoke/preprocessing/summary.json)。

## 命令与产物

在外层工作区激活本目录环境后执行：

```bash
python -m unittest discover -s compared_methods/tests -v
python compared_methods/wu/wu.py --config compared_methods/wu/smoke.json --stage all --workers 1
python compared_methods/wu/wu.py --config compared_methods/wu/smoke.json --stage all --workers 2 --output-dir artifacts/wu_smoke_workers2
python compared_methods/yu/yu.py --config compared_methods/yu/smoke.json --stage all --device cpu
python compared_methods/yu/yu.py --config compared_methods/yu/smoke.json --stage valid --device cpu
python compared_methods/yu/yu.py --config compared_methods/yu/smoke.json --stage train --device cpu --output-dir artifacts/yu_smoke_resume --max-steps 4
python compared_methods/yu/yu.py --config compared_methods/yu/smoke.json --stage train --device cpu --output-dir artifacts/yu_smoke_resume --max-steps 8 --resume
python compared_methods/tests/check_workers.py --workers 2
```

以上是已执行的命令；在已有训练目录重复从头跑时，应换新 `--output-dir`。恢复命令保留
原目录并显式加 `--resume`。输出与缓存保存在忽略版本控制的 `artifacts/` 下：

- [整合检查结果](../artifacts/validation/integration.json)、[DataLoader 结果](../artifacts/validation/dataloader_workers.json)。
- [Wu 特征记录](../artifacts/wu_smoke/prepare.json)、[Wu 检索记录](../artifacts/wu_smoke/valid_summary.json)。
- [Yu 检测训练记录](../artifacts/yu_smoke/detector/summary.json)、[Yu 分类训练记录](../artifacts/yu_smoke/classifier/summary.json)。
- [Yu PCA/推断来源](../artifacts/yu_smoke/valid/provenance.json)、[Yu 检索记录](../artifacts/yu_smoke/valid/summary.json)。

未收敛的 smoke 分数仅用于确认报告生成正常，不作为 Wu 与 Yu 的性能比较结论。
