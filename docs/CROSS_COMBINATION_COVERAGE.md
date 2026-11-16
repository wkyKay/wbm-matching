# 排除同组合的 coverage 实验

日期：2026-11-10。协议标识：`coverage_cross_combination_v1`。

用户确认仅排除与 query **完整八维标签组合相同**的候选，保留跨组合完整覆盖、部分覆盖与零覆盖。例：query `{A,B}` 删除 `{A,B}`，保留 `{A,B,C}`（coverage=1）、`{A}`（0.5）、`{C}`（0）。不删除原图，不修改冻结 query/gallery CSV，不改变模型或训练。

这是原固定检索协议的独立派生实验。真实标签只用于评价器定义候选资格；各方法不接收真实标签打分。该条件依赖基准真值，应解释为跨组合检索诊断，不声称是部署时无标签可直接执行的候选过滤功能。它使用既有 test，属于新增分析实验，不能追溯表述为原 test 前已预注册的主评价。

## 候选、排序和指标

- 原始 query 身份全部保留。每个 query 的候选为原固定 gallery 除自身，再排除所有同完整组合候选。
- **先过滤，再评分/粗筛/门控/排序**。Wu 仍按欧氏距离粗筛最多 500，再 NCC 重排；此处必须重新粗筛，不能简单裁剪旧 Top-500。
- Yu 主比较仍为 `global`；Nakazawa 使用最后一轮；F 固定原 `best_nce`。所有模型无需为此重训。WBBS/WMHD 保留原公式。
- 同分规则、K=1/5/10/20/50/100/500 不变。新实验拒绝 K 超过任一 query 的剩余候选数量；可为小型 valid smoke 显式选更小 K。
- coverage 六项公式不变，版本仍为 `coverage_v1`，报告 `metadata.derived_protocol.protocol_version` 区分新候选条件。IDCG 和 FullCoverage Recall/AP 分母使用**过滤后的完整候选库**，不使用原 gallery 或 Top-K 数量。
- query 无完整覆盖候选时，不删除 query：FullCoverage Recall/AP 未定义并排除对应均值；Hit/Precision 仍计入。部分覆盖仍参与 nDCG/MeanCoverage。IDCG=0 时 nDCG 排除均值。有效计数、single/mixed、标签数分组、组合类别宏平均保持原规则。
- 因 exact 正样本已被全部排除，此模式只输出 coverage 指标，不生成无意义的 exact 指标。

2026-11-10 对本地冻结清单及标签进行身份审计（不是模型性能评价）：

| 集合 | query 数 | 每 query 剩余候选数范围 | 无跨组合完整覆盖候选的 query 数 |
| --- | ---: | ---: | ---: |
| valid | 500 | 3,499–3,585 | 72 |
| test | 7,199 | 6,999–7,170 | 999 |

这意味着 test 的 FullCoverage Recall/AP 有效 query 数与总数不同。不能把这 999 个 query 删除后再对所有指标统一平均。候选身份摘要（逐 query 的 `[query_id, eligible_ids]` 紧凑 ASCII JSON，每条追加换行，SHA-256）：

- valid：`e9e865ff288db79bf622f5016c17ac22d61d5ca071d39893a6202b9b0dd013b7`
- test：`68c6875330ed70841586c654da776f8d0bc72bcd4cd341b79cea29705543f0b8`

## 五个对比方法

从 `compared_methods/` 运行。使用新输出目录；以下为命令模板，本次没有执行正式 test。

```bash
python scripts/evaluate.py --config methods/wu/config.json --split test --exclude-exact --output-dir artifacts/cross_combination_v1/wu_test --workers 4
python scripts/evaluate.py --config methods/yu/config.json --split test --variant global --source-run <YU_RUN> --exclude-exact --output-dir artifacts/cross_combination_v1/yu_test --device cuda
python scripts/evaluate.py --config methods/wbbs/config.json --split test --exclude-exact --output-dir artifacts/cross_combination_v1/wbbs_test --workers 4 --rank-workers 4
python scripts/evaluate.py --config methods/nakazawa/config.json --split test --source-run <NAKAZAWA_RUN> --exclude-exact --output-dir artifacts/cross_combination_v1/nakazawa_test --device cuda
python scripts/evaluate.py --config methods/wmhd/config.json --split test --exclude-exact --output-dir artifacts/cross_combination_v1/wmhd_test --workers 4 --rank-workers 4
```

`<YU_RUN>`、`<NAKAZAWA_RUN>` 是原有完成训练并通过签名验证的运行目录，不是旧评价目录。不传 `--exclude-exact` 时保留原统一评价行为。先用对应 smoke 配置、`--split valid --ks 1 5 10` 检查实际环境；smoke 不可运行 test。

## F 主方法

独立脚本放在 `compared_methods/scripts/`，避免改动被 F 训练签名覆盖的源文件。读取原 suite manifest、缓存哈希、训练完成记录及 `best_nce.pt`，不调用训练、不调用会准备/写入来源目录的函数。

```bash
python scripts/evaluate_f_cross_combination.py \
  --f-project /data/wangkaiyan2024/wbm-region-retrieval \
  --source-run /data/wangkaiyan2024/wbm-region-retrieval/artifacts/formal_f_exact_20261109_132219/run \
  --split test --device cuda \
  --output-dir artifacts/cross_combination_v1/f_test
```

`--f-project` 默认相邻的 `wbm-region-retrieval/`。`--source-run` 指向有 `suite_manifest.json`、`cache/`、`seed_2026/F/` 的 suite 根目录。可用 `--config` 指定对应的 F 配置，默认当前 F 正式配置。任一来源签名不一致会拒绝运行，不允许重写 manifest 绕过检查；该脚本不是旧未去重模型迁移工具。

F 使用与既有实现相同的归一化 embedding、贪心一对一匹配和 query 面积平方根加权，只对符合新候选条件的图像输出分数/排名。对原缓存重新推理提取 embedding，不依赖被截断的旧排名。其冻结清单会与 `compared_methods/protocol/` 逐字节核对。

## 输出和验证

- 五方法：`OUT/protocol/cross_combination_candidates.csv`；F：`OUT/cross_combination_candidates.csv`。逐 query 保存原候选数、排除数、剩余数。
- `OUT/evaluation_manifest.json` 绑定原协议、派生规则、候选身份摘要、源码/模型及 K；五方法的 `candidates_per_query` 在本模式代表最小值，真实范围见 `derived_protocol` 和逐 query CSV。
- `OUT/{valid,test}/rankings.csv` 仍只保存 Top max(K)。同目录生成 `coverage_metrics.json`、`coverage_per_query.csv`、`coverage_per_class.csv`。
- 仅有 `completed.json` 的输出可视为完成。异常保留 `failed.json`，换新输出目录重跑；不覆盖历史产物。
- 新增测试覆盖同组合过滤、保留超集/部分/零覆盖、过滤后的 IDCG/AP/Recall、零完整覆盖、K 边界、真实 Wu 粗筛与 WBBS 多进程排序、F 独立入口全流程及来源字节/mtime 不变。F 测试使用合成 checkpoint/特征和模拟准备数据，不代表正式 CNN 性能或远程运行验证。

实现：`shared/cross_combination.py`、`shared/evaluation.py`、`scripts/evaluate.py`、`scripts/evaluate_f_cross_combination.py`。正式性能结果尚未产生，不能提前推断哪种方法在新条件下最好。

本地验证：`python -m unittest discover -s compared_methods/tests -q` 全套 **166 项通过**，其中新增跨组合测试 6 项。原始协议文件及 F 训练代码未修改；本次未连接远程、未同步或执行正式 test。
