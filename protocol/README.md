# 去重划分与固定检索协议

当前 Wu/Yu 正式及 smoke 配置使用 `deduplicated/split.csv`。原始 NPZ 不改写，样本 ID
始终等于原始 NPZ 行号；不把去重后的顺序重新编号为样本 ID。
正式检索另外读取同一份 `retrieval_v1/`；它只定义 query/gallery，不改变已有 train/valid/test 归属。
五方法现在可通过独立 `scripts/evaluate.py` 统一评价完整组合相关性，
命令和指标规则见[统一评价说明](../docs/EXACT_EVALUATION.md)。冻结清单保持原样。

## 生成规则

`../shared/data_protocol.py` 对整个源数组先做精确内容去重，再划分缺陷样本：

1. 使用完整晶圆原图的全部像素值，0/1/2/3 均参与判断；SHA-256 筛选后逐像素确认。
   不先滤波、不仅比较缺陷 mask，也不按旋转、翻转或近似外观合并。
2. 每个重复组仅保留最小原始 ID；保留 ID 与被过滤 ID 均可追溯。Normal 与缺陷图一起
   检查，任何同图不同标签均报错，不通过多数投票或忽略标签解决。
3. 对保留的非空标签图，按完整 multi-hot 组合分层，以 seed=2026 按 70%/10%/20%
   生成 train/valid/test。每类 train/valid 数量向下取整，剩余归 test。

当前源文件有 38,015 条记录。过滤 1,024 条重复记录后，保留 **36,991 张唯一图**，
其中缺陷图 35,991 张、Normal 1,000 张。37 类缺陷的新划分为：

| 集合 | 图数 |
| --- | ---: |
| train | 25,193 |
| valid | 3,599 |
| test | 7,199 |

Normal 当前没有重复，每次运行按 seed=2026 独立划分为 700 train、100 valid、200 holdout。
Normal 只用于 Yu 检测器；如果其他数据版本去重后的 Normal 数量与 `normal_split` 配置
不符，运行会报错，需显式确定新数量，不静默沿用或缩放。
Normal 分配仍依赖方法配置中的 `seed`；保持上述 Normal 协议时须沿用 2026。
检索 query/gallery 已存为固定清单，后续更换训练随机种子不会使它们重新抽样。

## 文件与重建

| 文件 | 用途 |
| --- | --- |
| `deduplicated/split.csv` | 去重代表的冻结缺陷划分；保留原始 `sample_id`、`original_index` 和源非空序列的 `valid_index` |
| `deduplicated/duplicates.csv` | 每条被过滤记录与其保留代表 ID 的对应关系 |
| `deduplicated/metadata.json` | 源文件 SHA-256、去重规则、种子、计数及生成文件哈希 |
| `retrieval_v1/valid_queries.csv` | 固定 500 个分层 valid query ID |
| `retrieval_v1/valid_gallery.csv` | 完整 3,599 个 valid 候选库 ID |
| `retrieval_v1/test_queries.csv` | 全部 7,199 个 test query ID |
| `retrieval_v1/test_gallery.csv` | 完整 7,199 个 test 候选库 ID |
| `retrieval_v1/metadata.json` | 原始数据与 split 哈希、四个清单哈希、分层配额、排除自身规则及协议签名所需信息 |
| `split.csv` | 旧协议原样保留，仅供历史追溯，当前配置不使用 |

正常运行直接使用仓库提供的新清单，不会自动重划。如果需要重新生成，使用显式入口，
并指定一个尚不存在的目录。例如从外层工作区执行：

```bash
python compared_methods/scripts/build_protocol.py \
  --data-file data/wm38k/Wafer_Map_Datasets.npz \
  --output-dir compared_methods/protocol/deduplicated_rebuild \
  --seed 2026
```

`--data-file` 和 `--output-dir` 的显式相对路径按当前 cwd 解析；省略时分别默认使用
仓库同级的 `data/wm38k/Wafer_Map_Datasets.npz` 和本仓库的 `protocol/deduplicated/`。
输出目录已存在时拒绝覆盖。使用新生成的目录时，应复制方法配置，将 `split_file`
指向该目录中的 `split.csv`，并为运行指定新的 `output_dir`。方法配置内的相对路径仍
以 `compared_methods/` 为基准。

固定检索协议使用独立入口，读取已有去重划分。例如从外层工作区生成一份核对副本：

```bash
python compared_methods/scripts/build_retrieval_protocol.py \
  --data-file data/wm38k/Wafer_Map_Datasets.npz \
  --split-file compared_methods/protocol/deduplicated/split.csv \
  --output-dir compared_methods/protocol/retrieval_v1_rebuild \
  --seed 2026
```

默认正式输出目录为 `compared_methods/protocol/retrieval_v1/`。输出已存在则拒绝覆盖；
正常运行直接读取仓库提供的清单，不必再次执行生成器。若确需使用新版本，修改方法配置的
`retrieval_protocol_dir` 并指定新的运行 `output_dir`。生成检索清单不会改写源 NPZ 或 split。

## Query 与候选的固定规则

valid 的 500 个 query 按完整 multi-hot 组合分层：先依据各类别占比确定配额，再施加
每类至少 5 张的约束（类别不足 5 张则全取），通过确定性的受约束余数分配使总数恰为 500。
在各类内部用 seed=2026 无放回抽样并保存 ID。这里保持类别代表性，但不是每类等量抽样。
当前配额为 Near-full 5 张、Random 11 张，其余 35 类中 29 类各 14 张、6 类各 13 张；
各类的具体配额与标签签名保存在元数据中。

| 检索集合 | Query | Gallery | 每个 query 的候选数 |
| --- | ---: | ---: | ---: |
| valid | 分层固定 500 | 全部 valid：3,599 | 3,598 |
| test | 全部 test：7,199 | 全部 test：7,199 | 7,198 |

候选规则始终为 `gallery 中 sample_id != query_id 的所有样本`。query 之间可以互为候选，
自身必须排除；所有图均来自去重后的代表。四个 ID 清单与该规则完整确定每个 query 的候选，
不需要展开保存 51,818,402 行 test 候选对。共享加载器按 query 惰性生成候选，避免二次方内存。

所有方法必须共用该目录，不能在各自运行中重新采样。加载时校验源数据、split 和四个 CSV
的哈希，以及 gallery 与冻结集合的完全一致、query 配额和 test 全量覆盖。
若清单缺失、被改写或与配置的 split 不匹配，程序报错，不静默生成替代协议。

smoke 仍在正式划分中取小子集：以正式 query 与 smoke valid ID 的交集作为查询，
smoke valid 全集除自身作为候选。该派生调试池与正式评价分开标识，不能用于 test；
不同方法应使用相同 smoke subset 配置。

## 后续方法接入

新增方法配置沿用以下路径；相对路径均以 `compared_methods/` 为基准：

```json
{
  "split_file": "protocol/deduplicated/split.csv",
  "retrieval_protocol_dir": "protocol/retrieval_v1"
}
```

方法入口使用 `shared.common.load_config`、`ensure_run` 和 `prepare_data` 完成路径解析、运行签名
与数据检查，再调用 `make_retrieval_pool(bundle, config, split_name="valid")` 或
`split_name="test"` 获取固定池。共享配置加载器允许新方法名称；各算法仍需实现自己的入口，
不能仅修改 `method` 名称来运行现有 Wu/Yu 代码。

固定池可按 `for query_id, candidate_ids in pool.items()` 逐查询使用，不能先转换成包含全部
候选对的巨大字典。图像与标签始终通过原始 `sample_id` 索引源数组，不能用 CSV 行号替代。
统一入口使用 `shared/exact_metrics.py`；原 Wu/Yu 的 `shared/common.py` 评价入口继续保留。
排序后调用 `evaluate_rankings(..., split_name="valid")` 或 `split_name="test"`；评价器接受
逐 query 的 `(query_id, ranked_ids)` 迭代器，并检查完整查询覆盖、集合归属、自身排除及候选合法性。
新方法使用独立输出目录，完整正式 valid/test 不允许任意减少 gallery 或替换 query。

## 运行时隔离与评价

每次准备数据都会重新核对原图重复组，要求 manifest 完整覆盖全部非空唯一代表，
拒绝包含重复副本、遗漏代表或标签不一致的划分。训练前做全源图像哈希属于数据审计，
不等于对 test/holdout 提取特征或进行性能评价。原始缓存可包含所有源行，后续方法计算
使用显式集合 ID；只有原方法的显式 `--stage test` 或统一入口的显式 `--split test`
才提取 test 特征并评价。

每个运行输出自身的去重摘要、37 类完整组合映射（仅由 train 建立）、实际 train/valid ID、
固定检索协议的来源信息及 Normal 补充分配。`all` 保持训练/valid 流程，不自动运行 test。
先用 valid 确定算法参数和模型，再显式评价 test；Yu 在 test 阶段仍只用 train 拟合 PCA
和编码缩放，不用 test 选择 checkpoint。Normal holdout 不加入缺陷检索。

统一 exact_v1 与既有 Wu/Yu 的相关性均按完整组合类别相同定义。筛选或门控可以减少某方法返回的结果，
但 AP/Recall 的相关样本分母仍由完整固定候选库确定，不能用方法筛出的子库替换。
exact_v1 增加 Hit@K、query/class 汇总和标签数量分组，使用固定 SHA-256 处理算法分数的剩余并列；
原始 ID 排序的旧结果不与之混比。它不计算覆盖率或 nDCG。统一评价单独签名并写入新目录，
不修改旧训练签名和来源运行；完整 CNN 训练、正式无 subset 配置及 `completed.json` 要求见上述说明。

旧 `split.csv` 来自 `../../wbm-region-retrieval/protocol/split.csv`，SHA-256 保持为
`e1785da387a70da63b65052975872da1377d15f690d4c37c5009840d08444cda`；其原始划分为
25,910/3,700/7,405。新的去重划分改变了训练和验证样本，旧模型、旧缓存和旧指标不能与
新协议混用。去重后的 `deduplicated/split.csv` 在本次检索协议升级中同样保持原样。
旧的随机 500-query/每 query 1,000 候选结果只作历史记录；当前方法配置使用新的输出目录，
运行签名包含协议源码和固定查询/候选清单。
