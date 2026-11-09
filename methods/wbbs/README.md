# WBBS（Wang & Wang，2023）

对应 survey 编号 1：*Similarity searching for fault diagnosis of defect patterns in wafer bin maps*，
DOI [10.1016/j.cie.2023.109679](https://doi.org/10.1016/j.cie.2023.109679)。这是根据论文重实现的
**无训练方法**，不是作者发布的代码。方法自身 CLI 保留 audit/prepare；
五方法统一 valid/test 及完整组合指标由独立入口提供，见[统一评价说明](../../docs/EXACT_EVALUATION.md)。
该入口对完整固定候选打分，用 SHA-256 处理同分，不改变 WBBS 公式或张量投票设置。

实现使用同一份 MixedWM38 原图、去重划分和固定检索清单，保持 37 种完整缺陷组合。
原文合并 Edge-Loc/Edge-Ring 得到 25 类的处理不在这里使用。输入只将像素 2 视为缺陷，
1/2 定义晶圆支撑，0/3 忽略。不重写原图或任何划分。

## 原文公式与实现

1. §2.1，式 (2)、(6)–(10)：每个缺陷点初始化为单位张量，累加其他缺陷点的闭式张量投票，
   衰减为 `exp(-||xi-xj||² / sigma_d)`，得到 `lambda_s=lambda1-lambda2` 和 `lambda_b=lambda2`。
   代码按块计算所有点对，没有抽点或邻域截断。对于 `Kj=I`，
   `R @ Kj @ (I - 0.5*r*r.T) @ R.T` 严格等于 `I - 0.5*r*r.T`；使用此代数恒等式加速，
   不是用局部点密度替代张量投票。
2. §2.1：只有 **stick 与 ball 同时严格小于各自最大值的 0.32 倍**时才去除该点。
   不是将两个条件取 OR，也不额外加入中值滤波或连通域去噪。
3. §2.2，式 (15)–(16)：点描述为归一化 `[stick, ball, rho]`，距离为
   `Δstick² + Δball² + mu*Δrho²`。原文位置描述仅有到晶圆中心的半径 `rho`，不包括方位角。
4. 式 (11)–(14)：统计互为最近邻的点对数 `B`，输出
   `B / min(N,M) * exp(-alpha*abs(N-M)/l)`，其中 N、M 是去噪后点数。
   **分母是 `min(N,M)`，不是 max，也不是仅按 query 点数归一化。** 原始 WBBS 分数是对称的，
   越大越相似；不加入 survey 建议的方向性变体。

主论文 PDF 已逐页核对 §2.1–§3 的公式和参数：
[本地论文](../../../related_works/Wang和Wang%20-%202023%20-%20Similarity%20searching%20for%20fault%20diagnosis%20of%20defect%20patterns%20in%20wafer%20bin%20maps.pdf)。

## 论文参数与未披露的实现约定

| 设置 | 当前值 | 依据 |
| --- | --- | --- |
| `noise_threshold` | 0.32 | 2023 原文 §3 |
| `alpha` | 5 | 2023 原文 §3 |
| `mu` | 1 | 2023 原文 §3 |
| `sigma_d` | 9.0 pixel² | **未核定的实现约定**；2023 本文没有给数值 |
| 投票次数、初始化 | 一次、全部 `Kj=I` | 初始化来自 §2.1；一次投票且排除自身是当前实现约定 |
| saliency 归一化 | 每图、逐通道除以去噪前最大值；全零通道置零 | 原文只要求映射到 [0,1]，未交代范围和常数通道处理 |
| rho 归一化 | 支撑框中心为原点，除以支撑点最大半径 | 原文只给径向距离和 [0,1] 范围；具体几何约定未披露 |
| `total_dies`（l） | 固定 2061 | MixedWM38 适配约定；常见支撑包含 2061 个 die，但部分原图的支撑数量不同 |
| 最近邻距离完全同分 | 另一点集中最小行号 | 原文没有 tie 规则；点按图像行列顺序排列 |
| 空点集 | 与任意图得分为 0 | 原式含 `1/min(N,M)`，此为未定义边界的实现约定 |
| `block_size` | 256 | 仅改变计算内存，不改变公式 |

**复现边界：** 2023 论文将详细投票设置指向 Wang & Chen (2022)，
[DOI 10.1109/TSM.2022.3183008](https://doi.org/10.1109/TSM.2022.3183008)；本地未有该文全文。
另一个可读前作 *Tensor Voting Based Similarity Matching...*（DSIT 2022）§III 给长度尺度
`sigma=3`，但采用不同的衰减表达式。当前 `sigma_d=3²=9` 仅是明确记录的尺度约定，
**不能据此声称已经恢复 2023 作者的精确默认参数或原文结果**。后续拿到上述引用全文后应核定，
如果改动参数或源码，须使用新运行目录。

径向描述可能令不同点具有完全相同的三维特征。单值 `argmin` 只选择一个最近邻，所以即使两张
图完全相同，WBBS 也不一定等于 1（例如三个完全相同描述只形成一个 BBP，得分为 1/3）。
代码不通过多对匹配或强制 self=1 改变原式；数学单测覆盖该情况。

## 运行

在 `compared_methods/` 中执行（也可以从其他目录以绝对脚本路径执行）：

```bash
.venv/bin/python methods/wbbs/wbbs.py --stage audit
.venv/bin/python methods/wbbs/wbbs.py --stage prepare --workers 1
```

小规模连通性检查：

```bash
.venv/bin/python methods/wbbs/wbbs.py --config methods/wbbs/smoke.json --stage prepare \
  --output-dir artifacts/wbbs_smoke_check_NEW
```

`audit` 读取并检查固定数据及检索协议，保存运行签名、数据审计和 query/gallery 身份。
`prepare` 缓存完整 valid 的单图特征；正式配置为 3599 张，smoke 是每类 2 张共 74 张的派生调试集。
smoke 没有重新抽取查询，仍取固定 500 个 valid query 与该子集的交集。
上述方法 CLI 无模型训练、无标签驱动选参，不计算 test 特征或检索指标；正式参数调优尚未进行。
统一入口另行显式选择 `--split valid/test`，必须使用新评价目录；smoke 只允许 valid。

运行结果默认位于 `artifacts/wbbs_retrieval_v1/`：

- `run_manifest.json`：数据、代码、配置、固定协议签名；配置或源码变化时拒绝混用旧结果。
- `audit.json`、`protocol/`：协议核查与原始 `sample_id` 身份信息。
- `features.npz`：`ids`、`offsets`、`points`、`descriptors`、`original_counts`。
  每张图的点数可变，由 offsets 索引；不用 pickle/object 数组。
- `prepare.json`：特征哈希、参数、点数、耗时及 `test_accessed=false`、`metrics_computed=false`。

相同签名已完成的缓存会校验哈希后复用，不重算、不覆盖。损坏、不完整或签名不匹配时要求新目录。

## 单图与候选打分接口

公共函数均在 `wbbs.py`：

```python
from methods.wbbs.wbbs import encode_wafer, score_pair, score_candidates

query = encode_wafer(query_map)
candidates = [encode_wafer(array) for array in candidate_maps]
scores = score_candidates(query, candidates)  # 顺序对应输入，分数越大越相似
```

指定非默认参数时，预处理与打分必须传入同一份 `settings`。缓存会同时保存这些参数：

```python
from pathlib import Path
from shared.common import load_config, ensure_run, prepare_data, make_retrieval_pool
from methods.wbbs import wbbs

config = load_config('methods/wbbs/config.json', require_ks=False)
ensure_run(config, [Path(wbbs.__file__)])
bundle = prepare_data(config)
pool = make_retrieval_pool(bundle, config, split_name='valid')
store = wbbs.load_prepared(config['output_dir'])
query_id = int(pool.queries[0])
ranked_ids, scores = wbbs.rank_from_pool(store, pool, query_id)
```

`rank_from_pool` 对一个 query 的完整固定候选集打分，排除自身；同分按原始 `sample_id` 升序。
它不计算标签相关性或任何评价指标。现有统一入口调度同一 `encode_wafer` /
`score_candidates` 接口并读取既有清单，最后使用统一哈希排序；不直接沿用此函数的 ID 并列结果。
方法 CLI 仍只开放 audit/prepare。
逐对精确匹配的时间随缺陷点数增长，完整 test 需要五千多万对；本轮不默认执行该计算。

## 检查

```bash
.venv/bin/python -m unittest discover -s tests -p test_wbbs.py -v
```

覆盖闭式公式与独立矩阵计算一致、投票 self 排除、严格双条件去噪、输入码值、径向旋转、
min 点数归一化、数量差异惩罚、重复特征 tie、空集、分数边界/对称性、原始 ID 缓存和完整池身份。
这些检查不是论文检索成绩复现。
