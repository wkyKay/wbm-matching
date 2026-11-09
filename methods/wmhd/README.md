# WMHD（Hsu et al., 2020）

本目录重实现 [Hsu et al., 2020](https://doi.org/10.1016/j.cie.2020.106358) 的
mountain function 和方向性 WMHD，默认使用论文最终的 **W2-O1**。无需训练。
方法 CLI 保留特征准备和打分范围；五方法统一 valid/test 及完整组合指标见
[统一评价入口](../../docs/EXACT_EVALUATION.md)。该入口保留 query → candidate 方向，以负距离排序，
允许边界情形的 `-inf` 最差分数，并使用 SHA-256 处理同分，不对距离做对称化。

**复现边界：**Eq. (1)–(6) 已实现。论文 §3.2 提及 Cartesian–Polar 坐标变换，
但未说明如何对齐、搜索旋转角度及合并多个姿态的分数。本版采用固定 Cartesian 方向，
没有复现该未披露的旋转对齐步骤，不能称为完整旋转预处理复现。
ESRN 数值设置及 mountain、容差参数也未在 Hsu 2020 中公布；以下逐项区分。
未使用官方源码，未复现原论文的私有数据或数值成绩。

## 方法与来源

代码中的原图解码只把 MixedWM38 的 `2` 视为缺陷，`1/2` 视为晶圆上的真实 die，
`0/3` 作为背景。这不改变中央协议在原始全部像素上精确去重的规则。

| 步骤 | 实现 | 依据／补足约定 |
| --- | --- | --- |
| 去噪 | King-move 八邻域分别判断好／坏 die 比例，支持信号增强与噪声消除 | Hsu §3.2 的 ESRN 文字规则；等权、排除中心／背景、同步一次更新、两阈值 `5/8` 是实现约定 |
| 坐标 | 以真实 die 包围盒中心为原点，最大真实 die 欧氏半径归一为 1，保持原方向 | MixedWM38 适配约定；与缺陷范围、类别和其他数据划分无关 |
| 密度 | `M(i) = Σ_j exp(-m * beta * d(i,j))`；`beta = N / Σ_j d(j,center)` | Eq. (1)；N 是所有真实 die 数，不是缺陷点数；密度用欧氏距离为实现约定 |
| 权重 | W0 为 1；W1 为中心 mountain 值相对差；W2 为中心及八邻域共九个相对差之**和** | Eq. (3)–(5)，不除以 9，不替换成 `1 + weight` |
| 匹配 | query 每点到 candidate 点的加权 city-block 距离取最小，按原始 query 点数平均 | Eq. (2)，方向固定为 query → candidate；不添加对称化 |
| 离群惩罚 | 空间距离严格 `< Sout` 的点才能匹配；未匹配 query 点计入 `No * Sout` | §3.3.2 和 Eq. (6)；按文字排除未匹配点的 h 分子，分母保留 Eq. (2) 的 NA，此衔接是明确的实现解释 |

`mountain_m=0.1` 借鉴 [Lee et al., 2021](https://doi.org/10.1109/TSM.2021.3102679)
§IV-C 的 Hsu 基线设置。该文的 `Sout=0.1*maxd` 在其数据上按 AP 选出；本版将
`matching_tolerance=0.1` 固定为**归一晶圆半径**的 0.1，既不拟合当前 valid，也不从
test 或其他划分读取最大缺陷半径。它们是可溯源的实现起点，不是 Hsu 2020 的已知原始值，
也不是针对 MixedWM38 调优过的参数。所有这些设置写入配置、run 签名及特征 manifest。

其他数值约定：晶圆外的 mountain 值补 0；两个值都是 0 时其相对差定义为 0；只有中心
一个缺陷导致 beta 分母为 0 时使用 `1e-12` 下界；保留所有去噪后的点，不抽样近似。
O1 对任一 query 点在容差内的**全部** candidate 点计算加权最小值，不先用最近空间点代替。
ESRN 后两图都空时距离为 0；仅 query 空时为 `+inf`，仅 candidate 空时 O1 为
`NA*Sout`（O0 为 `+inf`）。这些是原文未定义情形的边界约定，不会回退到原始缺陷点。
准备记录报告 `empty_after_esrn`，统一评价保留该诊断；空 query 的排序没有区分能力。

## 运行

从 `compared_methods/` 运行；配置中的路径均相对该目录解析：

```bash
.venv/bin/python methods/wmhd/wmhd.py --stage audit
.venv/bin/python methods/wmhd/wmhd.py --stage prepare
```

调试协议采用固定 query 与小 valid 子集的交集，只用于接口验证：

```bash
.venv/bin/python methods/wmhd/wmhd.py --config methods/wmhd/smoke.json --stage prepare
```

可用 `--workers 2` 并行单图特征计算。`--output-dir artifacts/wmhd_new_run` 可指定新目录。
源码、参数或数据签名变化时必须换目录；不改旧 manifest 或旧结果。默认输出
`artifacts/wmhd_retrieval_v1/`，smoke 为 `artifacts/wmhd_smoke_retrieval_v1/`。

两种 stage 都读取现有 `protocol/deduplicated/split.csv` 和 `protocol/retrieval_v1/`。
`audit` 检查来源及 valid query/gallery，不生成图像特征；`prepare` 只读取 valid gallery
图像。正式 valid 是固定 500 query、3,599 gallery，每个 query 排除自身。Normal 不用于
WMHD 特征。公共协议审计会加载原始数组、核对各划分 ID，并保存约定的 Normal 划分记录，
这不表示方法使用了 train/Normal/test 图像拟合或评分。

输出包括公共的 `run_manifest.json`、`resolved_config.json`、`environment.json`、
`protocol/` 审计记录（其中 `valid_pool.json` 保存本次 valid 池），以及方法的：

- `audit.json`：参数、固定 valid 规模、复现限制。
- `features.npz`：`sample_ids` 为原始 NPZ 行号；`offsets` 划分各图的 `points` 与
  `neighborhoods`；`scalars` 每行依次为归一半径、beta、原始缺陷数、真实 die 数。
- `prepare.json`：数据／协议／参数签名、特征文件 SHA-256、空图数量和耗时。

上述方法 CLI 不产生检索指标或 test 结果；统一入口须另行显式选择 `--split valid/test` 和新输出目录。
正式全候选逐对 WMHD 有较高计算成本，单图缓存
不会免除后续点集匹配成本；打分按候选迭代、query 点分块，不构建整个 test 分数矩阵。

## 单图与候选打分接口

```python
from methods.wmhd.wmhd import extract_features, distance, score_candidates, load_features

# 输入原始 MixedWM38 数组，不需要标签；settings 使用正式配置的 parameters。
query = extract_features(query_wafer, settings)
candidate = extract_features(candidate_wafer, settings)
d = distance(query, candidate, settings)     # 越小越相似
scores = score_candidates(query, candidate_features, settings)  # -d，越大越相似

# 已准备的 valid 缓存以原始 sample_id 为键，可复用同一个 feature 多次比较。
features = load_features(output_dir)
candidate_ids = frozen_pool[query_id]         # 来自 shared.make_retrieval_pool，排除自身
scores = score_candidates(features[query_id],
                          (features[int(i)] for i in candidate_ids), settings)
```

`rank_candidates(query, candidates, candidate_ids, settings)` 返回全部传入候选的排序 ID
及分数；同分按原始 ID 升序排列，不使用标签打破同分。可额外传入 `query_id=...` 来
强制移除自身。现有统一驱动调用单图／点集打分接口，再按统一哈希规则排序，
不沿用原始 ID 并列规则；本目录没有自行构造 test query/gallery 或重新采样的逻辑。

O1 的计算只对 `L1 < matching_tolerance` 的有效点对累加 W0/W1/W2 权重，再取每个
query 点的最小加权距离。距离矩阵、严格阈值、九项权重累加顺序、原始 query 点数分母
和 outlier 惩罚保持原式；O0 仍计算所有点对。这是等价的稀疏计算，所有候选图仍参与评分。
已用密集公式回归及真实 valid 点数分位数样本确认分数逐位一致，详见
[验证记录](../../docs/VALIDATION.md)。

本地原文依据：`related_works/Hsu 等 - 2020 - Similarity matching of wafer bin maps for manufacturing intelligence to empower Industry 3.5 for sem.pdf`
§3.2–3.3、Eq. (1)–(6)、§4.2；参数补充来源为同目录的 Lee 2021 全文 §IV-C。
