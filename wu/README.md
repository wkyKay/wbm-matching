# Wu 2015：论文方法复现

论文：Ming-Ju Wu、Jyh-Shing R. Jang、Jui-Long Chen，*Wafer Map Failure Pattern Recognition and Similarity Ranking for Large-Scale Data Sets*，IEEE TSM 28(1)，2015。DOI：[10.1109/TSM.2014.2364237](https://doi.org/10.1109/TSM.2014.2364237)。

这里独立实现论文的 WMSR 检索分支，不是作者原始代码。原论文研究 WM-811K；本实现把相同方法用于 MixedWM38，因此不把这里的结果称为原论文结果复现。SVM 分类属于独立的 WMFPR 分支，本任务不训练 SVM。

## 流程与特征

每张图生成一个 116 维向量，查询先按欧氏距离选出最多 500 张候选，再用整图及单一显著区域的 NCC 平均值重排。

每个原图/滤波版本包含 `40 + 18 = 58` 维，按“原图 Radon、原图几何、滤波图 Radon、滤波图几何”的顺序拼接：

| 特征 | 维度 | 定义与顺序 |
|---|---:|---|
| Radon | 40 | sinogram 整体 min-max，沿角度计算各 ρ 的均值和标准差；分别沿 ρ 轴 cubic 重采样为 20 点。先均值 20，后标准差 20 |
| 区域属性 | 14 | 最大面积区域 Sa 的 `area/(πr²)`；最大周长区域 Sp 的 `perimeter/r`；再依次取最大中心距、最小中心距、长轴/r、短轴/r、solidity、eccentricity，每项均按 Sa、Sp 顺序 |
| 缺陷比例 | 3 | 整个有效晶圆、外侧第 1 圈、外侧第 2 圈分别计算缺陷数/有效 die 数 |
| 线性属性 | 1 | Hough 检出的线段数量 |

这里不是第三方分类 Notebook 常用的 59 维特征。两个区域可能是同一个连通区域，仍保留 Table III 要求的重复属性。两条外圈分别取值，不能合并。

## 原文规定与补足约定

原文 pp.3–5 明确 Radon 的 min-max、均值/标准差、20+20 维及 cubic 重采样，区域属性及两个 8 邻域外圈，以及线长大于直径 10%、断裂空隙小于直径 3%。p.7 明确欧氏预筛、NCC 重排、β=0.5。p.8 明确 median 及原图/滤波图双版本共 116 维。

以下没有足够信息严格恢复作者程序，作为本实现公开约定，由 `config.json` 固定：

- 输入值 2 表示缺陷，1/2 表示有效 die；0 和 3 忽略，与冻结数据协议一致。有效区不会被缺陷 median 改写。
- median 使用 3×3、零填充；滤波后再次施加有效区 mask。
- Radon 使用 `0…179°` 的 180 个角度、`circle=False` 保留边缘缺陷；标准差 `ddof=1`；ρ 的完整长度映射到 `[0,1]` 后含端点重采样。极短输入不足四点时退化为线性插值。
- 缺陷区域按 8 连通标记；周长用 scikit-image 的 4 邻域实现。晶圆中心取有效区 bbox 中心，直径取 bbox 高宽的较小值，半径为其一半。同面积/周长时选行优先扫描标记更早的区域。
- Hough 使用 180 个 `[-90°,90°)` 角度；相对峰阈值 0.3、最多 20 峰、ρ 方向最小峰间距 2、角度最小峰间距 5。每个峰收集距离该直线不超过 0.5 像素的缺陷点，沿线投影排序。投影点按单位像素宽度计算空隙，空隙达到直径 3% 时断开；每段端点距离严格大于直径 10% 才计数。不同峰的有效段分别计数。该确定性实现不使用概率 Hough，也不等同于 MATLAB 未公开的峰提取设置。
- 不添加 z-score、L2 向量归一化、PCA 或其他特征权重。欧氏距离直接使用上述论文内归一化后的 116 维数值。
- 最终粗筛数量 `n=500`，独立于评估的 k；候选少于 500 时取全部。原文没有给足该 n 的设置。
- NCC 使用 52×52 画布；其他尺寸仅用最近邻调整整张画布。显著区域选择原始缺陷 mask 中的最大面积单一连通区域，保留其在整张晶圆中的坐标，其他位置置零。原文对显著区域的完整构造细节交代不足，这是一项补足约定。
- 不独立 crop 区域，不搜索平移或旋转，不取滑动相关的最大值，也不按 MixedWM38 标签拆成多个区域进行最佳匹配。

## 数值与排序

NCC 按论文式（24）实现为两个中心化单位向量的内积；最终分数是 `0.5 × 整图NCC + 0.5 × 显著区域NCC`。任一输入方差为零时，对应 NCC 为 0；两个空图不会被特判为完美匹配。所有空区域特征为 0；常量 sinogram 的 Radon 特征为 0；不允许 NaN/Inf 进入缓存或排序。

预筛按 `(欧氏距离升序, 样本ID升序)`；重排按 `(NCC分数降序, 欧氏距离升序, 样本ID升序)`。不参与预筛的候选不补排到结果末尾。公共评估对超出已返回排名的 P@K 位置按未命中处理。

## 冻结数据协议

使用外层 `compared_methods/common.py` 和 `protocol/split.csv`。只有冻结的 train/valid 可用，没有 test CLI 或 test 分支。Wu 无拟合过程，默认只为显式 `valid_ids` 提取特征；不会为了凑出训练阶段读取 train 图像，也不为 Normal 类额外提取特征。验证查询和候选均由公共验证池生成；本方法不自行采样、不使用标签重排、不调用项目覆盖指标。

正式配置使用 500 个验证查询，每个查询 1000 个候选；指标 k 为 1、5、10、20、50、100、500。小样本配置每个缺陷类只取 2 个 valid 样本，使用 37 个查询、每查询 73 个候选；仍保留 n=500，因此实际全体 73 个候选进入重排。小样本配置中的 train/Normal 数量用于统一公共数据协议，Wu 无需计算这些图像特征。

## 命令

先按[总说明](../README.md)安装独立依赖环境，再在外层工作区执行：

```bash
python compared_methods/wu/wu.py --config compared_methods/wu/config.json --stage audit
python compared_methods/wu/wu.py --config compared_methods/wu/smoke.json --stage all --workers 1
python compared_methods/wu/wu.py --config compared_methods/wu/config.json --stage prepare --workers 4
python compared_methods/wu/wu.py --config compared_methods/wu/config.json --stage valid
```

`--stage audit` 仅审计协议；`prepare` 提取特征；`valid` 使用已有缓存检索和评价；`all` 依次执行。没有训练阶段。`--output-dir` 可覆盖输出位置，相对路径按 `compared_methods/` 解释；更换配置或代码后使用新的输出目录，避免混用旧缓存。

输出包括公共运行签名/环境记录、`audit.json`、`prepare.json`、`feature_ids.npy`、`features.npy`、`ncc_full.npy`、`ncc_region.npy`、`rankings.json`、`rankings.csv`、公共评价文件和 `valid_summary.json`。特征缓存使用 float32；距离用 float64 计算。NCC 向量缓存避免逐候选重复中心化。`--workers` 仅控制 CPU 特征提取进程数，不启用 GPU。

可单独验证的纯数组函数包括 `wafer_masks`、`radon_features`、`regional_features`、`statistical_features`、`hough_line_count`、`extract_features`、`largest_region`、`ncc_vector`、`ncc_score` 和 `rank_candidates`。

本地真实数据 smoke、单/双进程一致性检查已通过，范围与证据见[验证记录](../VALIDATION.md)。
