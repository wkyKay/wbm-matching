# Yu 2019：MixedWM38 适配

来源：Yu、Xu、Wang，*Wafer Defect Pattern Recognition and Analysis Based on Convolutional Neural Network*，DOI `10.1109/TSM.2019.2937793`。本实现保留检测、分类、Fc2 特征和编码检索流程；将八种单缺陷分类改为 MixedWM38 的 37 种非空完整组合类别。它不是八标签 BCE 模型，也没有覆盖率训练损失。

当前命令只允许 train/valid；不会训练、评估或拟合 test/Normal holdout。共享协议复制已有冻结划分，Normal 另行固定划为 700 train、100 valid、200 holdout；不改原项目的 split。

## 网络与训练

| 项目 | 检测器 | 分类器 |
| --- | --- | --- |
| 任务 | Normal／有缺陷，两类 CE | 37 种完整组合类别，单标签 CE |
| 卷积组 | 32、64、64，每组一个卷积 | 32×2、64×2、128×3 |
| 卷积 | 3×3，stride 1，padding 1，ReLU | 同左 |
| 每组后的池化 | MaxPool 3×3，stride 3，无 padding | 同左 |
| 空间尺寸 | 224→74→24→8 | 同左 |
| 全连接层 | 4096→256→2 | 8192→4096→1024→37 |
| 隐层激活／dropout | Sigmoid；256 维层 dropout 0.5 | Sigmoid；4096、1024 维层 dropout 0.5 |
| 正式 batch size | 64 | 128 |
| 正式 epoch 预算 | 10 | 80 |
| 损失 | CE | CE + 10⁻⁶·Σw²，权重矩阵不含 bias |

Adam 学习率 10⁻⁴，betas=(0.9,0.999)、eps=10⁻⁸、weight_decay=0；不使用 AdamW、预训练权重或学习率调度。CE 直接接收 logits，避免重复 Softmax。10／80 epoch 来自原论文训练曲线的展示范围，不表示原文规定了早停协议。本实现用最低 valid CE 选择 checkpoint，同分保留较早模型；两个模型独立选模。

随机初始化采用 PyTorch 各层默认初始化。池化 stride=3 根据原图的尺寸变化确定。Adam 的 betas/eps、裁剪幅度、bias 不做 L2、选模规则均是明确的复现约定。

## 输入与增强

原始图使用 0=晶圆外、1=正常芯片、2=失效芯片；原数据中的值 3 按共享协议视为无效位置，先映射为 0。按用户确认，**先在 52×52 原图做 9×9 中值滤波，再 nearest 缩放为 224×224**，不调小滤波核。滤波使用 nearest 边界扩展，保留原 wafer mask，晶圆内滤波后变成 0 的位置恢复为正常芯片。

RGB 固定为白背景 `(255,255,255)`、青色正常 `(0,255,255)`、品红缺陷 `(255,0,255)`；使用 `I/127.5−1` 得到 [-1,1]。颜色映射和 mask 处理是实现约定；原文只给图示而未提供可执行预处理。原文文字称 [0,1]，但公式对应 [-1,1]，这里明确采用公式。

仅训练时使用 [0°,360°) 随机旋转和面积比例 [0.95,1.0]、正方形的随机裁剪后缩放；nearest 插值，旋转填充晶圆外值。随机增强由 seed、epoch、sample_id 确定。裁剪范围是原文未给出的复现默认；未加入翻转或额外噪声。验证、PCA 拟合和检索全部关闭随机增强。9×9 滤波可能消除原始细线，该设置在本轮按确认方案保留，不根据 valid/test 结果暗中调整。

## 特征与检索

1. 分类器切换到 eval，提取 FC2 的 Sigmoid 后 1024 维特征，关闭所有 dropout。
2. 仅在当前 train 中全部缺陷样本拟合 PCA256，whiten=False；Normal、valid、test 均不参与拟合。
3. 按 PCA 每一维的 train 最小／最大值缩放；常数维置 0，对后续数据 clip 到 [0,1]。若已保存编码器的 checkpoint 签名和 train 拟合 ID 完全匹配，验证阶段复用它，否则只用 train 重新拟合。
4. 每维编码为 `min(floor(10*v),9)`，得到 256 维十值整数向量。
5. 距离为标准 categorical Hamming：256 个位置中编码不相等的数量；距离越小排名越高，同距离按 sample_id 升序。

原文十值编码的 `⊕` 未说明具体实现。本实现采用常用的 categorical Hamming，不作整数位运算 XOR。PCA 使用固定种子的 randomized solver；min-max 的拟合范围是为避免信息泄漏明确补充的协议。

验证分别保存两个结果目录，不混为一个分数：

- `valid/global/`：主结果；共享冻结 valid 查询／候选池，每个候选都参与 Hamming 排序，不经过检测或类别门控。
- `valid/predicted_class/`：完整串联的辅助结果；query 预测为 Normal 时返回空，否则保留检测为缺陷且预测组合类别与 query 相同的候选。query 与候选都只使用模型预测门控，valid 真标签只交给评价器。

评价相关性采用完整组合类别相同，不是类别交集或覆盖率。共享评价器处理候选不足 K、空返回和自身排除；分支保留原固定候选池的相关样本总数，不因门控缩小分母。该预测门控是可执行适配约定，并非原文充分报告了每个细节。

## 命令

先按[总说明](../README.md)安装依赖，再从外层工作区执行：

```bash
python compared_methods/yu/yu.py --config compared_methods/yu/config.json --stage audit
python compared_methods/yu/yu.py --config compared_methods/yu/smoke.json --stage all
python compared_methods/yu/yu.py --config compared_methods/yu/config.json --stage train --device cuda
python compared_methods/yu/yu.py --config compared_methods/yu/config.json --stage train --device cuda --resume
python compared_methods/yu/yu.py --config compared_methods/yu/config.json --stage valid --device cuda
```

`audit`／`prepare` 只准备并审计协议；`train` 训练两个网络；`valid` 使用各自最优 checkpoint、拟合 train PCA 并评价 valid；`all` 串联训练和验证。若目录已有训练 checkpoint，训练必须显式 `--resume`。配置、方法源码或数据协议签名改变时使用新输出目录。

`--max-steps N` 是每个网络的累计 optimizer-step 上限，适合中断恢复验证。恢复时从已记录的 epoch、批次位置和固定随机顺序继续，保存 optimizer、Python/NumPy/PyTorch/CUDA RNG；未完成 epoch 标记 `truncated_epoch`，不伪装成整轮训练完成。checkpoint 每固定步数及验证后保存；CPU/CUDA 均可运行，但跨硬件浮点结果不保证逐位一致。

## 本地 smoke 的边界

smoke 使用**完整网络**、batch size 2、每个网络至多 8 个训练 step。37 类各取 10 train、2 valid；Normal 仅取独立 train/valid 中的 32／16 张。PCA 仍使用 370 张真实 train 图的 FC2 特征拟合完整 256 维表示；不能用 8 张或 32 张拟合后补零替代。valid 固定池含 37 个 query，每个 query 73 个候选。

smoke 的模型只用于验证执行、维度、保存、恢复和检索链路，指标不代表训练充分后的方法性能。PCA 至少要求 257 张真实 train 特征，确保中心化后有足够的潜在秩；本地推断可分批，不需要同时把全部图放入内存。本地完整网络 CPU smoke、PCA/检索、断点恢复与多进程数据加载均已通过，范围和原始产物见[验证记录](../VALIDATION.md)。目标 CUDA 硬件尚未实测。

关键产物：`detector/{last,best}.pt`、`classifier/{last,best}.pt`、各自 history/summary、`encoder.npz`（含 train 拟合 ID）、`valid/classification.json`、两个检索报告目录以及门控数量和 provenance。每个检索目录有逐 query 的 `rankings.csv`，记录候选 ID、名次、categorical Hamming 距离及双方预测类别；空返回通过门控统计和 per-query 评价表保留。原始标签不作为线上检索打分输入。

`preprocessing/per_sample.csv` 与 `summary.json` 记录过滤前后缺陷芯片数、保留比例、原有缺陷被全部消除的图像数，以及无有效芯片、值 3 等情况。仅处理本次明确选取的 train/valid ID，不处理 test/holdout。这些是固定预处理的诊断，不作为隐藏调参依据，也不把单个样本分数、可视化或 smoke 数字写成方法性能结论。

隔离测试入口为 `python -m unittest discover -s compared_methods/tests -p test_yu.py -v`。恢复测试中的极小模型只用于快速验证 checkpoint/optimizer/RNG 恢复机制；正式配置和 smoke 都执行上述完整双 CNN。
