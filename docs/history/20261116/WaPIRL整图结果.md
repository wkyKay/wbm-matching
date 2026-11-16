> 历史记录｜适用版本：F 对齐的整图 WaPIRL 适配；100 epoch，best_nce 第100轮。
> 来源：artifacts/wapirl_f_aligned_v1/、wapirl_valid_v1/、wapirl_test_v1/；登记 key wapirl_valid、wapirl_test。
> 后续状态：固定结果仍有效；不包含后续与 F 的融合，不宣称原论文所有设置原样复现。
> 阅读当前内容：[返回实验结果](../../项目说明.md#实验结果)。旧命令、未完成状态和建议均按原记录日期理解，不作为新运行指令。
> 原文版本：`cc46cd713e133bde2e62ed9db55a4313f3444f50`；原路径、版本和 SHA-256 见[来源清单](source_manifest.json)。本地副本仅修正引用和入口，历史指标与结论保留。

# WaPIRL 整图对比：正式训练与评价（2026-11-11）

已完成100轮正式训练、完整valid及完整test，并独立复核全部逐query指标。
方法是与F对齐的整图适配版，不将本次配置称为原论文所有设置的原样复现。

## 方法与训练

- 输入：整图96×96，缺陷/缺陷/有效晶圆三通道；不提取区域，不截取52×52窗口。
- F同构三通道ResNet18、stem stride1、h512、128维线性projector，仅直角旋转增强。
- SGD、batch256、100轮、lr0.01、momentum0.9、weight_decay0.001、WaPIRL cosine；温度0.07、memory EMA0.5、5000负样本。
- 训练单位是整图，每轮25193张train及3599张valid；共9900次更新。Normal不参与训练或检索。
- 选模固定为最低valid NCE：第100轮，NCE=1.811406688；不按test选模。
- NVIDIA A30单卡；100轮累计训练/验证循环耗时44.66分钟（不含准备数据、初始memory及最终检索）。
- 整图h512归一化余弦截断至[0,1]；无区域匹配或面积加权。训练不启用辅助best_retrieval选模。

## 正式数据与评价口径

沿用冻结去重划分及retrieval_v1：valid为500 query/3599 gallery；test为7199 query/7199 gallery。
每query排除自身，其余候选全部打分；同分使用统一SHA规则。两类报告使用同一份排名。
所有数值为query平均，以下表格用百分数表示（包括nDCG乘100）。
AP/Recall分母是完整候选库相关数；mAP@K为截断指标，不是full mAP。valid/test的候选库大小及query分布不同，不直接用mAP数值比较泛化差距。

## 第一类 exact_v1

查询与候选的完整八维标签组合相同才相关。

| 集合 | K | Hit (%) | Precision (%) | Recall (%) | 截断mAP (%) |
|---|---:|---:|---:|---:|---:|
| valid | 1 | 36.0000 | 36.0000 | 0.4207 | 0.4207 |
| valid | 5 | 73.6000 | 32.0400 | 1.9165 | 1.4789 |
| valid | 10 | 86.6000 | 29.6200 | 3.5749 | 2.4747 |
| valid | 20 | 96.2000 | 26.7000 | 6.1679 | 3.6912 |
| valid | 50 | 99.0000 | 22.0440 | 12.0261 | 5.7313 |
| valid | 100 | 99.8000 | 18.4920 | 19.6503 | 7.7045 |
| valid | 500 | 100.0000 | 10.1080 | 52.1472 | 12.7108 |
| test | 1 | 41.7141 | 41.7141 | 0.2232 | 0.2232 |
| test | 5 | 78.1081 | 37.0996 | 1.0021 | 0.7813 |
| test | 10 | 89.2485 | 34.5617 | 1.8738 | 1.3049 |
| test | 20 | 95.3327 | 31.4822 | 3.4233 | 2.1464 |
| test | 50 | 99.0693 | 26.9812 | 7.1669 | 3.7833 |
| test | 100 | 99.8055 | 22.8726 | 11.9350 | 5.3688 |
| test | 500 | 100.0000 | 13.6950 | 34.9552 | 10.1188 |

## 第二类 coverage_v1

覆盖率为候选覆盖查询缺陷类别的比例；不惩罚候选额外标签，不设置部分覆盖阈值τ。

| 集合 | K | Coverage-nDCG (%) | MeanCoverage (%) | FullCoverage Hit (%) | Precision (%) | Recall (%) | 截断mAP (%) |
|---|---:|---:|---:|---:|---:|---:|---:|
| valid | 1 | 72.1500 | 72.1500 | 46.4000 | 46.4000 | 0.2065 | 0.2065 |
| valid | 5 | 69.8520 | 69.3233 | 82.6000 | 43.0000 | 1.0090 | 0.8381 |
| valid | 10 | 68.9694 | 68.3100 | 91.8000 | 41.5800 | 1.9449 | 1.5019 |
| valid | 20 | 67.4892 | 66.2400 | 98.6000 | 39.0800 | 3.2821 | 2.2864 |
| valid | 50 | 64.7628 | 62.8077 | 99.8000 | 35.2080 | 6.2228 | 3.5980 |
| valid | 100 | 62.3107 | 59.9390 | 100.0000 | 31.9040 | 10.1331 | 5.0268 |
| valid | 500 | 61.1288 | 52.3906 | 100.0000 | 23.8896 | 31.6289 | 10.9450 |
| test | 1 | 76.3636 | 76.3636 | 53.3685 | 53.3685 | 0.0993 | 0.0993 |
| test | 5 | 74.1289 | 73.5021 | 85.3591 | 49.2402 | 0.4633 | 0.3855 |
| test | 10 | 72.8634 | 71.9899 | 93.4574 | 47.1079 | 0.8887 | 0.6848 |
| test | 20 | 71.1760 | 70.0227 | 97.5552 | 44.4756 | 1.6786 | 1.2078 |
| test | 50 | 68.4869 | 66.9604 | 99.5277 | 40.5481 | 3.5369 | 2.2354 |
| test | 100 | 65.7775 | 63.9863 | 99.9444 | 36.8687 | 5.9247 | 3.2841 |
| test | 500 | 60.8621 | 56.4462 | 100.0000 | 28.1848 | 19.2725 | 7.6204 |

## 检查与产物

- 本地及服务器全套172项测试通过；新增6项针对模型初始化、整图输入、旋转、打分、smoke限制及断点恢复/篡改保护的回归。
- CUDA smoke完成；正式valid/test各有completed.json，流程退出码0。
- 独立复核7699个query、3849500条Top500排名：全部候选数量、排名ID/分数、两类逐query指标及总体均值、所有产物哈希均通过。
- 原五方法与冻结协议35个保护文件，本地/远程旧仓库/新目录哈希一致。原始NPZ SHA256：`a19e791c85ccd5051c080169a3d2b17902a42545cf749e9869f6c96312bcdc69`。
- 本地下载归档SHA256与服务器一致：`c4f5d8678634122d9bc4ee66c492be6802151753f7a87a94ae052ee51d50ab86`；解包后按completed.json再次验证全部文件。

远程根目录：`172.18.36.192:/data/wangkaiyan2024/compared_methods/`。
模型及续跑状态位于`artifacts/wapirl_f_aligned_v1/{best_nce.pt,last.pt}`，权重保留在服务器。
本地已取回训练记录、完整Top500排名、特征和两类指标，不是仅保存汇总表。

- [方法与命令](../../../methods/wapirl/README.md)
- 训练记录（仅本地产物：`compared_methods/artifacts/wapirl_results_20261111/wapirl_f_aligned_v1/history.json`；不随 Git 发布）
- valid exact指标（仅本地产物：`compared_methods/artifacts/wapirl_results_20261111/wapirl_valid_v1/valid/metrics.json`；不随 Git 发布）
- valid coverage指标（仅本地产物：`compared_methods/artifacts/wapirl_results_20261111/wapirl_valid_v1/valid/coverage_metrics.json`；不随 Git 发布）
- test exact指标（仅本地产物：`compared_methods/artifacts/wapirl_results_20261111/wapirl_test_v1/test/metrics.json`；不随 Git 发布）
- test coverage指标（仅本地产物：`compared_methods/artifacts/wapirl_results_20261111/wapirl_test_v1/test/coverage_metrics.json`；不随 Git 发布）
- 独立验证记录（仅本地产物：`compared_methods/artifacts/wapirl_results_20261111/wapirl_pipeline_20261111/independent_verification.json`；不随 Git 发布）

训练曲线（本地和服务器产物：`compared_methods/artifacts/wapirl_results_20261111/training_curve.png`；不随 Git 发布）
