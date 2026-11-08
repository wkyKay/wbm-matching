# 本地验证记录

验证日期：2026-11-08。两种方法均已在真实 MixedWM38 小批量数据上跑通，28 项自动测试通过。
本记录验证实现能执行、数据隔离及恢复机制；没有开展完整训练、正式性能比较或 test 评价。

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
checkpoint 加载，远程预检及 CUDA smoke 命令见[总说明](README.md)。

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
逐图统计见 [Yu 预处理审计](artifacts/yu_smoke/preprocessing/summary.json)。

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

- [整合检查结果](artifacts/validation/integration.json)、[DataLoader 结果](artifacts/validation/dataloader_workers.json)。
- [Wu 特征记录](artifacts/wu_smoke/prepare.json)、[Wu 检索记录](artifacts/wu_smoke/valid_summary.json)。
- [Yu 检测训练记录](artifacts/yu_smoke/detector/summary.json)、[Yu 分类训练记录](artifacts/yu_smoke/classifier/summary.json)。
- [Yu PCA/推断来源](artifacts/yu_smoke/valid/provenance.json)、[Yu 检索记录](artifacts/yu_smoke/valid/summary.json)。

未收敛的 smoke 分数仅用于确认报告生成正常，不作为 Wu 与 Yu 的性能比较结论。
