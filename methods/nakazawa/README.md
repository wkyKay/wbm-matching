# Nakazawa & Kulkarni 2018

对应 survey 第 1.3 篇 [Wafer Map Defect Pattern Classification and Image Retrieval Using Convolutional Neural Network](https://doi.org/10.1109/TSM.2018.2795466)。这是按论文重实现并适配 MixedWM38 的 CNN 二值编码方法，不使用 Yu 的网络、预处理或选模流程。方法 CLI 提供训练、编码和 Hamming 打分，没有 `test` stage。

五方法统一 valid/test 评价使用[独立入口](../../docs/EXACT_EVALUATION.md)：只读已完成训练的
`--source-run`，保持最后一轮 checkpoint 和 256 位 Hamming 距离，使用 SHA-256 破同分，
在新目录输出完整组合指标。正式评价要求完成正式全部 epoch；smoke 来源只能评价 smoke valid。

## 原文依据与复现边界

已核对[原文正文 §II-B/C、§III-B](https://www.scribd.com/document/645154096/Wafer-Map-Defect-Pattern-Classification-and-Image-Retrieval-Using-Convolutional-Neural-Network)及[公开转载的原文 Table I、III 图片](https://aiegg-travel.tistory.com/entry/CNN%EC%9C%BC%EB%A1%9C-Wafer-Map-Defect-Pattern-%EB%B6%84%EB%A5%98%ED%95%98%EA%B8%B0-%EB%85%BC%EB%AC%B8-%EB%A6%AC%EB%B7%B0)。后一个链接只用于读取原表格，未采用博客作者额外推断。

| 项目 | 论文已披露 | 本实现 |
| --- | --- | --- |
| 网络 | 输入 286×400；三组 3×3/stride1 卷积，通道 32、32、64；每组 ReLU 和 2×2 maxpool；FC256 sigmoid → dropout → 类别 FC → softmax | 保留全部层次及输入大小；最终类别数由 22 改为 37；训练使用 logits 接 cross entropy，softmax 可由 `predict_proba` 获取 |
| 编码 | sigmoid 输出大于等于 0.5 置 1，形成 256 位；Hamming 距离检索 | 完整保留，使用整数 bit 差异；不使用 PCA、十值量化或预测类别门控 |
| 训练 | 报告 10 个 epoch 的训练/验证分类准确率；达到所需准确率后测试 | 默认训练 10 epoch，监测分类 loss/accuracy；不使用检索指标选模 |
| batch | Table III 列出 385、770、1540、3080 的内存用量，未指明最终结果使用哪个 | 选择表中最小的 385，属于实现约定；microbatch=8 累积梯度，每个有效 batch 只更新一次 |

**原文没有完整披露训练配方，不能声称逐项等价的官方复现。** 已查到的正文和表格未给出优化器、学习率、dropout 比例、卷积 padding、池化 stride、权重初始化、最终 checkpoint 选择规则，以及密度图送入网络时的颜色通道细节。本实现将这些补足约定写入配置：Adam(1e-3, β=0.9/0.999, ε=1e-8)、cross entropy、无 weight decay、dropout=0.5、valid convolution、pool stride2、Xavier uniform/零 bias、固定最后一个 epoch 的 checkpoint。它们是为了使重实现可运行的明确约定，不是论文报告值，也不是 Yu 的设置；验证分类曲线只监测，不用于择优或提前停止。

默认 valid padding 下特征图为 64×34×48，FC 输入为 104,448，网络约 2,678 万参数。即使 smoke 也保留这个网络和 286×400 输入。没有图像增强、滤波、预训练、类别重采样或额外训练数据。

## MixedWM38 适配

原文的密度值是每个 die 内的缺陷点数量除以全图最大 die 计数。MixedWM38 仅有 pass/fail，没有这些计数，不能还原原始密度。本实现用 `wafer == 2` 得到单通道 0/1 占用图，0/1/3 其他编码均映射为 0，再 nearest resize 到 286×400；不人为制造缺陷频率或用平滑密度替代。这是数据更换导致的必要输入适配，需在实验中披露。

训练只使用已有去重划分中的 25,193 个缺陷 train；完整 valid 3,599 个用于分类监测；输出类别为 train 决定的 37 种完整组合。Normal 不加入分类或检索。共享数据审计会写 Normal 清单，但该方法不读取它训练。

原方法编码阶段读取 `protocol/retrieval_v1` 的固定 valid gallery 和 query：3,599 个 gallery、500 个 query；代码不重新抽 query。原始 NPZ、去重清单和旧运行均不覆盖。test 不进入训练或选模；仅统一入口显式 `--split test` 才进行 test 编码和评价，共享原始身份去重核查不属于性能评价。

## 运行

从 `compared_methods/` 执行（所有配置内相对路径均相对此目录）：

```bash
.venv/bin/python methods/nakazawa/nakazawa.py --stage audit
.venv/bin/python methods/nakazawa/nakazawa.py --stage prepare
.venv/bin/python methods/nakazawa/nakazawa.py --stage train --device cuda
.venv/bin/python methods/nakazawa/nakazawa.py --stage encode --device cuda
```

没有 CUDA 时可使用 `--device cpu` 或明确使用可用的 `mps`。`all` 依次执行 prepare、train、valid 编码，不计算检索指标、不运行 test。`prepare` 核对协议并记录 ID，输入图像在读取时 resize，避免产生约 11 GB 的训练输入缓存。

小规模连通性验证：

```bash
.venv/bin/python methods/nakazawa/nakazawa.py --config methods/nakazawa/smoke.json --stage all --output-dir artifacts/nakazawa_smoke_mycheck
```

smoke 使用每类 1 个 train、2 个 valid，仅运行 1 个训练 batch 和 1 个验证 batch、1 epoch，并编码小 valid 全集。它与正式 query 取交集，不能作为实验结果。输出目录须使用尚未有其他设置产物的新目录。

训练中断后用同一配置、输出目录与 `--resume --stage train` 从最后一个完成的 epoch 继续；未完成 epoch 会重新训练。恢复训练要求使用原设备类型（CPU/CUDA/MPS），以恢复对应的 dropout 随机状态；推理编码可换设备。默认不重训已有 checkpoint。已完成的同一运行会校验并复用。训练、编码各自用运行目录锁串行化同目录写入。任何代码、科学参数或输入签名变化须使用新输出目录。

## 原方法产物与编码接口

默认输出 `artifacts/nakazawa_retrieval_v1/`：

- `run_manifest.json`、`resolved_config.json`、`environment.json`：源码、参数、数据和协议身份。
- `protocol/`、`audit.json`、`prepared.json`：原始 sample_id、类别映射与固定池记录。
- `checkpoint_last.pt`：最后完成 epoch 的模型、优化器、随机状态、run/data 签名和类别映射；不是按 valid loss 或检索指标选择的 best。
- `training_history.csv`、`training_complete.json`：分类训练监测及完成记录。只有完成指定 epoch 并通过签名校验的模型才允许编码。
- `valid_codes/sample_ids.npy`、`sigmoid_features.npy`、`binary_codes.npy`、`metadata.json`：原始 ID 顺序、N×256 float32 特征、N×256 uint8 代码、模型/文件哈希和固定查询身份。不会生成评价成绩。

供统一驱动使用的数组接口：

```python
from methods.nakazawa.nakazawa import (
    load_trained_model, encode_maps, hamming_distances, score_candidates,
    rank_candidates,
)

model, device = load_trained_model(bundle, config)
features, codes = encode_maps(model, bundle.maps, fixed_gallery_ids,
                              batch_size=8, device=device)
# codes 的行号对应 fixed_gallery_ids，不能把原始 sample_id 直接当行号。
distances = hamming_distances(query_code, candidate_codes)  # 越小越相似
scores = score_candidates(query_code, candidate_codes)     # -distance，越大越相似
ids, distances = rank_candidates(query_code, candidate_codes, candidate_ids,
                                 query_id=query_id)        # 排除自身；同分按原 ID
```

上述数组接口不读取标签、不抽样、不定义相关性；统一驱动负责传入既定 query/gallery。
原 `rank_candidates` 的 Hamming 同分按 sample_id 升序；新统一入口直接使用 Hamming 分数并采用
哈希规则，两者结果不可混比。不能将大量同分掩饰为模型有额外区分能力。

验证：`PYTHONPATH=. .venv/bin/python -m unittest discover -s tests -p 'test_nakazawa.py'`。
