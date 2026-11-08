# WM811K 论文在 MixedWM38 上的独立复现

本目录提供 Wu（2015）和 Yu（2019）的独立实现。只使用 train/valid，当前没有 test 入口。
论文没有公开说明的细节已明确记录，不声称与作者未提供的源码逐行一致。

| 方法 | 主实现 | 运行与参数说明 |
| --- | --- | --- |
| Wu：116 维手工特征粗筛＋NCC 重排 | [wu/wu.py](wu/wu.py) | [Wu README](wu/README.md) |
| Yu：检测/分类 CNN＋PCA256＋十值编码检索 | [yu/yu.py](yu/yu.py) | [Yu README](yu/README.md) |

详细论文依据、网络、复现约定及迁移差异见[单类缺陷对比方法](../对比方法说明/单类缺陷对比方法.md)。
实测环境和小规模验证见[验证记录](VALIDATION.md)。

## Git 管理范围

`compared_methods/` 是独立 Git 仓库。版本控制保留方法源码、正式/smoke 配置、测试、
依赖清单、README、验证记录，以及运行必需的 `protocol/split.csv`。
`.gitattributes` 保留冻结划分的原始字节及换行符，避免不同平台检出后改变数据签名。

`.gitignore` 排除本地虚拟环境、数据集、数组缓存、模型权重、实验输出和日志，以及
本地环境变量文件与编辑器缓存；这些文件仍保留在本机。远程运行时按下文重建环境、
单独放置数据，训练产物在服务器上生成。`VALIDATION.md` 中指向 `artifacts/` 的链接是
本地验证证据，克隆仓库后不会附带这些产物。

同级 `对比方法说明/` 和 `data/` 位于本仓库之外，不随本仓库提交；前者的详细说明链接
在完整工作区中可用，各方法 README 和运行代码已包含在仓库内。

## 环境和远程目录

从包含 `compared_methods/` 的外层目录运行以下命令。建议使用独立 Python 3.10–3.12 环境。
远程可沿用 `wbm-region-retrieval` 已正常运行的 PyTorch/CUDA 环境，再补齐本目录依赖。
若现有依赖已满足范围，无需升级 PyTorch；若是新环境，先安装与服务器驱动匹配的官方
CUDA PyTorch 包，再安装其余依赖。本目录不固定服务器 CUDA 版本。

```bash
python -m venv compared_methods/.venv
source compared_methods/.venv/bin/activate
python -m pip install -r compared_methods/requirements.txt

# 确认所用的 python 和 CUDA。GPU 正式训练应显式指定 --device cuda。
python -c "import sys, torch; print(sys.executable); print(torch.__version__, torch.version.cuda); print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

macOS 本地安装的是 CPU/Mac 版本 PyTorch，不能把本地 `.venv` 复制到 Linux 服务器。
远程同步代码、`compared_methods/protocol/split.csv` 和数据即可，不依赖原工程的 Python 模块。
不需要下载预训练权重，也不依赖本机 Zotero、临时 PDF 或绝对路径。

默认目录关系为：

```text
工作区/
├── compared_methods/
└── data/wm38k/Wafer_Map_Datasets.npz
```

配置中的 `data_file`、`split_file`、`output_dir`（以及可选 `cache_dir`）以
`compared_methods/` 为基准，不随启动命令的 cwd 改变；绝对路径也支持。
`--config` 参数本身按命令当前目录解析。数据位置不同时复制配置并修改 `data_file`。
默认原始数组缓存位于输出目录的同级 `_data/`，按输入 SHA-256 分组，允许两方法共用。

## 数据与评价

- 保留已有 37 类缺陷样本划分：train/valid/test=25,910/3,700/7,405。
- Normal 独立固定为 700 train、100 valid、200 holdout；只用于 Yu 检测器。
- 37 个分类标签由 train 的完整 multi-hot 组合建立，不使用八维 BCE。
- 本轮不对 test/holdout 提取特征、训练、拟合 PCA、推理或评价；读取原始 NPZ/核对划分
  不等于运行测试。原始缓存可包含原数组，所有计算均经显式 train/valid ID 选择。
- 正式 valid 为 500 个 query，各有 1,000 个 valid candidate，排除自身，seed=2026。
  两方法使用同一池；相关性为完整缺陷组合相同，不采用项目的类别覆盖率。
- P@K 分母为 K，少于 K 个返回时未返回位置按未命中；AP@K 的分母为原固定候选池中
  全部相关样本数。零相关 query 不计入 mAP/Recall，仍计入 Precision，并报告其数量。
  这个 AP 分母与 Yu 原文的 Top-500 内相关数不同，名称和口径已明确区分。

## 先做小规模验证

```bash
python -m unittest discover -s compared_methods/tests -v
python compared_methods/wu/wu.py --config compared_methods/wu/smoke.json --stage all --workers 1
python compared_methods/yu/yu.py --config compared_methods/yu/smoke.json --stage all --device cpu

# 目标 GPU 上使用独立目录做实际 CUDA 验证。
python compared_methods/yu/yu.py --config compared_methods/yu/smoke.json --stage all --device cuda --output-dir artifacts/yu_cuda_smoke

# 可选：核验 spawn/多进程 DataLoader 与串行预处理一致。
python compared_methods/tests/check_workers.py --workers 2
```

smoke 使用完整模型，但每个网络最多 8 次更新；没有完成正式 epoch，不能据其指标判断方法优劣。
PCA 始终保留 256 维，实际使用 370 张 train 图拟合。重复训练已有目录需 `--resume`，或者
指定新输出目录；若只需重新检索，可运行 `--stage valid`。

## 正式 train/valid

```bash
# 无训练方法；CPU 并行提取特征并评价 valid。
python compared_methods/wu/wu.py --config compared_methods/wu/config.json --stage all --workers 4

# Yu：先审计，再训练两个 CNN；检测器 10 epoch，分类器 80 epoch。
python compared_methods/yu/yu.py --config compared_methods/yu/config.json --stage audit
python -u compared_methods/yu/yu.py --config compared_methods/yu/config.json --stage train --device cuda

# 中断后继续。
python -u compared_methods/yu/yu.py --config compared_methods/yu/config.json --stage train --device cuda --resume

# 冻结各网络最优 valid CE checkpoint，用 train 拟合 PCA 后进行 valid 检索。
python -u compared_methods/yu/yu.py --config compared_methods/yu/config.json --stage valid --device cuda
```

Yu 使用全 train 标签，FP32、无预训练、无自动混合精度；默认检测/分类 batch 为 64/128。
如果目标显存不足，复制配置并减少对应 batch，使用新输出目录并记录该适配；不在运行时
静默降低 batch。`num_workers`、`torch_threads` 和推断 batch 可以在配置中指定。
Wu 的 SciPy/scikit-image 特征处理主要在 CPU 上执行，不要求 GPU 支持。

## 产物和恢复

每个运行保存配置、源码/数据签名、类别映射和实际样本 ID，拒绝把不同科学设置混入同一目录。
更换方法源码、训练配置、数据或划分时应使用新输出目录。输出路径和执行设备可以变化。

Wu 保存 valid 特征与 NCC 缓存、逐候选排名及评价；它只精排粗筛 Top-n，默认正式 n=500，
不声称对原 1,000 张候选生成完整精排。Yu 保存两个网络的 best/last checkpoint、optimizer/RNG、
epoch/批次位置、训练日志、PCA 编码器，以及 global 和 predicted_class 两个独立检索报告。

`--max-steps` 是 Yu 每个网络的累计步数上限。中途停止会标记 `truncated_epoch`，后续恢复
继续尚未处理的批次。小规模验证不是性能实验；本地 CPU 成功不代表已在目标 CUDA 硬件实测。

52×52 上的 9×9 滤波会消除部分图像的缺陷像素。输入顺序按用户确认保持，逐图影响保存在
Yu 输出的 `preprocessing/` 中，不根据 valid 结果暗中更改算法。
