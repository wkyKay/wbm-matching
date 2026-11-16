# MixedWM38 晶圆检索方法复现

独立实现 Wu、Yu、WBBS、Nakazawa CNN、WMHD 和 WaPIRL 整图适配。六方法共享冻结数据与检索协议，保留各自的表示、训练和评分方式。

| 入口 | 内容 |
|---|---|
| [文档目录](docs/文档目录.md) | 四篇简明主题文档与历史索引 |
| [实验设计](docs/实验设计.md) | 方法区别、监督信息和复现边界 |
| [实验结果](docs/实验结果.md) | 完成情况、比较结论和保存位置 |
| [数据划分](docs/数据划分.md) | 去重、query/gallery、exact/coverage 公式 |
| [运行指南](docs/运行指南.md) | 训练、评价、时间 ID 与登记流程 |
| [协议文件说明](protocol/README.md) | 冻结清单、生成规则和哈希 |

六方法普通 test 和七方法跨组合 test 已完成。跨组合中的 F 是旧径向面积模型，不能替代当前 cc8。详细数值只维护在主项目的 [自动结果总览](https://github.com/wkyKay/wbm-region-retrieval/blob/main/results_catalog/CURRENT_RESULTS.md)。

## 目录与入口

| 目录 | 用途 |
|---|---|
| `methods/` | 六方法代码、配置、论文依据和实现约定 |
| `shared/` | 公共数据、协议和评价代码 |
| `scripts/evaluate.py` | 统一评价入口；训练来源只读 |
| `protocol/` | 去重划分及固定 query/gallery |
| `tests/` | 实现与回归检查 |
| `docs/` | 简明主题文档；旧报告在 history/ |
| `artifacts/` | 模型、缓存和运行产物，不随 Git 发布 |

具体说明：[Wu](methods/wu/README.md)、[Yu](methods/yu/README.md)、[WBBS](methods/wbbs/README.md)、[Nakazawa](methods/nakazawa/README.md)、[WMHD](methods/wmhd/README.md)、[WaPIRL](methods/wapirl/README.md)。使用已有 PyTorch/CUDA 环境或按 `requirements.txt` 安装依赖。统一评价必须指定新输出目录，不能覆盖历史结果。

## 保存与协作

本地及服务器活动目录均叫 `compared_methods/`，GitHub 仓库名为 `wkyKay/wbm-matching`，分支为 `main`。服务器路径 `/data/wangkaiyan2024/wbm-matching` 是指向 `/data/wangkaiyan2024/compared_methods` 的符号链接，不是第二份工作副本。

默认原始数据为 `../data/wm38k/Wafer_Map_Datasets.npz`。Normal 不进缺陷检索库。exact 要求完整标签相同；coverage 衡量同一排名上查询标签被覆盖的程度，不按真实标签重新排序。

唯一登记器在相邻主项目 `wbm-region-retrieval/results_catalog/`，本仓库不维护第二份成绩表。新运行使用真实时间型 run_id，先完成时间身份接入，再按 [AGENTS.md](AGENTS.md) 登记、汇总和同步。原始数据、模型、冻结协议和历史产物保持原样。
