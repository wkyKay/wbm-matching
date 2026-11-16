# 2026-11-11 历史实验脚本与归档

本目录于 2026-11-16 从服务器 `/data/wangkaiyan2024/` 根目录整理而来。脚本与原文件逐字节一致；本次不修改算法、配置或历史运行结果。

## 文件用途

| 实验 | 文件 | 用途 |
|---|---|---|
| `wapirl` | [run_wapirl_formal_20261111.sh](wapirl/run_wapirl_formal_20261111.sh) | F 对齐的 WaPIRL 整图方法训练、valid/test、smoke 和保护文件哈希记录。 |
| `wapirl` | [wapirl_pipeline_20261111.log](../../../archives/20261111/wapirl/logs/wapirl_pipeline_20261111.log) | F 对齐的 WaPIRL 整图方法训练、valid/test、smoke 和保护文件哈希记录。 |
| `wapirl` | [wapirl_protected_hashes.json](wapirl/wapirl_protected_hashes.json) | F 对齐的 WaPIRL 整图方法训练、valid/test、smoke 和保护文件哈希记录。 |
| `wapirl` | [wapirl_results_20261111.tar.gz](../../../archives/20261111/wapirl/bundles/wapirl_results_20261111.tar.gz) | F 对齐的 WaPIRL 整图方法训练、valid/test、smoke 和保护文件哈希记录。 |
| `wapirl` | [wapirl_smoke_20261111.log](../../../archives/20261111/wapirl/logs/wapirl_smoke_20261111.log) | F 对齐的 WaPIRL 整图方法训练、valid/test、smoke 和保护文件哈希记录。 |
| `wapirl` | [wapirl_smoke_eval_20261111.log](../../../archives/20261111/wapirl/logs/wapirl_smoke_eval_20261111.log) | F 对齐的 WaPIRL 整图方法训练、valid/test、smoke 和保护文件哈希记录。 |
| `wapirl` | [wapirl_source_20261111.tar.gz](../../../archives/20261111/wapirl/bundles/wapirl_source_20261111.tar.gz) | F 对齐的 WaPIRL 整图方法训练、valid/test、smoke 和保护文件哈希记录。 |
| `wapirl` | [wapirl_tests_20261111.log](../../../archives/20261111/wapirl/logs/wapirl_tests_20261111.log) | F 对齐的 WaPIRL 整图方法训练、valid/test、smoke 和保护文件哈希记录。 |

## 运行和保留约定

- Python 脚本通过原有命令行参数指定项目、来源运行、输出目录；迁移没有改变参数。仍应使用全新输出目录，遵守脚本已有的冻结来源和哈希检查。
- `wapirl/run_wapirl_formal_20261111.sh` 保留当时服务器的绝对项目路径、Conda Python 路径和 GPU 设置；这是历史执行脚本，不是可直接在任意本机运行的通用启动器。该脚本包含正式训练、valid 和 test，本次整理未执行。
- 日志与压缩包只在服务器相应仓库的 `archives/20261111/<实验>/logs/` 或 `bundles/` 中归档，保留文件名、内容及 mtime；不要求将大文件下载到本机。归档目录被 Git 忽略。
- `relocation-manifest.json` 记录全部旧路径、新路径、原始 SHA-256、大小和 mtime；其中归档文件的相对链接仅在服务器存有归档的工作副本可用。
- `artifacts/` 内原有运行产物与脚本快照保持原位、原样；旧 manifest 中的绝对路径是当时运行的历史记录，不为本次目录整理改写。核验旧根目录脚本时，应根据路径映射定位保留的原件并核对哈希和原 mtime。
- 原始根目录脚本及小型哈希 JSON 保留在服务器 `.sync-backups/20261116-organize-root/original-scripts/`；仓库内另有完全相同的受 Git 管理副本。
- 忽略规则针对归档目录、压缩包、数组、缓存和权重；Git 忽略规则不按文件大小自动筛选。既有受跟踪的冻结协议和第三方测试资料不取消跟踪。
- 本次没有删除任何原文件，也没有重建协议、重训模型或重跑正式检索。
