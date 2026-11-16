# 跨项目结果登记与汇总

更新日期：2026-11-16。六种对比方法与 F 系列统一登记在 `wbm-region-retrieval/results_catalog/`，本仓库保留方法实现、协议和原始产物。

- 在线查看：[当前结果总览](https://github.com/wkyKay/wbm-region-retrieval/blob/main/results_catalog/CURRENT_RESULTS.md)、[登记与生成说明](https://github.com/wkyKay/wbm-region-retrieval/blob/main/results_catalog/README.md)。
- 两仓库相邻时，本地目录为 `../wbm-region-retrieval/results_catalog/`；服务器为 `/data/wangkaiyan2024/wbm-region-retrieval/results_catalog/`。
- `registry.csv` 登记每项评价身份和原始路径；`metrics.csv` 保存全部分组/K的原始比例；四类分表隔离普通 test、分割、融合和跨组合协议。
- `snapshot.json` 保存核验后的报告及来源 manifest，可在没有服务器大产物时离线重新生成。历史时间型 run_id 基于完成标记文件时间，明确不当作真实实验开始时间；已有 ID 保持稳定，不重命名原目录。

五方法 exact 的来源为 `artifacts/formal_exact_20261109_120738/evaluation/`；WBBS test 和 WMHD valid/test 读取已完成的 `_r2`。对应 test coverage 读取 `artifacts/coverage_v1_test_20261109_191909/`，并校验与 exact 引用的排名哈希一致。WaPIRL valid/test 位于 `artifacts/wapirl_{valid,test}_v1/`，同时包含两套评价。早期五方法 valid 只登记已有 exact，不补造 coverage。

`artifacts/cross_coverage_seven_test_20261111_v1/` 七方法已全部完成；派生候选池排除同组合，单独汇总。这里 F 是旧径向面积来源，不是当前 cc8 主方法。详见 [跨组合完成记录](CROSS_COVERAGE_SEVEN_TEST_20261111.md)。

原始 artifacts、archives、模型、缓存和排名保持原位，不纳入此次 Git 汇总；GitHub 代码同步不替代服务器产物备份。后续先更新主项目 `sources.json` 并在服务器 collect，再同步生成的目录，避免两仓库手工维护不同数值表。
