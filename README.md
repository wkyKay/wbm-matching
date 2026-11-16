# MixedWM38 晶圆检索方法复现

实现 Wu、Yu、WBBS、Nakazawa、WMHD 和 WaPIRL 整图适配，使用统一的冻结数据与检索协议。

- [项目说明](docs/项目说明.md)：方法、数据与评价、运行、结果和登记，统一在一个文件中维护。
- [自动结果总览](../wbm-region-retrieval/results_catalog/CURRENT_RESULTS.md)：唯一的跨项目成绩入口。
- [协议文件说明](protocol/README.md)：冻结清单与生成规则。
- [协作规范](AGENTS.md)：时间型 run_id、历史保护与三端同步规则。

服务器：`/data/wangkaiyan2024/compared_methods/`，旧 `wbm-matching` 是其符号链接。数据、模型和原始产物保持原位，不随 Git 发布。
