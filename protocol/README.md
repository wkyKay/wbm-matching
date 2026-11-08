# 冻结划分

`split.csv` 原样复制自 `../../wbm-region-retrieval/protocol/split.csv`，SHA-256 为
`e1785da387a70da63b65052975872da1377d15f690d4c37c5009840d08444cda`。

原始缺陷样本的 train/valid/test 分别是 25,910/3,700/7,405，样本 ID 等于原始 NPZ 行号。
本轮仅使用 train/valid；不提供 test 运行入口。保留 test 行只用于检查分割完整性。

每次运行在其输出目录下生成协议记录：37 类完整组合映射（仅由 train 建立）、实际使用的
train/valid ID、固定 valid 查询/候选池以及 Normal 的补充划分。Normal 的 1,000 张图使用
种子 2026 独立划分为 700 train、100 valid、200 holdout，后者留待将来统一测试；不修改
此处原始缺陷划分，也不会自动并入原 test 查询清单。

正式 valid 从 3,700 张缺陷图中均匀、无放回抽取 500 个 query；每个 query 从其它 valid
图中无放回抽取 1,000 个候选，排除自身。抽样顺序与原工程 valid_pool 一致，两种方法
使用同一随机种子。相关性仅按完整组合类别相同定义。Normal 只用于 Yu 检测器。

共享缓存只保存原始数组的紧凑副本；特征生成、训练及评价均使用显式 train/valid ID。
