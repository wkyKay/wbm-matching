# 六方法与径向＋面积 F：排除同组合候选的完整 test

2026-11-11，经用户授权在远程 `172.18.36.192` 启动。本文是启动记录，最终成功以远程 `completed.json` 为准。

## 协议和模型

- 方法：Wu、Yu(global)、WBBS、Nakazawa(last epoch)、WMHD、WaPIRL(best valid NCE)、F＋径向＋相对面积(best_nce)。不重新训练或搜索 test 参数。
- 原冻结 test 7199 query/gallery 不改动；每 query 在评分、粗筛、排序前排除自身及完整八维标签相同候选，保留严格超集、部分覆盖、零覆盖。
- 派生协议 `coverage_cross_combination_v1`；候选身份 SHA-256：`68c6875330ed70841586c654da776f8d0bc72bcd4cd341b79cea29705543f0b8`。
- 每 query 6999–7170 candidates。999 个 query 无 FullCoverage 正样本，仅 Recall/AP 不纳入对应均值，有效 query=6200；不删除 query。IDCG 与相关样本分母取过滤后完整候选库。
- K=1/5/10/20/50/100/500，六种 coverage 指标；保存 Top500，不把它称为完整排名。
- F 保持 beta=1、16 bins、sigma_scale=1.5、scale_weight=.15、sqrt_relative_area；使用原始归一化 h512、贪心一对一匹配及 query 区域面积平方根权重。

## 远程执行

- 工作目录：`/data/wangkaiyan2024/compared_methods/`。
- Python：`/home/wangkaiyan2024/miniconda3/envs/wbm/bin/python`。
- tmux：`coverage-seven-20261111`，已脱离客户端。关闭本机或 SSH 客户端不会中断；不能关闭远程服务器电源或终止 tmux。
- 输出目录：`/data/wangkaiyan2024/compared_methods/artifacts/cross_coverage_seven_test_20261111_v1/`。
- 驱动日志：`/data/wangkaiyan2024/compared_methods/artifacts/cross_coverage_seven_test_20261111_v1_driver.log`。
- 各方法独立 `.log`、`*_status.json`、输出子目录。WMHD rank_workers=48，Wu/WBBS=8；三个神经网络分别使用 GPU 0/1/2。其余计算限制 BLAS 线程数，避免过度并发。
- 全部方法退出成功且各自完成标记存在后，驱动核对所有 Top500 候选、逐 query 指标和总体均值，生成 `independent_verification.json`、`comparison.csv`、`comparison.json`、`REPORT.md`，最后写根目录 `completed.json`。错误保存 `failed.json`，不会将部分完成称作全部完成。

```bash
ssh 172.18.36.192 'tail -n 30 /data/wangkaiyan2024/compared_methods/artifacts/cross_coverage_seven_test_20261111_v1_driver.log'
ssh 172.18.36.192 'cat /data/wangkaiyan2024/compared_methods/artifacts/cross_coverage_seven_test_20261111_v1/completed.json'
```

## 已完成的启动前验证

- 远程跨组合 6 项回归测试全部通过（包括 Wu 粗筛前过滤、WBBS 多进程和零完整覆盖的指标行为）。
- Yu、Nakazawa、WaPIRL 来源训练签名通过；Wu/WBBS/WMHD 与原正式方法文件逐字节一致。
- F 来源与已完成径向＋面积 test 的冻结文件哈希匹配。历史脚本迁移路径使用新位置但仍严格匹配原哈希，不改历史清单。
- 16 个正式 test query 的原 Top500 ID 完全复现，分数 atol=1e-12；过滤后保留样本的分数与原候选池中对应分数一致。候选身份摘要与统一派生协议一致。
- 本地实现：`scripts/cross_coverage_20261111/evaluate_f_area.py`、`run_seven.py`；远程对应 `scripts/evaluate_f_area_cross_20261111.py`、`run_cross_seven_20261111.py`。未修改旧模型、冻结协议或历史结果。

## 后续连接状态

已确认七个子进程运行：Wu 2700986、Yu 2700988、WBBS 2700990、Nakazawa 2700992、WMHD 2700994、WaPIRL 2700995、F 2700996。WBBS/WMHD 已进入 7199 张正式 test 特征准备。随后本机 `wbm-f-eval.sock` 消失，新 SSH 连接被服务器关闭；后续进度和全部完成状态尚未确认。不能将本记录视为七方法计算已完成。
