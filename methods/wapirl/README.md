# WaPIRL：与 F 对齐的整图适配

这是用户指定的 MixedWM38 整图对比方法，不宣称是原论文所有设置的原样复现。
训练单位由 F 的区域改成整张晶圆；不提取连通域、不做面积过滤、top-k 或区域匹配。

## 与 F 的对应

- 全原图最近邻缩放至 96×96；通道为 `raw==2`、`raw==2`、`(raw==1)|(raw==2)`。
  用户确认用全部缺陷代替目标区域，因而前两通道相同；不使用 F 的 52×52 区域窗口。
  原始 0/1/2/3 图像仍完整参与冻结数据去重。
- 完整 ResNet18 [2,2,2,2]，3 通道，首层 stride1，保留 maxpool，512维 GAP。
  初始化（含 BN bias=1 和 seed+52 三通道卷积）与 F 相同。
- 128维线性 projector，仅原图/0、90、180、270度旋转视图；无开运算、平移、尺度或噪声增强。
- SGD：100 epoch、batch256、lr0.01、momentum0.9、weight_decay0.001、WaPIRL cosine。
- 训练 NCE 温度0.07；memory EMA0.5；5000个有放回负样本，排除当前 batch。
  原图及旋转视图各占一半损失；memory 更新使用原图投影。
- valid NCE 沿用 F 的原图作为 detached anchor、旋转作为 positive，以及固定 RNG 的 train memory 负样本。
  遍历全部 valid 图，不使用标签反向传播或选模。
- 最低 valid NCE 的 `best_nce.pt` 用于正式评价，并列保留早轮次。保留 `last.pt` 支持续跑。
  不启用 F 的辅助 best_retrieval 或训练中标签检索监测；正式训练完成后统一评价。
- 检索使用 L2 归一化 h512 的余弦，截断至 [0,1]，与 F 的区域对分数相同。
  整图间直接打分，不作区域匹配或面积加权。

`f_reference.json` 记录复制的 F 源文件哈希及符号；实现独立置于 `wapirl.py`，
运行不依赖旁边的 F 仓库。训练单位数不同意味着每 epoch 的优化步数不同，不能声称计算预算完全相同。

## 数据与评价

复用 `protocol/deduplicated/split.csv` 和 `protocol/retrieval_v1/`，Normal 不参与训练和检索。
训练/valid/test 为 25193/3599/7199；valid 500 query，test 7199 query，完整 gallery 排除自身。
统一入口同时输出 exact_v1 和 coverage_v1。分数计算不读取真实标签；同分采用统一 SHA 规则。
`--exclude-exact` 可另开跨组合覆盖实验，不替代默认两类完整候选评价。

## 命令（从 compared_methods 目录执行）

```bash
python -u methods/wapirl/wapirl.py --config methods/wapirl/config.json --device cuda
# 中断恢复：相同配置/源码/数据/输出目录
python -u methods/wapirl/wapirl.py --config methods/wapirl/config.json --device cuda --resume
python -u scripts/evaluate.py --config methods/wapirl/config.json --split valid --output-dir artifacts/wapirl_valid_v1 --device cuda
python -u scripts/evaluate.py --config methods/wapirl/config.json --split test --output-dir artifacts/wapirl_test_v1 --device cuda
```

smoke 使用 `methods/wapirl/smoke.json`，每类2张 train/valid、1轮、batch16；仍使用5000负样本。
只评价 valid；其池为正式 valid query 与小 valid 的交集，不得评价 test。
评价目录必须全新；来源训练目录只读，完成和 checkpoint 哈希核验后才允许正式评价。
所有训练参数、源码、数据及冻结协议均纳入运行签名，不覆盖旧 checkpoint 或旧评价产物。

## 已完成的正式运行

2026-11-11已完成100轮、完整valid与test，两类指标及全部逐query独立复核通过。
详见[结果报告](https://github.com/wkyKay/wbm-matching/blob/6074d3775bd734f7a42d776f1e192f919c51b6ca/docs/history/20261116/WaPIRL%E6%95%B4%E5%9B%BE%E7%BB%93%E6%9E%9C.md)。
