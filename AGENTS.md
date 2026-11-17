# 项目协作与结果管理约定

本文件适用于 `compared_methods/` 全部目录。修改前检查 Git 状态、相关目录更具体的 `AGENTS.md` 和当前代码，保留已有改动。用户仅询问或要求方案时，不擅自修改代码、启动训练或正式评价。

## 方法与查阅入口

- 本仓库包含 Wu、Yu、WBBS、Nakazawa CNN、WMHD、WaPIRL 整图适配。先读目标方法 README、`protocol/README.md`、`docs/项目说明.md#数据与评价` 和相关验证记录；复现约定不因结果整理而改变。
- 新方法放在 `methods/<方法>/`，复用公共数据与评价接口；不私自重划数据、抽样 query 或缩小正式 gallery。方法评分不得读取真实标签；跨组合标签过滤是显式的评价候选资格规则。
- 最新成绩入口是 [对比方法实验结果](docs/对比方法实验结果.md)。唯一的登记器和汇总位于相邻主项目 `../wbm-region-retrieval/results_catalog/`，不要在本仓库维护第二套独立登记表；结果文档数值区由 `scripts/summarize_results.py` 从主项目指标长表生成，不手工改数。
- 服务器活动目录为 `/data/wangkaiyan2024/compared_methods/`；旧 `wbm-matching` 是符号链接，不能当作另一个需分别更新的仓库。历史 manifest 的旧绝对路径保留。

## 新实验必须使用时间型 run_id

- 每个新实验在实际启动时分配并持久化 `run_id` 和带时区的 `started_at`。统一采用 Asia/Shanghai：`YYYYMMDDTHHMMSS+0800__稳定语义key`；`started_at` 用 ISO 8601，如 `2026-11-17T09:30:00+08:00`。
- 时间必须是真实启动时间，不用计划时间、当前整理时间或文件复制时间代替。后缀区分方法、变体、seed、checkpoint、split及同秒运行，确保唯一；示例：`20261117T093000+0800__wapirl_seed2026_best_nce_test`。
- 结果登记单位是“方法／变体 × split”的评价；训练运行单独记录其身份并由评价引用，同一排名的 exact 与 coverage 合并登记。训练启动时间不能冒充后续独立评价的启动时间。批量任务可有带时间的父目录，各评价保留自己的身份。
- 新运行使用含时间身份的新输出根，通过显式输出参数传入。实现新入口时支持该规则；不修改旧入口默认目录来冒充新版本，不重命名既有目录。科学设置和签名一致的断点续跑沿用原 ID、started_at 和目录。
- 科学设置、源码签名、数据协议、模型、checkpoint 或候选规则变化时，创建新 key、run_id 和输出目录；用来源字段关联旧实验，禁止改写旧 manifest 或完成标记以强行复用。
- 历史登记仍保留原 ID 和 `id_time`。`completion_marker_mtime` 仅是明确标注的历史回退依据，不是开始时间；不能给旧运行补造 started_at，也不因同步 mtime 改变而重分配 ID。
- 当前主项目登记器对历史数据使用完成标记 mtime。接入第一项带真实 started_at 的新实验时，必须先扩展 `../wbm-region-retrieval/results_catalog/build.py` 读取并验证持久化的时间身份，保留已登记 ID，并验证重采集稳定性；不能让新实验继续静默回退到 mtime。此项是新实验接入要求，不授权现在启动实验或重跑历史结果。

## 完成后必须登记并生成汇总

1. 实验开始前先确认当前协议、来源模型、输出隔离和时间身份；将运行命令、选择规则、监督信息及是否为事后实验写入新运行的配置／manifest 或独立元数据。
2. 以完成标记和其引用文件的校验结果确认完成，不以进程退出、tmux 消失或部分 CSV 存在判断成功。失败、运行中、smoke 和历史不同协议不得加入正式成绩表，应在运行记录或目录盘点中明确状态。
3. 完成后更新唯一的跨项目来源清单 `../wbm-region-retrieval/results_catalog/sources.json`，记录唯一 key、方法、变体、split、协议、报告类型、来源路径、完成标记、manifest、选择记录和设计边界。检查原有记录，不给同一结果重复分配身份。
4. 在持有完整产物的服务器执行：

   ```bash
   cd /data/wangkaiyan2024/wbm-region-retrieval
   python3 results_catalog/build.py collect --workspace /data/wangkaiyan2024
   python3 results_catalog/build.py check
   ```

5. 将新的快照和生成文件同步到本地；在主项目根目录可离线执行 `python3 results_catalog/build.py render` 和 `check`。若改变生成器或新增协议适配，运行主项目的 `python3 -m unittest discover -s tests -p 'test_results_catalog.py' -v` 及适用的新增验证。
6. `snapshot.json` 由采集生成，`registry.csv`、`metrics.csv`、`tables/*.csv` 和 `CURRENT_RESULTS.md` 由脚本生成。禁止手工改数、绕过签名或修改快照来掩盖失败；来源清单改变后须重新 collect，不能继续使用旧快照。
7. 统一更新本仓库三份主题文档的对应内容及主项目当前入口；方法 README 仅补充专属参数，注明完成状态、原始路径、结果及限制；不得留下失效的“已启动、尚未确认完成”。有保存价值且被引用的旧分析保存为本地历史参考副本，并在项目说明的“验证与文档整理”节提供相对链接，不把旧结论改写为新实验结论。

## 汇总与论文比较边界

- 复用冻结去重协议：train/valid/test=25,193/3,599/7,199；valid 固定 500 query，每 query 3,598 候选；test 全 7,199 query，每 query 7,198 候选。排除自身，Normal 不入缺陷检索库，sample_id 保留原始 NPZ 行号。标签顺序为 center、donut、edge-loc、edge-ring、loc、near-full、scratch、random。
- 普通 `retrieval_v1` 与排除相同完整标签组合的 `coverage_cross_combination_v1` 必须分表。新增协议须独立命名、显式适配并验证，不能仅给旧成绩改标签。
- exact 与 coverage 使用同一排名、分别定义相关性；AP/Recall 分母及 coverage IDCG 取对应完整候选库。Top-500 不称为完整排名，mAP@K 不称为 full mAP。Wu 粗筛不改变相关样本分母。
- 保留 K=1/5/10/20/50/100/500、总体、类别宏平均、单／混合及标签数量分组；不得混合 split、group、checkpoint 或候选池求均值。缺失报告不补造，缺失数值不填零。
- CSV 保存原始比例；展示百分数乘 100，差值乘 100 为百分点。Yu／Nakazawa 的监督训练、WaPIRL／F 的自监督训练、其他方法无训练，以及选模规则、实现适配和事后实验必须显式区分。重复端点或相同排名的对照角色不是独立重复实验。
- 五方法旧 valid 只有 exact，不补造 coverage。旧 test 的 coverage 后处理须核对与 exact 引用同一排名；WBBS test、WMHD valid/test 使用已完成 `_r2`，不将早期未完成尝试混入汇总。
- 七方法跨组合 test 中的 F 是 2026-11-11 旧径向面积来源，不是当前 cc8 主实验。主项目 cc8 身份是在查看历史 test 后确认，角向＋最优匹配组合为事后补充；保留这些设计边界。

## 历史保护与同步

- 原始数据、冻结协议、已有模型／特征／排名／报告、`artifacts/`、`archives/` 和历史 manifest 保持原位、原样；未经明确要求，不删除、覆盖、改名、重建或批量去重。新登记不搬迁原实验目录。
- 新评估及后处理写入独立的新目录，来源只读。历史绝对路径和签名不改写；登记中以映射／来源字段说明迁移关系。
- Git 管理代码、配置、冻结协议、文档、登记器及轻量快照／汇总；不加入模型、原始数据、大型缓存或完整排名。GitHub 同步不等于大产物备份。
- 按本会话已确认的工作约定，完成本地修改和验证后，同步相关仓库的本地、服务器和 GitHub。提交前检查三方分支和未提交改动，采用普通推送及服务器快进；不强推、不覆盖他人改动。若远程有新提交，先安全整合并验证。
- 两项目统一使用 `main`；本仓库 origin 为 `https://github.com/wkyKay/wbm-matching.git`。修改主项目登记与汇总时须同时提交并同步主项目，不能仅推送本仓库文档。同步后核对提交号、受 Git 管理文件及工作区状态，准确报告未完成的同步或验证，不宣称原始大产物也已同步。

## 文档整合与维护

- 用户于2026-11-17明确要求三份主题文档，替代此前“全部集中到项目说明”的约定：[对比方法设计](docs/对比方法设计.md)写六方法论文、数据和实验设计；[对比方法实验结果](docs/对比方法实验结果.md)总结已完成成绩、排序和边界；[项目说明](docs/项目说明.md)写公共协议、评价、跨组合规则、运行与登记。
- 使用二级标题区分主题；复用三份文档，不另建重复运行指南、结果导航或日期版项目说明。根README仅保留简介和入口，方法README维护专属公式、参数与复现依据。
- 唯一成绩源仍是相邻主项目 `results_catalog/`。结果文档正文可解释结论；数值区由 `python3 scripts/summarize_results.py` 生成，`--check`核对。更新时先完成主项目collect/check，再生成本文表格，不手工修改生成区，不补造缺失报告。
- 结果表按协议、split、group、指标、K分别比较；保留原始精度排序、百分数展示和缺失值。用户要求复用已有结果时，不因整理文档重复训练或运行test；单seed结果不包装成重复实验或显著性结论。
- 已完全整合的六份历史说明已删除，迁移关系、旧版本与哈希保留在 `docs/history/20261116/source_manifest.json`；其中removed_archive是历史路径，不是现存文档。尚有独立证据的[验证记录](docs/history/20261116/验证记录.md)保留，仅更新导航；旧阶段状态不能代表当前实验状态。
- 文档使用简明中文与相对链接，跨项目指向工作区同级仓库；改路径和锚点时同步更新引用、来源清单及主项目入口。原始实验报告、数据、模型、冻结协议和历史产物不因文档整合而删除或修改。
- 每次更新后检查数据来源、生成区、链接和Git差异，按前述同步规则提交、普通推送、服务器快进，核对三方提交和受管理文件一致；无法完成的同步要明确报告。
