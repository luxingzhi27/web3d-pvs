# M4-v2 执行记录

日期：2026-08-03  
状态：协议已冻结，A 变体训练队列已启动；正式矩阵结果待队列完成

## 目的

本轮评价修复旧 M4 仅使用有效剔除率和错误剔除率判断方向遮挡代理贡献的问题。方向遮挡代理作为一个输入因子，显式遮挡抑制作为另一个结构因子，四个正式变体组成完整的 2x2 设计：A `geometry_context_ray_no_inhibition`、B `geometry_context_proxy_ray_no_inhibition`、C `geometry_context_ray`、D `full`。`aabb_ray` 和 `geometry_ray` 仅作为逐级输入参考。

## 已完成的协议和代码

- 旧 M4 的 15 份 intervention 文件已只读审计；每个成员包含 664 个 validation pose 的 TP、FP、FN、TN、候选哈希和阈值来源。
- 所有旧成员的候选身份一致，候选摘要为 `8bd3e6a840c7624e2de459ef8057b24380c91936383c29f2368d93801f4c17bf`；A 的三个随机种子缺失，不能用旧 checkpoint 伪造。
- 旧成员的阈值与本轮严格安全工作点多数不一致（例如同一 checkpoint 的旧冻结阈值和本轮 calibration 选择值不同）。因此旧 intervention 仍保持只读、不覆盖；B/C/D 和逐级基线仅复用旧 checkpoint，在新校准阈值下重新生成独立的 v2 validation intervention。这样保持模型、pose、候选和 GT 可比，同时不把旧阈值结果误当作本轮工作点。
- 新汇总器分别生成 pose-level 宏平均和 aggregate 合并指标，报告 recall、weighted recall、precision、F1、Jaccard、accuracy、balanced accuracy、specificity、useful cull、bad cull、平均 TP/FP/FN/TN、平均预测数、预测/候选和预测/GT。
- 新汇总器同时生成 `geometry_minus_aabb` 与 `context_minus_geometry` 两个逐级 paired comparison，用于单独展示离线几何和上下文表征增益；它们不参与方向代理路线门。
- 新 bootstrap 按 seed 聚类、再在 seed 内按 pose 重采样，默认 10,000 次；完整输出 B-A、C-A、D-C、D-B 和 D-B-C+A 的差值、95% 区间、方向和是否跨零。
- 每个 checkpoint 的阈值从自己的 calibration threshold rows 冻结，新增 pose recall >= 0.95、weighted recall > 0.99 和 weighted recall 单侧下界 > 0.99 三重安全约束；validation 不选阈值，test 未读取。
- 新路线判定分为安全层、分类/剔除层和系统层。系统层需要同位姿图像或 GLB 成本证据；当前 intervention 文件不含这些数据时显式标记 `not_available`。
- 回归审计期间修复了 `evaluate_proxy_interventions.load_threshold` 对旧式程序化调用缺少 `threshold_source` 属性的兼容问题；修复后 M4 专项测试、self-test、Python 编译和 benchmark 全量 unittest 均通过。

## 执行入口

```bash
conda run --no-capture-output -n slm_pvs \
  python neural_instance_culling/benchmark/audit_m4_formal_matrix_v2.py \
  --output neural_instance_culling/benchmark/out/m4_formal_matrix_validation_v2/input_manifest.json

bash neural_instance_culling/benchmark/run_formal_m4_v2.sh
```

长任务写入 `neural_instance_culling/benchmark/out/m4_formal_matrix_validation_v2/queue.log`，训练成员各自写入 `train_stdout.log` 和 `train_stderr.log`，评价成员写入各自的 `logs/`。

## 资源和不可变边界

A 变体使用独立的 `pvs_m4_v2_ablation_geometry_context_ray_no_inhibition_*` 输出目录；B/C/D 只复用既有 checkpoint，不覆盖旧 intervention。旧 M4 summary、旧 route JSON 和旧报告均未修改。运行时默认模型、前端资产和 Route B 文档结论未修改。

## 待完成

1. 等待当前 A 三个 seed 的 40 epoch 训练完成；M5 图像队列独立运行，不读取其未完成结果。
2. 生成 calibration workpoint，使用同一 664 个 validation pose 和同一后退相机候选重评六个成员变体；其中 A-D 用于 2x2 因子计算，AABB 和几何基线只用于逐级参考。
3. 完成 10,000 次 paired bootstrap、schema 校验、路线 JSON 和 M4-v2 正式报告。
4. 若没有同位姿图像或严格 GLB 成本输入，报告继续保留 `not_available`，不由 useful cull 推断系统收益。

## 2026-08-03 评价链路校验补强

在正式矩阵仍训练期间，对 v2 汇总后的 schema 校验入口做了一个不影响结果的加强：除了要求全部成员彼此共享候选身份外，最终校验还会默认比对预登记的数据集候选摘要
`8bd3e6a840c7624e2de459ef8057b24380c91936383c29f2368d93801f4c17bf`，并提供显式参数覆盖该登记值。这样可以发现所有变体共同发生的候选集合漂移，而不只是发现变体之间的不一致。修改已通过 `slm_pvs` 环境下的 Python 编译、M4 self-test 和 M4 相关 10 项 unittest；训练、旧 M4 产物和默认前端资产未修改。

同一轮收尾审计发现原始 pose 编号是数据集全局编号，冻结 validation 子集的 664 是行数而不是编号上界；例如合法记录包含 pose `2991` 和 `193`。已修正行校验器，改为检查非负编号、唯一性和总记录数，并加入稀疏原始 pose 编号单元测试。此前失败的汇总没有写出 summary，18 份 v2 intervention 仍保留并将在修复后直接复用。

随后发现汇总器仍把 A 之外的变体指向旧 M4 intervention，导致旧阈值与 v2 calibration workpoint 不一致。已修正为六个变体全部读取独立 v2 members；旧 intervention 仍只作为不可变审计输入，不进入 M4-v2 数值汇总。该问题不会改变任何 checkpoint 或候选集合。

汇总重试时又发现 pose 宏平均的候选数/GT 数字段没有映射到 bootstrap 注册的 `avg_candidate_count`/`avg_gt_count` 名称，已补齐映射并加入单元测试。该问题发生在统计字段组织阶段，未产生正式 summary，也未改变逐 pose TP/FP/FN/TN。

## 2026-08-03 正式结果

三组 A 变体均完成 40 epoch 训练、checkpoint 保存和 calibration 产物导出，训练日志中的非有限 loss/gradient 跳过计数均为 `0`。六个变体的三个 seed 共 `18` 个 v2 validation intervention 均使用同一 `664` 个 pose、同一后退相机候选集合和候选摘要
`8bd3e6a840c7624e2de459ef8057b24380c91936383c29f2368d93801f4c17bf`；阈值来自各自 calibration，summary 标记 `testRead=false`。最终通过了 M4 相关 `11` 项 unittest、self-test、候选摘要校验和 schema 校验。

路线判定为 `route_b_system`。方向代理无抑制时的 `B-A` aggregate recall 差值为 `-0.030582`，95% CI 为 `[-0.034099, -0.007174]`；有显式抑制时的 `D-C` aggregate recall 差值为 `-0.001581`，95% CI 为 `[-0.038721, -0.001008]`。两组比较的 weighted recall 和 bad cull 没有形成足以抵消召回风险的安全证据，且成员级 validation 安全工作点并未全部满足要求。因此，即使分类诊断表中部分 precision、F1、balanced accuracy 或 useful-cull 差值为正，也不能把方向代理写成安全约束下的独立贡献。

完整 pose 宏平均、aggregate、TP/FP/FN/TN、分类指标、剔除指标、候选/GT 比、五组因子差值、交互项和 10,000 次分层 paired bootstrap 均写入 `neural_instance_culling/benchmark/out/m4_formal_matrix_validation_v2/summary.json`，正式报告为 `docs/evaluation/m4_formal_matrix_validation_v2_2026-08-03.md`。当前 intervention 没有同位姿图像、GLB 字节曲线或浏览器计时，因此这些系统层指标均保留为 `not_available`；本轮不修改旧 M4 结果、默认模型、默认前端资产或旧 Route B 文档结论。
