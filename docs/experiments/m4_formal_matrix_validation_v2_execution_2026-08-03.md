# M4-v2 执行记录

日期：2026-08-03  
状态：协议已冻结，A 变体训练队列已启动；正式矩阵结果待队列完成

## 目的

本轮评价修复旧 M4 仅使用有效剔除率和错误剔除率判断方向遮挡代理贡献的问题。方向遮挡代理作为一个输入因子，显式遮挡抑制作为另一个结构因子，四个正式变体组成完整的 2x2 设计：A `geometry_context_ray_no_inhibition`、B `geometry_context_proxy_ray_no_inhibition`、C `geometry_context_ray`、D `full`。`aabb_ray` 和 `geometry_ray` 仅作为逐级输入参考。

## 已完成的协议和代码

- 旧 M4 的 15 份 intervention 文件已只读审计；每个成员包含 664 个 validation pose 的 TP、FP、FN、TN、候选哈希和阈值来源。
- 所有旧成员的候选身份一致，候选摘要为 `8bd3e6a840c7624e2de459ef8057b24380c91936383c29f2368d93801f4c17bf`；A 的三个随机种子缺失，不能用旧 checkpoint 伪造。
- 新汇总器分别生成 pose-level 宏平均和 aggregate 合并指标，报告 recall、weighted recall、precision、F1、Jaccard、accuracy、balanced accuracy、specificity、useful cull、bad cull、平均 TP/FP/FN/TN、平均预测数、预测/候选和预测/GT。
- 新汇总器同时生成 `geometry_minus_aabb` 与 `context_minus_geometry` 两个逐级 paired comparison，用于单独展示离线几何和上下文表征增益；它们不参与方向代理路线门。
- 新 bootstrap 按 seed 聚类、再在 seed 内按 pose 重采样，默认 10,000 次；完整输出 B-A、C-A、D-C、D-B 和 D-B-C+A 的差值、95% 区间、方向和是否跨零。
- 每个 checkpoint 的阈值从自己的 calibration threshold rows 冻结，新增 pose recall >= 0.95、weighted recall > 0.99 和 weighted recall 单侧下界 > 0.99 三重安全约束；validation 不选阈值，test 未读取。
- 新路线判定分为安全层、分类/剔除层和系统层。系统层需要同位姿图像或 GLB 成本证据；当前 intervention 文件不含这些数据时显式标记 `not_available`。

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

1. 等待当前 M5 队列结束后训练 A 的三个 seed。
2. 生成 calibration workpoint，使用同一 664 个 validation pose 和同一后退相机候选重评六个成员变体；其中 A-D 用于 2x2 因子计算，AABB 和几何基线只用于逐级参考。
3. 完成 10,000 次 paired bootstrap、schema 校验、路线 JSON 和 M4-v2 正式报告。
4. 若没有同位姿图像或严格 GLB 成本输入，报告继续保留 `not_available`，不由 useful cull 推断系统收益。
