# M11 泛化矩阵审计

日期：2026-08-04  
范围：航向留出、跨场景零样本迁移和 Metropolis 少样本适配  
状态：诊断证据已齐全；严格安全门未通过，不能作为通用泛化结论

## 评价口径

所有结果使用模型输入视场角 66°、真实显示视场角 60°、保存的后退视锥候选集合和原始实例 GT。候选集合没有补入真实可见实例，阈值来自各 checkpoint 自己的 calibration；test 只读取冻结阈值并执行一次。跨场景迁移重新生成目标场景的实例特征表，未复用源场景实例编号、AABB 或场景缓冲区。

当前严格安全门为普通 pose recall `>=0.95`、weighted recall 点估计 `>=0.9925` 且单侧 bootstrap 下置信界 `>0.99`。M11 的历史校准入口已保证 weighted recall 条件，但部分旧方向/少样本工作点没有强制普通 pose recall 下限，因此下表按当前门重新解释：普通召回不足的结果标记为诊断，不标记为安全通过。

## Frozen test 结果

| 工作点 | 训练信息 | 阈值 | pose recall | weighted recall | pose precision | aggregate precision | useful cull | bad cull | 判断 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| HKUST 航向留出 | HKUST 方向训练，test 378 pose | 0.100 | 0.925663 | 0.993071 | 0.724988 | 0.248328 | 0.838552 | 0.002150 | 普通召回不达安全门 |
| Metropolis 零样本 | 仅迁移查询参数，目标场景重建特征 | `1.7783e-7` | 0.999996 | 0.999983 | 0.091107 | 0.091107 | 约 0 | 约 0 | 高召回但无有效剔除 |
| Metropolis 1% 适配 | 205 个训练 pose | 0.050 | 0.945077 | 0.993608 | 0.156026 | 未在该摘要复算 | 0.279348 | 0.004880 | 普通召回不达安全门 |
| Metropolis 5% 适配 | 1,023 个训练 pose | 0.050 | 0.947238 | 0.990595 | 0.307265 | 0.280205 | 0.626039 | 0.005463 | 普通/weighted 门均不完整 |
| Metropolis 10% 适配 | 2,045 个训练 pose | 0.075 | 0.943922 | 0.991585 | 0.359480 | 0.326031 | 0.679711 | 0.005971 | 普通召回不达安全门 |

Metropolis 少样本结果显示适配比例增加后 useful cull 从 1% 的 `0.279348` 增至 5% 的 `0.626039` 和 10% 的 `0.679711`，同时 precision 提升；但 10% 的普通 pose recall 仍为 `0.943922`，不能据此宣称满足画面安全约束。Zero-shot 的高 recall 伴随几乎全量预测，说明共享查询参数可以维持安全召回，却不能单独恢复目标场景的有效剔除。

## Calibration 审计

| 工作点 | calibration pose recall | weighted recall | weighted recall LCB | bootstrap |
|---|---:|---:|---:|---:|
| HKUST 航向留出 | 0.917674 | 0.996999 | 0.994986 | 10,000 |
| Metropolis 1% | 0.951636 | 0.994991 | 0.994765 | 10,000 |
| Metropolis 5% | 0.955944 | 0.992787 | 0.992347 | 10,000 |
| Metropolis 10% | 0.946283 | 0.993030 | 0.992628 | 10,000 |

加权召回的点估计和置信界在这些历史 calibration 中基本满足预设要求，但 HKUST 航向和 Metropolis 10% 的 calibration 普通 pose recall 不足 0.95。后续若要把该组结果用于正式主表，应使用带普通召回下限的独立严格 calibration 导出；不能在 test 上调阈值。

## 结论边界

1. 航向留出证明了同一场景未见方向上的性能不等于随机 split 结果；本次 HKUST 航向工作点的普通召回不足，不能写成方向泛化通过。
2. 零样本跨场景迁移只能维持高召回，无法维持有效剔除，支持“场景特定离线特征表仍是必要条件”的工程结论。
3. 少样本适配呈现明显的资源效率恢复趋势，但 1%/5%/10% 都没有同时满足当前严格安全门，尤其不能只用 useful cull 上升来替代 recall 和 weighted recall。
4. 当前论文表述应限定为“目标场景重新建立固定实例表并进行少样本适配”，不使用“通用编码器”或“跨场景零样本高效泛化”。

## 证据与复现入口

- 航向协议：`docs/experiments/m11_directional_generalization_protocol_2026-08-01.md`。
- HKUST 航向 frozen test：`neural_instance_culling/benchmark/out/m11_formal_hkust_directional_frozen_test_20260802/summary.json`。
- Metropolis 1% frozen test：`neural_instance_culling/benchmark/out/m11_formal_metropolis_fewshot_1pct_frozen_test_20260802_protocolfix/summary.json`。
- Metropolis 5% frozen test：`neural_instance_culling/benchmark/out/m11_pvs_m11_fewshot_5pct_metropolis_yaw20_rvl_strong_v2_full40_seed20260801_retry3_frozen_test/summary.json`。
- Metropolis 10% frozen test：`neural_instance_culling/benchmark/out/m11_pvs_m11_fewshot_10pct_metropolis_yaw20_rvl_strong_v2_full40_seed20260801_retry3_frozen_test/summary.json`。
- 零样本迁移记录：`docs/experiments/m11_directional_generalization_protocol_2026-08-01.md` 的跨场景结果节。

本报告只整理已有产物，没有修改默认模型、前端资产、M4/M5 结果或 M13 test 协议。
