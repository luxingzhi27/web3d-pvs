# M11 Metropolis 5% 少样本冻结评测

日期：2026-08-02
阶段：M11 跨场景少样本适配
状态：5% 子门已完成；10% 适配仍在运行；不能替代 M13 三种子正式主结果

## 目的与协议

本实验检验 HKUST 方向模型迁移到 Metropolis 后，在目标场景只使用原生训练视点的 5% 进行适配时，能否保持画面安全并恢复有效剔除能力。迁移时只复制形状一致的可学习参数；Metropolis 的实例几何、实例到 GLB 映射、方向遮挡证据和固定运行特征均在目标场景重新建立。

模型查询视场角为 `66°`，真实渲染视场角为 `60°`。候选集合使用数据集保存的严格后退视锥候选，不补入真实可见实例、不截断候选。训练使用 40 个 epoch、每 epoch 900 steps、单 pose 训练批次、AMP 和 `feature-export-batch-size=128`；这些参数只用于控制显存，不改变数据语义。

训练输出：

```text
neural_instance_culling/model/out/pvs_m11_fewshot_5pct_metropolis_yaw20_rvl_strong_v2_full40_seed20260801_retry3/
```

冻结清单和 test 输出：

```text
neural_instance_culling/benchmark/out/m11_pvs_m11_fewshot_5pct_metropolis_yaw20_rvl_strong_v2_full40_seed20260801_retry3_frozen_manifest.json
neural_instance_culling/benchmark/out/m11_pvs_m11_fewshot_5pct_metropolis_yaw20_rvl_strong_v2_full40_seed20260801_retry3_frozen_test/summary.json
```

训练稳定性审计：

```text
neural_instance_culling/benchmark/out/m11_audit/pvs_m11_fewshot_5pct_stability.json
```

## 数据与冻结来源

| 项目 | 数值 |
|---|---:|
| 适配训练视点 | 1,023 |
| 完整原生 train 视点 | 20,454 |
| validation / calibration / test | 1,149 / 2,271 / 2,271 |
| test pose digest | `dd157d2252edf65b` |
| checkpoint 选择 | 完整 validation |
| 阈值选择 | 独立 calibration |
| test 遍历 | 2,271 个唯一 pose，无放回 |
| test 阈值扫描 | 未执行 |
| test 评测次数 | 1 |

冻结阈值为 `0.05000000074505806`，来源是 checkpoint 的 calibration 工作点。calibration 点估计 weighted recall 为 `0.992787`，10,000 次 bootstrap 的单侧 95% 下界为 `0.992347`，满足点估计 `>=0.9925` 和下界 `>=0.99` 的预注册安全门。

## Frozen test 结果

| 指标 | 数值 |
|---|---:|
| pose weighted recall | 0.990595 |
| pose recall | 0.947238 |
| pose precision | 0.307265 |
| pose F1 | 0.454879 |
| pose accuracy | 0.741969 |
| pose balanced accuracy | 0.828415 |
| pose specificity | 0.709592 |
| aggregate recall | 0.938765 |
| aggregate precision | 0.280205 |
| aggregate accuracy | 0.774713 |
| useful cull = TN / candidate | 0.626039 |
| bad cull = FN / candidate | 0.005463 |
| 平均 candidate / GT / prediction | 11,481.53 / 1,046.06 / 3,504.58 |
| 平均候选 GLB 字节 / 预测 GLB 字节 | 137.04 MB / 30.94 MB |
| GLB 字节削减 | 0.7741 |

这里的 weighted recall 按 `visible_weights` 对真实可见实例加权，表示重要可见实例被找回的比例；它不惩罚 false positive。普通 pose recall 表示每个 pose 内真实可见实例集合被找回的比例，因而仍需与 weighted recall 同时报告。useful cull 只统计候选中被正确剔除的不可见实例，bad cull 统计被错误剔除的真实可见实例，分别反映效率和画面风险。

## 稳定性与质量门判断

完整 40 epoch 历史审计结果为 `warning`：

- 非有限 loss：`0`；
- 非有限指标：`0`；
- AMP 非有限梯度跳过：`19` 次，共 `36,000` 个记录 step；
- 梯度跳过率：`0.0528%`。

因此该训练过程可以作为“优化过程保持有限、但发生过少量 AMP 梯度跳过”的正式证据，不能写成“全程无异常”。如果最终主线需要 clean stability claim，应另行注册 FP32 或稳定 AMP 重跑，不能覆盖本次 checkpoint 或 test。

5% 适配满足 weighted-recall 安全门，普通 pose recall 为 `0.947238`，略低于项目期望的 `0.95`，useful cull 已明显高于此前 1% 适配的 `0.279348`。因此本结果支持“5% 目标场景适配能够恢复一部分有效剔除能力”的有限结论，不支持跨场景零样本通用、图像近似无损或最终泛化主张。M11 是否通过更高层质量门，仍需 10% 适配、按航向的分层结果和与完整场景训练的同口径比较。

## 可复现命令与保留决定

训练由以下已注册队列执行：

```bash
bash neural_instance_culling/benchmark/run_m11_metropolis_fewshot.sh
```

冻结 test 由队列在 calibration-ready 后执行 `evaluate_frozen_test.py prepare/evaluate`，test 只读取冻结阈值。5% 输出、manifest、stdout/stderr 和稳定性审计均保留；该结果不进入最终 M13 主表，直到最终架构、数据和三种子协议冻结。
