# `pvs_hierarchical_relation_survival_integrated_spectral_quality_rvl_v1` Pilot

日期：2026-08-13

## 结论

本报告汇总三个 pilot validation summary。pilot 的 calibration workpoint 只在
32 pose 上选择；完整 validation replay 覆盖 213 pose。三个矩阵的所有成员在
完整 validation 上 weighted recall 都低于 0.99，因此当前结果不能作为画面安全
主门通过，也不能进入 formal80。后续应先修正或解释 calibration 与完整
validation 的分布/口径差异，再重新进行 pilot。

`testRead` 在三个汇总文件中均为 `false`。本报告没有读取或使用 test split。
当前默认模型、默认 checkpoint、前端资产和默认 runner 均保持不变。

## 输入与口径

读取的三个独立汇总文件：

- `neural_instance_culling/benchmark/out/pvs_hierarchical_relation_survival_integrated_spectral_quality_rvl_v1_pilot_20260813_validation_summary.json`
- `neural_instance_culling/benchmark/out/pvs_hierarchical_relation_survival_integrated_spectral_quality_rvl_v1_loss_pilot_20260813_retry2_validation_summary.json`
- `neural_instance_culling/benchmark/out/pvs_hierarchical_relation_survival_integrated_spectral_quality_rvl_v1_spectral_pilot_20260813_retry3_validation_summary.json`

三个文件的 schema 均为
`pvs-hierarchical-relation-survival-integrated-validation-summary-v1`，split 均为
`validation`，seed 均为 `20260801`，并使用相同的原生候选集合。

校准阶段记录的 32 pose 只用于冻结阈值，不能替代完整 validation。完整 validation
指标使用汇总中的 `aggregate.weightedRecall`；画面安全目标为 weighted recall
大于 0.99，且其单侧下界也大于 0.99。普通 recall、precision、useful cull 和
bad cull 只作诊断，不能抵消 weighted recall 未达安全门的事实。

## 成员结果

数值均为 aggregate 结果；`val LCB` 是完整 validation 的 weighted recall
下界。每行的 calibration pose 数均为 32，完整 validation pose 数均为 213。

| 矩阵 | 成员 | calibration weighted recall / LCB | validation weighted recall / LCB | validation recall | validation precision | useful cull | bad cull | avg pred |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| relation | `R0_free_survival` | 0.995806 / 0.992172 | 0.867953 / 0.838086 | 0.046117 | 0.424528 | 0.952965 | 0.042226 | 24.88 |
| relation | `R1_single_scale_relation` | 0.996771 / 0.993871 | 0.883763 / 0.853835 | 0.072701 | 0.308060 | 0.948504 | 0.041049 | 54.06 |
| relation | `R2_hierarchical_relation` | 0.995208 / 0.990436 | 0.846363 / 0.809942 | 0.032466 | 0.461135 | 0.954053 | 0.042830 | 16.13 |
| relation | `R3_hierarchical_shuffled_source` | 0.995358 / 0.990417 | 0.860248 / 0.825271 | 0.044949 | 0.401648 | 0.952769 | 0.042278 | 25.63 |
| loss | `L0_rvl_strong_v2` | 0.995088 / 0.990876 | 0.818244 / 0.775147 | 0.026194 | 0.463547 | 0.954391 | 0.043108 | 12.94 |
| loss | `L1_quality_risk` | 0.995089 / 0.990880 | 0.820079 / 0.777253 | 0.026502 | 0.464607 | 0.954381 | 0.043094 | 13.07 |
| loss | `L2_quality_resource_r005` | 0.998421 / 0.997076 | 0.870189 / 0.838840 | 0.044129 | 0.421661 | 0.953053 | 0.042314 | 23.97 |
| loss | `L3_quality_resource_r010` | 0.998421 / 0.997076 | 0.870137 / 0.838760 | 0.044272 | 0.421299 | 0.953041 | 0.042307 | 24.07 |
| spectral | `S0_fourier117` | 0.995205 / 0.990704 | 0.838184 / 0.798902 | 0.031052 | 0.463444 | 0.954141 | 0.042893 | 15.35 |
| spectral | `S1_learned_spectral_point` | 0.999667 / 0.999427 | 0.877422 / 0.847091 | 0.046855 | 0.412189 | 0.952775 | 0.042193 | 26.04 |
| spectral | `S2_integrated_spectral` | 0.999667 / 0.999427 | 0.878301 / 0.848181 | 0.047244 | 0.410508 | 0.952729 | 0.042176 | 26.36 |

所有完整 validation weighted recall 均未达到 0.99；最高为
`R1_single_scale_relation` 的 0.883763，仍明显低于安全门。`S2` 相比同矩阵的
`S1` 只有轻微的 validation weighted recall 增量，尚不足以证明区域积分频谱带来
可靠的安全收益。

## 解释与限制

- calibration 的 32 pose 选点均可出现 weighted recall 及下界超过 0.99 的记录，
  但完整 213 pose replay 明显下降。这说明当前 pilot 的阈值冻结证据覆盖不足，
  不能把 calibration 的 `safeWorkpoint` 字段解释为完整 validation 安全通过。
- 三个矩阵共享同一 validation 候选集合，比较口径一致；但每个矩阵
  只有一个 seed，结果仍属于快速 pilot，不能替代多 seed 正式实验。
- 当前 evaluator 的 image metrics 标记为 `not_available`，因此本报告没有图像
  miss pixel、wrong ID pixel 或浏览器硬件延迟结论。
- `testRead=false` 只说明本次训练/校准/validation 流程没有读取 test；它不表示
  模型已通过正式 test 验收。

## 决策

当前路线暂缓 `formal80`。不修改默认模型，不替换默认 checkpoint，不更新前端
资产。下一次实验至少应扩大 calibration 覆盖并复核 32 pose 与 213 pose 的 split
和采样口径，随后重新完成 validation safety gate；在此之前不得把本 pilot 的
任何成员标记为主线模型。
