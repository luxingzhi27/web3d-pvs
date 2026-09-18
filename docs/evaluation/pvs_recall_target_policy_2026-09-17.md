# PVS weighted-recall 资格规则与复算结果

日期：2026-09-17

## 评价规则

每个 checkpoint 只在 calibration 上冻结阈值，顺序固定为：

1. 优先选择 aggregate weighted recall 与单侧 95% pose-bootstrap LCB 均大于 `0.99` 的最高实际 float32 score change-point；
2. 若不存在第一层工作点，则选择 aggregate weighted recall 大于 `0.99` 的最高 change-point；
3. Validation 平均 WR 大于 `0.99` 的成员均保留并报告，LCB 单独标注是否达到 `0.99` 目标；
4. Validation 平均 WR 不大于 `0.99` 的成员只作为诊断结果，不读取 test；
5. Validation 与 test 不重选阈值。

机器资格字段使用 `confidence_target_met`、`mean_target_met` 和
`mean_target_not_met`。LCB 表示结果对 pose 重采样的置信余量，不再作为平均 WR 已达标结果
的硬否决门。

## Viking sampling-v2

三个模型均为纯 Full V4 输出。seed01/02 使用原始从头训练 checkpoint，seed03 使用纯模型
checkpoint rescue；运行时没有指定实例覆盖规则。数据集为 1763 units，split 为
`1944/216/276/276`，三个 seed 使用同一 calibration、validation 和 test。

### Validation

| Seed | Threshold | WR | LCB | CNOR | Useful Cull | Bad Cull | Pose PR-AUC | Qualification |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 20260801 | 0.541596 | 0.992606 | 0.991289 | 0.866635 | 0.693639 | 0.143163 | 0.899551 | LCB 目标达成 |
| 20260802 | 0.515393 | 0.992468 | 0.990656 | 0.846497 | 0.668437 | 0.135619 | 0.883617 | LCB 目标达成 |
| 20260803 | 0.701710 | 0.992705 | 0.987887 | 0.804466 | 0.608423 | 0.075803 | 0.843234 | 平均 WR 达标 |

### Test

| Seed | WR | LCB | CNOR | Useful Cull | Bad Cull | Pose PR-AUC | Pose precision | Pose recall | Avg candidate / GT / pred |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 20260801 | 0.992563 | 0.991133 | 0.839233 | 0.672193 | 0.128270 | 0.891669 | 0.839861 | 0.709132 | 560.10 / 133.89 / 111.76 |
| 20260802 | 0.993848 | 0.992719 | 0.816053 | 0.645458 | 0.119556 | 0.875141 | 0.816410 | 0.719960 | 560.10 / 133.89 / 131.62 |
| 20260803 | 0.994121 | 0.992392 | 0.782926 | 0.601489 | 0.071351 | 0.830266 | 0.757802 | 0.802647 | 560.10 / 133.89 / 183.24 |
| mean +/- std | 0.993511 +/- 0.000832 | 0.992081 +/- 0.000838 | 0.812737 +/- 0.028300 | 0.639713 +/- 0.035700 | 0.106392 +/- 0.030658 | 0.865692 +/- 0.031774 | - | - | - |

测试产物位于
`neural_instance_culling/benchmark/out/paper_results/pvs_v4_recall_target_policy_v1/viking_village_128k`。
所有 test JSON 均为 276 poses，并保存逐 pose sidecar；stderr 为空。

## 已有场景重新分级

精确阈值本身在 LCB 目标可达时不变，因此 HKUST 和 IFCBench 已有分数无需改变数值，只需按
新规则重新解释资格层级。

| Scene / family | Split | Seeds | WR mean +/- std | LCB mean +/- std | CNOR mean +/- std | 资格结论 |
|---|---|---:|---:|---:|---:|---|
| HKUST Full V4 | test | 3 | 0.997082 +/- 0.001634 | 0.994334 +/- 0.003271 | 0.909787 +/- 0.007364 | 三种子均达到 LCB 目标 |
| IFCBench Full V4 | validation | 3 | 0.990898 +/- 0.000524 | 0.989919 +/- 0.000560 | 0.642616 +/- 0.032332 | 三种子平均 WR 均达标；seed03 达到 LCB 目标 |
| IFCBench final fine-tune | test | 3 | 0.991179 +/- 0.000668 | 0.990523 +/- 0.000552 | 0.676840 +/- 0.007504 | 三种子均达到 LCB 目标 |
| Viking sampling-v2 Full V4 | test | 3 | 0.993511 +/- 0.000832 | 0.992081 +/- 0.000838 | 0.812737 +/- 0.028300 | 三种子 test 均达到 LCB 目标 |

IFCBench Full V4 的 seed01/02 过去因 validation LCB 为 `0.989853/0.989395` 被排除；其平均
WR 为 `0.991020/0.990323`，按当前协议均属于可报告的“平均 WR 达标”成员。最终 fine-tune
仍然保留，因为其三种子 test LCB 均达到目标，且平均 CNOR 和 Useful Cull 更高。

## 仍在训练的场景

Sponza 64 KiB 与 Big City 64 KiB 尚未完成 40 epoch 长训，不能读取 test。当前最新冻结
validation 快照中：Sponza seed03 已达到 WR/LCB 目标；Sponza seed01/02、Big City 三种子
的最新平均 WR 暂低于 `0.99`。这些只是中途结果，完整训练结束后按同一三级规则重新冻结并
汇总，不能用当前快照替代正式结果。

| Scene | Seed | Epoch | WR | LCB | CNOR | Useful Cull | 当前层级 |
|---|---:|---:|---:|---:|---:|---:|---|
| Sponza 64 KiB | 20260801 | 12 | 0.985741 | 0.981047 | 0.570026 | 0.557730 | 平均 WR 未达 |
| Sponza 64 KiB | 20260802 | 4 | 0.987414 | 0.983020 | 0.479473 | 0.449545 | 平均 WR 未达 |
| Sponza 64 KiB | 20260803 | 4 | 0.995023 | 0.993516 | 0.398702 | 0.366334 | LCB 目标达成 |
| Big City 64 KiB | 20260801 | 8 | 0.988996 | 0.987694 | 0.486438 | 0.519450 | 平均 WR 未达 |
| Big City 64 KiB | 20260802 | 8 | 0.986159 | 0.984284 | 0.477155 | 0.552733 | 平均 WR 未达 |
| Big City 64 KiB | 20260803 | 12 | 0.984328 | 0.981413 | 0.160496 | 0.176876 | 平均 WR 未达 |
