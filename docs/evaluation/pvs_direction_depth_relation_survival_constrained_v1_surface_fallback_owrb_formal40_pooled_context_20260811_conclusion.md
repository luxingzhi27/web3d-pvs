# 视线关系场、生存场与 OWRB 正式结论

日期：2026-08-11

## 评价范围

- 完整三因素矩阵：8 个变体 × 3 个随机种子 × 40 epoch。
- validation：213 个 pose；test：238 个 pose。
- 所有成员使用相同的后退相机候选集合、实例 GT 和 pose 顺序。
- 阈值由每个 checkpoint 的 calibration split 独立冻结；test 只在 validation 选择完成后读取一次。
- paired bootstrap：10,000 次，外层按 seed 聚类，内层在 seed 内按 pose 重采样。
- Color-ID 图像评价：24 个成员均通过 NVIDIA Vulkan/ANGLE 硬件 WebGL 门。
- WebGPU parity：使用 Playwright 无头 Chrome 完成 8 个 case 的软件后端数值校验，最大绝对误差为 `2.861e-6`、最大相对误差为 `6.787e-6`；适配器为 `google/swiftshader`，`formalReady=false`，因此不能把该结果或延迟写成 NVIDIA WebGPU 硬件性能。

完整逐成员指标和所有成对置信区间见[validation 明细报告](pvs_direction_depth_relation_survival_constrained_v1_surface_fallback_owrb_formal40_pooled_context_20260811_formal_validation.md)。

## 主要结果

| 变体 | pose precision | pose weighted recall | pose useful cull | pose bad cull | aggregate weighted recall | aggregate useful cull | 平均预测数 | GLB 字节削减 | 前向 p95 (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pooled / 无生存场 / RVL | 0.3486 | 0.9887 | 0.5861 | 0.0258 | 0.9894 | 0.9071 | 400.9 | 46.6% | 1.61 |
| directional / 无生存场 / RVL | 0.3368 | 0.9907 | 0.5733 | 0.0214 | 0.9916 | 0.8945 | 481.5 | 45.4% | 1.76 |
| pooled / 生存场 / RVL | 0.4318 | 0.9900 | 0.6683 | 0.0309 | 0.9907 | 0.9255 | 311.9 | 55.3% | 2.93 |
| directional / 生存场 / RVL | 0.4319 | 0.9903 | 0.6673 | 0.0306 | 0.9912 | 0.9241 | 324.7 | 55.2% | 3.89 |
| pooled / 无生存场 / 安全约束 | 0.2889 | 0.9945 | 0.5442 | 0.0201 | 0.9942 | 0.9039 | 414.6 | 36.1% | 2.16 |
| directional / 无生存场 / 安全约束 | 0.2659 | 0.9959 | 0.5249 | 0.0170 | 0.9956 | 0.8833 | 535.4 | 31.9% | 1.69 |
| pooled / 生存场 / 安全约束 | 0.2221 | 0.9979 | 0.4211 | 0.0111 | 0.9979 | 0.5682 | 2214.9 | 23.9% | 2.15 |
| directional / 生存场 / 安全约束 | 0.2248 | 0.9979 | 0.4376 | 0.0108 | 0.9979 | 0.5985 | 2058.3 | 25.4% | 2.13 |

这些是三个 seed 的均值，用于概览，不替代报告中的 pose 宏平均、aggregate 合并值和置信区间。安全约束损失确实提高了 weighted recall 并降低了 bad cull，但代价是预测数量上升、precision 和有效剔除下降；因此不能仅凭 recall 提升判定为系统效率提升。

## 因素判定

| 因素 | 观察到的稳定变化 | 正式判定 |
|---|---|---|
| 球面方向上下文 | pose precision 的差值区间为 `[-0.0186, -0.0004]`，aggregate precision 也为负；useful cull、资源字节和图像 miss-pixel 没有稳定改善 | 暂不作为贡献，上下文表示需要重新设计 |
| 单调遮挡生存场 | pose precision `+0.0053~+0.0290`、pose F1 `+0.0069~+0.0230`，但 validation 安全资格跨越包含 weighted recall 不足的 RVL 成员，且 aggregate useful cull 区间为负 | 有分类诊断价值，但当前不能作为安全系统主线 |
| 安全约束效用损失 | weighted recall 增益稳定，但 precision、balanced accuracy、F1、useful cull 和预测资源均恶化；图像 miss-pixel 虽下降，代价是过预测和更高下载压力 | 作为高召回控制组保留，不替换当前主线 |

综合路线判定为 `degrade_to_auxiliary_or_failed_ablation`，没有因素进入默认模型或默认前端资产。该结论不是说生存场没有任何信号，而是说明在预先登记的安全、分类、剔除和资源联合规则下，尚未证明它能成为可部署的主线改进。

## 冻结选择与 test

validation 选择了：

- 变体：`context_off_survival_off_safety`；
- seed：`20260802`；
- checkpoint：`pvs_direction_depth_relation_survival_constrained_v1_surface_fallback_owrb_formal40_pooled_context_20260811/context_off_survival_off_safety_seed20260802_e40/best.pt`；
- 固定阈值：`0.20000000298023224`；
- runtime 固定表：156 维，`5,875,272` bytes；
- 可见性头输入：177 维；ray 查询：9 维。

冻结成员在 test 的结果为：

| 指标 | pose 宏平均 | aggregate |
|---|---:|---:|
| precision | 0.2544 | 0.3623 |
| recall | 0.8659 | 0.6309 |
| weighted recall | 0.9957 | 0.9944 |
| F1 | 0.3404 | 0.4603 |
| balanced accuracy | 0.7266 | 0.7867 |
| useful cull | 0.5414 | 0.8962 |
| bad cull | 0.0245 | 0.0181 |
| 平均预测数 | 381.3 | 381.3 |
| GLB 字节削减 | 40.0% | 40.0% |

test 前向延迟为 mean `2.153 ms`、p50 `1.539 ms`、p95 `1.603 ms`、p99 `3.099 ms`。该 test 结果只验证冻结选择，不能反向用于修改阈值、路线或默认前端资产。

## 论文表述边界

本轮实验支持以下谨慎结论：离线生存场表征能够在部分控制条件下改善实例级分类诊断，安全约束损失能够有效把 weighted recall 推高到安全区间；但球面上下文当前没有稳定增益，安全约束的高召回收益伴随明显过预测和资源代价，生存场也尚未通过全因子安全路线。后续工作应优先改进上下文监督与预算感知损失，并保持无头 Playwright 的硬件 WebGL 采样证据、固定离线特征表和轻量前端查询路径。
