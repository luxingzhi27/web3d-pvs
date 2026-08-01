# M0 Metropolis 正式 frozen test

日期：2026-08-01
状态：正式 one-shot test 已完成；结果可作为当前空间协议的 test 证据，M0 其他场景和图像/设备门仍按各自报告判断。

## 评价口径

本次使用 IFCBench Fantasy Metropolis 实例化 v2 的空间隔离 Pose CSR 数据集，读取预先保存的后退相机候选集合。
阈值先在独立 calibration split 选择并冻结，test 只使用一个阈值运行一次；评测器不扫描 test 阈值、不把真实可见实例
补回候选、不裁剪候选，也不使用放回抽样。test 遍历全部 `2,340` 个唯一 pose，`testEvaluationCount=1`。

模型输入和候选后退相机使用 66°视场角，真实前端渲染视场角保持 60°。`weighted recall` 使用采样记录的
`visible_weights`，它是当前数据管线的弱可见性权重，不是真实像素覆盖率。`useful cull` 只统计正确剔除的不可见候选，
`bad cull` 统计被错误剔除的真实可见候选。

冻结 test 清单和摘要：

```text
neural_instance_culling/benchmark/out/m0_frozen_pvs_directional_occlusion_proxy_encoder_rvl_strong_v2_full40_metropolis_spatial_fov66_seed20260801_protocolfix_bs1_gpu2/frozen_manifest.json
neural_instance_culling/benchmark/out/m0_frozen_pvs_directional_occlusion_proxy_encoder_rvl_strong_v2_full40_metropolis_spatial_fov66_seed20260801_protocolfix_bs1_gpu2/test/summary.json
```

## 结果

冻结阈值为 `0.3199999928`。以下是 pose-level 指标的平均值，和跨全部 test pose 合并后的 aggregate 指标分开列出：

| 指标 | Pose-level 平均 | Aggregate | 中文含义 |
|---|---:|---:|---|
| Precision | 0.268367 | 0.170008 | 被预测保留的候选中，实际可见候选的比例 |
| Recall | 0.931414 | 0.922928 | 每个 pose 中真实可见候选被找回的比例 |
| Weighted recall | 0.992120 | 未提供同义 aggregate 主结论 | 按 `visible_weights` 加权后的可见实例找回比例 |
| F1 | 0.383984 | 0.287126 | precision 与 recall 的调和平均 |
| Accuracy | 0.662666 | 0.668754 | 候选集合中 TP 与 TN 的总体正确率 |
| Balanced accuracy | 0.777225 | 0.785940 | 可见正类 recall 与不可见负类 specificity 的平均 |
| Specificity | 0.623036 | 0.648951 | 不可见候选被正确剔除的比例 |
| Useful cull | 0.523903 | 未提供同义 aggregate 主结论 | 正确剔除的不可见候选占候选集合的比例 |
| Bad cull | 0.007593 | 未提供同义 aggregate 主结论 | 错误剔除的真实可见候选占候选集合的比例 |

资源规模和预测数量：

| 项目 | 数值 |
|---|---:|
| Test pose 数 | 2,340 |
| 平均候选实例数 | 9,982.64 |
| 平均真实可见实例数 | 721.53 |
| 平均预测保留实例数 | 3,917.02 |
| 平均候选 GLB 数 | 1,357.87 |
| 平均预测 GLB 数 | 724.34 |
| 平均候选 GLB 字节 | 53.10 MB |
| 平均预测 GLB 字节 | 21.58 MB |

## 安全解释

weighted recall 点估计为 `0.992120`，高于当前 `0.99` 安全目标；普通 pose recall 为 `0.931414`，说明
大量低权重实例仍存在漏检，不能把 weighted recall 单独解释成所有构件的画面安全。`bad cull=0.007593`
表示约 0.76% 的候选决策属于错误剔除，后续图像级评价仍需要检查这些漏检是否集中于高视觉贡献构件。

当前结果也说明 Metropolis 的候选规模明显高于 HKUST，逐实例 precision 下降和前端推理负担增加具有直接关系。
因此不能只用 `avg_pred / avg_candidate` 作为效率结论，必须结合有效剔除、错误剔除、图像 miss-pixel 和下载字节结果。

## 协议与保留判断

- 本次使用严格保存候选集合；candidate 中没有真实可见集合之外的修补步骤。
- test 没有阈值扫描，`testEvaluationCount=1`，并使用独立冻结清单。
- 本结果作为 Metropolis M0 正式 test 证据保留。
- M5 实例级图像安全、M8 真实网络轨迹、M10 真实设备 WebGPU 和 M11 泛化结果不能由本报告替代。
