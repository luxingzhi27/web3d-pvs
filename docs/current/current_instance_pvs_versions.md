# 当前实例级 PVS 版本

更新时间：2026-08-13

本文只记录当前仍可运行、仍有明确用途的版本。历史实验输出可以留在独立的 `model/out` 或 `benchmark/out` 目录中，但不属于默认前端或默认训练入口。

## 场景与默认前端模型

| 场景 | 实例数 | GLB 数 | 前端模型 | 前端阈值 |
|---|---:|---:|---|---:|
| HKUST v3 | 18,831 | 3,273 | `pvs_directional_occlusion_proxy_encoder_rvl_strong_v2_full40_best` | `0.02` |
| IFCBench Fantasy Metropolis 实例化 v2 | 41,298 | 3,669 | `pvs_directional_occlusion_proxy_ifcbench_fantasy_metropolis_instanced_v2_k4_full40_best` | 由各自导出元数据冻结 |

前端实际模型映射以 `slm2viewer/src/neuralCullingBackendMode.js` 为准。训练 benchmark 中的 `pvs_directional_occlusion_proxy_encoder_rvl_w042_full40_best` 是 HKUST 历史/基线 runner，不等同于当前前端的 `rvl_strong_v2` 资产。

## 保留训练输出

当前默认前端对应的主要训练输出：

```text
neural_instance_culling/model/out/pvs_directional_occlusion_proxy_encoder_rvl_strong_v2_full40_hkust_spatial_fov66_seed20260801_protocolfix_retry2
neural_instance_culling/model/out/pvs_directional_occlusion_proxy_ifcbench_fantasy_metropolis_instanced_v2_k4_full40
```

当前正式研究证据和后续关系数据源也必须保留：

```text
neural_instance_culling/model/out/pvs_direction_depth_relation_survival_constrained_v1_surface_fallback_owrb_formal40_pooled_context_20260811
neural_instance_culling/benchmark/out/pvs_direction_depth_relation_survival_constrained_v1_surface_fallback_owrb_formal40_pooled_context_20260811
neural_instance_culling/model/out/pvs_ray_context_survival_owrb_supplement_v1_formal40_20260812
neural_instance_culling/benchmark/out/pvs_ray_context_survival_owrb_supplement_v1_formal40_20260812
neural_instance_culling/benchmark/out/pvs_ray_context_survival_owrb_v1_subpose5_20260811_directchrome
```

前四个目录包含 40 epoch 三种子 checkpoint、固定特征表、训练日志、校准摘要、10,000 次 paired bootstrap 和硬件 Color-ID 评价。最后一个目录保存 train-only 三角形深度层、surface fallback、合并缓存及 NVIDIA Vulkan/ANGLE GPU evidence，是后续构建无来源 top-k 截断遮挡关系 CSR 的依赖。新的优化实验必须使用计划登记的新目录，不能覆盖或清理上述正式结果与证据。

## 前端运行资产

```text
slm2viewer/assets/neural_instance_culling/pvs_directional_occlusion_proxy_encoder_rvl_strong_v2_full40_best
slm2viewer/assets/neural_instance_culling/pvs_directional_occlusion_proxy_ifcbench_fantasy_metropolis_instanced_v2_k4_full40_best
slm2viewer/public/assets/neural_instance_culling/pvs_directional_occlusion_proxy_encoder_rvl_strong_v2_full40_best
slm2viewer/public/assets/neural_instance_culling/pvs_directional_occlusion_proxy_ifcbench_fantasy_metropolis_instanced_v2_k4_full40_best
slm2viewer/public_deploy/assets/neural_instance_culling/
```

运行资产包含 FP16 固定实例特征、模型权重和 `instance_model_meta.json`。浏览器不读取训练 checkpoint、点云缓存或方向证据源列表。

## 当前运行链路

1. 后退 `66°` 相机对实例 AABB 做候选筛选。
2. Worker/WebGPU 对候选实例输出可见性分数和 GLB 下载优先级。
3. 主线程使用真实 `60°` 相机再次做实例级视锥过滤。
4. 按实例更新 instanced mesh；GLB 只负责下载、解码和缓存聚合。
5. 只有相机超出 `CameraPredictionGate` 的预测范围时才重新推理；未命中的历史实例不能永久驻留显示。

## 数据与模型口径

- 采样和模型查询垂直 FOV 固定为 `66°`，真实渲染垂直 FOV 固定为 `60°`。
- 前端运行时只使用离线固定几何、上下文和遮挡代理特征，不运行 PointNet++、Graph U-Net、dynamic-pool、Triplane 或动态图邻居传播。
- 实例可以复用同一个 GLB 原型，但 AABB、实例编号和显示状态始终独立。
- `visible_weights` 用于重要性召回和效用监督；如果来源是历史 rvcServer `component_weights`，不能称为真实像素覆盖率。

## 阈值和评价

正式阈值在 calibration split 冻结，优先满足 `weighted recall > 0.99` 及其置信下界要求，再比较 precision、F1、useful cull、GLB 字节和延迟。当前优化阶段暂不把 `bad cull` 置信区间上界作为路线否决条件，但 `bad cull` 仍必须报告。

当前论文模型、正式 validation 结果和代码入口见 [`pvs_mainline_2026-08-23.md`](pvs_mainline_2026-08-23.md)。2026-08-11 修正正式矩阵与 [Fourier 补充矩阵](../evaluation/pvs_ray_context_survival_owrb_supplement_v1_formal40_20260812_formal_validation.md)继续作为保留基线，不再作为当前训练入口。
