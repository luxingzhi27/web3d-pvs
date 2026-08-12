# NeuralStreamWeb3D 文档索引

更新时间：2026-08-13

这里的文档只服务于当前可运行版本、正式保留证据和正在执行的唯一模型优化计划。重复路线报告、被修正的旧汇总、失败 pilot 及其本地输出已经清理；当前前端资产、正式训练权重、正式 benchmark、采样数据和硬件证据保持不变。

## 先读这几份

| 文档 | 内容 |
|---|---|
| [`current/optimization_restart_2026-08-10.md`](current/optimization_restart_2026-08-10.md) | 当前优化阶段、最新 validation 结果和后续实验规则 |
| [`experiments/pvs_hierarchical_occlusion_survival_viewcell_spectral_query_2026-08-13.md`](experiments/pvs_hierarchical_occlusion_survival_viewcell_spectral_query_2026-08-13.md) | 当前唯一实施计划：分层关系生存网络、视点区域积分频谱、质量风险与资源排斥 RVL；先快速验证，再进行 80 epoch 三种子长训和留一消融 |
| [`current/current_instance_pvs_versions.md`](current/current_instance_pvs_versions.md) | 两个场景、默认模型、前端资产和保留边界 |
| [`current/neuralstreamweb3d_architecture_technical.md`](current/neuralstreamweb3d_architecture_technical.md) | 离线编码、模型查询、实例渲染和 GLB 调度的系统架构 |
| [`current/neuralstreamweb3d_model_pipeline.md`](current/neuralstreamweb3d_model_pipeline.md) | 模型输入、方向遮挡代理、RVL 损失、训练和导出 |
| [`current/neuralstreamweb3d_dataset_protocol.md`](current/neuralstreamweb3d_dataset_protocol.md) | 66° view-cell/subpose 采样、Color-ID、候选集合和 CSR 语义 |
| [`current/hardware_gpu_execution_policy.md`](current/hardware_gpu_execution_policy.md) | Chrome NVIDIA 硬件 GPU 采样和评价的准入与证据 |
| [`experiments/pvs_ray_context_survival_owrb_protocol_2026-08-11.md`](experiments/pvs_ray_context_survival_owrb_protocol_2026-08-11.md) | 2026-08-11 正式矩阵的历史协议，仅供复核保留结果和数据/GPU 口径；不得作为当前 8/16/80 epoch 计划的训练协议 |
| [`experiments/pvs_ray_context_survival_owrb_literature_matrix_2026-08-11.md`](experiments/pvs_ray_context_survival_owrb_literature_matrix_2026-08-11.md) | 相关工作与本项目实现边界的对照矩阵 |

## 前端与部署

| 文档 | 内容 |
|---|---|
| [`frontend/neuralstreamweb3d_runtime_implementation.md`](frontend/neuralstreamweb3d_runtime_implementation.md) | Worker 候选、WebGPU 推理、真实视锥过滤和实例级渲染 |
| [`frontend/lightweight_frontend_pvs_scheduler.md`](frontend/lightweight_frontend_pvs_scheduler.md) | 当前调度器、预测门控、驻留和运行资产 |
| [`frontend/neuralstreamweb3d_deployment_assets.md`](frontend/neuralstreamweb3d_deployment_assets.md) | 导出、按场景打包、模型压缩和 nginx 目录 |
| [`../slm2viewer/README_DEPLOY.md`](../slm2viewer/README_DEPLOY.md) | 发布包上传、远端目录和 nginx 验证命令 |

## 评价协议与保留结果

| 文档 | 内容 |
|---|---|
| [`evaluation/unified_pvs_metrics_evaluation.md`](evaluation/unified_pvs_metrics_evaluation.md) | pose/aggregate、weighted recall、accuracy、balanced accuracy、useful cull 和资源指标定义 |
| [`evaluation/test_split_benchmark_protocol.md`](evaluation/test_split_benchmark_protocol.md) | calibration、validation、test 的冻结关系 |
| [`evaluation/viewcell_image_per_evaluation.md`](evaluation/viewcell_image_per_evaluation.md) | Color-ID 图像漏检、错误实例和额外实例指标 |
| [`evaluation/pvs_direction_depth_relation_survival_constrained_v1_surface_fallback_owrb_formal40_pooled_context_20260811_formal_validation.md`](evaluation/pvs_direction_depth_relation_survival_constrained_v1_surface_fallback_owrb_formal40_pooled_context_20260811_formal_validation.md) | 2026-08-11 修正 pooled-context 定义后的正式 8×3×40 validation 汇总、因子效应、图像评价和硬件门状态 |
| [`evaluation/pvs_direction_depth_relation_survival_constrained_v1_surface_fallback_owrb_formal40_pooled_context_20260811_conclusion.md`](evaluation/pvs_direction_depth_relation_survival_constrained_v1_surface_fallback_owrb_formal40_pooled_context_20260811_conclusion.md) | 同一正式矩阵的路线判定、冻结成员、test 结果和论文结论 |
| [`evaluation/pvs_ray_context_survival_owrb_supplement_v1_formal40_20260812_formal_validation.md`](evaluation/pvs_ray_context_survival_owrb_supplement_v1_formal40_20260812_formal_validation.md) | 5 个补充机制变体、15 个成员、10,000 次 paired bootstrap 和硬件 Color-ID 图像评价 |
| [`evaluation/pvs_ray_context_survival_owrb_supplement_v1_formal40_20260812_conclusion.md`](evaluation/pvs_ray_context_survival_owrb_supplement_v1_formal40_20260812_conclusion.md) | 补充矩阵的上下文容量、Fourier ray、单调参数化和 AABB 证据结论 |
| [`evaluation/hkust_fov66_rvl_w042_full40_test_2026-07-31.md`](evaluation/hkust_fov66_rvl_w042_full40_test_2026-07-31.md) | HKUST 已部署模型的 FOV66 基线 test 结果 |
| [`evaluation/ifc_metropolis_instanced_v2_test_eval_2026-07-15.md`](evaluation/ifc_metropolis_instanced_v2_test_eval_2026-07-15.md) | IFCBench Metropolis 场景基线 test 结果 |

## 场景和入口

当前保留两个场景：

| 场景 | 路由 | 实例数 | GLB 数 |
|---|---|---:|---:|
| HKUST v3 | `?scene=hkust-v3` | 18,831 | 3,273 |
| IFCBench Fantasy Metropolis 实例化 v2 | `?scene=ifcbench_fantasy_metropolis_instanced_v2` | 41,298 | 3,669 |

前端默认模型映射以 [`slm2viewer/src/neuralCullingBackendMode.js`](../slm2viewer/src/neuralCullingBackendMode.js) 为准，训练 benchmark 默认映射以 [`neural_instance_culling/benchmark/model_runners.py`](../neural_instance_culling/benchmark/model_runners.py) 为准。两者的用途不同，不能仅凭目录名判断默认模型。

统一相机口径为：真实渲染 `60°`，采样、后退候选和模型查询 `66°`。正式浏览器采样和 Color-ID 评价必须使用 NVIDIA Vulkan/ANGLE 硬件 GPU，并保存页面后端、Chrome 日志及 `nvidia-smi`/`pmon` 证据。

## 编写规则

- 当前状态写入 `current/`，评价口径和结果写入 `evaluation/`，前端和部署写入 `frontend/`，正在进行的实验写入 `experiments/`。
- 每个新实验必须有独立名称、输入/输出、指标、阈值来源和保留条件。
- 结果必须区分画面安全、有效剔除、资源效率和运行成本；不能只写 precision、F1 或候选削减。
- test split 只能在模型、阈值和资产冻结后读取一次；validation 结果不能写成 test 结论。
