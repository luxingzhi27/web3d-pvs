# NeuralStreamWeb3D 文档索引

更新时间：2026-08-23

仓库文档只索引当前可运行系统、当前论文模型和仍需复现的正式基线。失败的旧整合网络、尾部分离探针、独立安全前沿 runner 及重复计划已删除；历史结论仍可从 Git 提交记录追溯，不再占用当前入口。

## 当前主线

| 文档 | 内容 |
|---|---|
| [`current/pvs_mainline_2026-08-23.md`](current/pvs_mainline_2026-08-23.md) | 当前模型架构、代码入口、数据依赖、正式指标与保留边界 |
| [`experiments/pvs_mainline_training_2026-08-21.md`](experiments/pvs_mainline_training_2026-08-21.md) | 八组扫描、三种子完整模型和四个核心消融的固定训练协议 |
| [`evaluation/pvs_mainline_validation_2026-08-23.md`](evaluation/pvs_mainline_validation_2026-08-23.md) | 10,000 次 bootstrap 重审计、三种子 validation 指标和消融结论 |
| [`evaluation/unified_pvs_metrics_evaluation.md`](evaluation/unified_pvs_metrics_evaluation.md) | pose/aggregate、weighted recall、accuracy、balanced accuracy、useful cull 和资源指标定义 |
| [`evaluation/test_split_benchmark_protocol.md`](evaluation/test_split_benchmark_protocol.md) | train、calibration、validation 和冻结 test 的职责边界 |

当前代码入口：

```text
neural_instance_culling/model/pvs_model.py
neural_instance_culling/model/train_pvs.py
neural_instance_culling/model/export_pvs.py
neural_instance_culling/benchmark/run_pvs.py
neural_instance_culling/benchmark/evaluate_pvs.py
neural_instance_culling/benchmark/summarize_pvs.py
neural_instance_culling/benchmark/reaudit_pvs.py
```

## 数据与系统

| 文档 | 内容 |
|---|---|
| [`current/current_instance_pvs_versions.md`](current/current_instance_pvs_versions.md) | 当前部署场景、模型、阈值和前端资产边界 |
| [`current/neuralstreamweb3d_architecture_technical.md`](current/neuralstreamweb3d_architecture_technical.md) | 离线编码、实例可见性查询、渲染过滤和 GLB 调度架构 |
| [`current/neuralstreamweb3d_model_pipeline.md`](current/neuralstreamweb3d_model_pipeline.md) | 当前部署方向代理模型的训练与导出链 |
| [`current/neuralstreamweb3d_dataset_protocol.md`](current/neuralstreamweb3d_dataset_protocol.md) | 66° view-cell、subpose、Color-ID、候选和 CSR 数据语义 |
| [`current/hardware_gpu_execution_policy.md`](current/hardware_gpu_execution_policy.md) | Chrome NVIDIA Vulkan/ANGLE 硬件采样和 WebGPU 证据要求 |
| [`evaluation/viewcell_image_per_evaluation.md`](evaluation/viewcell_image_per_evaluation.md) | Color-ID 图像漏检、错误实例和额外实例评价 |

统一相机口径为真实渲染 `60°`，采样、后退候选和模型查询 `66°`。阈值只能由 checkpoint 自己的 calibration split 冻结；validation 用于配置与 checkpoint 比较，test 只能在模型和阈值冻结后读取。

## 前端与部署

| 文档 | 内容 |
|---|---|
| [`frontend/neuralstreamweb3d_runtime_implementation.md`](frontend/neuralstreamweb3d_runtime_implementation.md) | Worker 候选、WebGPU 推理、真实视锥过滤和实例级渲染 |
| [`frontend/lightweight_frontend_pvs_scheduler.md`](frontend/lightweight_frontend_pvs_scheduler.md) | 当前预测门控、资源驻留和调度逻辑 |
| [`frontend/neuralstreamweb3d_deployment_assets.md`](frontend/neuralstreamweb3d_deployment_assets.md) | 导出、按场景打包和 nginx 部署资产 |
| [`../slm2viewer/README_DEPLOY.md`](../slm2viewer/README_DEPLOY.md) | 发布包上传和远端验证命令 |

当前部署仍使用已冻结的方向遮挡代理模型。论文主线训练结果尚未替换默认 checkpoint、阈值或前端资产。

## 保留实验

以下历史实验仍有正式复现价值，因此保留代码、文档和本地结果：

| 文档 | 保留原因 |
|---|---|
| [`experiments/pvs_ray_context_survival_owrb_protocol_2026-08-11.md`](experiments/pvs_ray_context_survival_owrb_protocol_2026-08-11.md) | 2026-08-11 修正正式矩阵及硬件深度层缓存的数据协议 |
| [`experiments/pvs_ray_context_survival_owrb_literature_matrix_2026-08-11.md`](experiments/pvs_ray_context_survival_owrb_literature_matrix_2026-08-11.md) | 相关工作与实现边界 |
| [`evaluation/pvs_direction_depth_relation_survival_constrained_v1_surface_fallback_owrb_formal40_pooled_context_20260811_formal_validation.md`](evaluation/pvs_direction_depth_relation_survival_constrained_v1_surface_fallback_owrb_formal40_pooled_context_20260811_formal_validation.md) | 8×3×40 正式矩阵及硬件图像评价 |
| [`evaluation/pvs_ray_context_survival_owrb_supplement_v1_formal40_20260812_formal_validation.md`](evaluation/pvs_ray_context_survival_owrb_supplement_v1_formal40_20260812_formal_validation.md) | Fourier、上下文容量和单调参数化补充矩阵 |
| [`evaluation/hkust_fov66_rvl_w042_full40_test_2026-07-31.md`](evaluation/hkust_fov66_rvl_w042_full40_test_2026-07-31.md) | 当前 HKUST 部署基线的冻结 test 结果 |
| [`evaluation/ifc_metropolis_instanced_v2_test_eval_2026-07-15.md`](evaluation/ifc_metropolis_instanced_v2_test_eval_2026-07-15.md) | Metropolis 部署基线的冻结 test 结果 |

## 文档规则

- `current/` 记录当前系统和研究主线，`evaluation/` 记录指标协议与正式结果，`frontend/` 记录运行和部署，`experiments/` 只保留仍可复现的正式实验协议。
- 结果必须同时报告画面安全、分类诊断、有效剔除、资源效率和运行成本。
- 已失败且不再复现的实验直接从当前文档和 runner 删除，通过 Git 历史追溯，不保留兼容入口。
