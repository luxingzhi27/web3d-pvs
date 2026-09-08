# NeuralStreamWeb3D 文档索引

更新时间：2026-09-08

仓库文档只索引当前可运行系统、当前论文模型和仍需复现的正式基线。失败的旧整合网络、尾部分离探针、独立安全前沿 runner 及重复计划已删除；历史结论仍可从 Git 提交记录追溯，不再占用当前入口。

## 当前主线

| 文档 | 内容 |
|---|---|
| [`current/pvs_mainline_2026-08-23.md`](current/pvs_mainline_2026-08-23.md) | 当前模型架构、代码入口、数据依赖、正式指标与保留边界 |
| [`experiments/pvs_mainline_training_2026-08-21.md`](experiments/pvs_mainline_training_2026-08-21.md) | 当前无对比 Full、三种子长训和六个核心消融的固定训练协议 |
| [`evaluation/pvs_mainline_core_ablation_paper_analysis_2026-08-28.md`](evaluation/pvs_mainline_core_ablation_paper_analysis_2026-08-28.md) | 论文风格说明分层生存场、区域矩包络、综合损失、消融原理及结果分析 |
| [`evaluation/unified_pvs_metrics_evaluation.md`](evaluation/unified_pvs_metrics_evaluation.md) | 论文统一指标协议：pose/aggregate、PR-AUC 与正样本比例、安全门、图像/资源/运行指标及主表组织 |
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
| [`current/neuralstreamweb3d_dataset_protocol.md`](current/neuralstreamweb3d_dataset_protocol.md) | 66° view-cell、subpose、Color-ID、候选和 CSR 数据语义 |
| [`current/hardware_gpu_execution_policy.md`](current/hardware_gpu_execution_policy.md) | Chrome NVIDIA Vulkan/ANGLE 硬件采样和 WebGPU 证据要求 |
| [`experiments/ifcbench_sparse_triangle_relation_pipeline_2026-09-02.md`](experiments/ifcbench_sparse_triangle_relation_pipeline_2026-09-02.md) | IFCBench 六层硬件深度剥离的分片即时稀疏化、关系矩归并、缓存压缩与验证协议 |
| [`experiments/pvs_third_scene_and_geometry_shell_hzb_baseline_2026-09-08.md`](experiments/pvs_third_scene_and_geometry_shell_hzb_baseline_2026-09-08.md) | 独立来源第三 BIM 场景选择、图形学场景实例化边界及纯几何外壳 HZB 系统基线计划 |
| [`current/browser_instance_visibility_streaming_patent_application_2026-09-02.md`](current/browser_instance_visibility_streaming_patent_application_2026-09-02.md) | 面向大规模三维场景的区域可见性计算、统一资源状态调度与增量加载完整专利文本，含摘要、权利要求、说明书及附图 |
| [`current/browser_visibility_streaming_patent_figures_plan_2026-09-01.md`](current/browser_visibility_streaming_patent_figures_plan_2026-09-01.md) | 专利附图集合、统一参考标号、语义配色规范和可编辑 draw.io 图源位置 |
| [`evaluation/viewcell_image_per_evaluation.md`](evaluation/viewcell_image_per_evaluation.md) | Color-ID 图像漏检、错误实例和额外实例评价 |

统一相机口径为真实渲染 `60°`，采样、后退候选和模型查询 `66°`。阈值只能由 checkpoint 自己的 calibration split 冻结；validation 用于配置与 checkpoint 比较，test 只能在模型和阈值冻结后读取。

## 前端与部署

| 文档 | 内容 |
|---|---|
| [`frontend/pvs_v4_runtime_and_deployment.md`](frontend/pvs_v4_runtime_and_deployment.md) | V4 运行资产、Worker/WebGPU 查询、实例级显示、数值 parity 和部署 |
| [`../slm2viewer/README_DEPLOY.md`](../slm2viewer/README_DEPLOY.md) | 发布包上传和远端验证命令 |

当前 HKUST 前端只加载 `pvs_mainline_v4`，阈值为 `0.6800000071525574`。没有匹配 V4 资产的场景使用实例 AABB 视锥模式，不复用 HKUST 权重或旧神经模型。

## 保留实验

以下历史实验仍有正式复现价值，因此保留代码、文档和本地结果：

| 文档 | 保留原因 |
|---|---|
| [`experiments/pvs_ray_context_survival_owrb_protocol_2026-08-11.md`](experiments/pvs_ray_context_survival_owrb_protocol_2026-08-11.md) | 2026-08-11 修正正式矩阵及硬件深度层缓存的数据协议 |
| [`experiments/pvs_ray_context_survival_owrb_literature_matrix_2026-08-11.md`](experiments/pvs_ray_context_survival_owrb_literature_matrix_2026-08-11.md) | 相关工作与实现边界 |
| [`evaluation/pvs_direction_depth_relation_survival_constrained_v1_surface_fallback_owrb_formal40_pooled_context_20260811_conclusion.md`](evaluation/pvs_direction_depth_relation_survival_constrained_v1_surface_fallback_owrb_formal40_pooled_context_20260811_conclusion.md) | 8×3×40 正式矩阵及硬件图像评价的精简结论；完整数值保存在 benchmark JSON |
| [`evaluation/pvs_ray_context_survival_owrb_supplement_v1_formal40_20260812_conclusion.md`](evaluation/pvs_ray_context_survival_owrb_supplement_v1_formal40_20260812_conclusion.md) | Fourier、上下文容量和单调参数化补充矩阵的精简结论 |
| [`experiments/pvs_survival_rank_capacity_sweep_v1_2026-08-28.md`](experiments/pvs_survival_rank_capacity_sweep_v1_2026-08-28.md) | 生存场方向秩 2/4/8/12 的三种子完整容量实验及 rank-4 保留结论 |
| [`evaluation/hkust_fov66_rvl_w042_full40_test_2026-07-31.md`](evaluation/hkust_fov66_rvl_w042_full40_test_2026-07-31.md) | HKUST 历史方向代理基线的冻结 test 结果 |
| [`evaluation/ifc_metropolis_instanced_v2_test_eval_2026-07-15.md`](evaluation/ifc_metropolis_instanced_v2_test_eval_2026-07-15.md) | Metropolis 实例化场景历史基线的冻结 test 结果 |

## 文档规则

- `current/` 记录当前系统和研究主线，`evaluation/` 记录指标协议与正式结果，`frontend/` 记录运行和部署，`experiments/` 只保留仍可复现的正式实验协议。
- 结果必须同时报告画面安全、分类诊断、有效剔除、资源效率和运行成本。
- 已失败且不再复现的实验直接从当前文档和 runner 删除，通过 Git 历史追溯，不保留兼容入口。

## 2026-09-05 文档清理

本次清理删除了已被当前报告替代的执行计划、旧含表征对比损失的主线结论、失败的 96 维调制记录、过时的 Metropolis 训练说明、重复的自动生成明细报告和无引用架构图。当前核心消融、数据划分和正式指标分别收敛到主线训练协议、论文式消融分析和统一指标协议。

2026-08-11/12 正式矩阵的精简结论继续保留，完整逐成员数值仍位于原 benchmark JSON；模型 checkpoint、benchmark 输出、硬件采样证据、IFCBench 当前流水线、前端部署文档、专利正文和专利附图均未删除。前端 README 已改为引用当前版本清单，不再引用过时仓库布局文档。
