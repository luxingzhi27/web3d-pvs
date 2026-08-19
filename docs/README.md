# NeuralStreamWeb3D 文档索引

更新时间：2026-08-19

这里的文档只服务于当前可运行版本、正式保留证据和 v4 实验链。重复路线报告和被修正的旧汇总已经清理；当前前端资产、正式训练权重、正式 benchmark、采样数据和硬件证据保持不变。v2 `formal80` 及 v3 纠错设计只作为历史诊断依据。108 维尾部分离旧矩阵已经停止，不再续跑；逐位姿平衡与安全前沿实验已完成两轮参数扫描和三种子 40 epoch 从头训练。该损失改善了阈值尺度但未提高总体分类和剔除效果，因此默认 checkpoint、阈值和前端资产保持不变。

## 先读这几份

| 文档 | 内容 |
|---|---|
| [`evaluation/pvs_direction_conditioned_counterfactual_occlusion_v1_quick8_2026-08-19.md`](evaluation/pvs_direction_conditioned_counterfactual_occlusion_v1_quick8_2026-08-19.md) | 4.53 MiB 神经资产、5.23 MiB 完整运行包下的方向门控关系与同实例反事实损失 2x2 快速评价；反事实损失有效，当前门控关系公式不保留 |
| [`experiments/pvs_direction_conditioned_counterfactual_occlusion_v1_2026-08-19.md`](experiments/pvs_direction_conditioned_counterfactual_occlusion_v1_2026-08-19.md) | 方向条件化关系表征、视觉加权同实例反事实损失、四 GPU 快速矩阵和前端运行硬约束 |
| [`evaluation/pvs_cross_pose_operating_exposure_representation_v1_2026-08-19.md`](evaluation/pvs_cross_pose_operating_exposure_representation_v1_2026-08-19.md) | 跨视点工作边界与 view-cell 区域暴露监督的 8 epoch 扫描、单种子 16 epoch 2x2 快速评价、GLB 资源指标和不增长的前端运行契约 |
| [`experiments/pvs_cross_pose_operating_exposure_representation_v1_2026-08-19.md`](experiments/pvs_cross_pose_operating_exposure_representation_v1_2026-08-19.md) | 当前训练期组合创新的输入、损失、暴露监督、快速矩阵、运行资产上限和执行状态 |
| [`evaluation/pvs_pose_balanced_frontier_visibility_loss_v1_formal40_2026-08-19.md`](evaluation/pvs_pose_balanced_frontier_visibility_loss_v1_formal40_2026-08-19.md) | 逐 pose 平衡 BCE 加动态安全前沿损失的两轮扫描、三种子 40 epoch 正式结果及与旧 V4 的同口径比较；结果未晋级默认模型 |
| [`experiments/pvs_pose_balanced_frontier_visibility_loss_v1_2026-08-19.md`](experiments/pvs_pose_balanced_frontier_visibility_loss_v1_2026-08-19.md) | 新损失的设计、输入输出、扫描配置、从头训练协议和保留规则 |
| [`experiments/pvs_joint_108d_query_tail_separator_from_scratch_v1_2026-08-19.md`](experiments/pvs_joint_108d_query_tail_separator_from_scratch_v1_2026-08-19.md) | 已停止的 108 维联合分离器矩阵计划；只保留历史协议和已有产物，不再训练或续跑 |
| [`evaluation/pvs_train_owned_nonlinear_tail_posterior_longtrain_v1_2026-08-19.md`](evaluation/pvs_train_owned_nonlinear_tail_posterior_longtrain_v1_2026-08-19.md) | 历史诊断：冻结旧 v4 主干后训练尾部残差，并在结果上拟合探针；只能说明旧 checkpoint 的后验可修复性，不能视为从头联合训练 |
| [`experiments/pvs_train_owned_nonlinear_tail_posterior_longtrain_ablation_2026-08-19.md`](experiments/pvs_train_owned_nonlinear_tail_posterior_longtrain_ablation_2026-08-19.md) | 历史 refinement 计划：记录固定旧 checkpoint 上的线性、分段线性和小型 MLP 探针比较 |
| [`current/optimization_restart_2026-08-10.md`](current/optimization_restart_2026-08-10.md) | 当前优化阶段、最新正式证据、v4 当前入口和统一实验规则 |
| [`current/pvs_bounded_relation_survival_moment_v4_implementation_2026-08-15.md`](current/pvs_bounded_relation_survival_moment_v4_implementation_2026-08-15.md) | v4 已完成实现：共享关系先验加逐实例校准残差、正式关系/置乱数据、真实 CUDA smoke、关系负边语义与 55.2 倍训练性能纠错、checkpoint 专属 124 维运行表、validation 安全门、图像回填协议和 4.53 MiB 导出预算 |
| [`experiments/pvs_bounded_relation_survival_moment_v4_run_and_ablation_2026-08-15.md`](experiments/pvs_bounded_relation_survival_moment_v4_run_and_ablation_2026-08-15.md) | 已完成的 v4 基础长训与后处理协议；作为当前从头联合训练的基础模型和历史对照，不再是唯一执行入口 |
| [`experiments/pvs_bounded_relation_survival_moment_envelope_safety_reserve_plan_2026-08-14.md`](experiments/pvs_bounded_relation_survival_moment_envelope_safety_reserve_plan_2026-08-14.md) | v3 历史纠错设计依据；其共享生存生成器已由 v4 的逐实例校准残差修正，不再直接执行 |
| [`experiments/pvs_full_innovation_hyperparameter_scan_longtrain_ablation_2026-08-13.md`](experiments/pvs_full_innovation_hyperparameter_scan_longtrain_ablation_2026-08-13.md) | 已完成的 v2 扫参、冻结配置和 15 成员 formal80 预登记；保留用于解释历史训练，不再作为后续代码改进入口 |
| [`experiments/pvs_hierarchical_occlusion_survival_viewcell_spectral_query_2026-08-13.md`](experiments/pvs_hierarchical_occlusion_survival_viewcell_spectral_query_2026-08-13.md) | v1 分层关系生存网络、视点区域积分频谱和质量/资源 RVL 的架构、实现、pilot 与学习曲线历史记录 |
| [`experiments/pvs_integrated_spectral_survival_field_diagnosis_2026-08-14.md`](experiments/pvs_integrated_spectral_survival_field_diagnosis_2026-08-14.md) | 对当前 validation 结果、积分公式/中心、退化关系层级、固定生存观察、事件语义、置乱对照和资源梯度的完整代码审计 |
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
| [`evaluation/pvs_direction_conditioned_counterfactual_occlusion_v1_quick8_2026-08-19.md`](evaluation/pvs_direction_conditioned_counterfactual_occlusion_v1_quick8_2026-08-19.md) | 单种子快速结果：同实例反事实损失改善安全工作点和跨视角分离，当前方向门控关系特征发生负交互，未修改默认模型 |
| [`evaluation/pvs_cross_pose_operating_exposure_representation_v1_2026-08-19.md`](evaluation/pvs_cross_pose_operating_exposure_representation_v1_2026-08-19.md) | 单种子快速结果：组合训练机制在 4.53 MiB 运行资产不变的前提下改善分类、有效剔除和 GLB 削减；等待三种子与图像/移动端评价 |
| [`evaluation/pvs_pose_balanced_frontier_visibility_loss_v1_formal40_2026-08-19.md`](evaluation/pvs_pose_balanced_frontier_visibility_loss_v1_formal40_2026-08-19.md) | 新损失两轮参数扫描、三种子从头 40 epoch 结果、安全阈值、分类/剔除/资源指标和未晋级结论 |
| [`evaluation/unified_pvs_metrics_evaluation.md`](evaluation/unified_pvs_metrics_evaluation.md) | pose/aggregate、weighted recall、accuracy、balanced accuracy、useful cull 和资源指标定义 |
| [`evaluation/test_split_benchmark_protocol.md`](evaluation/test_split_benchmark_protocol.md) | calibration、validation、test 的冻结关系 |
| [`evaluation/viewcell_image_per_evaluation.md`](evaluation/viewcell_image_per_evaluation.md) | Color-ID 图像漏检、错误实例和额外实例指标 |
| [`evaluation/pvs_bounded_relation_survival_moment_v4_score_separation_diagnosis_2026-08-17.md`](evaluation/pvs_bounded_relation_survival_moment_v4_score_separation_diagnosis_2026-08-17.md) | v4 三 seed 正式结果、极端分数尾部、RVL 梯度动力学和三轮单种子最小原因实验；未晋级新模型 |
| [`evaluation/pvs_direction_depth_relation_survival_constrained_v1_surface_fallback_owrb_formal40_pooled_context_20260811_formal_validation.md`](evaluation/pvs_direction_depth_relation_survival_constrained_v1_surface_fallback_owrb_formal40_pooled_context_20260811_formal_validation.md) | 2026-08-11 修正 pooled-context 定义后的正式 8×3×40 validation 汇总、因子效应、图像评价和硬件门状态 |
| [`evaluation/pvs_direction_depth_relation_survival_constrained_v1_surface_fallback_owrb_formal40_pooled_context_20260811_conclusion.md`](evaluation/pvs_direction_depth_relation_survival_constrained_v1_surface_fallback_owrb_formal40_pooled_context_20260811_conclusion.md) | 同一正式矩阵的路线判定、冻结成员、test 结果和论文结论 |
| [`evaluation/pvs_ray_context_survival_owrb_supplement_v1_formal40_20260812_formal_validation.md`](evaluation/pvs_ray_context_survival_owrb_supplement_v1_formal40_20260812_formal_validation.md) | 5 个补充机制变体、15 个成员、10,000 次 paired bootstrap 和硬件 Color-ID 图像评价 |
| [`evaluation/pvs_ray_context_survival_owrb_supplement_v1_formal40_20260812_conclusion.md`](evaluation/pvs_ray_context_survival_owrb_supplement_v1_formal40_20260812_conclusion.md) | 补充矩阵的上下文容量、Fourier ray、单调参数化和 AABB 证据结论 |
| [`evaluation/hkust_fov66_rvl_w042_full40_test_2026-07-31.md`](evaluation/hkust_fov66_rvl_w042_full40_test_2026-07-31.md) | HKUST 已部署模型的 FOV66 基线 test 结果 |
| [`evaluation/ifc_metropolis_instanced_v2_test_eval_2026-07-15.md`](evaluation/ifc_metropolis_instanced_v2_test_eval_2026-07-15.md) | IFCBench Metropolis 场景基线 test 结果 |
| [`evaluation/pvs_hierarchical_relation_survival_integrated_spectral_quality_rvl_v1_pilot.md`](evaluation/pvs_hierarchical_relation_survival_integrated_spectral_quality_rvl_v1_pilot.md) | 2026-08-13 新路线三个单 seed pilot 的完整 validation 诊断；calibration 仅 32 pose，weighted recall 安全门未通过，未读取 test |
| [`evaluation/pvs_hierarchical_relation_survival_learning_curve_e32_2026-08-13.md`](evaluation/pvs_hierarchical_relation_survival_learning_curve_e32_2026-08-13.md) | R0/R2 双 seed、32 epoch 学习曲线；R2 达到 weighted recall 安全门但未通过效率晋级门，旧 Formal80 未启动 |

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
