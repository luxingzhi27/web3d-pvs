# Docs Index

本目录只保留当前运行版本、通用评价口径、前端接入记录、实验记录和当前汇报图示。旧 fixed-geo、contextual U-Net、dynamic-pool、screen-grid、topology、hypergraph、froxel 等实验报告已从当前目录清理，避免后续误读为主线；dynamic-pool 只在阶段总结中作为历史演进背景出现。

截至 2026-08-04，当前事实以本索引、`current/repository_layout_2026-07-31.md`、`current/current_instance_pvs_versions.md`、`current/hardware_gpu_execution_policy.md` 和 `slm2viewer/assets/config.json` 为准。子目录 README 只提供操作入口，不替代这些当前状态文档。

## 快速阅读路径

1. 当前仓库布局与保留边界：`current/repository_layout_2026-07-31.md`
2. 当前总架构与边界：`current/neuralstreamweb3d_architecture_technical.md`
3. 当前模型、损失与训练：`current/neuralstreamweb3d_model_pipeline.md`
4. 数据集与 NeuralPVS view-cell 采样协议：`current/neuralstreamweb3d_dataset_protocol.md`
5. 当前前端运行模型与训练记录：`experiments/pvs_directional_occlusion_proxy_encoder_2026-06-10.md`
6. 当前保留版本清单：`current/current_instance_pvs_versions.md`
7. 阶段周报总结：`current/dynamic_pool_to_directional_proxy_summary_2026-06-11.md`
8. 本周参考周报：`current/references/weekly_reference_2026-05-30_2026-06-05.md`
9. 前端 Worker、WebGPU、部署和严格实例级渲染记录：`frontend/neuralstreamweb3d_runtime_implementation.md`
10. 导出、按场景打包和 nginx 资产边界：`frontend/neuralstreamweb3d_deployment_assets.md`
11. 通用打包和 nginx 流程：`frontend/slm2viewer_deploy_guide_2026-06-26.md`
12. Metropolis 当前实例化与训练：`experiments/ifc_metropolis_instanced_v2_training_2026-07-15.md`
13. Metropolis 当前评测：`evaluation/ifc_metropolis_instanced_v2_test_eval_2026-07-15.md`
14. Metropolis 当前部署：`frontend/ifc_metropolis_instanced_v2_deploy_2026-07-15.md`
15. 合成数据泛化训练待办：`experiments/synthetic_viewcell_generalization_plan_2026-06-23.md`
16. 指标口径：`evaluation/unified_pvs_metrics_evaluation.md`、`evaluation/test_split_benchmark_protocol.md`、`evaluation/viewcell_image_per_evaluation.md`
17. HKUST 独立远端 GLB 部署：`frontend/2026-07-28-hkust-liteweb3d-standalone-deploy.md`
18. HKUST FOV66 重采样、RVL full40 训练与现有 test-calibrated 探索性评测：`evaluation/hkust_fov66_rvl_w042_full40_test_2026-07-31.md`
19. 2026 年 9 月投稿研究、实验和实施规划：`experiments/neuralstreamweb3d_submission_plan_2026-09.md`
20. M0 固定验证/校准/test 协议执行记录：`experiments/m0_fixed_validation_calibration_protocol_2026-08-01.md`
21. M2 空间隔离 split 执行记录：`experiments/m2_spatial_split_execution_2026-08-01.md`
22. M1 正式资源语义复审与历史失败记录：`evaluation/m1_resource_semantics_audit_2026-08-01.md`
23. M6 第一批冷启动可见性 baseline runner：`experiments/m6_visibility_baseline_runners_2026-08-01.md`
24. M5 实例级 60° Color-ID manifest、绑定预检和 schema smoke：`evaluation/m5_instance_id_buffer_schema_2026-08-01.md`
25. M0/M1 中间校准提前退出的审计与协议恢复：`experiments/m0_training_protocol_recovery_2026-08-01.md`
26. M7/M8 离线 pose-index 轨迹下载回放、冷/温缓存和完成事件：`experiments/m7_m8_trajectory_replay_2026-08-01.md`
27. M9 浏览器运行时、Cold-0、实例绑定 fail-closed 和远端 GLB smoke：`frontend/m9_browser_runtime_benchmark_2026-08-01.md`
28. M9 空间 AABB 索引与全量扫描集合一致性、候选规模和耗时审计：`evaluation/m9_spatial_aabb_index_audit_2026-08-02.json`（旧版 2026-08-01 结果保留为历史记录）
29. M13 复现资产清单工具：`neural_instance_culling/tools/build_artifact_manifest.py`（只记录路径、字节数和 SHA-256，不复制大文件）
30. M8 确定性 pose-index 轨迹生成器：`neural_instance_culling/benchmark/build_pose_index_trajectory.py`（明确标注为非真实导航）
31. M5 真实浏览器实例级 Color-ID 渲染 smoke：`evaluation/m5_image_pipeline_smoke_2026-08-01.md`
32. M7 独立 RankNet 下载排序器和调度对照：`experiments/m7_unified_download_scheduling_2026-08-01.md`
33. M6 真实三角形 HZB 生成与 validation/calibration 执行：`experiments/m6_triangle_hzb_baseline_2026-08-01.md`
34. M9 空间特征分页格式、候选集合审计与运行时边界：`evaluation/m9_spatial_feature_page_audit_2026-08-01.json`、`frontend/m9_browser_runtime_benchmark_2026-08-01.md`
35. M6 NeuralPVS 对照实现缺口与公平边界：`experiments/m6_neuralpvs_baseline_audit_2026-08-01.md`
36. M10 桌面/移动设备 benchmark 可复核性审计：`frontend/m10_device_benchmark_2026-08-01.md`
37. M5 HKUST 正式模型 validation 图像门：`evaluation/m5_hkust_formal_validation_image_2026-08-01.md`
38. M3 两场景正式代理干预与配对 bootstrap：`experiments/m3_formal_execution_2026-08-01.md`
39. M7 HKUST 正式 validation/calibration 下载调度结果：`experiments/m7_unified_download_scheduling_2026-08-01.md`
40. M0 Metropolis 正式 frozen test：`evaluation/m0_metropolis_formal_frozen_test_2026-08-01.md`
41. M5 HKUST 正式 calibration 图像门：`evaluation/m5_hkust_formal_calibration_image_2026-08-01.md`
42. M11 航向泛化与跨场景适配协议：`experiments/m11_directional_generalization_protocol_2026-08-01.md`
43. NeuralStreamWeb3D 投稿计划执行记录：`experiments/neuralstreamweb3d_submission_execution_2026-08-01.md`
44. M0 frozen test 关键产物清单：`evaluation/m0_frozen_artifact_manifest_2026-08-01.json`
45. M5 HKUST 图像安全门失败分析：`evaluation/m5_hkust_image_failure_analysis_2026-08-01.md`
46. M5 图像安全修复实验预注册协议：`experiments/m5_visual_safety_repair_protocol_2026-08-02.md`
47. M4 注册输入分支消融矩阵 validation 评价与路线判定：`evaluation/m4_formal_matrix_validation_2026-08-02.md`
48. M4-v2 完整 2x2 因子消融协议与独立评价：`experiments/m4_formal_matrix_validation_v2_protocol_2026-08-03.md`；正式报告：`evaluation/m4_formal_matrix_validation_v2_2026-08-03.md`；路线判定为 `route_b_system`
49. M5 视觉安全修复分块图像评价：`evaluation/m5_visual_safety_repair_image_chunked_2026-08-03.md`；完整 validation/calibration 结果仍为 `No-Go`
50. M5 硬件 GPU 渲染路径、Chrome Vulkan 参数和软件回退硬门：`evaluation/m5_hardware_gpu_renderer_2026-08-04.md`
51. M5 全量 dense subpose 硬件 GPU 图像评价：`evaluation/m5_visual_safety_repair_image_dense_hw_2026-08-04.md`；536,256 个 validation/calibration 样本，硬件门通过，视觉安全门为 `No-Go`

52. 当前正式采样/浏览器光栅化硬件 GPU 执行政策：`current/hardware_gpu_execution_policy.md`
53. M5 dense subpose 鲁棒监督实验协议：`experiments/m5_subpose_robust_supervision_protocol_2026-08-04.md`
54. M5 dense subpose 鲁棒监督独立图像评价入口：`neural_instance_culling/benchmark/run_m5_subpose_robust_image_evaluation.sh`；结果目录为 `neural_instance_culling/benchmark/out/m5_subpose_robust_image_dense_hw_20260804/`
55. M7/M8 同轨迹配对 bootstrap 汇总：`evaluation/m7_m8_trajectory_paired_bootstrap_v2_2026-08-04.md`；独立结果为 `neural_instance_culling/benchmark/out/m7_m8_trajectory_paired_bootstrap_v2_20260804.json`
56. M5-v2 dense subpose 严格汇总器：`neural_instance_culling/benchmark/summarize_m5_subpose_robust_image.py`；它强制校验硬件 GPU、严格 calibration 和 `testRead=false`，不允许软件渲染结果进入正式汇总
57. M5 视觉贡献安全损失诊断 pilot：`experiments/m5_visual_safety_subpose_pilot_2026-08-04.md`；单 seed、10 epoch，仅用于判断高视觉贡献漏检是否改善，不是正式质量门结果

正式采样和图像评价的 GPU 规则：默认必须使用 Chrome 的硬件 Vulkan/NVIDIA 后端，并开启 `--require-hardware-gpu`；结果必须同时保存浏览器 `gpuBackend`/`gpuGate`、Chrome 日志和同一窗口的 `nvidia-smi`/`pmon` 证据。检测到 SwiftShader、llvmpipe、softpipe、swrast 或缺少硬件证据时，只能作为语义调试/历史结果，不能进入硬件性能结论。完整规则见 `current/hardware_gpu_execution_policy.md`。
注意：WebGL 的硬件门不等于 WebGPU 的硬件门。WebGPU 推理或 WGSL 性能实验必须读取并单独核验 WebGPU adapter；若两种 API 的后端不一致，按 API 分别报告，不能用 WebGL 的 NVIDIA 证据替代 WebGPU 证据。

58. M12 WebGPU 硬件门核验：`evaluation/m12_webgpu_hardware_gate_2026-08-04.md`；本次 WebGL 硬件路径通过，但 WebGPU adapter 为 SwiftShader，因此硬件 WebGPU 性能门失败
59. M11 泛化矩阵审计：`evaluation/m11_generalization_matrix_2026-08-04.md`；航向与少样本结果已整理，但严格普通召回安全门未封存为通过
60. M6 三角形 HZB 硬件链路 smoke：`evaluation/m6_hardware_gpu_renderer_2026-08-04.md`；硬件 WebGL 子门通过，完整 M6 仍为 No-Go
61. M9 全量 pose 空间 AABB 候选审计：`evaluation/m9_spatial_aabb_index_full_2026-08-04.md`；候选集合正确，但强制空间索引未通过加速门

## 当前场景

| 场景 | 路由 | 实例数 | GLB 数 | 运行模型 |
|---|---|---:|---:|---|
| HKUST v3 | `?scene=hkust-v3` | 18,831 | 3,273 | `pvs_directional_occlusion_proxy_encoder_rvl_strong_v2_full40_best` |
| IFCBench Fantasy Metropolis 实例化 v2 | `?scene=ifcbench_fantasy_metropolis_instanced_v2` | 41,298 | 3,669 | `pvs_directional_occlusion_proxy_ifcbench_fantasy_metropolis_instanced_v2_k4_full40_best` |

当前主线只使用一套相机口径：前端真实渲染垂直视场角为 60°，模型采样、后退候选相机和运行时推理垂直视场角为 66°（60° × 1.1）。运行时不读取历史数据中的候选视场角字段。

## 当前保留文档

| 文档 | 用途 |
|---|---|
| `current/current_instance_pvs_versions.md` | 当前模型、训练输出、前端资产和默认 benchmark 名称 |
| `current/neuralstreamweb3d_architecture_technical.md` | 后端离线编码、训练、前端推理和资源调度的总架构 |
| `current/neuralstreamweb3d_model_pipeline.md` | PointNet++ 风格编码器、方向遮挡代理、损失和导出前检查 |
| `current/neuralstreamweb3d_dataset_protocol.md` | view-cell/subpose、Color-ID、Pose CSR 和候选语义 |
| `current/repository_layout_2026-07-31.md` | 当前保留场景、数据、模型和删除边界 |
| `current/project_context_2026-06-01.md` | 项目背景和当前仓库边界 |
| `current/dynamic_pool_to_directional_proxy_summary_2026-06-11.md` | 从 dynamic-pool 消融到当前方向遮挡代理模型的阶段周报 |
| `experiments/pvs_directional_occlusion_proxy_encoder_2026-06-10.md` | 当前 directional occlusion proxy encoder 训练、RVL 搜索、full40 导出和前端接入 |
| `experiments/ifc_metropolis_instanced_v2_training_2026-07-15.md` | 当前最大 IFC 场景的完整构件实例化、粒度解耦和 full40 训练记录 |
| `experiments/synthetic_viewcell_generalization_plan_2026-06-23.md` | 合成 view cell 数据集、泛化预训练、真实场景微调和留一验证待办 |
| `experiments/neuralstreamweb3d_submission_plan_2026-09.md` | Eurographics 2027 主目标下的 Cold-0 资源模型、可见性-效用-下载级联、文献缺口、基线、实验矩阵、排期和 Go/No-Go 门槛 |
| `frontend/lightweight_frontend_pvs_scheduler.md` | 当前前端默认资产、调度、严格渲染驻留和部署包记录 |
| `frontend/neuralstreamweb3d_runtime_implementation.md` | 当前 Worker、WebGPU、预测门控、真实视锥和实例渲染执行路径 |
| `frontend/m9_browser_runtime_benchmark_2026-08-01.md` | M9 浏览器正确性、Cold-0 请求顺序、实例绑定安全门和主体 GLB smoke 证据 |
| `frontend/neuralstreamweb3d_deployment_assets.md` | 模型导出、按场景打包、压缩策略和 nginx 目录 |
| `frontend/slm2viewer_deploy_guide_2026-06-26.md` | 通用部署流程:打包、上传、nginx配置、加/删场景 |
| `frontend/2026-07-28-hkust-liteweb3d-standalone-deploy.md` | HKUST 独立前端包、远端 GLB 基址和验证结果 |
| `frontend/ifc_metropolis_instanced_v2_deploy_2026-07-15.md` | 当前 metropolis 前端接入、浏览器 smoke、服务器替换和旧版本删除记录 |
| `evaluation/unified_pvs_metrics_evaluation.md` | pose-level、aggregate、weighted、useful cull、bad cull 等指标定义 |
| `evaluation/test_split_benchmark_protocol.md` | 正式 test split 和 unique viewcell 口径 |
| `evaluation/viewcell_image_per_evaluation.md` | 图像 PER、miss pixel 和 wrong-ID pixel 口径 |
| `evaluation/m5_image_pipeline_smoke_2026-08-01.md` | M5 真实浏览器 Color-ID 链路 smoke 和质量门边界 |
| `experiments/m5_formal_split_alignment_2026-08-01.md` | M5 正式图像评价 split 对齐与 schema smoke |
| `evaluation/ifc_metropolis_instanced_v2_test_eval_2026-07-15.md` | 当前 metropolis full40 test 指标与有效剔除口径 |
| `evaluation/hkust_fov66_rvl_w042_full40_test_2026-07-31.md` | HKUST FOV66 采样、训练、test-calibrated 探索性评测、导出和旧数据清理记录；不能作为投稿正式 test 结果 |
| `experiments/m0_fixed_validation_calibration_protocol_2026-08-01.md` | 固定 validation、独立 calibration、冻结阈值和 one-shot test 的实现与门控 |
| `experiments/m2_spatial_split_execution_2026-08-01.md` | HKUST/Metropolis 空间隔离四路 split、guard 和类别稀疏风险 |
| `evaluation/m1_resource_semantics_audit_2026-08-01.md` | 正式候选/点云/实例映射复审，以及旧数据失败记录 |
| `experiments/m6_neuralpvs_baseline_audit_2026-08-01.md` | NeuralPVS-style baseline 所需深度片段、视锥体体素和冷启动资源审计 |
| `frontend/m10_device_benchmark_2026-08-01.md` | 当前 Chrome/WebGPU smoke、设备缺口和 M10 门控判断 |
| `evaluation/m5_hkust_formal_validation_image_2026-08-01.md` | HKUST 正式模型 213 个 validation view-cell 的实例级图像质量与长尾诊断 |
| `experiments/m3_formal_execution_2026-08-01.md` | HKUST/Metropolis 正式代理、上下文干预和 paired bootstrap 机制诊断 |
| `experiments/m7_unified_download_scheduling_2026-08-01.md` | M7 主线、独立排序器和严格 GLB 字节预算的正式比较 |
| `evaluation/m0_metropolis_formal_frozen_test_2026-08-01.md` | Metropolis 空间隔离数据的独立 calibration、冻结阈值和 one-shot test 结果 |
| `evaluation/m5_hkust_formal_calibration_image_2026-08-01.md` | HKUST 168 个 calibration view-cell 的实例级图像质量、长尾和真实渲染代价 |

## 当前图示资产

| 文件 | 用途 |
|---|---|
| `experiments/figures/generated_architecture/neuralstreamweb3d_integrated_scheduling_rich_16x9_2026-06-18.png` | NeuralStreamWeb3D 算-传-解-渲-缓一体化调度富信息架构图 |
| `experiments/figures/generated_architecture/neuralstreamweb3d_backend_frontend_architecture_16x9_2026-06-17.png` | NeuralStreamWeb3D 后端离线建模、神经推理、前端算传渲调度整体架构图 |

## 写作规则

新增文档仍按类别落盘：

- 当前默认方案、保留版本、正式指标报告放入 `current/`。
- 新评价管线和指标解释放入 `evaluation/`。
- 前端调度、Worker、WebGPU、部署和浏览器 smoke 放入 `frontend/`。
- 新模型计划、实验过程、失败分析和消融尝试放入 `experiments/`。

报告中第一次出现内部英文名或缩写时，必须先用中文解释它输入什么、输出什么、为什么需要它。普通 precision、F1 或 sample-level 二分类指标不能单独作为 PVS 主结论。
