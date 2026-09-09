# NeuralStreamWeb3D 文档索引

更新时间：2026-09-09

当前主线是 HKUST 场景的 V4 实例级可见性模型：离线固定实例特征和遮挡生存系数，浏览器以 `66°` 后退视锥建立候选，以真实 `60°` 视锥完成显示过滤。Metropolis 当前使用显式实例 AABB 视锥模式。

2026-09-09 已将源码收敛到当前 V4 训练、六项核心消融、容量实验、数据重建、统一评价和前端部署链路。旧 M4-M12、OWRB、方向代理、失败的额外打分头、完整三角形 HZB 及迁移前 Windows/ROCm 入口已从当前源码删除；清理前状态保存在 Git 提交 `30e15f7`，不再以兼容开关保留。

同次清理删除了被忽略目录中的旧 checkpoint、旧 benchmark 和重复深度缓存，只保留 HKUST/IFCBench 当前主数据、V4 关系资源、正式主线与核心消融、rank 容量实验和图像评价所需缓存。Pose CSR 候选文件统一为 `candidate_offsets.bin` 与 `candidate_ids.bin`，不再生成重复的历史别名文件；前端发布也统一使用 `npm run package:deploy`。

## 论文主线

| 文档 | 内容 |
|---|---|
| [`current/pvs_mainline_2026-08-23.md`](current/pvs_mainline_2026-08-23.md) | V4 模型结构、代码入口、主 split、当前工作点和 validation 摘要 |
| [`experiments/pvs_mainline_training_2026-08-21.md`](experiments/pvs_mainline_training_2026-08-21.md) | 三种子 `40 × 900` 正式训练及六项核心消融协议 |
| [`evaluation/pvs_mainline_core_ablation_paper_analysis_2026-08-28.md`](evaluation/pvs_mainline_core_ablation_paper_analysis_2026-08-28.md) | 论文式方法说明、正式消融结果和配对 bootstrap 分析 |
| [`experiments/pvs_survival_rank_capacity_sweep_v1_2026-08-28.md`](experiments/pvs_survival_rank_capacity_sweep_v1_2026-08-28.md) | 生存场方向秩 2/4/8/12 的完整容量结果及 rank-4 选择依据 |

## 数据与评价

| 文档 | 内容 |
|---|---|
| [`current/neuralstreamweb3d_dataset_protocol.md`](current/neuralstreamweb3d_dataset_protocol.md) | view-cell、subpose、Color-ID、候选集合、GT 和 Pose CSR 数据语义 |
| [`experiments/ifcbench_sparse_triangle_relation_pipeline_2026-09-02.md`](experiments/ifcbench_sparse_triangle_relation_pipeline_2026-09-02.md) | IFCBench 三角形深度层的分片稀疏化、关系归并和验证记录 |
| [`evaluation/unified_pvs_metrics_evaluation.md`](evaluation/unified_pvs_metrics_evaluation.md) | pose-macro/aggregate、weighted recall、分类、剔除、资源和运行指标口径 |
| [`evaluation/pvs_frontend_inference_latency_protocol_2026-09-09.md`](evaluation/pvs_frontend_inference_latency_protocol_2026-09-09.md) | 桌面和真实移动设备的 WebGPU 候选模型前向耗时协议 |
| [`evaluation/test_split_benchmark_protocol.md`](evaluation/test_split_benchmark_protocol.md) | train、calibration、validation、test 的职责及冻结 test 规则 |
| [`evaluation/viewcell_image_per_evaluation.md`](evaluation/viewcell_image_per_evaluation.md) | 真实 `60°` 相机下的 Color-ID 图像漏检与额外像素评价 |
| [`evaluation/pvs_mainline_image_evaluation_2026-09-09.md`](evaluation/pvs_mainline_image_evaluation_2026-09-09.md) | HKUST 与 IFCBench validation 的硬件 Color-ID 正式图像结果 |

## 当前系统与部署

| 文档 | 内容 |
|---|---|
| [`current/current_instance_pvs_versions.md`](current/current_instance_pvs_versions.md) | 场景映射、V4 资产、阈值和运行 schema |
| [`current/neuralstreamweb3d_architecture_technical.md`](current/neuralstreamweb3d_architecture_technical.md) | 离线编码、运行时查询、实例显示和 GLB 调度架构 |
| [`current/hardware_gpu_execution_policy.md`](current/hardware_gpu_execution_policy.md) | Chrome NVIDIA Vulkan/ANGLE 采样、WebGL/WebGPU 硬件证据和软件路径边界 |
| [`frontend/pvs_v4_runtime_and_deployment.md`](frontend/pvs_v4_runtime_and_deployment.md) | Worker WebGPU/WASM、统一资源状态机、前端 parity 和运行验证 |
| [`frontend/pvs_runtime_benchmark_site.md`](frontend/pvs_runtime_benchmark_site.md) | 桌面/手机自助推理测试页、结果回传服务和部署方法 |
| [`../slm2viewer/README_DEPLOY.md`](../slm2viewer/README_DEPLOY.md) | 发布包生成、场景 GLB 上传、nginx 配置和远端验证 |

## 计划与专利

| 文档 | 内容 |
|---|---|
| [`experiments/pvs_third_scene_and_geometry_shell_hzb_baseline_2026-09-08.md`](experiments/pvs_third_scene_and_geometry_shell_hzb_baseline_2026-09-08.md) | 独立来源第三 BIM 场景选择与 Geometry-only shell HZB 基线计划 |
| [`current/browser_instance_visibility_streaming_patent_application_2026-09-02.md`](current/browser_instance_visibility_streaming_patent_application_2026-09-02.md) | 浏览器区域可见性、资源调度和增量加载专利正文 |
| [`current/browser_visibility_streaming_patent_figures_plan_2026-09-01.md`](current/browser_visibility_streaming_patent_figures_plan_2026-09-01.md) | 专利附图规划、参考标号、可编辑图源和导出规范 |
