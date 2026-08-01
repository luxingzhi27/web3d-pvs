# Project Context

Updated: 2026-06-10

Current repository boundary update: 2026-07-31. The authoritative current layout is recorded in
`repository_layout_2026-07-31.md`; this document keeps the project background and model evolution.

当前项目目标是用轻量、可部署的神经 PVS 前端链路统一处理实例可见性和 GLB 下载优先级。仓库已清理为当前运行版，不再维护旧 fixed-geo、dynamic-pool、screen-grid、topology、hypergraph 或 froxel 实验版本。

## Current Mainline

当前前端运行模型有两套场景专用导出：HKUST 使用
`pvs_directional_occlusion_proxy_encoder_rvl_w042_full40_hkust_fov66_best`，IFCBench Metropolis
使用 `pvs_directional_occlusion_proxy_ifcbench_fantasy_metropolis_instanced_v2_k4_full40_best`。

它的核心数据流是：

1. 离线点云编码器生成每个实例的上下文特征和方向遮挡代理特征。
2. 前端运行时只读取固定特征表和轻量查询头权重。
3. Worker 先用后退相机扩展视锥得到候选实例。
4. WebGPU 模型输出实例可见性分数、可见 bit 和 GLB 下载优先级。
5. 主线程再用真实相机做实例级视锥过滤。
6. 渲染按实例裁剪，下载按 GLB 聚合排序。

## Historical Baseline Note

旧 dynamic-pool 方案及其 P0 消融已经从代码、输出和默认前端资产中移除；只保留与当前方向遮挡代理模型有关的周报总结。

保留原因：

- 它是上一版统一“实例可见性 + GLB 下载优先级”的完整论文叙事。
- 它证明 dynamic occlusion pool 可以压缩遮挡关系资产。
- 它也暴露了前端运行成本偏高的问题，是当前 directional proxy 路线的反面证据。

## Current Data

保留数据：

```text
neural_instance_culling/dataset/out/pose_csr_hkust_v3_viewcell_colorid_fov66
neural_instance_culling/dataset/out/directional_occlusion_evidence_hkust_v3_fov66
neural_instance_culling/dataset/out/ifcbench_fantasy_metropolis_instanced_v2_viewcell_colorid_k4
neural_instance_culling/dataset/out/directional_occlusion_evidence_ifcbench_fantasy_metropolis_instanced_v2_k4
```

保留训练输出：

```text
neural_instance_culling/model/out/pvs_directional_occlusion_proxy_encoder_rvl_w042_full40_hkust_fov66
neural_instance_culling/model/out/pvs_directional_occlusion_proxy_encoder_rvl_w042_full40_hkust_fov66_best_eval
neural_instance_culling/model/out/pvs_directional_occlusion_proxy_ifcbench_fantasy_metropolis_instanced_v2_k4_full40
```

保留前端资产：

```text
slm2viewer/public/assets/neural_instance_culling/pvs_directional_occlusion_proxy_encoder_rvl_w042_full40_hkust_fov66_best
slm2viewer/public/assets/neural_instance_culling/pvs_directional_occlusion_proxy_ifcbench_fantasy_metropolis_instanced_v2_k4_full40_best

IFCBench Metropolis 的 41298 个原始构件仍保存在
`ifcbench_fantasy_metropolis_source/assets/task-0/glb/LOD0/sub_*.glb`，实例化后的
3669 个原型 GLB 和运行时元数据保存在 `ifcbench_fantasy_metropolis_instanced_v2/assets`。
```

## Current Validation Rules

正式报告必须同时说明：

- pose-level 和 aggregate precision / recall / F1 / Jaccard。
- weighted recall 的权重来源。
- useful cull 和 bad cull。
- 平均预测实例数、GLB 数量和 GLB 字节削减。
- 前端 Worker/WebGPU 推理耗时、调度耗时和实际 Three.js drawn instance 数。

普通 precision 或 F1 不能单独作为主结论。
