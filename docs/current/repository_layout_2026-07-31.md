# 当前仓库布局与保留边界

复核日期：2026-08-13

本文件定义当前场景、数据、模型、前端和工具的保留边界。它不描述已经删除的历史实验过程；历史 benchmark 输出若仍有复现实验价值，可留在独立输出目录，但不能被默认 runner 或前端引用。

## 场景资产

```text
hkust-v3/
ifcbench_fantasy_metropolis_source/
ifcbench_fantasy_metropolis_instanced_v2/
```

IFCBench 的 `sub_*.glb` 是实例化工具的原始构件源；`ifcbench_fantasy_metropolis_instanced_v2/assets` 保存经验证的原型 GLB、实例映射、AABB 和运行时元数据。显示粒度始终是实例级，GLB 只作为资源粒度。

## 训练与数据

当前两个场景的主要数据入口：

```text
neural_instance_culling/dataset/out/pose_csr_hkust_v3_viewcell_colorid_fov66
neural_instance_culling/dataset/out/directional_occlusion_evidence_hkust_v3_fov66
neural_instance_culling/dataset/out/ifcbench_fantasy_metropolis_instanced_v2_viewcell_colorid_k4
neural_instance_culling/dataset/out/directional_occlusion_evidence_ifcbench_fantasy_metropolis_instanced_v2_k4
neural_instance_culling/dataset/out/glb_points_v3.bin
neural_instance_culling/dataset/out/ifcbench_fantasy_metropolis_instanced_v2_glb_points_v3.bin
```

当前研究证据只保留下列正式生成目录：

```text
neural_instance_culling/model/out/pvs_direction_depth_relation_survival_constrained_v1_surface_fallback_owrb_formal40_pooled_context_20260811
neural_instance_culling/benchmark/out/pvs_direction_depth_relation_survival_constrained_v1_surface_fallback_owrb_formal40_pooled_context_20260811
neural_instance_culling/model/out/pvs_ray_context_survival_owrb_supplement_v1_formal40_20260812
neural_instance_culling/benchmark/out/pvs_ray_context_survival_owrb_supplement_v1_formal40_20260812
neural_instance_culling/benchmark/out/pvs_ray_context_survival_owrb_v1_subpose5_20260811_directchrome
```

前四项是修正后的 40 epoch 正式矩阵与 Fourier 补充矩阵；最后一项保存 train-only 三角形深度层、表面补全关系和硬件 GPU 证据，是下一阶段关系 CSR 的数据源。旧 `flat_id`、关系残差、选择性纠错、gamma/surface pilot 和被修正版替代的正式输出不在保留边界内。

## 当前代码入口

| 任务 | 入口 |
|---|---|
| 实例化原型分析 | `neural_instance_culling/tools/glb_instancer/` |
| view-cell 同方向 subpose 计划 | `neural_instance_culling/sampler/build_neuralpvs_viewcell_pose_plan.mjs` |
| Three.js Color-ID 采样 | `neural_instance_culling/sampler/run_scene_viewcell_colorid_sampling.mjs` |
| Pose CSR 构建 | `neural_instance_culling/dataset/build_color_id_pose_csr.py`、`build_rvc_viewcell_pose_csr.py` |
| 方向遮挡证据 | `neural_instance_culling/dataset/build_directional_occlusion_evidence.py` |
| 模型训练 | `neural_instance_culling/model/train_pvs.py` |
| 前端导出 | `neural_instance_culling/model/export_pvs.py` |
| benchmark runner | `neural_instance_culling/benchmark/run_pvs.py` |
| 前端查询与 Worker | `slm2viewer/src/PVSQuerySession.js`、`PVSDispatcher.js`、`PVSWorker.js` |
| 前端渲染 | `slm2viewer/src/RendererRuntime.js`、`RendererEffects.js`、`InstancePVS.js`、`RenderVisibilitySystem.js` |
| 按场景打包 | `slm2viewer/scripts/package_deploy.mjs`、`package_scene_glb.mjs` |

## 前端资产边界

```text
slm2viewer/assets/scenes/<scene>/
slm2viewer/assets/neural_instance_culling/<model>/
slm2viewer/public_deploy/
```

部署包只包含场景元数据、代理几何、固定实例特征、模型权重和混淆后的前端代码。训练 checkpoint、原始点云、遮挡证据和原始 sub-GLB 不进入前端包；主体 GLB 在 `scene_glbs/<scene>/` 独立提供。

## 数据和相机契约

- 采样、后退候选和模型查询使用垂直 FOV `66°`。
- 真实渲染使用垂直 FOV `60°`。
- View-cell 在局部空间盒内生成同朝向 subpose；候选集合是 subpose 候选并集，可见集合是 subpose 可见并集。
- 正式构建要求 `visible_ids ⊆ candidate_ids`，禁止用 GT 补入候选修复数据。
- Color-ID 正式采样必须通过硬件 GPU 门；证据规则见 `hardware_gpu_execution_policy.md`。

## 删除边界

本阶段删除过时文档、失败实验专用代码及其生成输出，不删除当前训练、采样、评测、导出、前端和实例化核心代码，也不删除默认模型、数据集、前端资产或上面列出的正式 validation 与深度证据。删除任何资源前必须检查 runner、配置和前端引用。
