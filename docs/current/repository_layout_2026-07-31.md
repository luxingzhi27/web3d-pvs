# 当前仓库布局与保留边界

日期: 2026-07-31

## 目的

本次整理把仓库收敛到当前可复现主线：HKUST 场景、IFCBench Metropolis 实例化 v2
场景，以及两者所需的模型训练、评测、采样和前端接入代码。旧场景和旧模型输出不再作为
本地运行版本，避免旧路径被误用。

## 保留场景

```text
hkust-v3/
ifcbench_fantasy_metropolis_source/
ifcbench_fantasy_metropolis_instanced_v2/
```

`ifcbench_fantasy_metropolis_source/assets/task-0/glb/LOD0/sub_*.glb` 是 IFCBench 的原始
构件源，每个文件仍代表一个完整构件。`ifcbench_fantasy_metropolis_instanced_v2/assets`
是从这些源构件严格验证复用关系后生成的 3669 个原型 GLB、实例映射、AABB 和 proxy；
41298 个实例的 ID 和可见性粒度保持不变。源目录中的单体合并 GLB 已删除，避免后续工具
误把合并网格当作当前实例化输入。

## 保留模型与数据

```text
neural_instance_culling/model/out/pvs_directional_occlusion_proxy_encoder_rvl_w042_full40_hkust_fov66
neural_instance_culling/model/out/pvs_directional_occlusion_proxy_encoder_rvl_w042_full40_hkust_fov66_best_eval
neural_instance_culling/model/out/pvs_directional_occlusion_proxy_ifcbench_fantasy_metropolis_instanced_v2_k4_full40

neural_instance_culling/dataset/out/pose_csr_hkust_v3_viewcell_colorid_fov66
neural_instance_culling/dataset/out/directional_occlusion_evidence_hkust_v3_fov66
neural_instance_culling/dataset/out/glb_points_v3.bin
neural_instance_culling/dataset/out/glb_points_v3_meta.json
neural_instance_culling/dataset/out/ifcbench_fantasy_metropolis_instanced_v2_viewcell_colorid_k4
neural_instance_culling/dataset/out/directional_occlusion_evidence_ifcbench_fantasy_metropolis_instanced_v2_k4
neural_instance_culling/dataset/out/ifcbench_fantasy_metropolis_instanced_v2_glb_points_v3.bin
neural_instance_culling/dataset/out/ifcbench_fantasy_metropolis_instanced_v2_glb_points_v3_meta.json
```

前端运行时只保留：

```text
slm2viewer/assets/neural_instance_culling/pvs_directional_occlusion_proxy_encoder_rvl_w042_full40_hkust_fov66_best
slm2viewer/assets/neural_instance_culling/pvs_directional_occlusion_proxy_ifcbench_fantasy_metropolis_instanced_v2_k4_full40_best
slm2viewer/assets/scenes/hkust-v3
slm2viewer/assets/scenes/ifcbench_fantasy_metropolis_instanced_v2
```

前端不运行 PointNet++、Graph U-Net、dynamic-pool 或动态图传播；离线特征表随模型导出，
运行时只做后退相机候选、WebGPU 查询、真实相机实例过滤和 GLB 级下载排序。

当前统一数据协议使用 66° 采样/模型相机和 60° 真实渲染相机。运行时元数据只记录并使用这两项
口径，不从历史 Pose CSR 行中读取候选 FOV。

前端包不再保留根目录的 `assets/glbIndex.json` 和
`assets/runtimeVisibilityMeta.json` 历史副本；加载器只读取对应场景目录中的元数据，防止
场景之间发生错误回退。

## IFCBench 工具链

当前可复用的处理入口包括：

- `neural_instance_culling/tools/glb_instancer/`：从原始 `sub_*.glb` 分析和生成实例化场景。
- `neural_instance_culling/dataset/build_ifcbench_composite_scene.py`：从 IFC 源构造完整构件场景。
- `neural_instance_culling/dataset/build_scene_runtime_meta.mjs`：生成 GLB 索引、实例 AABB 和运行时元数据。
- `neural_instance_culling/sampler/build_neuralpvs_viewcell_pose_plan.mjs`：生成 NeuralPVS 语义的 view cell 同朝向 subpose。
- `neural_instance_culling/sampler/run_scene_viewcell_colorid_sampling.mjs`：使用 Three.js color-ID 光栅化采样。
- `neural_instance_culling/dataset/build_color_id_pose_csr.py`：构造候选集合与可见集合的 CSR 数据集。
- `neural_instance_culling/dataset/build_rvc_viewcell_pose_csr.py`：从 rvcServer 或 color-ID 的 view-cell subpose 原始 JSONL 构造正式 PoseCSR 数据集。
- `neural_instance_culling/dataset/build_directional_occlusion_evidence.py`：从采样 GT 构造方向遮挡代理的弱监督证据。
- `neural_instance_culling/dataset/generate_glb_points_v3.mjs`：生成离线实例点云输入。

`build_viewcell_pose_csr_for_fixed_geo.py` 是保留的历史格式转换器，默认输入
`dataset/out/proxy_viewcell_pvs_v1`，该中间数据目录当前不在仓库中，因此不属于当前可复现入口。
新的 view-cell 数据集应使用 `build_rvc_viewcell_pose_csr.py` 并显式传入原始采样目录、输出目录和运行时元数据。

实例化工具要求显式传入 `--input` 或 `--source-assets`，不再提供指向旧合并 GLB 的默认
路径。推荐重建命令见 `neural_instance_culling/tools/glb_instancer/README.md`。

## 删除范围

本次删除了旧场景资产、旧场景前端元数据、过期的场景专用候选门控和分组部署脚本、旧模型
checkpoint、未完成采样输出、过期场景报告和单体合并 GLB。保留的
动态遮挡池阶段总结只用于说明当前方向遮挡代理模型的演进，不再进入默认训练或前端路径。

## 远端部署边界

远端 `/var/www/slm2viewer` 只保留 `public_deploy` 当前发布包、
`scene_glbs/hkust-v3` 和 `scene_glbs/ifcbench_fantasy_metropolis_instanced_v2`。
旧 Block、District、旧 Metropolis 目录和部署备份不属于当前运行资源；nginx 的站点配置
仍由服务器系统目录维护，不随场景资源清理。

## 复现约束

- 训练命令使用 Linux CUDA conda 环境 `slm_pvs`。
- 所有长任务写入 stdout/stderr 日志。
- 正式评测使用完整 test split，并同时报告画面安全、有效剔除、资源节省和运行延迟。
- 删除或重命名生成目录后，训练、导出、前端配置和文档必须使用同一场景名。
