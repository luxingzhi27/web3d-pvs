# Current Instance-PVS Versions

本文记录当前仓库仍保留的实例级 PVS 版本。旧 fixed-geo、contextual U-Net、dynamic-pool、screen-grid、topology、hypergraph、froxel 和早期 scan 输出已经清理，不再作为默认 runner、训练入口或前端资产。当前目录边界和删除范围以 `repository_layout_2026-07-31.md` 为准。

## 当前运行版本

| 名称 | 输入 | 输出 | 当前用途 |
|---|---|---|---|
| `pvs_directional_occlusion_proxy_encoder_rvl_strong_v2_full40_best` | FOV66 后退相机扩展候选、固定离线上下文/遮挡代理特征、当前相机到实例的视线查询特征 | 实例可见性分数、实例可见集合、GLB 下载优先级 | HKUST 当前前端默认运行模型 |
| `pvs_directional_occlusion_proxy_ifcbench_fantasy_metropolis_instanced_v2_k4_full40_best` | 41298 个实例的固定离线特征、3669 个实例化 GLB 映射、后退相机 ray-space 查询 | 实例可见性与原型 GLB 下载优先级 | metropolis 当前部署模型 |

场景粒度和资产粒度如下：

| 场景 | 实例数 | GLB 数 | 前端显示粒度 |
|---|---:|---:|---|
| HKUST v3 | 18,831 | 3,273 | 实例级 |
| IFCBench Fantasy Metropolis 实例化 v2 | 41,298 | 3,669 | 实例级；GLB 只用于下载、解码和缓存聚合 |

## 保留训练输出

```text
neural_instance_culling/model/out/pvs_directional_occlusion_proxy_encoder_rvl_strong_v2_full40_hkust_spatial_fov66_seed20260801_protocolfix_retry2
neural_instance_culling/model/out/pvs_directional_occlusion_proxy_ifcbench_fantasy_metropolis_instanced_v2_k4_full40
```

Metropolis 训练数据：

```text
neural_instance_culling/dataset/out/ifcbench_fantasy_metropolis_instanced_v2_viewcell_colorid_k4
neural_instance_culling/dataset/out/directional_occlusion_evidence_ifcbench_fantasy_metropolis_instanced_v2_k4
neural_instance_culling/dataset/out/ifcbench_fantasy_metropolis_instanced_v2_glb_points_v3.bin
neural_instance_culling/dataset/out/ifcbench_fantasy_metropolis_instanced_v2_glb_points_v3_meta.json
```

保留内容包括：

- 两套模型的 `best.pt` 和 `last.pt`。
- HKUST 的 `epoch14_best_snapshot.pt`、`epoch24_snapshot.pt`，以及 Metropolis 的 `checkpoint_epoch_*.pt`。
- `eval_summary.json` / `eval_summary_best.json`。
- `instance_runtime_features_fp16.bin`、几何/上下文/遮挡代理特征表和 meta。
- 训练日志、训练历史和导出评估日志。

## 前端运行资产

```text
slm2viewer/public/assets/neural_instance_culling/pvs_directional_occlusion_proxy_encoder_rvl_strong_v2_full40_best
slm2viewer/public/assets/neural_instance_culling/pvs_directional_occlusion_proxy_ifcbench_fantasy_metropolis_instanced_v2_k4_full40_best
slm2viewer/assets/neural_instance_culling/pvs_directional_occlusion_proxy_encoder_rvl_strong_v2_full40_best
slm2viewer/assets/neural_instance_culling/pvs_directional_occlusion_proxy_ifcbench_fantasy_metropolis_instanced_v2_k4_full40_best
slm2viewer/public_deploy/assets/neural_instance_culling/pvs_directional_occlusion_proxy_encoder_rvl_strong_v2_full40_best
slm2viewer/public_deploy/assets/neural_instance_culling/pvs_directional_occlusion_proxy_ifcbench_fantasy_metropolis_instanced_v2_k4_full40_best
```

2026-08-01 HKUST 已导出正式空间训练模型 `rvl_strong_v2_full40_best`。其阈值来自
`calibration_ready_summary.json` 的冻结校准工作点 `0.02`，校准 weighted recall 为
`0.9930808`，bootstrap 单侧 95% 下界为 `0.9909417`；导出器不会在 test 上重新扫描阈值。
Metropolis 仍等待当前正式训练结束，不能提前替换其旧运行资产。Metropolis 的运行元数据不再引用已删除的
`ifcbench_fantasy_metropolis_v1` 数据集路径，训练期资源清单也不再随运行元数据展开。

当前混淆多场景发布目录：

```text
slm2viewer/public_deploy
```

当前没有保留多场景 `public_deploy.tar.gz`；需要归档时直接对该目录执行 `tar -czf`。另有 HKUST 独立包 `slm2viewer/hkust_v3_public_deploy_liteweb3d.tar.gz`，其 GLB 从 `https://www.liteweb3d.com/data/hkust-v3/` 请求，不包含本地 GLB 本体。

当前服务器上的 Metropolis 已替换为完整构件实例化版本。旧 Metropolis 模型、旧 41298 GLB 场景目录和历史部署备份已从服务器删除；HKUST 与 Metropolis 的现行 GLB 目录仍由 nginx 单独提供。

## 默认 Benchmark 名称

`neural_instance_culling/benchmark/model_runners.py` 当前默认暴露：

```text
baseline_aabb_hzb（显示名：baseline_aabb_depth_proxy）
pvs_directional_occlusion_proxy_encoder_rvl_strong_v2_full40_best
pvs_directional_occlusion_proxy_ifcbench_fantasy_metropolis_instanced_v2_k4_full40_best
```

`baseline_aabb_hzb` 是规则对照，不是神经实验版本；它只对 AABB 投影矩形执行 CPU 深度代理，不是真实三角形 HZB。论文和新报告使用显示名 `baseline_aabb_depth_proxy`，旧 key 仅为兼容。

## 当前运行逻辑

前端显示链路必须保持：

1. 使用后退相机扩展视锥生成候选实例。
2. 当前模型对候选实例输出可见性分数和 GLB 下载优先级。
3. 前端用真实相机对模型输出的实例集合再做一次实例级视锥过滤。
4. 渲染按实例过滤，不允许退化为“GLB 中一个实例可见就显示整个 GLB”。
5. 两次模型预测之间不低频刷新预测结果；只有 `CameraPredictionGate` 判断超出范围后才重新预测。
6. 不属于当前渲染实例集合推导出的 GLB 会从 Three.js `rootScene` 摘除，但缓存可以保留。

## 数据口径与复现警告

当前统一协议使用 66° 采样/模型相机和 60° 真实渲染相机。两个场景的运行时元数据均遵循这一
口径，候选 FOV 不再从数据记录中回读。

HKUST 旧 `w042` 资产的点云特征元数据曾记录 3,273 个 GLB 原型，而运行时实例数为 18,831，不能继续作为当前主线证据。
当前 `rvl_strong_v2` 训练输出已按实例运行特征表导出，并通过正式资源审计；旧资产仅作为历史复现实验保留。

完整的系统边界、模型、数据集、前端和部署说明见：

- `docs/current/neuralstreamweb3d_architecture_technical.md`
- `docs/current/neuralstreamweb3d_model_pipeline.md`
- `docs/current/neuralstreamweb3d_dataset_protocol.md`
- `docs/frontend/neuralstreamweb3d_runtime_implementation.md`
- `docs/frontend/neuralstreamweb3d_deployment_assets.md`

## 指标口径

正式结论不能只看普通 precision 或 F1。当前验收优先级为：

- 画面安全：pose recall、weighted recall、image PER、miss pixel rate。
- 剔除效率：useful cull、bad cull、平均预测数量、GLB 数量/字节削减。
- 运行成本：前端 Worker/WebGPU 推理耗时、调度耗时、实际 Three.js visible mesh / drawn instance 数。

普通 precision、F1、Jaccard 和逐实例 accuracy 必须报告，但不能单独决定主线。

当前统一选择规则：训练保存 `best.pt`、阈值校准和前端导出均先要求 `weighted recall > 0.99`，再选择 `pose precision` 最高的工作点。`weighted recall == 0.99` 不合格；普通 pose recall 只作为并行安全诊断指标，不再作为第一筛选条件。

代码层面，训练、benchmark、运行时阈值读取和前端导出共享同一个安全筛选函数。没有合格工作点时不会覆盖旧 `best.pt`，也不会把摘要中的 `bestF1` 或其他不安全工作点回退成默认阈值；手动不安全阈值仅用于显式诊断。

旧 HKUST `w042` 的 test 校准工作点 `0.64` 仅作为 exploratory 历史结果保留。
当前正式 `rvl_strong_v2` 的默认前端阈值为校准冻结值 `0.02`；正式 frozen test 和图像质量门仍按 M0/M5
收尾结果更新，不能用旧 `0.64` 代替。
