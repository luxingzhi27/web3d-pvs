# Lightweight Frontend PVS Scheduler

Date: 2026-08-12

本文只记录当前前端运行路径。旧 fixed-geo viewcell、dynamic-pool、epoch24 过渡资产和其它历史前端模型已从部署资产中清理。

本文是运行记录；完整且经过代码核对的实现说明见
`docs/frontend/neuralstreamweb3d_runtime_implementation.md`，导出/打包边界见
`docs/frontend/neuralstreamweb3d_deployment_assets.md`。当前两个场景统一使用真实渲染 60°、
模型查询 66°的口径。

## Current Runtime Assets

当前前端神经剔除资产：

```text
slm2viewer/public/assets/neural_instance_culling/pvs_directional_occlusion_proxy_encoder_rvl_strong_v2_full40_best
slm2viewer/public/assets/neural_instance_culling/pvs_directional_occlusion_proxy_ifcbench_fantasy_metropolis_instanced_v2_k4_full40_best
```

运行时模型名：

```text
pvs_directional_occlusion_proxy_encoder_rvl_strong_v2_full40_best
```

前端默认路径：

```text
slm2viewer/src/neuralCullingBackendMode.js
```

## Runtime Data Flow

1. `SLM2Loader` 捕获当前相机状态。
2. `CameraPredictionGate` 判断是否需要重新预测；如果相机仍在上次 viewcell / delta 包络内，不重新运行模型。
3. `LightweightPVSWorker` 用后退相机扩展视锥做实例 AABB 候选过滤。
4. WebGPU 模型对候选实例输出可见性分数、实例可见 bit 和 GLB 下载优先级。
5. 主线程用真实相机对模型输出的实例集合再做一次实例级视锥过滤。
6. 过滤后的实例集合用于 `_applyInstancedVisibility()`，按 component instance 裁剪显示。
7. 过滤后的实例集合同时推导渲染 GLB working set；GLB 只负责资源挂载边界，不能把“一个实例可见”扩展成“整个 GLB 可见”。
8. 下载队列使用模型输出的 GLB 下载优先级排序；当前真实视锥内进入 immediate，后退扩展视锥多出来的预测结果进入 prefetch。

## Prediction Trigger

模型预测的是后退扩大视锥下的保守可见集合，因此前端不每帧推理。默认触发由 `CameraPredictionGate` 控制：

| Trigger | Default | Meaning |
|---|---:|---|
| position movement | `2.0m` | HKUST 水平 XZ 圆盘半径；Y 方向无扰动，越界即重新预测 |
| orientation | fixed | 与上次预测的四元数方向保持一致，不使用 yaw/pitch 容差 |
| FOV | render `60 deg`, model `66 deg` | 显示相机保持 60°；Worker 查询相机固定 66° |
| aspect change | `0.08` | 视口比例明显变化 |
| minimum interval | `350ms` | 仅作状态记录/并发保护；view-cell 越界优先于该时间值 |

两次模型预测之间不允许低频刷新预测结果或实例集合。

HKUST 的正式 view-cell 是半径 `2 m`、垂直扰动 `0 m` 的水平圆盘，所有 subpose 使用同一朝向。`CameraPredictionGate` 以一次当前相机查询作为预测锚点：只要相机仍在该水平圆盘和固定朝向内，就复用一次模型结果；越过位置、朝向、FOV 或 aspect 边界后才建立新锚点。浏览器不展开 subpose，也不为一个 view-cell 增加多次模型推理。

2026-08-12 验证：`node slm2viewer/scripts/test_camera_prediction_gate.mjs`。

## Strict Render Residency

2026-06-10 修复“漫游一圈后帧率下降”的问题后，当前渲染驻留语义为：

- 神经 PVS 默认隐藏延时为 `0 ms`。
- 不属于本次“模型输出 + 当前真实视锥过滤”渲染集合的 resident GLB 会从 Three.js `rootScene` 摘除。
- 摘除只影响场景树，不释放缓存对象；后续再次命中时可以重新挂回。
- 新下载完成的 GLB 只有在属于当前渲染 GLB working set 时才会加入场景。
- 预取 GLB 和后退视锥中多出来的资源只进入缓存，不显示。
- `CacheMgr.refreshVisible()` 返回 `detachedCount`，用于检查本次刷新实际从场景树摘除了多少历史 resident GLB。

这保证调试面板中的当前视锥候选数量低时，Three.js 场景树也不会继续保留大量历史不可见节点。

## Instance-Level Display

本项目当前 `runtimeVisibilityMeta.json` 中的多构件 GLB 使用 `EXT_mesh_gpu_instancing`，因此前端必须按实例裁剪：

- `componentModelList` 是实例级输出，用于渲染过滤。
- `immediateGlbIds` 和 `prefetchGlbIds` 是下载/缓存调度输出，不用于直接决定整包显示。
- `_applyInstancedVisibility()` 根据当前实例集合重排 instanced matrix，并设置 `mesh.count`。
- 如果某个 GLB 中只有部分 component 在当前真实视锥内，只显示这些 component 对应的 instances。

## Deployment

部署脚本：

```bash
cd slm2viewer
npm run package:deploy
```

`package_deploy.mjs` 当前把 HKUST 和 IFCBench Metropolis 实例化 v2 的运行资产打入 `public_deploy`：

```text
assets/neural_instance_culling/pvs_directional_occlusion_proxy_encoder_rvl_strong_v2_full40_best/instance_model_meta.json
assets/neural_instance_culling/pvs_directional_occlusion_proxy_encoder_rvl_strong_v2_full40_best/instance_pvs_assets.bin
assets/neural_instance_culling/pvs_directional_occlusion_proxy_ifcbench_fantasy_metropolis_instanced_v2_k4_full40_best/instance_model_meta.json
assets/neural_instance_culling/pvs_directional_occlusion_proxy_ifcbench_fantasy_metropolis_instanced_v2_k4_full40_best/instance_pvs_assets.bin
```

Metropolis 的 `sceneWeb.json`、`glbIndex.json`、`runtimeVisibilityMeta.json` 和 proxy
随部署包保留；训练端 checkpoint、点云缓存和原始 sub-GLB 不进入部署包，原始 sub-GLB
只用于离线重建，远端场景 GLB 单独管理。

## Validation

最近一次验证：

- `node --check` passed for `CacheMgr.js`, `SLM2Loader.js`, and `viewer.js`.
- `npm run build` passed in `slm2viewer`.
- `public/` synced to `dist/`.
- `npm run package:deploy` passed with JS obfuscation enabled.

仍建议在浏览器中做一次实际漫游 smoke：绕场景一圈后，观察 `actualRender.objectCount`、`actualRender.drawnInstanceCount` 和 `renderRefresh.detachedCount`，确认历史不可见 GLB 不再持续累积在 `rootScene`。

## Current Scene Routes

当前配置只注册两个场景：

| 场景 | 路由参数 | 运行模型 |
|---|---|---|
| HKUST | `?scene=hkust-v3` | `pvs_directional_occlusion_proxy_encoder_rvl_strong_v2_full40_best` |
| IFCBench Metropolis | `?scene=ifcbench_fantasy_metropolis_instanced_v2` | `pvs_directional_occlusion_proxy_ifcbench_fantasy_metropolis_instanced_v2_k4_full40_best` |
