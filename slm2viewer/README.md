# NeuralStreamWeb3D Viewer

这是当前 NeuralStreamWeb3D 前端工程，负责加载 HKUST v3 和 IFCBench Fantasy Metropolis
实例化 v2 场景。HKUST 使用 V4 共享设备 WebGPU 查询执行实例级可见性推理、真实视锥过滤和
GLB 下载调度；Metropolis 没有匹配的 V4 权重，使用实例 AABB 视锥模式。

当前仓库状态以 [`../docs/README.md`](../docs/README.md) 和
[`../docs/current/repository_layout_2026-07-31.md`](../docs/current/repository_layout_2026-07-31.md)
为准。部署细节见 [`README_DEPLOY.md`](README_DEPLOY.md)。

## 本地运行

```bash
npm install
npm run dev
```

场景路由：

```text
http://localhost:3000/?scene=hkust-v3
http://localhost:3000/?scene=ifcbench_fantasy_metropolis_instanced_v2
```

## 构建和打包

```bash
npm run build
npm run package:deploy:direct
npm run package:deploy:direct -- --scene hkust-v3
npm run package:deploy:direct -- --scene ifcbench_fantasy_metropolis_instanced_v2
```

部署包默认写入 `public_deploy/`，前端代码混淆默认开启；场景 GLB 本体使用
`npm run package:scene-glb -- --scene <scene>` 单独打包，不和前端元数据一起传输。

详细运行、资产和部署说明见 [`../docs/frontend/pvs_v4_runtime_and_deployment.md`](../docs/frontend/pvs_v4_runtime_and_deployment.md)。

模型预测集合用于后退视锥预取，最终显示仍要经过真实相机的实例级 AABB 过滤；GLB 只用于下载、解码和缓存聚合，不能因为一个 GLB 中有一个可见实例就整体显示该 GLB。硬件 WebGPU 可用时，候选过滤、模型推理和真实视锥过滤与渲染共享一个 GPUDevice；否则由 WebGL2 渲染和 Worker WASM SIMD 完成同一查询语义。
