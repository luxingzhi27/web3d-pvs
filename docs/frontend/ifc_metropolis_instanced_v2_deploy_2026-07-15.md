# IFC Metropolis Instanced V2 前端部署

## 日期

2026-07-15

## 前端接入

当前路由使用 `?scene=ifcbench_fantasy_metropolis_instanced_v2`，与实例化后的场景、模型和本地目录保持一致:

- 场景:`ifcbench_fantasy_metropolis_instanced_v2`
- 模型:`pvs_directional_occlusion_proxy_ifcbench_fantasy_metropolis_instanced_v2_k4_full40_best`
- 实例数:41298
- 原型 GLB 数:3669
- 可见性阈值:0.64（严格 weighted recall > 0.99 后的最高 pose precision 工作点）

前端模型映射修改于 `slm2viewer/src/neuralCullingBackendMode.js`。场景元数据位于
`slm2viewer/assets/scenes/ifcbench_fantasy_metropolis_instanced_v2`，GLB 本体通过
`/scene_glbs/ifcbench_fantasy_metropolis_instanced_v2/` 提供。

无纹理 IFC 场景补充了空的 `task-0/images/image_lod.json`,避免 nginx SPA fallback 返回 HTML 后触发 JSON 解析异常。

## 打包

执行 `npm run package:deploy:direct`,JavaScript 混淆已开启。最终包不包含旧 metropolis 模型目录。

2026-07-31 重新导出运行元数据并重建混淆包后，当前发布包以如下数据为准；旧表中的大体积 meta 是紧凑运行元数据之前的历史结果。阈值 `0.220 / primaryHighRecall` 的浏览器 smoke 记录也是旧工作点；当前模型元数据使用 `0.640 / primaryWeightedPrecision`，切换阈值后的浏览器性能需单独复测。

| 项目 | 结果 |
| --- | ---: |
| public deploy 原始字节 | 125979663 |
| public deploy Brotli 传输字节 | 72921441 |
| 新模型 bin 原始字节 | 29931228 |
| 新模型 meta 原始字节 | 400864 |
| 新模型 bin 预压缩 | 不生成 |
| 新模型 meta Brotli 字节 | 11445 |

## 浏览器 Smoke

本地和公网均使用本机 Chrome、Playwright、WebGPU Worker 进行验证。

- `navigator.gpu` 可用。
- Worker 后端为 `worker-webgpu`。
- 模型状态为就绪,无降级原因。
- 模型元数据应显示为 `threshold 0.640 / primaryWeightedPrecision`；本节其余浏览器请求与绘制数量来自旧 `0.220 / primaryHighRecall` smoke，不能当作新阈值的性能结论。
- 本地累计 2719 个 GLB 请求返回 200,404 数为 0。
- 公网复测累计 2722 个 GLB 请求返回 200,404 数为 0。
- 画面统计为 786 个可见 Mesh,其中 666 个 InstancedMesh,实际绘制 6921 个实例。
- 补充 `image_lod.json` 后 page error 为 0。

浏览器关闭时仍可能中止少量尚在队列中的 fetch,表现为 `ERR_FAILED`;请求不是服务器 404,不影响已完成的模型推理和场景渲染。

## 远端部署与删除

服务器:`139.196.34.161`,域名:`https://web3d.ysnb.asia/`。

新 GLB 和 `public_deploy` 先上传到 staging,验证实例数、GLB 数、模型阈值和旧模型缺失后再切换。切换完成后:

- `/var/www/slm2viewer/scene_glbs/ifcbench_fantasy_metropolis_instanced_v2` 仅保留新 3669 个 GLB,约 183 MB。
- 旧 41298 GLB 的 metropolis 场景目录已删除。
- 旧模型 `pvs_directional_occlusion_proxy_ifcbench_fantasy_metropolis_v1_k4_best` 已删除,其 HTTP URL 返回 404。
- 新页面、配置、元数据、模型 meta 和样例 GLB 均返回 HTTP 200。
- `nginx -t` 通过并已 reload。

HKUST 场景保留；旧 block、district 和 Huayi 场景不再作为当前部署资源。

## 实例化视锥提前剔除修复

2026-07-15 在相机移动检查中发现，模型路径和强制 AABB 视锥路径都会出现仍在真实 60 度视锥内的构件提前消失。该问题与模型阈值无关，也不是实例化后的构件 AABB 写错。

前端先使用 66 度后退查询相机形成候选并执行模型查询，再使用真实 60 度渲染相机对模型输出的构件编号逐实例进行 AABB 筛选。下载和缓存按原型 GLB 聚合，最终渲染仍通过构件编号选择对应的实例矩阵。

对实例化资产进行了完整一致性检查：

| 检查项 | 结果 | 含义 |
| --- | ---: | --- |
| 构件 AABB | 41298/41298 通过 | 用实际 GLB 顶点和实例矩阵重算世界包围盒，未发现包围盒漏包几何 |
| 最大 AABB 数值误差 | 0.000605 | 来自浮点变换，不足以造成视锥提前剔除 |
| 原型实例数量 | 3669/3669 通过 | GLB 内实例数量与 `sceneWeb.json` 映射一致 |
| 构件与矩阵顺序 | 3669/3669 通过 | 构件编号、manifest 和 GPU 实例矩阵顺序一致 |
| GLB 级显示降级 | 0 | 没有原型触发“命中一个实例后整组显示”的兼容分支 |

根因位于 `slm2viewer/slm2/SLM2Loader.js`。前端根据当前可见构件集合压缩 `InstancedMesh` 的实例矩阵和 `count` 后，没有同步重建 Three.js 使用的包围球。相机移动后，Three.js 仍按上一批实例的旧包围球执行 draw-call 级视锥剔除。修复前一次移动测试中，1908 个已加载实例化网格有 702 个包围球过期，其中 682 个存在大于 1 米的中心或半径变化。

修复后，每次实例矩阵集合变化和恢复完整实例集合时都会清空并重建实例化网格包围体。执行命令：

```bash
cd slm2viewer
npm run package:deploy:direct
```

混淆部署包分别在强制 AABB 后端和真实 WebGPU 模型后端完成 Playwright 相机移动复测：

| 路径 | 结果 |
| --- | ---: |
| 强制 AABB 后端 | 1908 个已加载实例化网格，过期包围体 0，映射降级 0，浏览器错误 0 |
| WebGPU 模型后端 | 1481 个当前可见实例化网格，过期包围体 0，映射降级 0，浏览器错误 0 |

该修复保留为 metropolis 当前前端主线。模型阈值当前为 `0.64`，来源是严格 weighted recall 安全筛选后的最高 pose precision 工作点；普通 recall 阈值 `0.22` 仅作为历史诊断点，不能作为默认前端阈值。

修复包已同步到 `139.196.34.161` 并完成 nginx 原子切换。公网 Playwright 复测结果与本地一致：强制 AABB 后端在相机移动后检查 1849 个可见实例化网格，WebGPU 模型后端检查 1481 个可见实例化网格；两条路径的过期包围体数、GLB 级兼容降级数和浏览器错误数均为 0。

## 右上角控制面板默认状态调整

2026-07-28。为避免控制面板遮挡各场景首屏画面，将 dat.GUI 根控制面板和其中的 `PVS调试`子面板都改为默认收起。用户仍可通过右上角入口手动展开，运行时数据刷新、模型调试操作和场景控制项没有移除。

执行命令：

```bash
cd slm2viewer
npm run package:deploy:direct
```

本地和公网 Chrome smoke 均确认：根面板 `ul.closed`、`PVS调试`子面板 `ul.closed`、根面板高度为 `0`，页面错误数为 `0`。混淆包已部署到 `139.196.34.161`，nginx 配置测试和 reload 通过，公网脚本版本为 `app.a6a4d504.js?v=8483db3b8c41`。
