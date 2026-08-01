# M9 浏览器运行时与 Cold-0 请求审计

日期：2026-08-01  
范围：`slm2viewer/scripts/benchmark_webgpu_runtime.mjs` 的离线自测与当前源码前端的 Chromium smoke。  
说明：本文前面的浏览器记录保留原始审计时点的事实；本文末尾的修复记录描述随后补入的 M9 运行时修复及其回归验证。

## 1. 目标与审计边界

M9 工具通过 Playwright 观察现有页面，不向生产运行时注入业务逻辑。它记录：

- 页面导航、XHR/fetch 和 Chromium 网络响应的开始/结束时间；
- 目标场景 GLB、代理 GLB、模型权重和运行时元数据的分类；
- 首次可见性预测调度与目标 GLB 请求之间的顺序；
- 页面和 Worker 中的 WebGPU adapter、device、shader、pipeline、command encoder、queue 事件；
- Worker 初始化/预测消息、运行时预测快照、backend 切换和调度统计；
- 当前相机与后退相机的实例 AABB 视锥查询；
- `componentModelList` 的集合关系、实例绑定、实例化网格状态和实际 Three.js 场景树状态。

工具只负责观测和报告。尤其需要区分以下三种证据：

1. `predict` 消息是预测调度，不等于预测已经完成；报告同时记录运行时预测快照。
2. 主动调用一次 AABB 视锥查询只能证明查询函数对相机变化有响应，不能证明独立渲染可见性系统在渲染循环中持续刷新。
3. 模型输出的实例 ID 集合不等于实例已经成功显示；必须有有效 GLB、实例状态或实际 `InstancedMesh` 证据才能确认显示。

## 2. 工具入口

离线自测：

```bash
node slm2viewer/scripts/benchmark_webgpu_runtime.mjs --self-test
```

已有服务时直接观测：

```bash
node slm2viewer/scripts/benchmark_webgpu_runtime.mjs \
  --url 'http://127.0.0.1:3000/?scene=hkust-v3' \
  --duration-ms 30000 \
  --settle-ms 5000 \
  --wait-for prediction \
  --out /tmp/m9.json \
  --jsonl-out /tmp/m9.jsonl
```

工具启动本地 Parcel 并在结束后清理进程：

```bash
node slm2viewer/scripts/benchmark_webgpu_runtime.mjs \
  --start-server \
  --scene hkust-v3 \
  --port 34357 \
  --duration-ms 40000 \
  --poll-ms 250 \
  --settle-ms 12000 \
  --wait-for prediction \
  --out /tmp/m9_local_hkust_release_final.json \
  --jsonl-out /tmp/m9_local_hkust_release_final.jsonl
```

`--settle-ms` 用于避免首次 fallback 预测出现后立即结束观测，使 Worker 有机会完成 WebGPU 初始化并产生后续预测。JSON 是汇总报告，JSONL 是带时间戳的事件流；本次 `/tmp` 输出是可复核的临时产物，没有加入仓库。

## 3. 离线自测

执行：

```bash
node --check slm2viewer/scripts/benchmark_webgpu_runtime.mjs
node slm2viewer/scripts/benchmark_webgpu_runtime.mjs --self-test
```

结果：语法检查通过，6/6 自测通过。

| 自测项 | 结果 |
|---|---|
| 目标场景 GLB 分类 | 通过 |
| 代理 GLB 不被当作目标几何 | 通过 |
| 神经模型权重分类 | 通过 |
| Cold-0 正常顺序 | 通过 |
| Cold-0 违规顺序 | 通过 |
| 没有首轮预测时不能误报通过 | 通过 |

## 4. Chromium smoke 实际输出

执行时间为 2026-08-01（JSON 中浏览器 wall time 使用 UTC）。浏览器为本机 Chromium，页面使用 `1280x720`、headless Chromium 和 `--enable-unsafe-webgpu`。实际报告：

```text
/tmp/m9_local_hkust_release_final.json
/tmp/m9_local_hkust_release_final.jsonl
```

### 4.1 Cold-0 请求审计

Cold-0 定义为：第一个目标场景 GLB 请求的开始时间不能早于第一个 Worker `postMessage(type=predict)`。

| 事件 | 浏览器时间 |
|---|---:|
| 首个 `predict` 调度 | 24609.4 ms |
| 首轮预测完成时间戳 | 24945.8 ms |
| 首个目标 GLB 请求 | 24949.3 ms |

实际摘要：

```text
cold0.status = pass-no-target-glb-request-before-first-prediction
requestsBeforeFirstPrediction = 0
requestsBeforeFirstPredictionCompletion = 0
completionObserved = true
completionTimestampSource = frontend-visibility.timestamp
```

因此本次源码前端 smoke 支持“首轮预测前没有目标 GLB 请求”，并且第一个目标 GLB 请求也发生在首轮预测完成快照之后。该结论只针对请求顺序，不代表 GLB 已经成功加载。

### 4.2 WebGPU 初始化与后端切换

观测到页面和 Worker 的 WebGPU 初始化事件，包括 Worker 的 `requestAdapter`、`requestDevice`、shader module、compute pipeline、command encoder 和 queue submit。页面日志还出现：

```text
[InstancePVS] Using WebGPU compute backend
[LightweightPVSDispatcher] Worker backend updated: worker-webgpu
```

摘要为：

```text
webgpu.status = observed-success
visibility.scheduledPredictions = 2
visibility.backendTransitions = [
  "uninitialized",
  "worker-aabb-fallback",
  "worker-webgpu"
]
visibility.finalBackend = worker-webgpu
visibility.finalRuntimePrediction.backend = worker-webgpu
visibility.finalRuntimePrediction.serial = 2
visibility.finalRuntimePrediction.rawInstanceCount = 658
visibility.finalRuntimePrediction.renderInstanceCount = 615
visibility.finalRuntimePrediction.rawGlbCount = 533
visibility.finalRuntimePrediction.renderGlbCount = 514
visibility.finalRuntimePrediction.totalMs = 2175.2 ms
visibility.finalRuntimePrediction.inferenceMs = 2169.2 ms
visibility.finalRuntimePrediction.postMs = 6.0 ms
```

这里的 `2175.2 ms` 是本次 headless Chromium、当前机器和当前源码包的一次观测值，不是稳定性能基准，也不能外推到移动端。

### 4.3 目标 GLB 资源失败

当前本地 Parcel 只提供了 `slm2viewer/assets/scenes/...` 下的运行元数据，默认配置请求的主体路径是：

```text
./scene_glbs/hkust-v3/task-*/glb/LOD0/sub_*.glb
```

本地不存在 `slm2viewer/scene_glbs/hkust-v3`，Parcel 对这些未知路径返回 `HTTP 200 text/html` 的应用 fallback。GLTFLoader 随后把 HTML 当作 GLB/JSON 解析，浏览器报：

```text
SyntaxError: Unexpected token '<', "<!DOCTYPE "... is not valid JSON
```

实际统计：

```text
targetGlb.requestsObserved = 1329
targetGlb.completedObserved = 1291
targetGlb.responseSummary.statuses = { "200": 1291 }
targetGlb.responseSummary.htmlFallbackResponses = 1291
targetGlb.responseSummary.nonHtmlResponses = 0
runtimeAssetErrors.loadErrorCount = 1469
runtimeAssetErrors.pageErrorCount = 1
runtimeAssetErrors.requestFailureCount = 0
```

因此工具将本次运行标记为：

```text
completed-observation-with-invalid-target-glb
```

这不是 Cold-0 失败；它是主体 GLB 未部署到本地观察服务器造成的资源完整性失败。工具没有把 `200 text/html` 当作有效 GLB，也没有把当前结果当作完整场景渲染性能结果。

### 4.4 视锥与实例级约束

探针读取到：

```text
mode = neural
idMode = global-glb-priority
raw instance count = 658
render instance count = 615
renderOutsideRaw = 0
unknownRender = 0
filterEnabled = true
filterSource = back-camera
```

主动对当前真实相机的 AABB 视锥进行一次小位移查询，结果为：

```text
activeBefore.count = 5800
activeAfterSmallMove.count = 5799
activeChangedForSmallMove = true
backBefore.count = 5933
cameraRestored = true
```

这说明实例 AABB 视锥查询对相机变化有响应，且探针恢复了相机位置。`renderOutsideRaw=0` 和 `unknownRender=0` 说明当前观测到的模型输出集合没有越过 raw 集合，也没有发现无法映射到运行时元数据的实例 ID。

但是实例显示仍未验证：

```text
loadedInstancedVisibilityStates = 0
actualInstancedMeshCount = 0
actualDrawnInstanceCount = 0
instanceLevelEvidence = not-observed
```

独立 `RenderVisibilitySystem` 的持续刷新也未验证：

```text
renderVisibilityStats = null
renderVisibilityLastUpdateAt = null
independentFrustumRefreshEvidence = not-observed
```

原因仍是目标 GLB 全部解析失败，页面没有可供 Three.js 挂载和统计的有效实例化网格。`aabbFrustumQueryEvidence=observed` 仅代表主动查询通过，不能替代真实 GLB 加载后的渲染循环证据。

## 5. 有效主体 GLB 复测

本地 Parcel 默认没有主体 GLB，上一节的 HTML fallback 运行只能作为资源失败诊断。为完成有效实例显示验证，本次仍使用本地前端、场景元数据和模型资产，但通过查询参数把主体 GLB 基址指向已部署的 HKUST 远端资源：

```bash
node slm2viewer/scripts/benchmark_webgpu_runtime.mjs \
  --start-server --scene hkust-v3 --port 34360 \
  --url 'http://127.0.0.1:34360/?scene=hkust-v3&glbResourcesBaseUrl=https%3A%2F%2Fwww.liteweb3d.com%2Fdata%2Fhkust-v3%2F' \
  --duration-ms 40000 --settle-ms 12000 --poll-ms 250 \
  --wait-for prediction --executable-path /usr/bin/google-chrome \
  --out /tmp/m9_hkust_remote_glb_final_20260801.json \
  --jsonl-out /tmp/m9_hkust_remote_glb_final_20260801.jsonl
```

正式保留的机器可读证据位于：

```text
neural_instance_culling/benchmark/out/m9_browser_runtime_hkust_remote_glb_20260801/report.json
neural_instance_culling/benchmark/out/m9_browser_runtime_hkust_remote_glb_20260801/events.jsonl
```

本次报告状态为 `completed-observation`，没有运行时资源错误：

| 证据 | 结果 |
|---|---:|
| Cold-0 首次预测调度 | 16812.7 ms |
| 首轮预测完成 | 17163.3 ms |
| 首个目标 GLB 请求 | 17170.2 ms |
| 首次预测前/完成前目标 GLB 请求 | 0 / 0 |
| 目标 GLB 响应 | 314 个 HTTP 200，0 个 HTML fallback |
| 最终后端 | `worker-webgpu` |
| 最终预测 | 1225.2 ms，其中推理 1219.2 ms、后处理 6.0 ms |
| 最终 raw/render 实例 | 658 / 615 |
| 最终 raw/render GLB | 533 / 514 |
| 实际 Three.js mesh / InstancedMesh | 203 / 16 |
| 实际绘制实例数 | 73 |
| 运行时错误 | 0 |

独立视锥探针先把真实相机沿 X 轴移动 2 个场景单位，再调用现有渲染可见性系统的强制更新，随后恢复相机并再次更新。结果为：

```text
filterEnabled = true
filterSource = back-camera
active AABB count: 5800 -> 5784
independentFrustumRefreshObserved = true
renderRefreshAfterActiveMove.skippedByGate = false
renderRefreshAfterActiveMove.frustumRejected = 4
renderRefreshAfterActiveMove.durationMs = 6.4
renderRefreshRestored.skippedByGate = false
cameraRestored = true
instanceLevelEvidence = observed
renderOutsideRaw = 0
unknownRender = 0
```

这证明当前页面的最终显示路径实际经过实例绑定和真实相机渲染可见性更新；它不证明 10k 候选下达到计划中的 `<20 ms` 桌面 p95 门槛。该次浏览器运行使用远端网络资源，仍属于单次 desktop smoke，不能替代 M10 真实设备 benchmark 或 M13 统计实验。

## 6. 验证与未覆盖项

已验证：

- M9 脚本语法和 6 项离线自测；
- Chromium 启动、页面导航和 Worker 观测注入；
- 首轮预测前无目标 GLB 请求；
- WebGPU Worker 初始化、pipeline 创建、queue submit 和 backend 切换；
- 预测快照的实例/GLB 数量；
- raw/render 实例集合约束和当前/后退相机 AABB 查询响应；
- Parcel 子进程在 smoke 结束后已清理，无残留服务进程。

未覆盖：

- 有效主体 GLB 加载后的实际 `InstancedMesh` 数量、`mesh.count`、实际绘制实例数；
- 有效主体 GLB 加载后的真实视锥刷新、摘除历史 resident 和画面正确性；
- 远程服务器的 CORS、Range、CDN 缓存和网络带宽；
- 移动端或低功耗 GPU 的推理/渲染延迟；
- 多次重复运行的统计置信区间。

后续要得到完整实例显示结论，需要给本地服务器挂载与 `config.json` 相同结构的 `scene_glbs/hkust-v3/task-*/glb/LOD0/sub_*.glb`，或直接使用已部署且主体 GLB 可访问的页面再次运行同一工具。当前报告不修改模型阈值，也不修改任何 test 结果。

## 7. M9 运行时修复记录（2026-08-01）

### 7.1 变更目的

原始审计发现两个影响 M9 准确性的缺口：正常 `SLM2Loader.update()` 没有接入独立的驻留渲染可见性更新；实例绑定数量不一致时，旧路径会把整个 `InstancedMesh` 恢复为完整数量，常驻显示路径还可能再次触发这种退化。前者会让旧的可见状态继续驻留，后者会把实例级错误扩大为 GLB 级错误。

### 7.2 实现

- `SLM2Loader.update()` 在 `modelCacheMgr.update()` 之后调用 `RenderVisibilitySystem.update()`，不传 `force`。因此它只使用已有系统的时间间隔、相机移动和驻留状态门控，不调用 `sceneCulling()`，也不改变 `neuralPredictionGate` 的预测频率。
- 加载实例化 GLB 时校验实例化网格数量、组件绑定数量、组件到实例槽位的连续索引和实例化网格是否存在。发现错误时记录明确的绑定错误原因，并隐藏 GLB 根节点、所有实例化网格，将 `InstancedMesh.count` 设为零。
- `CacheMgr.applyRenderState()` 识别无效绑定标记，避免后续缓存刷新或常驻显示把已拒绝的 GLB 根节点重新设为可见。`_showAllResidentInstancedMeshes()` 使用同一 fail-closed 路径。

### 7.3 修改文件与依赖

- `slm2viewer/slm2/SLM2Loader.js`
- `slm2viewer/slm2/CacheMgr.js`
- `slm2viewer/scripts/test_m9_frontend.mjs`
- `slm2viewer/package.json`：增加 `test:m9` 脚本

依赖资源仍为现有运行时元数据、实例化 GLB 和模型资产；没有修改模型权重、阈值、采样数据或 benchmark test split。

### 7.4 回归命令与结果

```bash
node --check slm2viewer/slm2/SLM2Loader.js
node --check slm2viewer/slm2/CacheMgr.js
npm --prefix slm2viewer run test:m9
node slm2viewer/scripts/benchmark_webgpu_runtime.mjs --self-test
git diff --check
```

`test:m9` 通过 4 个用例：正常更新路径每次调用一次渲染状态更新且不启动神经预测；缓存层拒绝无效根节点；绑定数量不匹配时整个 GLB 和实例化网格均隐藏且绘制数量为零；有效绑定仍保持实例级过滤。M9 浏览器审计脚本的 6 项离线自测也通过。

### 7.5 风险与未覆盖项

绑定错误现在会故意隐藏整包资源，可能表现为局部缺失，但这是可诊断的保守结果，避免错误几何进入画面；需要在资产发布前修复元数据与 GLB 的实例数量。此次回归未重新执行带远端主体 GLB 的完整 Chromium 长时 smoke，因此不把该用例结果当作新的浏览器延迟或画面正确性结论；后续仍需用有效主体 GLB 验证正常渲染循环中的 `renderVisibility` 统计和实际绘制实例数。

## 8. 子审查后的修复与当前门状态（2026-08-01）

### 8.1 新发现

后续只读审查发现，上一节的修复仍有三个证据缺口：隐藏 resident 只在缓存刷新路径中摘除，`RenderVisibilitySystem` 自己把对象设为不可见时仍可能把节点留在 `rootScene`；加载路径只对已经存在绑定表的 GLB 做实例过滤，缺少绑定的多实例 GLB 存在整包显示旁路；渲染循环在 `renderer.render()` 之后才调用 `SLM2Loader.update()`，无法证明本帧绘制使用的是刚完成的可见性状态。

### 8.2 本轮实现

- `RenderVisibilitySystem._setVisible()` 在默认剔除模式下通过 `CacheMgr.detachFromScene()` 摘除隐藏 resident，并记录 `detachedCount`；因此 `isVisible=false` 不再等价于仍然驻留在场景树中。
- resident 调试策略在渲染可见性系统中显式跳过，避免“显示全部”被下一帧的普通剔除逻辑覆盖。
- 从 `runtimeVisibilityMeta.globalGlbRecords` 建立“应当具有实例绑定”的哈希集合。多组件 GLB 若没有对应绑定，加载时设置 `missing-runtime-binding`，隐藏根节点和所有 `InstancedMesh`，并将绘制数量置零。该路径和数量不一致路径使用相同的 fail-closed 约束。
- `viewer.animate()` 将 `SLM2Loader.update()` 移到 `renderer.render()` 之前，使下载队列、实例矩阵、场景树驻留和实际绘制使用同一帧的状态。该更新不会启动新的神经预测。
- `InstancePVS` 和 Worker 结果新增候选 AABB 查询耗时；审计器记录候选数、摘除数、实例绑定统计，并对最多 8 个目标 GLB 响应检查 `glTF` 二进制魔数。HTML fallback、无效魔数、未探测的大响应和 Range 片段分别记录，不能混为成功几何响应。

修改文件：

- `slm2viewer/src/RenderVisibilitySystem.js`
- `slm2viewer/src/viewer.js`
- `slm2viewer/src/InstancePVS.js`
- `slm2viewer/src/LightweightPVSWorker.js`
- `slm2viewer/slm2/SLM2Loader.js`
- `slm2viewer/scripts/benchmark_webgpu_runtime.mjs`
- `slm2viewer/scripts/test_m9_frontend.mjs`

### 8.3 回归证据

```bash
npm --prefix slm2viewer run test:m9
npm --prefix slm2viewer test
node slm2viewer/scripts/benchmark_webgpu_runtime.mjs --self-test
npm --prefix slm2viewer run build
```

结果：

| 检查 | 结果 |
|---|---|
| M9 前端回归用例 | 7/7 通过 |
| 当前场景/模型完整性 smoke | HKUST 18,831 实例、3,273 GLB；Metropolis 41,298 实例、3,669 GLB，均通过 |
| 审计器 self-test | 7/7 通过 |
| Parcel 生产构建 | 通过 |
| Python/模型/数据 | 未修改 |

本轮本地 Chromium smoke 输出在 `/tmp/m9_local_after_fix.json`。它观察到 WebGPU API 可用和 Cold-0 请求顺序通过，但最终仍是 `worker-aabb-fallback-warming`；由于本地没有主体 `scene_glbs/hkust-v3`，96 个目标 GLB 响应被识别为 HTML fallback，运行状态为 `completed-observation-with-invalid-target-glb`。该结果只证明审计器能识别资源失败，不能证明完整实例渲染。

随后使用同一修复后的源码、Chromium 和已部署的 HKUST 主体 GLB 远端地址完成复测。机器可读报告保存在：

```text
neural_instance_culling/benchmark/out/m9_browser_runtime_hkust_remote_glb_after_fix_20260801/report.json
neural_instance_culling/benchmark/out/m9_browser_runtime_hkust_remote_glb_after_fix_20260801/events.jsonl
```

远端复测结果：

| 证据 | 结果 |
|---|---:|
| 报告状态 | `completed-observation` |
| Cold-0 首次预测完成前目标 GLB 请求 | 0 |
| 最终后端 | `worker-webgpu` |
| 最终候选实例数 | 5,959 |
| 候选 AABB 查询耗时 | 4.9 ms |
| 最终推理总耗时 | 1,345.6 ms |
| 其中 WebGPU 推理 | 1,340.0 ms |
| raw/render 实例 | 658 / 615 |
| raw/render GLB | 533 / 514 |
| 实例绑定期望/映射/无效 | 384 / 384 / 0 |
| 抽样验证的有效 GLB 响应 | 8/8 通过 `glTF` 魔数 |
| HTML fallback / 无效内容 | 0 / 0 |
| 实际 InstancedMesh / 绘制实例 | 12 / 67 |
| 主动视点刷新摘除 resident | 3 |
| 页面错误 / 请求失败 | 0 / 0 |

其余 208 个成功响应没有读取完整响应体，只按响应状态和非 HTML 内容记录为 `unvalidatedContentResponses`，因此不能把 8 个样本的魔数验证外推为全部资源内容证明。该次运行是单次桌面 smoke，不能替代移动端或 p95 统计。

### 8.4 当前 M9 状态与后续条件

当前结论是：

- 代码级正确性门：通过回归测试；
- 缺失绑定安全门：已实现 fail-closed，并有回归覆盖；
- 本地有效主体 GLB 门：未通过，因为本地没有主体 GLB；
- 修复后远端主体 GLB 长时 smoke：已通过单次 desktop smoke；
- 移动端、p95、多次漫游和真实设备测量：未覆盖。

因此，M9 的代码正确性、有效主体 GLB 运行链路和单次桌面 smoke 已有证据，但完整质量门仍未全绿。下一步仍需多次重复和移动端测量，至少报告 p50/p95、漫游过程中的 resident 摘除统计、不同候选规模下的候选查询/推理/调度耗时，并确认完整资源集的 Range、缓存和网络行为。

## 8.5 空间桶索引降低重复候选扫描（2026-08-01）

上一版 M9 的候选阶段虽然在 Worker 中执行，但每次预测仍遍历全部实例 AABB。当前在
`src/InstancePVS.js` 中加入保守的空间桶索引：初始化时按 64 米默认桶边长建立 AABB 到桶的映射；占据超过
128 个桶的实例进入溢出列表；查询时用 66°后退视锥的世界空间包围范围枚举桶，再使用原有
`Frustum.intersectsBox` 对桶内和溢出实例做精确判定。桶查询范围非有限或超过 100,000 个桶时回退原始全量扫描。

该修改不改变候选集合口径，不改变真实 60°视锥的第二次过滤，也不把空间桶当作遮挡判断。`lastPredictTimings.candidateSelection`
记录索引/回退来源、查询桶数、溢出数量和最终候选数，便于单独报告 CPU 候选成本。生产 Parcel 构建、Node 语法检查和
M9 离线审计自测通过；本次 3 秒本地 Chromium smoke 因本地没有主体 GLB，未观察到完整 WebGPU 预测，不把它作为性能结论。

当前质量状态：空间索引代码门通过，远端有效主体 GLB 的长时 p95 和移动端规模门仍未完成。必须在有效主体资源上比较
索引与全扫描的候选集合哈希、漏失数量、候选耗时和端到端 p95，确认 `candidateSelection.source=spatial_aabb_index`
不会改变画面安全性后，才能将其计入 M9 性能结果。
