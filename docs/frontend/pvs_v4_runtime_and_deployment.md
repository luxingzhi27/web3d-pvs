# PVS V4 前端运行与部署

更新时间：2026-08-28

本文是当前前端神经剔除的唯一运行说明。浏览器只接受 V4 运行包；旧方向代理、dynamic-pool、相机哈希、空间分页和二阶段后端升级接口已经从当前代码与部署包移除。

## 当前模型

HKUST 使用 `pvs_mainline_v4`，运行 schema 为：

```text
pvs-bounded-relation-prior-instance-calibrated-moment-runtime-v4
```

校准阈值为 `0.6800000071525574`，由 checkpoint 自己的 calibration split 冻结。该工作点的 aggregate weighted recall 为 `0.9978366`，单侧置信下界为 `0.9963827`。当前导出对应 seed `20260802` 的无对比尾部分离消融成员，最佳安全 checkpoint 位于 epoch 36；前端不得自行重选阈值。

运行时每个实例保存 `96` 维几何特征和 `28` 维已融合的逐实例生存场系数，共 `124` 维 FP16 固定特征。离线分层关系网络、逐实例校准、点云编码和遮挡证据生成均不在浏览器执行。默认计算后端为 `auto`：优先在 Worker 内运行 WebGPU；WebGPU 不存在、adapter/device 初始化失败或运行中设备失效时，自动在同一个 Worker 内切换到 WASM SIMD 后端继续运行同一模型。纯 JavaScript CPU 神经计算已经删除。

## 运行资产

目录：

```text
slm2viewer/assets/neural_instance_culling/pvs_mainline_v4
```

文件包括：

| 文件 | 用途 |
|---|---|
| `model_meta.json` | schema、阈值、FOV、维度和校准信息 |
| `instance_runtime_features_fp16.bin` | 逐实例 124 维固定特征 |
| `instance_aabb_fp32.bin` | 逐实例世界 AABB |
| `instance_to_glb_uint32.bin` | 实例到下载 GLB 的映射 |
| `query_weights_fp16.bin` | V4 轻量查询网络权重 |
| `frequency_cycles_fp32.bin` | 16 组视点区域频率 |
| `chi_table_fp32.bin` | 视点区域矩的固定查表数据 |
| `assets/wasm/instance_pvs_v4.wasm` | WebGPU 不可用时的 V4 SIMD 查询内核 |

六个二进制文件合计约 `5.03 MiB`，加元数据约 `5.48 MB`。运行包不包含 checkpoint、训练点云、关系边、子视点、邻居表或场景 GLB。

源码配置中的 HKUST GLB 根地址固定为 `https://www.liteweb3d.com/data/hkust-v3/`；默认场景和显式 `hkust-v3` 入口使用同一地址。`resourcesBaseUrl` 仍指向本地的小型场景元数据目录，只有实际 GLB、纹理等大型构件资源从该远端根地址下载。

## 首屏相机与下载队列

HKUST 首屏相机位置为 `[-547.7782649980463, 23.54278676456619, 303.0768747325337]`，目标点为 `[-542.2631449548929, 22.259696118085966, 294.83447645881984]`，垂直 FOV 为 `60°`。浏览器运行时的 aspect 始终由实际画布尺寸决定；首屏优先队列使用 `694×552` 视口采集，对应 aspect `1.2572463768115942`。

`glbIndex.json` 只保存完整 GLB 编号、路径和 task 等映射信息，不能兼作首屏下载队列。首屏队列单独保存在：

```text
slm2viewer/assets/scenes/hkust-v3/initialGlbLoadOrder.json
```

队列在上述真实 `60°` 相机位置采集。模型输出先经过真实视锥实例过滤，再按每个 GLB 内实例的最高可见性概率排序；该位置共有 `3633` 个最终显示实例，涉及 `1602` 个 GLB。首屏文件只保存分数最高的前 `100` 个 GLB，Loader 的实际消费入口也固定限制为 `100`，不会在模型初始化前下载完整可见集合。

启动时序为：先读取完整 GLB 映射和独立首屏队列，立即发起前 `100` 个 GLB 的隐藏预加载；随后启动运行元数据和神经模型资产请求，不等待这些 GLB 下载完成。模型完成首次预测后，神经下载队列接管后续资源调度，首屏预加载不改变实例级显示集合，也不绕过真实 `60°` 视锥过滤。

## GLB 资源管线

GLB 资源只使用 HTTP，处理分为三个独立阶段：

1. **下载**：`fetch()` 将 GLB 读为 `ArrayBuffer`，每个请求保存 `AbortController`。新预测使某资源离开当前工作集时，可取消尚未完成的旧请求。
2. **解析**：下载完成的 buffer 进入有界 `GLTFLoader.parse()` 队列。桌面端默认并发 `2`，移动端默认并发 `1`，防止多个大 GLB 同时解析占满主线程和内存。
3. **挂载**：解析完成的场景进入 `pendingSceneInsertions`，按帧预算分批加入 Three.js 场景。神经模式默认每帧最多占用约 `4 ms`。

远端请求失败后，只要该 GLB 仍属于当前下载计划，就会重新进入下载队列，最多重试三次；切换视点后已经离开计划的资源不重试。该规则同时适用于立即队列和预取队列，不再只为全量加载模式重试。

真实 `60°` 视锥重过滤产生新增 GLB 后，Loader 立即将它从预取队列移除，并分别提升尚未发起的下载、已下载待解析项和已解析待挂载项。已缓存或在途的资源不重复请求。场景分组切换使用管线序号隔离旧异步任务：旧下载会被取消，已经进入解析的旧任务完成后也不会挂载到新场景。

调试面板严格区分两种数量：`modelPlan` 是最近一次预测产生的计划总数，计划完成后不会自行归零；`activeQueues` 才是尚未完成的立即下载、预取、HTTP、解析和挂载数量。不能用 `modelPlan.prefetchTotal` 判断下载器是否停住。当前静态首屏长窗口检查中累计挂载从 `111` 增长到 `2112`，最终计划内立即队列和预取队列均归零；未进入当前视点下载计划的场景 GLB 不会被全场后台下载。

重新调整首屏相机后，可在开发页面已经启动的情况下重建队列：

```bash
cd slm2viewer
npm run capture:initial-glb-order
```

该命令当前允许 SwiftShader 完成功能采集，只用于生成确定性优先级队列，不构成 WebGPU 硬件性能结果。正式延迟测试仍必须使用 NVIDIA/Vulkan 硬件门。

## 完整预测与视锥重过滤

1. 主线程把真实相机位置、旋转、宽高比和裁剪范围发送给 Worker。
2. Worker 从真实相机建立后退 `3.4641 m`、垂直 FOV `66°` 的候选相机，并把候选相机和真实 `60°` 相机的十二个视锥平面写入 WebGPU 常量缓冲。
3. 第一个 GPU 计算阶段以线程编号作为实例编号，对全部实例执行后退视锥 AABB 测试；只有通过测试的实例才运行 V4 查询网络。浏览器不会展开离线 subpose，也不会在线查询邻居。
4. 同一 GPU 线程完成阈值判断、后退区域可见实例压缩和真实 `60°` 视锥过滤，并以原子最大值把实例可见性分数聚合到所属 GLB。
5. 第二个 GPU 计算阶段遍历 GLB 聚合表，压缩得到下载队列及其最高可见性分数。
6. 普通运行只回读后退区域可见实例编号、真实视锥最终实例编号和 GLB 下载队列，不回读逐候选概率。Worker 只做队列排序与数量限制，主线程只更新实例化渲染状态。

完整预测结束后，GPU `resultBuffer` 中的后退区域模型可见实例集合保持驻留。相机仍在已登记的水平 `2 m` view-cell 内且方向、垂直位置、FOV 和宽高比契约没有变化时，浏览器不再运行可见性 MLP，也不改变下载/预取队列。Worker 只启动独立的 WebGPU 重过滤管线：读取缓存实例编号，使用最新真实 `60°` 相机重新测试实例 AABB，在 GPU 上压缩最终实例编号并聚合最终 GLB 编号。到达 view-cell 边界或契约发生变化时才重新运行完整预测。

重过滤请求与完整预测共用串行 GPU 队列。相机在请求期间继续移动时，主线程只登记一次待处理更新，并在当前请求结束后立即处理最新相机；不会并发改写统一缓冲，也不会使用固定低频定时器偷偷重跑模型。

### WebGPU 不可用时的同模型兼容

兼容路径不是纯 AABB 视锥剔除。WASM SIMD 后端仍按当前 V4 权重完整计算关系条件、视点区域频谱矩、生存场、边界摘要、可见性概率、冻结阈值、真实 `60°` 显示过滤和 GLB 最高分聚合。AABB 只负责与 WebGPU 路径完全相同的后退 `66°` 候选生成和最终真实视锥重过滤，不能直接决定神经可见集合。

`src/InstancePVSRuntime.js` 统一管理后端。`auto` 先尝试 WebGPU；初始化失败时切换 WASM，运行期 WebGPU 预测失败时切换 WASM 并重做当前查询，缓存重过滤失败时先在 WASM 上重建最近一次完整神经预测再过滤。模型资产、冻结阈值、候选相机和输出 schema 不改变，也不会调用 `setCullingMode('frustum')`。强制测试入口为：

```text
?neuralRuntimeBackend=wasm
?neuralRuntimeBackend=webgpu
?neuralRuntimeBackend=auto
```

WASM 推理始终位于 `LightweightPVSWorker`，不会阻塞 Three.js 渲染主线程。FP16 特征与权重只在初始化时解码一次，八维关系条件也只预计算一次；这些数据随后常驻 WASM 线性内存。每个 pose 只跨 JS/WASM 边界调用一次，内核内完成候选过滤、频谱矩、生存场、MLP、阈值判断、真实视锥过滤和 GLB 聚合。正式兼容实现使用单线程 SIMD，不要求 COOP/COEP，也不启用 WASM threads。

当前浏览器功能 smoke 中，初始视点得到 `9398` 个候选、`3728` 个神经可见实例和 `3633` 个真实视锥显示实例；强制 WASM 和模拟 WebGPU 获取失败后的自动 WASM 输出一致，单次完整推理分别约为 `65.2 ms` 和 `67.2 ms`。原 JavaScript Worker 路径约为 `420 ms`，WASM SIMD 在该服务器上约快 `6.3` 倍。主线程 `10 ms` 定时器在推理期间持续触发。该结果是服务器无头 Chrome 功能数据，不是移动端性能结论。

未达到阈值但分数不低于预取阈值 `0.04` 的实例可以参与 GLB 预取。当前 checkpoint 没有独立下载头，GLB 优先级由所属实例的最高可见性概率聚合得到；这属于当前部署实现，不应描述成模型已经学习了独立资源效用。

HKUST 普通模式的固定回读布局为 `4 + 2×18831 + 3×3273` 个 32 位字，即 `189940 bytes`。其中只预留最终实例列表和 GLB 队列容量，不包含逐候选概率。开启 `neuralDebugLogs=true` 时才额外回读候选编号、概率和中间特征，用于数值 parity；调试模式的传输量和延迟不能代表生产运行。

缓存重过滤使用实例和 GLB 位图，固定回读布局为 `2 + ceil(18831/32) + ceil(3273/32)` 个 32 位字，即 `694 words / 2776 bytes`。它不回读概率、不运行查询网络，也不重新生成下载优先级。Worker 直接对新旧位图求差，只把新增/移除的实例和 GLB 编号传给主线程，不再传送完整集合后重建 `Set`。

## 增量显示更新

当前神经后端输出的真实 `60°` 最终实例集合是神经模式下唯一的显示依据。旧 `RenderVisibilitySystem` 中的 CPU GLB AABB、八角点投影、屏幕面积阈值、迟滞阈值和隐藏延时已经删除；屏幕面积以后只能用于 LOD 或下载排序，不能再次删除神经后端判定为显示的实例。

GLB 工作集直接消费 Worker 的新增/移除差量，只处理变化项和刚完成下载的 GLB。普通相机更新不再遍历全部驻留资源。主线程长期保存实例和 GLB 位图，不再长期维护完整实例编号数组；只有冻结、调试或 benchmark 明确读取时才临时展开编号。

每个实例化原型维护一个无序稠密槽位表：新增实例追加到尾部；删除实例时把尾部实例交换到空槽，因此 `InstancedMesh.count` 范围内始终没有空洞。每次差量只复制新增槽位和交换槽位的 16 个矩阵分量，并通过 `instanceMatrix` update range 标记实际变化区间；离散变化过多时才合并上传范围。该实现不再移动整个有序后缀，不构造逗号字符串、不逐实例调用 `setMatrixAt()`，也不反复计算实例化包围球。由于实例已经通过神经后端的显式真实视锥过滤，这些 `InstancedMesh` 设置为 `frustumCulled=false`，避免 Three.js 使用过期动态包围体再次提前剔除。

已加载 GLB 的 Three.js 根节点在当前场景内保持稳定驻留。可见性变化只切换根节点 `visible` 和实例槽位，不频繁执行 `scene.add()` / `removeFromParent()`；只有场景切换、显式清空或缓存回收时才真正移除节点和释放资源。这避免了场景树遍历、父子关系修改和挂载队列在相机移动期间反复抖动。

GLB 编号仍只负责资源下载、驻留和根节点挂载；最终显示始终由实例编号控制。一个实例化 GLB 中只有进入最终集合的实例矩阵会计入 `InstancedMesh.count`。

## 静态场景压平与运行时合批

远端 GLB 文件和下载地址保持不变。每个 GLB 解析完成后，前端检查其场景结构；只有一个静态 Mesh 时，将该 Mesh 的世界变换烘焙到自身矩阵，直接提升到场景根，并设置 `matrixAutoUpdate=false`、`matrixWorldAutoUpdate=false`。因此原 GLTF 中只用于组织的多层 `Group/Object3D` 不进入每帧场景遍历。

已有 `InstancedMesh` 保持原实例化结构，只进行上述场景压平。普通、不透明、无骨骼、无动画、无 morph、多材质分组不超过一个且只对应单个运行时实例的 Mesh，可以进入 `BatchedMesh`。合批键由 task、共享材质和几何 attribute/index 布局共同决定；透明、多材质、骨骼、动画以及不兼容几何保留普通 Mesh 路径。

合批在浏览器端渐进完成，不改变资产服务器。每个 batch 最多容纳 `64` 个构件，并限制总顶点和索引容量；同组达到批量或停止新增 `1.5 s` 后才执行一次合并，避免每到一个 GLB 就扩容和复制。下载计划结束后，剩余两个以上的兼容构件也会合并，单个剩余构件继续使用原 Mesh。

普通单构件 GLB 的 component ID、GLB hash 和 batch object ID 保持一一映射。神经结果变化时只调用对应 batch 对象的 `setVisibleAt()`；调试红色高亮使用 `setColorAt()`，不会把整个共享材质改红。缓存淘汰时删除对应 batch instance 和 geometry，batch 为空后释放 GPU buffer。

2026-08-26 的完整当前视点加载检查中，2123 个成功返回的 GLB 有 1471 个普通构件进入 118 个 batch，理论减少 1353 个独立场景绘制对象；另有 172 个 GLB 保持原 `InstancedMesh`。该次检查有一个远端连接被重置，暴露并修正了神经下载计划不重试的问题。

## 固定原生分辨率

渲染器始终直接使用完整 `window.devicePixelRatio`，不再设置分辨率档位、像素比例上限或自动升降逻辑。例如 `1600×900 @ DPR 2` 的绘制缓冲固定为 `3200×1800`。窗口尺寸、浏览器缩放或跨显示器变化时重新读取设备 DPR；连续 resize 事件使用 `150 ms` 尾部合并，相同尺寸和 DPR 不重建渲染目标。AO、SMAA 和颜色链保持启用。

场景不创建 Three.js Fog，配置文件和控制面板也不再提供雾化、远景雾区或动态分辨率选项。原有相关控制器、GPU timer 和测试入口已经删除，不保留禁用状态的兼容代码。

## 按需渲染与低频维护

页面不再无条件每帧递归请求 RAF。相机交互、键盘/触摸移动、自动旋转、轨迹采集和显式动画期间保持连续渲染；GLB 挂载、实例可见性差量、batch 变化、材质纹理完成和窗口 resize 会主动请求新帧。控制面板折叠、相机静止且没有待挂载工作时停止主 RAF。

相机剔除不再由 Loader 每 `80 ms` 无条件轮询。OrbitControls、键盘和触摸输入只在相机实际变化时登记一次待更新，达到既有 `80 ms` 合并间隔后执行最新相机的剔除。下载请求和 GLB 解析仍由异步回调继续推进，不依赖连续 RAF；只有解析完成后的有界挂载需要逐帧预算。

`stats.update()` 只在右上角控制面板展开时运行。运行指标 DOM 和独立队列面板保持 `500 ms` 更新间隔，面板折叠时不再每帧读取完整运行统计。完整下载计划结束后的检查确认控制面板折叠时 `animationFrameId` 为空；仍在进行的纹理任务只通过低频 timer 唤醒，不恢复持续渲染。

2026-08-26 的硬件 WebGL 诊断在 `694×552` 视口回报 NVIDIA/Vulkan。早期场景约有 `158` 个 draw call，计划内资源继续挂载后可见 Mesh 接近 `1600`。低帧率同时伴随较低 GPU 功耗，更符合主线程场景遍历、大量小 WebGL draw submission、GLB 解析和挂载争用造成的 CPU/浏览器提交瓶颈；降低像素分辨率不能解决这些开销，因此当前主线固定原生 DPR，后续性能优化应集中在减少 draw call、合批和限制每帧主线程资源处理预算。

## 冻结结果检查

2026-08-25 修正了调试面板的冻结语义。冻结按钮保存当前一次神经查询经过真实 `60°` 相机视锥过滤后的最终实例编号和最终 GLB 编号。实例编号决定实际显示，GLB 编号只决定需要下载哪些资源；冻结期间不再请求全场 GLB，也不再显示全部已驻留对象。

冻结后即使移动检查相机，Loader 仍保持这份实例级快照。冻结前已经开始但尚未返回的预测会被丢弃，后续才下载完成的实例化 GLB 也会立即按照同一份冻结实例编号压缩实例矩阵。解除冻结后恢复自动缓存调度并强制发起一次新预测。冻结路径与普通路径共用精确工作集和增量实例更新，不再依赖额外 CPU 视锥或面积判断。

静态契约要求冻结入口只能读取 `renderComponentIds` 和 `renderGlbIds`，并禁止冻结分支调用全驻留显示或读取后退视锥原始实例集合。验证命令为 `cd slm2viewer && npm test && npm run build`。

HKUST 页面功能 smoke 中，冻结前的最终集合为 `3769` 个实例和 `1268` 个 GLB；冻结后页面快照、Loader 快照和实例矩阵过滤集合完全一致，下载工作集保持 `1268/3273` 个 GLB。检查相机平移 `1000 m` 后实例集合仍未变化。该 smoke 没有记录 WebGPU 硬件性能数据。

## 生产控制台诊断

2026-08-25 修正了材质 LOD 配置的资源根路径。代理 GLB 继续来自本地小型场景元数据目录；`image_lod.json` 和后续纹理统一来自场景的 `glbResourcesBaseUrl`。JSON 响应解析增加了受控失败处理，某个 task 失败时保留原 task 索引，不再让后续材质组错位。Three.js 已由弃用的 `RGBELoader` 迁移到 `HDRLoader`。

生产构建关闭 Parcel 缓存并使用独立构建缓存目录，构建结束时主动拒绝任何包含 HMR runtime 的 JavaScript。`npm test` 和 `npm run build` 均通过；静态生产页面检查确认五个 HKUST `image_lod.json` 均返回 `200 application/json`，材质组为 `5/5`，没有 JSON 解析错误、HMR WebSocket 或 RGBE 弃用警告。`content_main.js` 不属于仓库或生产包，是浏览器扩展注入脚本。

## 代码边界

| 模块 | 责任 |
|---|---|
| `src/InstancePVS.js` | V4 资产校验、GPU 候选、WGSL 查询、GPU 压缩和 GLB 聚合 |
| `src/InstancePVSWasm.js` | WASM 资产装载、常驻线性内存、单次批量调用和最终结果解码 |
| `wasm/instance_pvs_v4/src/lib.rs` | SIMD 候选筛选、ray/频谱/生存场、V4 MLP、重过滤和 GLB 聚合 |
| `src/InstancePVSRuntime.js` | `auto/webgpu/wasm` 后端选择及运行期故障切换 |
| `src/InstancePVSBackendPolicy.js` | 后端参数规范化与 WebGPU 故障识别 |
| `src/LightweightPVSWorker.js` | 构造 66°/60° 相机、运行统一后端、排序下载队列并传递最终结果 |
| `src/LightweightPVSDispatcher.js` | 相机快照、完整预测/缓存重过滤消息和请求串行号 |
| `src/CameraPredictionGate.js` | 判断是否越过 view-cell 需要完整预测，以及 cell 内是否需要重过滤 |
| `src/neuralCullingBackendMode.js` | 只为有 V4 资产的场景启用神经模式 |
| `src/RenderVisibilitySystem.js` | 按神经后端最终 GLB 集合增量挂载和移除驻留资源 |
| `src/IdBitsetState.js` | 保存主线程实例/GLB 可见位图并按需展开编号 |
| `src/DenseInstancedSlots.js` | 稠密实例槽位、交换删除和矩阵变化区间上传 |
| `src/StaticSceneOptimizer.js` | 单 Mesh 场景压平、材质/task 渐进合批和实例到 batch 对象映射 |
| `src/RenderSurfacePolicy.js` | 固定设备原生 DPR 和渲染表面尺寸计算 |
| `slm2/SLM2Loader.js` | HTTP 下载/解析/挂载管线、Worker 差量编排和场景资源生命周期 |
| `scripts/test_current.mjs` | 当前单模型静态契约检查 |
| `scripts/capture_v4_frontend_parity.mjs` | 从真实 V4 页面采集一次候选、概率和 WebGPU 后端证据 |
| `scripts/verify_v4_frontend_parity.py` | PyTorch 与 WebGPU 同位姿数值比较 |

HKUST 映射到 `pvs_mainline_v4`。Metropolis 尚未导出 V4 权重，因此明确使用实例 AABB 视锥模式；不得复用 HKUST 权重或回退到旧神经模型。

## 数值一致性

WGSL 必须与训练端依次对齐九维中心视角、`9×2` 视点区域轴、64 维频谱矩和最终概率。频谱方差采用数值稳定的等价公式，避免 GPU 三角函数微小误差在相近大数相减时被放大。

2026-08-25 的 GPU 融合同位姿检查覆盖 `5959` 个候选：

| 项目 | 结果 |
|---|---:|
| 平均概率绝对误差 | `0.0000039` |
| 最大概率绝对误差 | `0.0005391` |
| 阈值判定差异 | `0` |
| Worker 可见集合差异 | `0` |
| GPU 候选与 Three.js CPU 视锥集合差异 | `0` |
| GPU 60° 最终集合与 Three.js CPU 参考差异 | `0` |

重复剔除与增量更新修正后的浏览器 smoke 结果：

| 项目 | 结果 |
|---|---:|
| 首屏优先队列 / 实际预加载 GLB | `100 / 100` |
| 完整预测真实视锥实例 / GLB | `3633 / 1602` |
| 完整预测与 Loader 初始实例 / GLB 集合差异 | `0 / 0` |
| view-cell 内水平平移 `0.5 m` 后实例数 | `3634` |
| 平移后缓存重过滤与相同 AABB CPU 参考差异 | `0` |
| 重过滤 GPU 回读 | `2776 bytes` |
| Worker 向主线程传送的变化 ID | `5` |
| 新进入真实视锥并提升优先级的 GLB | `2` |
| 主线程实例 / GLB 位图计数 | `3634 / 1604` |
| 稠密槽位与反向映射 | 一致 |
| 主线程完整实例数组常驻 | 否 |
| 完整加载样本的合批构件 / batch | `1471 / 118` |
| 合批减少的独立绘制对象 | `1353` |
| 控制面板折叠且静止后的主 RAF | 停止 |
| 渲染像素比例 | 等于页面 `window.devicePixelRatio` |
| 雾化对象与控制项 | 不存在 |
| 平移期间完整预测序号变化 | `0` |
| 重过滤网络推理时间字段 | `0 ms` |
| Loader 最终实例集合与 Worker 重过滤差异 | `0` |

WASM SIMD 与 WebGPU 同位姿数值对照覆盖 `5959` 个候选。候选集合、阈值后可见实例集合和真实视锥实例集合完全一致；概率平均绝对误差为 `0.00000340`，最大绝对误差为 `0.0002782`。该 capture 的 WebGPU adapter 为 SwiftShader，只用于数值一致性验证，不作为硬件性能结果。

该 smoke 的无头 Chrome WebGPU adapter 回报 `google/swiftshader`，仅证明功能和集合一致性。`totalMs` 不构成硬件或移动端性能结果；正式 WebGPU 延迟必须读取 adapter 信息并通过仓库 NVIDIA/Vulkan 硬件门，WebGL 硬件证据不能替代 WebGPU adapter 证据。

## 验证与打包

WASM 源码使用用户级 Rust 工具链构建，不需要 root。构建机必须安装 `wasm32-unknown-unknown` target；`npm run build` 会先执行 `build:wasm`，以 `-C target-feature=+simd128` 编译当前内核并复制到 `assets/wasm/instance_pvs_v4.wasm`，随后才构建 Parcel 页面。发布包不能只复制 JavaScript 而遗漏该文件。

```bash
cd slm2viewer
npm test
npm run build
npm run smoke:refilter
npm run smoke:wasm-fallback
npm run package:deploy -- --scene hkust-v3
```

部署包内置与当前 Three.js 版本匹配的 Draco 和 KTX2 Basis 解码运行文件，不依赖
`unpkg` 等第三方 CDN。HKUST 独立包继续保留五个材质代理 `proxy.glb`、场景调度元数据
和神经运行资产；体量较大的 `task-*/glb/LOD0/sub_*.glb`、纹理及材质图像配置由
`glbResourcesBaseUrl` 指向的 liteweb3d 场景资源目录提供。

`smoke:refilter` 显式使用 `--allow-software-gpu`，验证缓存重过滤、CPU AABB 参考、Loader 最终集合、“未重跑 MLP”语义、渲染 DPR 等于设备原生 DPR、场景无 Fog，以及桌面/移动视口下 AO、SMAA 和画布非空。该 smoke 不输出硬件性能结论。

`smoke:wasm-fallback` 分别验证强制 WASM 和 `auto` 模式下 WebGPU 初始化失败后的自动 WASM 路径，要求两者都运行完整神经模型、产生比候选集合更小的预测集合、完成缓存重过滤，并确认主线程在推理期间保持响应。WASM/WebGPU 概率对照可在生成现有 parity capture 后运行 `scripts/verify_instance_pvs_wasm_webgpu_capture.mjs --capture=<capture.json>`。

同位姿数值检查分为页面采集和 PyTorch 对照两步：

```bash
node slm2viewer/scripts/capture_v4_frontend_parity.mjs \
  --viewer-dir slm2viewer/public \
  --out /tmp/pvs_v4_capture.json \
  --allow-software-gpu

conda run -n slm_pvs python slm2viewer/scripts/verify_v4_frontend_parity.py \
  --checkpoint <best_safe.pt> \
  --asset-dir slm2viewer/assets/neural_instance_culling/pvs_mainline_v4 \
  --capture /tmp/pvs_v4_capture.json
```

正式硬件性能采集必须把 `--allow-software-gpu` 换成 `--require-hardware-gpu`，且只有 capture
中的 WebGPU adapter、WebGL renderer 和同窗口 NVIDIA 证据共同通过时才可报告硬件耗时。

生产构建必须只携带 `pvs_mainline_v4`。部署脚本会检查运行 schema、文件集合、实例数和场景元数据；不再打包旧模型目录。

## 渲染位图边界

当前场景渲染使用 Three.js `WebGLRenderer`，神经查询主后端使用 WebGPU，兼容后端使用 Worker WASM SIMD。浏览器没有让 WebGL 着色器直接读取 WebGPU storage buffer 的零拷贝互操作接口。把可见性位图从 WebGPU 回读后再上传为 WebGL 纹理不会减少跨设备传输，而且隐藏实例仍会进入顶点阶段；因此本版继续把神经后端压缩后的最终实例编号交给现有实例矩阵压缩逻辑，它会实际降低 `InstancedMesh.count` 和绘制实例数。

只有把场景渲染整体迁移到 Three.js `WebGPURenderer` 后，渲染着色器才能与可见性计算共享 GPU 位图。该迁移会同时影响材质、后处理、加载器和浏览器兼容性，必须作为独立前端项目验证，不能在当前 WebGL 主线中加入一次 GPU 回读再上传的伪共享路径。
