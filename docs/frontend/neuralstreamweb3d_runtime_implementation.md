# NeuralStreamWeb3D 前端运行时实现

更新时间：2026-08-01

本文描述浏览器端从相机变化到实例显示和 GLB 调度的真实执行路径。这里的“可见实例”必须指实例编号集合；GLB 只用于下载、解码和缓存聚合。

## 1. 初始化与场景选择

`assets/config.json` 注册场景名称、初始相机、场景元数据基址和 GLB 基址。当前配置包含：

- `hkust-v3`：HKUST v3；
- `ifcbench_fantasy_metropolis_instanced_v2`：IFCBench Fantasy Metropolis 实例化 v2。

场景元数据包括场景树、GLB 索引和 `runtimeVisibilityMeta.json`。后者提供每个实例的 AABB、实例到 GLB 的映射和场景边界。模型目录由 `src/neuralCullingBackendMode.js` 按场景映射，未知场景会回退到默认 HKUST 模型，因此新增场景必须同时更新配置和模型映射。

加载器初始化神经剔除为默认模式，并先让 Worker 使用 AABB fallback 进入可用状态，再异步初始化 WebGPU。这样模型权重尚未完成 warm-up 时不会阻塞场景初始化；WebGPU 初始化失败则保持 Worker AABB fallback，并在调试信息中保留失败原因。

## 2. 相机与预测触发

主循环约每 80ms 调用一次场景剔除流程。`CameraPredictionGate` 比较当前相机与上一次已提交预测的状态：

- 位置变化约 2m；
- 偏航变化约 6°；
- 俯仰变化约 5°；
- FOV 变化约 2°；
- aspect 变化约 0.08；
- 两次预测的最小间隔约 350ms。

这些阈值是“是否重新推理”的门控，不是“是否更新真实视锥显示”的门控。未超过门控时，旧预测结果会继续使用；新资源到达或实际相机过滤变化仍可触发渲染状态处理。当前 `neuralRenderRetainMs=0`，应用新结果时不会因为历史驻留时间把已隐藏实例长期保留。

Dispatcher 将相机位置、四元数、FOV、aspect、near/far 以及矩阵快照发送给 Worker。Worker 自己构造预测相机，不直接依赖主线程中的旧 `backCamera` 对象。

## 3. 候选、推理和真实视锥过滤

一次神经预测的实际顺序如下：

1. Worker 根据模型元数据决定预测 FOV、aspect 和后退距离；
2. 初始化阶段把完全落在单个空间桶内的实例 AABB 放入保守的空间桶索引；跨桶 AABB 放入溢出列表；
3. 查询时用预测视锥的世界空间包围范围访问相交桶，再对桶内实例和溢出实例执行原有 `Frustum.intersectsBox` 精确测试；
4. 将相交的实例编号压缩为候选数组；索引查询范围异常时回退全量 AABB 扫描；
5. 通过 WebGPU 对候选实例执行神经查询；
6. 解码可见性标志、可见性分数和 GLB 下载优先级；
7. 用真实渲染相机再次对模型预测的实例集合做实例 AABB 过滤；
8. 返回 `componentModelList`、GLB 级 immediate/prefetch 列表和调试计时。

空间特征分页是独立的实验运行模式，不是当前默认模型的替换。它把固定实例特征和精确 FP32 AABB 按空间页组织：页目录先用页联合包围盒筛选，命中的页再逐实例做 66° 后退视锥 AABB 测试，最后只把命中实例的 AABB 与 FP16 固定特征拼成当前推理的候选缓冲。这样可以把首个相机区域的网络读取与候选特征上传延后，但页被访问过后会保留在 Worker 内存中；它没有把场景压缩成固定数量 token，也没有改变实例级输出。

空间索引只改变候选枚举方式，不改变 66° 后退视锥、AABB 相交判定或实例级输出语义。默认桶边长为
64 米，也可以在模型元数据中用 `spatialAabbCellSizeM` 指定。由于 `Frustum.intersectsBox` 对跨桶大盒体可能产生
只有完整盒体才通过、子桶都不通过的保守结果，任何跨越桶边界的实例都进入溢出列表，只有单桶实例使用索引；
这比单纯限制桶数量更保守，但能保证候选集合与全量扫描一致。调试计时中的
`candidateSelection.source` 会标记 `spatial_aabb_index` 或 `full_aabb_scan`，同时记录查询桶数、溢出实例数和最终候选数。

因此，后退视锥输出用于“安全候选和预取”，真实 60° 视锥输出用于“当前显示”。模型预测结果不能直接绕过真实视锥。

当前 FOV 契约只有两项：真实渲染相机为 60°，模型查询相机为 66°。唯一配置源是 `neural_instance_culling/config/neuralpvs_viewcell_protocol.json`；Worker 和实例候选过滤直接使用这两个固定值，只读取 `modelInputFovYDeg`，不再读取旧的 PVS、buffer、training 或 runtime inference FOV 字段；两个场景均遵循这一口径。

## 4. WebGPU 运行协议

当前运行 schema 是 `directional-occlusion-proxy-scheduler-v1`。权重和固定实例表以 FP16 保存，两个半精度值打包到一个 32 位字中，WGSL 通过 `unpack2x16float` 解码。

Bind group 的当前布局为：

| binding | 内容 | 访问 |
|---:|---|---|
| 0 | 64 字节相机/阈值/候选数量等 uniform | compute uniform |
| 1 | FP16 模型权重存储缓冲 | read-only storage |
| 2 | 候选实例编号 | read-only storage |
| 3 | 当前方向遮挡代理模型不使用；空间特征分页模式下为动态候选 AABB 与固定特征 | 默认模式不绑定；分页实验模式 read-only storage |
| 4 | 每个候选实例的输出 | storage |

每个候选实例对应一个逻辑线程，工作组大小为 64。每个实例写两个 `u32`：

- 第一个的 bit0 是可见标志，后续位保存量化后的可见性分数；
- 第二个保存量化后的 GLB 下载优先级。

视觉效用分数在模型头和 WGSL 下载优先级计算中作为中间量使用，当前没有作为独立字段回传给主线程。前端调试面板不应把它写成第三个模型输出。

默认的 `pvs_directional_occlusion_proxy_encoder_rvl_strong_v2_full40_best` 仍使用完整固定特征表，因此 binding 3 不存在。实验分页模型
`pvs_directional_occlusion_proxy_encoder_rvl_strong_v2_full40_spatial_pages_m9` 才启用 binding 3；两种布局不能混用，前端根据
`usesSpatialFeaturePages` 选择 WGSL 和 bind group。

## 5. 实例级显示与 GLB 级调度

模型输出的实例 ID 先形成 `componentModelList`。GLB 级调度通过实例到 GLB 映射做聚合：

- 当前真实视锥内的 GLB 进入 immediate 队列；
- 后退视锥内但不在当前视锥的 GLB 进入 prefetch 队列；
- 队列内部按模型下载优先级排序，并受 `maxImmediate`、`maxPrefetch` 限制；
- CacheMgr 负责实际请求、解码、缓存和挂载。

渲染时，`SLM2Loader._applyInstancedVisibility()` 按 GLB 内的实例矩阵重排并设置 `InstancedMesh.count`，因此同一 GLB 可以只显示其中一部分实例。运行时会校验分组构件数量、每个实例索引和 GLB 中所有实例化网格的数量；任一绑定不一致都会将整个 GLB 标记为无效并把根节点与所有实例化网格隐藏，禁止退化为整网格显示。实例化转换和部署 smoke 仍必须检查数量一致性，因为 fail-closed 的代价是该资源暂时不显示。

真实渲染状态的主要执行者是 `CacheMgr.refreshVisible()`、`RenderVisibilitySystem.update()` 和 `_applyInstancedVisibility()`。`SLM2Loader.update()` 在缓存更新后调用 `RenderVisibilitySystem.update()`，该调用只处理已经驻留资源的真实视锥/投影状态，并沿用其时间与相机门控，不触发新的神经预测；神经预测频率仍由 `sceneCulling()` 内的预测门控控制。

## 6. fallback 与错误边界

Worker 启动阶段会先构造 AABB fallback。以下问题会使前端保守降级或直接丢失一层安全过滤：

- WebGPU 初始化失败；
- 模型二进制或元数据 schema 不匹配；
- `runtimeVisibilityMeta.json` 无法加载或解析；
- 实例 AABB 数量与模型实例数不一致；
- 实例化 GLB 的矩阵数量与场景树分组数量不一致。

尤其要注意，运行时元数据加载失败时，部分路径仍可能继续运行，但会跳过真实相机的实例级过滤。部署验证必须把 `runtimeVisibilityMeta.json` 的 HTTP 状态、JSON 解析、实例数量和 AABB 有限性列为硬检查，不应只测试首页是否能打开。

## 7. 调试指标的正确解读

调试面板中需要区分：

- 后退视锥候选数；
- 模型预测实例数；
- 真实视锥过滤后的实例数；
- 当前实际绘制实例数；
- 已加载 GLB 数和下载队列数；
- Worker/WebGPU 推理延迟与总调度延迟。

部分历史统计字段使用模型原始实例列表作为 `visibleInstanceCount`，不一定等于真实视锥后的数量。判断“是否真的少画了”应优先看 `renderComponentModelList` 和 `actualRender.drawnInstanceCount`，不能只看模型预测数或 GLB 数。

## 8. 当前前端性能边界

当前候选 AABB 过滤仍在 Worker 中执行，并非 GPU 剔除；新版本优先使用空间桶索引，WebGPU 只负责候选上的神经推理。
索引初始化仍需遍历一次实例 AABB，且溢出列表仍会逐实例检查。对大场景而言，候选生成、主线程应用实例矩阵和
GLB 解码可能分别成为瓶颈，需要分开计时；空间索引只能降低重复预测时的全场景扫描成本，不能宣称总运行成本与实例数量无关。

当前没有低频“强制保留可见构件”的隐藏逻辑；预测结果只在 CameraPredictionGate 未触发时暂时复用。若出现“看过所有构件后帧率下降”，应按实际绘制实例数、实例化矩阵更新次数、GLB 解码驻留和 runtime meta 是否失效逐项排查，而不能仅根据模型预测数量判断。

空间特征分页的 2026-08-01 审计包含 18,831 个实例和 45 个空间页。128 个 HKUST CSR pose 中，分页查询与完整 AABB 扫描的候选集合
`0/128` 不一致；静态模型权重约 `361,860` bytes，首批命中页约 `9.85 MB`，全部页约 `13.79 MB`。桌面 Chromium 的固定自动相机
smoke 已进入 `worker-webgpu`，但该相机覆盖几乎整个场景，分页路径完整推理约 `4.36 s`，完整特征表路径约 `2.62 s`；本地没有主体
GLB，且没有进行移动端或多姿态性能验收。因此分页模式目前只作为实验和后续优化入口，默认前端仍使用完整特征表。
