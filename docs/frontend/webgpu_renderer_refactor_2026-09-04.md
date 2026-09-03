# WebGPU 渲染与共享 PVS 运行时重构

更新时间：2026-09-04

## 目标

本分支 `frontend/webgpu-renderer-refactor` 将当前 WebGLRenderer、WebGL 专用后处理、Worker WebGPU 推理和页面调度之间交叉的职责拆开。主路径使用 Three.js `0.183.2` 的 `WebGPURenderer`；浏览器没有 WebGPU或初始化失败时，由同一渲染器自动切换到 WebGL2 backend，不维护第二套 Viewer。

WebGPU 主路径由页面先创建一个高性能 adapter 和 device，再把同一个 device 交给 Three.js 渲染器和实例可见性查询。这样避免渲染与预测各自创建 GPUDevice，也消除 WebGPU 查询的 Worker 消息复制。真实实例编号仍需回到资源与实例矩阵调度层，因此本阶段不宣称完成零回读可见性位图；后续若改用节点材质读取 storage buffer，可以在当前共享 device 边界上继续实现。

## 模块边界

- `RendererRuntime`：异步创建 WebGPURenderer、自动 WebGL2 fallback、原生 MSAA、尺寸和后端信息。
- `PVSQuerySession`：统一 66° 后退候选、60° 真实视锥、V4 查询、缓存重过滤和结果差量。
- `PVSDispatcher`：WebGPU 渲染时在主线程提交共享 device 查询；WASM 或 WebGPU 故障时使用 Worker。
- `GlbResourceScheduler`：只消费查询结果并管理 GLB 下载、解析、挂载和驻留状态，不参与模型计算。
- `Viewer`：只编排相机、交互、场景和上述模块，不直接持有后端初始化细节。

## 简化边界

旧 `EffectComposer + N8AO + SMAAEffect` 和 RawShaderMaterial 背景已经删除。背景改为 TSL 节点，后处理改为 Three.js `RenderPipeline`、半分辨率 GTAO 和 TSL SMAA；同一份节点图由 WebGPU backend 编译为 WGSL，由 WebGL2 backend 编译为 GLSL。环境贴图使用 WebGPU 版 PMREM，PBR 材质、灯光、实例化、BatchedMesh 和资源调度保持。

软件 WebGPU adapter 不进入主路径。检测到 SwiftShader、llvmpipe、softpipe、swrast 或 software 标记时，按“没有硬件 WebGPU device”处理，自动使用 WebGL2 渲染和 Worker WASM SIMD。显式要求 `renderBackend=webgpu` 时则直接报错，不伪装成硬件 WebGPU。

## 验证

必须分别验证自动 WebGPU、强制 WebGL2 backend 和强制 WASM 查询；检查相同相机的实例/GLB 集合、共享 GPUDevice、WebGL2 fallback、画布非空、GLB 持续下载、缓存重过滤和主线程响应。硬件性能数据仍按 `hardware_gpu_execution_policy.md` 读取 adapter 与 NVIDIA 证据。

截至 2026-09-04，生产构建、前端单元测试、WebGL2 + Worker WASM 浏览器 smoke、桌面/移动尺寸非空画布和缓存重过滤均已通过。本服务器 Chrome 146 的 WebGPU adapter 返回 `google / swiftshader`，即使指定 `/etc/vulkan/icd.d/nvidia_icd.json` 仍不是 NVIDIA adapter，因此硬件 WebGPU 渲染、共享设备延迟和画面 parity 尚不能在本机形成正式结论；这不影响无 WebGPU device 的兼容路径运行。

验证过程中修正了两项启动时序问题。第一，TSL GTAO 的单通道纹理必须读取 `r` 标量后再调制场景 RGB，直接按 RGBA 相乘会让画面偏红。第二，首个模型结果可能早于较大的 `runtimeVisibilityMeta.json` 到达；实例和 GLB 位图现在优先使用 `model_meta.json` 中的固定总数初始化，避免缓存重过滤差量落到零长度位图。连续三次真实场景 smoke 的实例差量、GLB 差量和 NVIDIA WebGL/Vulkan 后端检查均通过。

生产构建中主应用未压缩 JavaScript 约 5.46 MB。WASM 兼容 Worker 改为只引入所需的 Three.js 数学与相机模块，从约 3.54 MB 降至约 414 KB；部署脚本已识别新的 `PVSWorker.*.js` 并通过独立目录打包检查。
