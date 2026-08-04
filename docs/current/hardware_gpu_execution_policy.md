# 正式浏览器采样与评价的硬件 GPU 执行政策

更新时间：2026-08-04

## 目的

Three.js Color-ID 采样、实例级 Color-ID 图像评价和三角形 HZB 构建都依赖浏览器光栅化。浏览器即使创建了 WebGL 上下文，也可能实际使用 SwiftShader 或其他 CPU 软件光栅器。为了避免把软件执行误写成 GPU 采样或硬件性能结果，当前正式路径采用“硬件门失败即停止”的规则。

## 正式路径

正式运行必须满足以下条件：

1. 使用系统 Chrome/Chromium，通过 Playwright 或登记的 Node 入口启动。
2. 启动参数包含 `--enable-gpu`、`--enable-webgl`、`--use-angle=vulkan`、`--enable-accelerated-2d-canvas`、`--enable-zero-copy`、`--ignore-gpu-blocklist` 和 `--disable-software-rasterizer`。
3. 浏览器页面读取 `WEBGL_debug_renderer_info`，保存实际 WebGL vendor、renderer 和 version。
4. renderer 字段不能含有 `SwiftShader`、`llvmpipe`、`softpipe`、`swrast` 或其他软件后端标记，且不能为空。
5. 正式批处理保存浏览器日志、`gpuBackend`、`gpuGate`，并保存同一时间段的 `nvidia-smi` 与 `nvidia-smi pmon` 证据。

### Color-ID 采样的实际执行链

正式采样的“使用 GPU”具体指下面这条链路，而不是仅仅在 CPU 上调用 Three.js API：

1. `run_sampler.mjs` 或 view-cell wrapper 用 Playwright 启动系统 Chrome，并把场景 GLB 交给浏览器加载；
2. 页面中的 Three.js `WebGLRenderer` 创建 WebGL 上下文和离屏颜色目标，每个实例绑定可解码的颜色 ID；
3. `renderer.render(scene, camera)` 由 Chrome 的 Vulkan/ANGLE 硬件光栅化路径执行，深度测试和遮挡关系在 GPU 光栅化阶段完成；
4. CPU 只调用 `readPixels` 读取已经完成的 Color-ID 缓冲，统计可见实例和屏幕覆盖权重，不在 CPU 上重建三角形光栅化；
5. 页面回报 `vendor`、`renderer`、`version`，外层脚本据此生成 `gpuBackend`/`gpuGate`，并在硬件门不通过时终止任务。

因此，“Chrome 能创建 WebGL context”或“输出了 PNG/JSON”都不是 GPU 证据。正式结果必须同时保留页面后端字段、Chrome 日志和同时间段的 `nvidia-smi`/`pmon` 记录；只有确认后端为 NVIDIA Vulkan/ANGLE 等硬件路径后，才允许把结果写成硬件 GPU 采样或评价。

当前机器的已核验硬件为 NVIDIA RTX A6000；有效路径示例为：

```text
vendor   = Google Inc. (NVIDIA)
renderer = ANGLE (NVIDIA, Vulkan ... NVIDIA RTX A6000 ...)
gpuGate  = required=true, hardware=true
```

仅看到“WebGL context created”、`WebGLRenderer` 初始化成功或页面能够输出图片，都不能证明使用了硬件 GPU。

## 入口与默认行为

| 任务 | 入口 | 默认规则 |
|---|---|---|
| 单场景 Color-ID 采样 | `neural_instance_culling/sampler/run_sampler.mjs` | 默认要求硬件 GPU |
| NeuralPVS view-cell 分片采样 | `neural_instance_culling/sampler/run_scene_viewcell_colorid_sampling.mjs` | wrapper 强制传入 `--require-hardware-gpu` |
| M5 实例级 Color-ID 图像评价 | `neural_instance_culling/benchmark/render_local_glb_color_id_browser.mjs` 与 `run_m5_component_image_batch.py` | 默认要求硬件 GPU |
| 三角形 HZB 浏览器构建 | `neural_instance_culling/benchmark/build_triangle_hzb_cache_browser.mjs` | 默认要求硬件 GPU |

正式采样的典型命令：

```bash
node neural_instance_culling/sampler/run_sampler.mjs \
  --assets-dir <scene-assets> \
  --pose-plan <pose-plan.jsonl> \
  --output <formal-output.jsonl> \
  --require-hardware-gpu
```

M5 renderer 的正式命令必须包含：

```bash
node neural_instance_culling/benchmark/render_local_glb_color_id_browser.mjs \
  --manifest <manifest.json> \
  --output-dir <output-dir> \
  --require-hardware-gpu
```

### 采样证据文件

`run_sampler.mjs` 每完成一个分片，会在采样 JSONL 旁写出同名的
`<分片>.jsonl.gpu_evidence.json`。该文件固定记录页面回报的 `gpuBackend`、`gpuGate`、实际 Chrome
可执行文件和启动参数，以及渲染期间的 `nvidia-smi` 与 `nvidia-smi pmon` 快照。正式 view-cell wrapper
`run_scene_viewcell_colorid_sampling.mjs` 在所有分片完成后，额外写出输出目录下的
`gpu_execution_summary.json`；它会逐分片检查证据，任何分片缺失证据、硬件门失败或出现错误都会使 wrapper
以失败退出。

因此，正式采样结果不能只依据 JSONL 行数或“浏览器成功输出”判定。若旧采样目录没有这些证据文件，不能事后
推断它使用了硬件 GPU；需要在独立输出目录中按当前入口重新采样，不能覆盖旧结果。

## 软件路径边界

`--allow-software-gpu` 只允许用于小规模的颜色语义、实例绑定或代码调试。它必须在日志和输出元数据中明确标记为非正式；不得用它重建正式数据集，不得用它填写 GPU/浏览器性能，不得覆盖同名硬件输出目录，也不得作为移动设备性能证据。

如果硬件门失败，应先检查 Chrome 可执行文件、NVIDIA 驱动、用户对 `/dev/nvidia*`/`/dev/dri/renderD*` 的访问权限、`--use-angle=vulkan` 和 `nvidia-smi`，然后重新运行。不能通过删除硬件门、加入 SwiftShader 参数或把失败结果继续汇总来“修复”问题。

## WebGL 与 WebGPU 分开核验

WebGL 的硬件门只证明 WebGL/ANGLE 的光栅化路径；它不能推断 WebGPU 适配器也使用硬件 GPU。凡是涉及 WebGPU 推理、WGSL 延迟或 WebGPU buffer/dispatch 的实验，必须额外读取 `navigator.gpu.requestAdapter()` 返回的适配器信息，并对该 API 单独执行软件后端拒绝规则。若 WebGL 报告 NVIDIA、但 WebGPU 报告 `SwiftShader`、`software` 或空适配器，结果只能记为“WebGL 硬件路径通过、WebGPU 硬件门失败”，不得把 WebGPU 延迟或显存结论写成硬件结果。

同理，`nvidia-smi` 看到 Chrome 进程只能作为辅助证据，不能替代页面实际回报的 API 后端字段。正式报告必须注明被测 API（WebGL 或 WebGPU）、对应的 renderer/adapter、硬件门结果和同时间段的 NVIDIA 进程证据。

## 结果验收

正式结果缺少任意一项时，只能写“浏览器渲染完成”，不能写“硬件 GPU 采样/评价”：

- 页面 `gpuBackend`；
- `gpuGate.required=true` 且 `gpuGate.hardware=true`；
- Chrome stdout/stderr；
- 同时间窗口的 `nvidia-smi` 快照或 `pmon` 记录。

对正式 Color-ID 采样，`nvidia-smi` 和 `nvidia-smi pmon` 均必须可读取；只存在其中一个时仍视为证据不完整。

模型质量门和硬件门是两件事。硬件门通过只说明执行后端可信，不代表模型的 recall、miss-pixel rate、延迟或移动端性能已经达标；这些指标必须按各自的评价协议单独判断。历史 SwiftShader 结果可以保留作语义诊断，但必须与硬件结果分目录、分报告，不能混合汇总。
