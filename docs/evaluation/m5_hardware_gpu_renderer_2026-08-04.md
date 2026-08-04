# M5 硬件 GPU 渲染路径核验

日期：2026-08-04  
范围：Three.js Color-ID 采样器与 M5 实例级 Color-ID 图像评价 renderer

## 结论

本机浏览器可以使用硬件 GPU。机器有 4 张 NVIDIA RTX A6000，驱动版本为 `535.183.01`；当前用户在 `video` 和 `render` 组，`/dev/nvidia*` 与 `/dev/dri/renderD*` 可访问。之前的 M5 图像评价确实使用过 SwiftShader，旧结果仍按软件渲染历史结果保留，不能改写为硬件性能结果。

本次核验复用了已经成功用于数据采样的 Chrome 启动参数，并为 M5 renderer 增加了浏览器端后端识别和硬件门。独立合成 smoke 回报：

```text
vendor   = Google Inc. (NVIDIA)
renderer = ANGLE (NVIDIA, Vulkan 1.3.242 (NVIDIA NVIDIA RTX A6000 (0x00002230)), NVIDIA)
gpuGate  = hardware=true
```

同时运行 `nvidia-smi pmon` 时可以看到 Chrome GPU 进程以 `C+G` 状态出现在 GPU 0，显存占用约 23--27 MiB。Color-ID smoke 的预期实例漏检为 `224` 个像素，说明图像语义测试仍正常完成；该数值不是模型质量结论。

随后用真实 HKUST GLB inventory 完成了独立 dense pilot：32 个 subpose、3,181 个按空间提交的 GLB、GLB 加载失败 0、渲染失败 0、self-consistency PER 0。浏览器页面仍回报同一 RTX A6000 后端，GPU 门为通过；页面总耗时为 `22.59 s`。该 pilot 只证明真实实例绑定和硬件渲染链路可运行，不替代完整 validation/calibration 图像质量门。

## 采样器的正确启动方式

当前成功采样的入口是 `neural_instance_culling/sampler/run_sampler.mjs`。它使用 Playwright 启动系统 Chrome，默认要求硬件 GPU；关键参数必须保持如下：

```text
--enable-gpu
--enable-webgl
--use-angle=vulkan
--enable-accelerated-2d-canvas
--enable-zero-copy
```

采样器在页面中读取 `WEBGL_debug_renderer_info`，日志必须出现 `NVIDIA` 和 `RTX A6000`，不能只看到“WebGL context created”。已有采样日志中的有效证据示例为：

```text
[browser:log] [sampler] WebGL backend vendor="Google Inc. (NVIDIA)" renderer="ANGLE (NVIDIA, Vulkan 1.3.242 ... RTX A6000 ...)"
```

采样器本身也执行硬件门：`run_sampler.mjs` 默认拒绝软件后端，并把 `gpuBackend`/`gpuGate` 写入最终摘要。只有调试时显式传入 `--allow-software-gpu` 才会放宽该门；该模式必须标记为非正式，不能用于重建正式数据集或填写 GPU 性能结果。

采样使用的是浏览器 GPU 光栅化：Three.js 将实例化 GLB 提交给 WebGL，Color-ID 片元写入离屏颜色缓冲，随后读取颜色缓冲统计实例 ID。CPU 负责加载资源、组织相机和汇总颜色，不应把这个链路描述成 CPU 光栅化。

## M5 renderer 的硬件门

`neural_instance_culling/benchmark/render_local_glb_color_id_browser.mjs` 现在默认带有采样器的 Vulkan 参数，并支持：

```bash
node neural_instance_culling/benchmark/render_local_glb_color_id_browser.mjs \
  --manifest <manifest.json> \
  --output-dir <output-dir> \
  --chrome-exe /usr/bin/google-chrome \
  --require-hardware-gpu
```

`--require-hardware-gpu` 会要求浏览器页面返回非软件 WebGL 后端，并拒绝含有 `SwiftShader`、`llvmpipe`、`softpipe`、`swrast` 或软件渲染标记的结果。失败时 renderer 写入 `failed_hardware_gpu_gate`，批处理不会把该结果汇总为正式评价。

正式 M5 批处理默认开启这个门：`run_m5_component_image_batch.py` 的 `--require-hardware-gpu` 默认值为启用。只有明确进行非正式的语义调试时才允许传入 `--no-require-hardware-gpu`；这种结果必须标注为软件或未核验后端，不能进入硬件性能表。任何 Chrome 参数都不得加入 `--use-angle=swiftshader*` 或 `--disable-gpu`。

Chrome 可以使用 `--headed` / `--browser-mode headed` 配合 `--display`，但当前服务器无当前用户可用的 X11 cookie；因此正式 headless 评价使用 Vulkan headless 路径。不要因为 `DISPLAY` 未设置就把 renderer 改回 `--use-angle=swiftshader-webgl`。若 Vulkan 后端不可用，应让硬件门失败并记录原因，而不是静默降级。

## 复核命令

代码和 GPU 后端的最小复核：

```bash
node --check neural_instance_culling/benchmark/render_local_glb_color_id_browser.mjs
conda run --no-capture-output -n slm_pvs \
  python -m unittest \
  neural_instance_culling.benchmark.tests.test_m5_visual_safety_image_evaluation \
  neural_instance_culling.benchmark.tests.test_instance_id_render_schema
nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.used,utilization.gpu --format=csv
```

正式结果中必须同时保存：Chrome stderr、renderer 的 `gpuBackend`/`gpuGate` 字段和一次 `nvidia-smi` 进程或监控快照。没有这三类证据时，只能写“浏览器渲染完成”，不能写“使用硬件 GPU”。

## 与历史结果的边界

- `m5_visual_safety_repair_image_chunked_20260803` 以及此前 M5 图像报告中的 SwiftShader 说明保持不变；这些结果仍可用于像素语义和模型比较，但不用于硬件延迟结论。
- 本文只确认硬件 GPU 路径可用，不改变模型、阈值、候选集合、GT、validation/calibration split 或默认前端资产。
- dense M5 正式评价必须使用独立输出目录，例如 `m5_visual_safety_repair_image_dense_hw_20260804`，不能覆盖软件路径的中断目录。

## 全量 dense 评价结果

`m5_visual_safety_repair_image_dense_hw_20260804` 已完成 48 个硬件浏览器分块和 `536,256` 个 validation/calibration 样本。所有分块均回报同一 NVIDIA RTX A6000 Vulkan 后端，GLB 加载失败和渲染失败均为 0；全量 miss-pixel rate 为 `0.7488%`。这条结果通过硬件渲染链路核验，但未通过预登记的 `<0.5%` 图像安全门，详细指标见 `docs/evaluation/m5_visual_safety_repair_image_dense_hw_2026-08-04.md`。这两个门必须分开记录，不能因硬件门通过而把模型质量写成通过。
