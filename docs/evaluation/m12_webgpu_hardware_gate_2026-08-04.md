# M12 WebGPU 硬件门核验

日期：2026-08-04  
范围：冻结 HKUST 前端模型的 WebGPU/WGSL parity 浏览器入口  
输出：`neural_instance_culling/benchmark/out/m12_webgpu_parity_hkust_strong_v2_hardware_20260804/m12_webgpu_capture.json`

## 结论

本次运行的 WebGPU probe 可以完成前端初始化和 16 个 parity case 的推理调用，但没有通过 WebGPU 硬件门，不能作为硬件 WebGPU 延迟或显存结果。页面报告：

```text
WebGPU adapter vendor      = google
WebGPU adapter architecture = swiftshader
WebGL renderer              = ANGLE (NVIDIA, Vulkan ... NVIDIA RTX A6000 ...)
gpuGate.required            = true
gpuGate.hardware            = false
gpuGate.softwareMarkers     = swiftshader
pageErrors                  = []
```

这说明当前 Chrome 在同一次启动中，WebGL/ANGLE 使用了 NVIDIA 硬件路径，但 WebGPU adapter 仍然选择了 SwiftShader。WebGL 的硬件证据不能替代 WebGPU adapter 证据；`nvidia-smi` 中出现 Chrome 进程也只能作为辅助信息。因此本次结果的正确表述是“WebGL 硬件路径通过、WebGPU 硬件门失败”，而不是“WebGPU 使用 GPU”。

## 执行与判定

入口：`slm2viewer/scripts/benchmark_m12_webgpu_parity.mjs`。运行使用了 `--require-hardware-gpu`，Chrome 参数包含 `--enable-gpu`、`--enable-webgpu`、`--enable-webgl`、`--use-angle=vulkan`、`--disable-software-rasterizer` 和 `--ignore-gpu-blocklist`。脚本从 `navigator.gpu.requestAdapter()` 读取 WebGPU adapter，并从 `WEBGL_debug_renderer_info` 读取 WebGL renderer；只要被测 WebGPU 证据含有软件后端标记，就以硬件门失败结束。

本次 probe 的 `backend` 字段为 `webgpu`、页面错误为空，但这只证明 WebGPU 代码路径可执行，不能解除硬件门失败。后续若要形成 WebGPU 性能结论，必须在相同的硬件门下重新运行，并保存 adapter 信息、浏览器日志、`gpuGate` 和同时间段的 NVIDIA 进程证据；不得通过删除 `--require-hardware-gpu`、加入 SwiftShader 参数或复用本目录填充硬件表。

## 硬件路径排查尝试

在不写入正式输出目录的独立探针中，依次测试了当前启动参数、增加 `--use-vulkan --enable-features=Vulkan`，以及再增加 `--force_high_performance_gpu --enable-features=Vulkan,UseSkiaRenderer` 的组合。三种 headless Chrome 组合均保持同样的结果：WebGL renderer 为 NVIDIA RTX A6000，WebGPU adapter architecture 为 `swiftshader`。因此这不是当前 M12 脚本漏传一个已知高性能参数造成的误判；该尝试保留为失败排查证据，不能成为放宽 WebGPU 硬件门的理由。

## 与 WebGL 采样的边界

Three.js Color-ID 正式采样和 M5 Color-ID 图像评价使用 WebGL/ANGLE 硬件门，已由独立报告确认 NVIDIA RTX A6000 路径可用。M12 的 WebGPU parity 属于另一条 API 路径，必须独立验收。完整规则见 [`docs/current/hardware_gpu_execution_policy.md`](../current/hardware_gpu_execution_policy.md)。
