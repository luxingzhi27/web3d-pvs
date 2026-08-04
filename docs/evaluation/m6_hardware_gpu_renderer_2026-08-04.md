# M6 三角形 HZB 硬件 GPU 链路核验

日期：2026-08-04  
范围：真实 HKUST GLB 的三角形深度光栅化与 HZB 下采样 smoke  
状态：硬件执行子门通过；完整 M6 baseline 仍未通过

## Smoke 结果

独立输出目录：`neural_instance_culling/benchmark/out/m6_triangle_hzb_hardware_smoke_20260804/`。

| 项目 | 结果 |
|---|---:|
| GLB | 1 个真实 HKUST GLB |
| pose | 1 个 validation pose |
| 分辨率 | `64x36` |
| 查询 FOV | `66°` |
| HZB 层数 | 7 |
| WebGL vendor | `Google Inc. (NVIDIA)` |
| WebGL renderer | `ANGLE (NVIDIA, Vulkan ... NVIDIA RTX A6000 ...)` |
| `gpuGate.required` | `true` |
| `gpuGate.hardware` | `true` |
| GLB 加载/渲染 | 成功 |

页面实际回报的 `gpuBackend`、`gpuGate` 和浏览器日志保存在 `triangle_hzb.json` 及其
`browserBuild` 字段中。该结果证明三角形 HZB 入口可以通过硬件 WebGL/ANGLE 路径，不能证明完整
validation/calibration 的质量或性能门。

## 后端边界

Chrome stderr 同时记录了 `vkCreateInstance() failed: -7`，但页面 WebGL/ANGLE 仍报告 NVIDIA Vulkan
renderer 并通过硬件门。该日志不能被忽略，也不能把它改写成 WebGPU 硬件证据；本 smoke 只验收 WebGL
光栅化 API。WebGPU 仍需按独立 adapter 门核验，规则见
[`hardware_gpu_execution_policy.md`](../current/hardware_gpu_execution_policy.md)。

## 质量边界

本次使用了单 GLB、单 pose 和非正式输出范围，元数据明确为 `formalReady=false`、
`geometryScope=explicit_glb_subset_non_formal`。它不进入 M6 主比较，不替代已有完整 warm-cache 语义
结果，也不改变 NeuralPVS 实例级适配尚未实现的结论。完整 M6 仍需同一候选 CSR、完整 GLB inventory、
validation/calibration 和冷/温资源核算；在这些条件完成前，M6 总门保持 No-Go。
