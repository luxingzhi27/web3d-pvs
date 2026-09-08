# 当前实例级 PVS 版本

更新时间：2026-09-09

本文记录当前前端可运行的神经模型、场景映射和运行资产。正式训练与消融结果见论文主线和评价报告。

## 场景映射

| 场景 | 实例数 | GLB 数 | 当前运行模式 | 阈值 |
|---|---:|---:|---|---:|
| HKUST v3 | 18,831 | 3,273 | `pvs_mainline_v4` | `0.6800000071525574` |
| IFCBench Fantasy Metropolis 实例化 v2 | 41,298 | 3,669 | 实例 AABB 视锥 | 不适用 |

映射唯一来源为 `slm2viewer/src/neuralCullingBackendMode.js`。只有 HKUST 有匹配实例顺序和场景元数据的 V4 权重；其他场景必须使用自己的实例表和运行模式。

## HKUST V4 来源

前端运行资产由以下 checkpoint 导出：

```text
neural_instance_culling/model/out/pvs_v4_integrated_visibility_mainline_v1/
paper_full_seed20260802_e40/
best_safe.pt
```

导出选中 epoch `36`，阈值来自 checkpoint 的 calibration 安全工作点。其 aggregate weighted recall 为 `0.9978366`，单侧置信下界为 `0.9963827`；导出元数据没有读取 test split。

前端资产：

```text
slm2viewer/assets/neural_instance_culling/pvs_mainline_v4
slm2viewer/public/assets/neural_instance_culling/pvs_mainline_v4
```

运行 schema 为 `pvs-bounded-relation-prior-instance-calibrated-moment-runtime-v4`。固定特征布局为 `96` 维几何和 `28` 维融合生存场，共 `124` 维 FP16；查询头输入为 `130` 维。

## 相机与显示契约

1. 后退 `3.4641 m`、垂直 FOV `66°` 的相机只负责建立候选集合。
2. 真实 `60°` 相机是视点区域查询中心，并负责最终实例 AABB 过滤。
3. Worker 对候选执行一次批量 V4 查询，WebGPU 优先、WASM SIMD 兼容，不展开离线 subpose。
4. 最终显示按实例编号更新；GLB 只承担下载、解码和缓存聚合。
5. 预测只在相机超出已登记的位置和方向复用范围后重新执行，不使用隐式低频轮询。

## 运行边界

浏览器只读取导出的固定实例表和轻量查询权重；关系编码、逐实例校准和训练期几何处理均在离线阶段完成。实例编号控制显示，GLB 编号只参与资源下载、解析和缓存聚合。

详细资产、代码和 parity 口径见 [`../frontend/pvs_v4_runtime_and_deployment.md`](../frontend/pvs_v4_runtime_and_deployment.md)。
