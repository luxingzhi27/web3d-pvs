# 当前实例级 PVS 版本

更新时间：2026-08-25

本文只记录当前前端实际可运行的神经模型。历史训练和正式消融结果继续保留在独立输出目录及评价报告中，但不作为浏览器兼容路径。

## 场景映射

| 场景 | 实例数 | GLB 数 | 当前运行模式 | 阈值 |
|---|---:|---:|---|---:|
| HKUST v3 | 18,831 | 3,273 | `pvs_mainline_v4` | `0.6800000071525574` |
| IFCBench Fantasy Metropolis 实例化 v2 | 41,298 | 3,669 | 实例 AABB 视锥 | 不适用 |

映射唯一来源为 `slm2viewer/src/neuralCullingBackendMode.js`。只有 HKUST 有匹配实例顺序和场景元数据的 V4 权重；其他场景不能复用该模型，也不能加载已删除的旧方向代理资产。

## HKUST V4 来源

前端运行资产由以下 checkpoint 导出：

```text
/mnt/sda/rhyang/slm-worktrees/pvs-v4-integrated-mainline-v1/
neural_instance_culling/model/out/
pvs_v4_integrated_visibility_mainline_v1_20260821/
formal40_s02_ablation_without_contrastive_separation_s02_guard030_sep020_mix025_seed20260802_e40/
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
3. Worker 对候选执行一次 WebGPU 批量推理，不展开离线 subpose。
4. 最终显示按实例编号更新；GLB 只承担下载、解码和缓存聚合。
5. 预测只在相机超出已登记的位置和方向复用范围后重新执行，不使用隐式低频轮询。

## 已移除运行路径

当前前端不再接受旧 V3、方向代理、dynamic-pool、camera hash、空间分页、M9/M12、在线邻居图或二阶段 WebGPU 升级接口。PointNet、分层关系编码和逐实例校准只在离线训练/导出阶段运行。

详细资产、代码和 parity 口径见 [`../frontend/pvs_v4_runtime_and_deployment.md`](../frontend/pvs_v4_runtime_and_deployment.md)。
