# Neural Instance Culling

本目录包含实例级可见性的数据采样、训练、评价、导出和场景实例化工具。当前论文模型与部署模型分开管理：论文模型已经完成三种子正式 validation，但尚未替换前端默认资产。

## 目录

```text
dataset/      view-cell CSR、关系证据和数据构建
sampler/      Three.js Color-ID 硬件采样
model/        当前 PVS、部署基线和固定特征导出
benchmark/    正式评价、图像评价、资源评价和实验编排
tools/        GLB/IFC 场景实例化工具
```

## 当前论文模型

主线代码使用短而固定的入口：

```text
model/pvs_model.py
model/train_pvs.py
model/export_pvs.py
benchmark/run_pvs.py
benchmark/evaluate_pvs.py
benchmark/summarize_pvs.py
benchmark/reaudit_pvs.py
```

训练前预检：

```bash
conda run -n slm_pvs python \
  neural_instance_culling/benchmark/run_pvs.py \
  preflight --data-root /mnt/sda/rhyang/slm
```

runner 固定主 split、关系证据、几何表、三种子和五个正式变体，不读取 test。具体架构、训练配置和结果见 [`../docs/current/pvs_mainline_2026-08-23.md`](../docs/current/pvs_mainline_2026-08-23.md)。

## 统一评价

阈值只能由 checkpoint 自己的 calibration split 冻结。正式结果必须同时报告 weighted recall 及其下界、普通 recall、precision、accuracy、balanced accuracy、specificity、useful cull、bad cull、平均预测数和资源指标。

通用多模型评价：

```bash
conda run -n slm_pvs python \
  neural_instance_culling/benchmark/evaluate_unified_pvs_metrics.py \
  --models baseline_aabb_hzb,<model-name> \
  --dataset-dir <pose-csr> \
  --runtime-meta <scene>/assets/runtimeVisibilityMeta.json \
  --output-dir neural_instance_culling/benchmark/out/<evaluation> \
  --device cuda
```

指标语义见 [`../docs/evaluation/unified_pvs_metrics_evaluation.md`](../docs/evaluation/unified_pvs_metrics_evaluation.md)。

## 当前部署基线

| 场景 | 实例数 | GLB 数 | 前端模型 |
|---|---:|---:|---|
| `hkust-v3` | 18,831 | 3,273 | `pvs_directional_occlusion_proxy_encoder_rvl_strong_v2_full40_best` |
| `ifcbench_fantasy_metropolis_instanced_v2` | 41,298 | 3,669 | `pvs_directional_occlusion_proxy_ifcbench_fantasy_metropolis_instanced_v2_k4_full40_best` |

部署模型仍由 `model/train_directional_occlusion_proxy_encoder.py` 和 `model/export_directional_occlusion_proxy_frontend.py` 维护。不得因论文模型代码整理而删除其 checkpoint、固定特征或前端资产。

## 数据与 GPU

统一相机契约为真实渲染 `60°`，采样、后退候选和模型查询 `66°`。多个同方向 subpose 只在离线阶段构造 view-cell 的保守可见并集；浏览器每个 view-cell 只运行一次模型。

正式 Color-ID、三角形深度层和图像评价必须使用 Chrome NVIDIA Vulkan/ANGLE 硬件路径并保存 GPU evidence。采样语义和硬件门分别见：

- [`../docs/current/neuralstreamweb3d_dataset_protocol.md`](../docs/current/neuralstreamweb3d_dataset_protocol.md)
- [`../docs/current/hardware_gpu_execution_policy.md`](../docs/current/hardware_gpu_execution_policy.md)

## IFCBench 实例化

`tools/glb_instancer` 从原始构件 GLB 分析几何复用并生成实例化场景。Metropolis 的源场景和转换结果分别位于 `ifcbench_fantasy_metropolis_source/assets` 与 `ifcbench_fantasy_metropolis_instanced_v2/assets`；运行工具时必须显式指定源目录，不能从已转换结果再次推导原型。
