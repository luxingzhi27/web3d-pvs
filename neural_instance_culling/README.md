# Neural Instance Culling

本目录包含 NeuralStreamWeb3D 的实例级可见性训练、采样、评测、特征导出和 IFCBench 实例化工具。当前默认前端模型和仓库边界以 [`../docs/README.md`](../docs/README.md) 与 [`../docs/current/current_instance_pvs_versions.md`](../docs/current/current_instance_pvs_versions.md) 为准。

## 当前场景

| 场景 | 实例数 | GLB 数 | 前端模型 |
|---|---:|---:|---|
| `hkust-v3` | 18,831 | 3,273 | `pvs_directional_occlusion_proxy_encoder_rvl_strong_v2_full40_best` |
| `ifcbench_fantasy_metropolis_instanced_v2` | 41,298 | 3,669 | `pvs_directional_occlusion_proxy_ifcbench_fantasy_metropolis_instanced_v2_k4_full40_best` |

## 训练数据

当前正式入口使用 66° view-cell/back-camera 数据和 Three.js Color-ID 采样结果：

```text
dataset/out/pose_csr_hkust_v3_viewcell_colorid_fov66
dataset/out/directional_occlusion_evidence_hkust_v3_fov66
dataset/out/ifcbench_fantasy_metropolis_instanced_v2_viewcell_colorid_k4
dataset/out/directional_occlusion_evidence_ifcbench_fantasy_metropolis_instanced_v2_k4
```

采样语义、候选集合和 GPU 证据要求见 [`../docs/current/neuralstreamweb3d_dataset_protocol.md`](../docs/current/neuralstreamweb3d_dataset_protocol.md) 与 [`../docs/current/hardware_gpu_execution_policy.md`](../docs/current/hardware_gpu_execution_policy.md)。

## 训练

训练使用 Linux CUDA conda 环境，并将 stdout/stderr 写入输出目录：

```bash
conda run --no-capture-output -n slm_pvs python -u \
  neural_instance_culling/model/train_directional_occlusion_proxy_encoder.py \
  --dataset-dir <pose-csr> \
  --runtime-meta <scene>/assets/runtimeVisibilityMeta.json \
  --evidence-dir <directional-evidence> \
  --glb-points <glb-points.bin> \
  --glb-index <scene>/assets/glbIndex.json \
  --glb-root <scene>/assets \
  --output-dir neural_instance_culling/model/out/<experiment> \
  --experiment-name <experiment> \
  --epochs 40 --device cuda \
  > neural_instance_culling/model/out/<experiment>/train_stdout.log \
  2> neural_instance_culling/model/out/<experiment>/train_stderr.log
```

训练前必须检查 `visible_ids ⊆ candidate_ids`、实例 AABB/GLB 映射、点缓存语义和证据维度。当前主线使用 RVL 可见性损失；具体结构、监督和导出格式见 [`../docs/current/neuralstreamweb3d_model_pipeline.md`](../docs/current/neuralstreamweb3d_model_pipeline.md)。

## 评测

统一评测必须同时报告画面安全、分类诊断、有效剔除、GLB 字节和运行成本：

```bash
conda run -n slm_pvs python neural_instance_culling/benchmark/evaluate_unified_pvs_metrics.py \
  --models baseline_aabb_hzb,<model-name> \
  --dataset-dir <pose-csr> \
  --runtime-meta <scene>/assets/runtimeVisibilityMeta.json \
  --output-dir neural_instance_culling/benchmark/out/<evaluation> \
  --device cuda
```

阈值只能由本 checkpoint 的 calibration split 冻结。当前优化阶段先要求 weighted recall 及其置信下界合格，再比较 precision、F1、useful cull、平均预测数和资源节省；bad cull 继续报告，但暂不作为路线否决条件。最新阶段结果见 [`../docs/current/optimization_restart_2026-08-10.md`](../docs/current/optimization_restart_2026-08-10.md)。

## 导出与前端

```bash
conda run --no-capture-output -n slm_pvs python -u \
  neural_instance_culling/model/export_directional_occlusion_proxy_frontend.py \
  --checkpoint <checkpoint>/best.pt \
  --runtime-features <checkpoint>/instance_runtime_features_fp16.bin \
  --runtime-meta <scene>/assets/runtimeVisibilityMeta.json \
  --eval-summary <checkpoint>/calibration_ready_summary.json \
  --output-dir slm2viewer/assets/neural_instance_culling/<model>
```

导出资产必须包含 `instance_model_meta.json` 和 `instance_pvs_assets.bin`，并与实例数量、AABB、实例到 GLB 映射和 60°/66° FOV 契约一致。前端不运行点云编码或动态图传播，具体运行逻辑见 [`../docs/frontend/neuralstreamweb3d_runtime_implementation.md`](../docs/frontend/neuralstreamweb3d_runtime_implementation.md)。

## IFCBench 实例化工具

原始 `sub_*.glb` 位于 `ifcbench_fantasy_metropolis_source/assets`，实例化结果位于 `ifcbench_fantasy_metropolis_instanced_v2/assets`。工具要求显式指定原始构件源：

```bash
cd neural_instance_culling/tools/glb_instancer
npm run analyze -- \
  --source-assets ../../../ifcbench_fantasy_metropolis_source/assets \
  --scene-name ifcbench_fantasy_metropolis_instanced_v2 \
  --report ./out/ifcbench_fantasy_metropolis_instanced_v2_analysis.json
```

Metropolis 的场景训练记录保留在 [`../docs/experiments/ifc_metropolis_instanced_v2_training_2026-07-15.md`](../docs/experiments/ifc_metropolis_instanced_v2_training_2026-07-15.md)。
