# Neural Instance Culling

本目录已收缩为当前 NeuralStreamWeb3D 实例级 PVS 主线。保留内容包括当前方向遮挡代理模型、当前数据集构建与采样链路、当前 benchmark 口径和前端导出脚本；旧 fixed-geo、contextual U-Net、dynamic-pool、screen-grid、topology、hypergraph、froxel 等实验代码和输出不再维护。

当前版本的完整状态以仓库顶层 `docs/README.md` 和 `docs/current/repository_layout_2026-07-31.md` 为准；本文只保留训练、评测和导出入口。

## 当前场景

| 场景 | 实例数 | GLB 数 | 前端模型 |
|---|---:|---:|---|
| `hkust-v3` | 18,831 | 3,273 | `pvs_directional_occlusion_proxy_encoder_rvl_w042_full40_hkust_fov66_best` |
| `ifcbench_fantasy_metropolis_instanced_v2` | 41,298 | 3,669 | `pvs_directional_occlusion_proxy_ifcbench_fantasy_metropolis_instanced_v2_k4_full40_best` |

## 保留数据

当前训练和评估依赖：

```text
neural_instance_culling/dataset/out/pose_csr_hkust_v3_viewcell_colorid_fov66
neural_instance_culling/dataset/out/directional_occlusion_evidence_hkust_v3_fov66
neural_instance_culling/dataset/out/glb_points_v3.bin
neural_instance_culling/dataset/out/glb_points_v3_meta.json
neural_instance_culling/dataset/out/ifcbench_fantasy_metropolis_instanced_v2_viewcell_colorid_k4
neural_instance_culling/dataset/out/directional_occlusion_evidence_ifcbench_fantasy_metropolis_instanced_v2_k4
neural_instance_culling/dataset/out/ifcbench_fantasy_metropolis_instanced_v2_glb_points_v3.bin
neural_instance_culling/dataset/out/ifcbench_fantasy_metropolis_instanced_v2_glb_points_v3_meta.json
```

`pose_csr_hkust_v3_viewcell_colorid_fov66` 是当前 HKUST 的正式 view-cell/back-camera PVS 数据集；`directional_occlusion_evidence_hkust_v3_fov66` 是方向遮挡代理模型的弱监督证据；`glb_points_v3` 是离线实例点云编码输入。

## 当前模型训练

当前运行版使用离线点云编码器生成视角无关实例上下文特征和方向遮挡代理特征。前端运行时只读取固定特征表，再用当前相机视线做轻量查询。

```bash
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n slm_pvs python -u neural_instance_culling/model/train_directional_occlusion_proxy_encoder.py \
  --dataset-dir neural_instance_culling/dataset/out/pose_csr_hkust_v3_viewcell_colorid_fov66 \
  --runtime-meta hkust-v3/assets/runtimeVisibilityMeta.json \
  --evidence-dir neural_instance_culling/dataset/out/directional_occlusion_evidence_hkust_v3_fov66 \
  --glb-points neural_instance_culling/dataset/out/glb_points_v3.bin \
  --glb-index hkust-v3/assets/glbIndex.json \
  --glb-root hkust-v3/assets \
  --output-dir neural_instance_culling/model/out/pvs_directional_occlusion_proxy_encoder_rvl_w042_full40_hkust_fov66 \
  --experiment-name pvs_directional_occlusion_proxy_encoder_rvl_w042_full40_hkust_fov66 \
  --loss-profile legacy \
  --rvl-mode evidence \
  --rvl-loss-weight 0.42 \
  --epochs 40 \
  --device cuda \
  --amp \
  > neural_instance_culling/model/out/pvs_directional_occlusion_proxy_encoder_rvl_w042_full40_hkust_fov66/train_stdout.log \
  2> neural_instance_culling/model/out/pvs_directional_occlusion_proxy_encoder_rvl_w042_full40_hkust_fov66/train_stderr.log
```

HKUST 当前权重实际使用 `legacy` 参数组合加 `rvl-mode=evidence`、`rvl-loss-weight=0.42`；这里的 `legacy` 只是保留该正式 checkpoint 的参数快照，不代表回退到 dynamic-pool。

Metropolis 的正式训练命令和参数记录在
`docs/experiments/ifc_metropolis_instanced_v2_training_2026-07-15.md`，其数据集、运行时元数据和模型输出均使用 `ifcbench_fantasy_metropolis_instanced_v2` 名称。

保留输出：

```text
neural_instance_culling/model/out/pvs_directional_occlusion_proxy_encoder_rvl_w042_full40_hkust_fov66
neural_instance_culling/model/out/pvs_directional_occlusion_proxy_encoder_rvl_w042_full40_hkust_fov66_best_eval
neural_instance_culling/model/out/pvs_directional_occlusion_proxy_ifcbench_fantasy_metropolis_instanced_v2_k4_full40
```

这些目录包含 `best.pt`、`last.pt`、epoch 快照、训练日志、`eval_summary.json`、固定实例特征表和 best checkpoint 的独立评估结果。

## 当前模型导出

```bash
conda run --no-capture-output -n slm_pvs python -u neural_instance_culling/model/export_directional_occlusion_proxy_frontend.py \
  --checkpoint neural_instance_culling/model/out/pvs_directional_occlusion_proxy_encoder_rvl_w042_full40_hkust_fov66/best.pt \
  --runtime-features neural_instance_culling/model/out/pvs_directional_occlusion_proxy_encoder_rvl_w042_full40_hkust_fov66/instance_runtime_features_fp16.bin \
  --runtime-meta hkust-v3/assets/runtimeVisibilityMeta.json \
  --eval-summary neural_instance_culling/model/out/pvs_directional_occlusion_proxy_encoder_rvl_w042_full40_hkust_fov66/eval_summary.json \
  --eval-model-name pvs_directional_occlusion_proxy_encoder_rvl_w042_full40_hkust_fov66_best \
  --output-dir slm2viewer/public/assets/neural_instance_culling/pvs_directional_occlusion_proxy_encoder_rvl_w042_full40_hkust_fov66_best
```

导出 Metropolis 时必须把 `--runtime-meta`、`--dataset-meta` 和输出目录替换为对应的实例化 v2 路径，并使用 `--workpoint primaryWeightedPrecision`；该工作点严格要求 `weighted recall > 0.99`，再选择 `pose precision` 最高的阈值。当前导出命令见 `docs/frontend/lightweight_frontend_pvs_scheduler.md` 和 Metropolis 训练记录。导出后用 `cd slm2viewer && npm run build` 同步到 `public/`，再用部署脚本生成混淆包。

Metropolis 导出示例：

```bash
conda run --no-capture-output -n slm_pvs python -u neural_instance_culling/model/export_directional_occlusion_proxy_frontend.py \
  --checkpoint neural_instance_culling/model/out/pvs_directional_occlusion_proxy_ifcbench_fantasy_metropolis_instanced_v2_k4_full40/best.pt \
  --runtime-features neural_instance_culling/model/out/pvs_directional_occlusion_proxy_ifcbench_fantasy_metropolis_instanced_v2_k4_full40/instance_runtime_features_fp16.bin \
  --feature-meta neural_instance_culling/model/out/pvs_directional_occlusion_proxy_ifcbench_fantasy_metropolis_instanced_v2_k4_full40/instance_features_meta.json \
  --runtime-meta ifcbench_fantasy_metropolis_instanced_v2/assets/runtimeVisibilityMeta.json \
  --eval-summary neural_instance_culling/model/out/pvs_directional_occlusion_proxy_ifcbench_fantasy_metropolis_instanced_v2_k4_full40/eval_summary.json \
  --eval-model-name pvs_directional_occlusion_proxy_ifcbench_fantasy_metropolis_instanced_v2_k4_full40_best \
  --output-dir slm2viewer/assets/neural_instance_culling/pvs_directional_occlusion_proxy_ifcbench_fantasy_metropolis_instanced_v2_k4_full40_best \
  --workpoint primaryWeightedPrecision \
  --dataset-meta neural_instance_culling/dataset/out/ifcbench_fantasy_metropolis_instanced_v2_viewcell_colorid_k4/dataset_meta.json
```

前端默认路径在 `slm2viewer/src/neuralCullingBackendMode.js` 中指向：

```text
./assets/neural_instance_culling/pvs_directional_occlusion_proxy_encoder_rvl_w042_full40_hkust_fov66_best
```

## Benchmark

默认 runner 保留 HKUST、IFCBench Metropolis 实例化 v2 和 AABB/HZB 规则基线。HKUST 评测示例：

```bash
conda run -n slm_pvs python neural_instance_culling/benchmark/evaluate_unified_pvs_metrics.py \
  --models baseline_aabb_hzb,pvs_directional_occlusion_proxy_encoder_rvl_w042_full40_hkust_fov66_best \
  --dataset-dir neural_instance_culling/dataset/out/pose_csr_hkust_v3_viewcell_colorid_fov66 \
  --runtime-meta hkust-v3/assets/runtimeVisibilityMeta.json \
  --output-dir neural_instance_culling/benchmark/out/current_directional_proxy \
  --device cuda
```

Metropolis 使用同一 runner 接口，但必须使用它自己的数据集和运行时元数据：

```bash
conda run -n slm_pvs python neural_instance_culling/benchmark/evaluate_unified_pvs_metrics.py \
  --models baseline_aabb_hzb,pvs_directional_occlusion_proxy_ifcbench_fantasy_metropolis_instanced_v2_k4_full40_best \
  --dataset-dir neural_instance_culling/dataset/out/ifcbench_fantasy_metropolis_instanced_v2_viewcell_colorid_k4 \
  --runtime-meta ifcbench_fantasy_metropolis_instanced_v2/assets/runtimeVisibilityMeta.json \
  --output-dir neural_instance_culling/benchmark/out/current_metropolis_directional_proxy \
  --device cuda
```

真实三角形 HZB 是 L2 warm-cache 基线。它要求完整本地 GLB 已可由浏览器光栅化，不能与冷启动
模型的资源预算直接混比。先用子集进行链路 smoke：

```bash
node neural_instance_culling/benchmark/build_triangle_hzb_cache_browser.mjs \
  --assets-dir slm2viewer/assets/scenes/ifcbench_fantasy_metropolis_instanced_v2 \
  --dataset-dir neural_instance_culling/dataset/out/pose_csr_metropolis_spatial_dense_subpose_union_fov66_v2 \
  --output /tmp/triangle_hzb_smoke/values.bin \
  --split test --max-poses 1 --glb-id-list 1109 \
  --width 64 --height 64 --chrome-exe /opt/google/chrome/google-chrome \
  --timeout-ms 120000 --force
```

只有完整 GLB 清单和完整 validation/calibration pose split 生成的缓存，且元数据中的
`formalReady=true`，才允许进入 HZB baseline 评测。`baseline_aabb_hzb` 仍只表示
`baseline_aabb_depth_proxy`，不能与 `baseline_triangle_hzb` 混称。

IFCBench Metropolis 的原始构件源保存在
`ifcbench_fantasy_metropolis_source/assets`，当前实例化展示资产保存在
`ifcbench_fantasy_metropolis_instanced_v2/assets`。实例化工具要求显式指定输入源，
避免误用已经删除的单体合并 GLB：

```bash
cd neural_instance_culling/tools/glb_instancer
npm run analyze -- \
  --source-assets ../../../ifcbench_fantasy_metropolis_source/assets \
  --scene-name ifcbench_fantasy_metropolis_instanced_v2 \
  --report ./out/ifcbench_fantasy_metropolis_instanced_v2_analysis.json
```

## 指标口径

PVS 结论优先看画面安全和有效剔除：

- 画面安全：pose recall、weighted recall、image PER、miss pixel rate。
- 有效剔除：useful cull、bad cull、平均预测数量、GLB 数量/字节削减。
- 运行成本：模型 forward latency、前端调度 latency、实际 visible mesh 和 drawn instance 数。

普通 precision、F1、Jaccard 和逐实例 accuracy 必须报告，但不能单独作为主结论。

阈值和 checkpoint 的统一选择规则是：先筛选 `weighted recall > 0.99` 的工作点，再选择 `pose precision` 最高者；F1、普通 recall、useful cull、bad cull、平均预测数和 GLB 字节削减继续作为诊断与工程评价指标。`weighted recall == 0.99` 不满足严格安全条件。

## 详细技术文档

- 总体架构：[`../docs/current/neuralstreamweb3d_architecture_technical.md`](../docs/current/neuralstreamweb3d_architecture_technical.md)
- 模型、损失和训练：[`../docs/current/neuralstreamweb3d_model_pipeline.md`](../docs/current/neuralstreamweb3d_model_pipeline.md)
- 数据集与 view-cell 采样：[`../docs/current/neuralstreamweb3d_dataset_protocol.md`](../docs/current/neuralstreamweb3d_dataset_protocol.md)

## 当前复现边界

新的统一协议使用 66° 采样/模型相机和 60° 真实渲染相机。运行时不从 Pose CSR 数据记录中回读候选 FOV；权重语义由各数据集的采样器元数据说明。

在重新训练 HKUST 前必须核对点云缓存的索引语义。当前模型代码按实例编号读取点云行，但现有 HKUST 点云元数据记录为 3,273 行，而运行时实例数为 18,831；这可能是历史缓存按 GLB 原型组织造成的语义不一致。不能仅运行本文旧命令就宣称严格复现 HKUST checkpoint，应先按模型专题文档完成资源审计。

训练输出中的 `eval_summary.json` 主要是训练阶段阈值扫描；正式 useful cull、bad cull、balanced accuracy、GLB 字节和完整图像指标以统一 benchmark 和评价文档为准。训练脚本保存 `best.pt`、导出脚本选择前端阈值时都执行同一条 weighted-recall-safe precision 规则，不再默认使用 `bestF1` 或普通 recall 工作点。

实现约束：当验证或评测阈值表没有 `weighted recall > 0.99` 的行时，训练不会替换已有 `best.pt`，benchmark/运行时读取器不会静默加载不安全阈值，前端导出也会终止。`bestF1`、普通 recall 工作点和手动不安全阈值只允许作为显式诊断，不得写入默认前端资产。
