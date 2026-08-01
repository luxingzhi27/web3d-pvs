# M0-M2 只读质量门审计

日期：2026-08-01
审计范围：`docs/experiments/neuralstreamweb3d_submission_plan_2026-09.md` 中的 M0、M1、M2
审计性质：只读；未运行 test 集模型推理，未修改数据集，未执行 `git reset` 或 `git checkout`，未重启正在运行的训练。

## 总结

| 门 | 当前判断 | 结论边界 |
|---|---|---|
| M0 固定 validation / calibration / frozen test | **未证明** | 协议代码和中间完整 validation/calibration telemetry 已存在，但正式训练未完成；bootstrap、固定运行特征、冻结清单和 one-shot test 均缺失 |
| M1 formal 资源语义 | **已证明，带残余说明** | 当前 formal CSR 的候选、AABB、实例到 GLB 映射和 GLB 点云缓存通过；HKUST 有 3 个空几何/退化 GLB，Metropolis 当前训练的 evidence 路径仍是旧别名 |
| M2 空间隔离 | **已证明核心门** | 两个 formal manifest 的 group 互斥、guard 距离、源数据哈希和应用后 hash 均通过；类别覆盖不完整，不能据此声称类别均衡泛化 |

本记录不把已有中间 `best.pt` 或旧阈值当作正式结果。M0 总门仍是当前投稿主线的阻断项。

## M0：固定协议与冻结测试

### 已证明的子门

协议实现和入口的证据见：

- `docs/experiments/m0_fixed_validation_calibration_protocol_2026-08-01.md`
- `docs/experiments/m0_frozen_test_audit_2026-08-01.md`
- `neural_instance_culling/benchmark/evaluate_frozen_test.py`
- `neural_instance_culling/benchmark/tests/test_frozen_test_entrypoint.py`

当前两个训练输出中的 `protocol_split.json` 都声明了完整 native 四路集合，且训练 telemetry 使用完整集合：

| 场景 | train | validation | calibration | test | 当前最近完整评估 epoch |
|---|---:|---:|---:|---:|---:|
| HKUST | 5,563 | 664 | 690 | 722 | 2 |
| Metropolis | 19,296 | 2,088 | 2,376 | 2,340 | 14 |

证据路径：

- `neural_instance_culling/model/out/pvs_directional_occlusion_proxy_encoder_rvl_strong_v2_full40_hkust_spatial_fov66_seed20260801_protocolfix/protocol_split.json`
- `neural_instance_culling/model/out/pvs_directional_occlusion_proxy_encoder_rvl_strong_v2_full40_hkust_spatial_fov66_seed20260801_protocolfix/train_metrics.jsonl`
- `neural_instance_culling/model/out/pvs_directional_occlusion_proxy_encoder_rvl_strong_v2_full40_metropolis_spatial_fov66_seed20260801_bs1_gpu1/protocol_split.json`
- `neural_instance_culling/model/out/pvs_directional_occlusion_proxy_encoder_rvl_strong_v2_full40_metropolis_spatial_fov66_seed20260801_bs1_gpu1/train_metrics.jsonl`

最近一次中间校准 telemetry 也遍历了完整 calibration 集合：HKUST 690 个 pose，Metropolis 2,376 个 pose。中间选出的 weighted recall 分别为 `0.9942044904` 和 `0.9944465558`，但这不是正式阈值证据。

### 尚未证明的子门

两套当前训练输出均没有以下正式产物：

- `calibration_ready_summary.json`；
- `instance_runtime_features_fp16.bin`；
- `eval_summary.json`；
- `frozenThreshold` 对应的正式冻结 manifest；
- `frozen_calibration_one_shot_test` 和 `testEvaluationCount=1`。

对应目录当前只有中间 checkpoint、`model_meta.json`、训练历史和日志。HKUST 进程使用了
`--skip-final-test`；Metropolis 没有该参数，但训练尚未完成，当前日志也没有正式 test 收尾摘要。因此本次审计没有执行 test 推理，也不能宣称 Metropolis 已满足 one-shot 计数。

中间 calibration 的 bootstrap 字段仍为：

```text
HKUST       replicates=0, lowerBoundFloor=null
Metropolis  replicates=0, lowerBoundFloor=null
```

所以计划要求的 `point estimate >= 0.9925` 且 view-cell bootstrap 单侧 95% 下界 `> 0.99` 尚未证明。中间阈值 `0.0099999998`（HKUST）和 `0.0299999993`（Metropolis）只能作为训练 telemetry，不能写入正式前端或论文主表。

训练状态快照：

- HKUST：当前约 `4/40` epoch，最近完成 epoch 3；`trainSkippedNonFiniteLoss=0`、`trainSkippedNonFiniteGrad=0`。
- Metropolis：审计收尾时约 `15/40` epoch，最近完整记录 epoch 14；`trainSkippedNonFiniteLoss=0`、`trainSkippedNonFiniteGrad=0`。

### M0 额外 provenance 风险

- HKUST 当前运行参数显式使用 `seed=20260801`，与空间 manifest 的 `selectionSeed=20260801` 一致。
- Metropolis 输出目录名带 `seed20260801`，但实际 `model_meta.json.args.seed=20260610`，`protocol_split.json.selectionSeed=20260610`。数据集自身的空间 manifest 仍是 `selectionSeed=20260801`；这不会改变已经写入的 native split 标签，但会造成训练随机性和目录名不一致。
- Metropolis 当前命令读取 `directional_occlusion_evidence_metropolis_spatial_fov66_v1`。该目录的四个二进制 evidence 文件与明确的 `..._dense_subpose_union_fov66_v2` 目录是同 inode、同 SHA-256，但 v1 的 `evidence_meta.json.datasetDir` 仍写旧路径。最终 formal artifact 需要记录这种别名等价性，或用显式 v2 路径重新运行新的实验，不能静默改名覆盖当前结果。

### M0 最小下一步

当前训练自然完成后，先只做不读取 test CSR 的准备阶段：

```bash
conda run --no-capture-output -n slm_pvs python \
  neural_instance_culling/benchmark/evaluate_frozen_test.py prepare \
  --model-checkpoint <name>=<completed-run>/best.pt \
  --runtime-features <name>=<completed-run>/instance_runtime_features_fp16.bin \
  --dataset-dir <formal-csr> \
  --output neural_instance_culling/benchmark/out/<name>_frozen_manifest.json
```

`prepare` 通过后才可以单独人工确认清单，再在独占输出目录运行唯一一次正式 test；本次审计没有运行后一步。

## M1：formal 数据资源语义

当前 authoritative 证据不是旧的随机 CSR 报告，而是：

- `neural_instance_culling/dataset/out/pose_csr_hkust_v3_spatial_raw_subpose_aabb_fov66_v1/resource_audit.json`
- `neural_instance_culling/dataset/out/pose_csr_metropolis_spatial_dense_subpose_union_fov66_v2/resource_audit.json`
- `neural_instance_culling/dataset/out/pose_csr_hkust_v3_spatial_raw_subpose_aabb_fov66_v1/dataset_meta.json`
- `neural_instance_culling/dataset/out/pose_csr_metropolis_spatial_dense_subpose_union_fov66_v2/dataset_meta.json`

`docs/evaluation/m1_hkust_spatial_resource_audit_2026-08-01.json` 和
`m1_metropolis_spatial_resource_audit_2026-08-01.json` 中的 `datasetDir` 仍指向较早的目录，不能单独代表当前 formal CSR；本审计以 formal CSR 内的 `resource_audit.json` 和实际二进制复核为准。

### 候选集合

| 场景 | 实例 | view-cell / pose | visible refs | candidate refs | raw candidate 独立文件 | raw 候选漏可见 |
|---|---:|---:|---:|---:|---|---:|
| HKUST | 18,831 | 7,999 | 860,150 | 40,110,522 | 有；与 candidate 文件 SHA 相同 | 0 |
| Metropolis | 41,298 | 27,252 | 25,991,985 | 272,275,134 | 有；与 candidate 文件 SHA 相同 | 0 |

两个数据集的 metadata 都声明 `candidateVisibleUnionAllowed=false`、`candidateVisibleUnionAdded=0`。本次独立按 pose 读取 `raw_candidate_ids.bin`，重新计算 `visible_ids - raw_candidate_ids`，两个场景均为 `missing_refs=0`、`missing_poses=0`。因此当前 formal 候选没有依赖 GT 可见正样本补入。

模型/前端视场语义也一致：两套 `dataset_meta.json` 都记录模型/后退相机 `66°`、真实渲染 `60°`。

### AABB、实例到 GLB 映射和资产路径

实际读取 `runtimeVisibilityMeta.json`、`glbIndex.json` 并核对反向映射：

| 场景 | component records | GLB records / index | ID 连续且唯一 | component 映射反向一致 | index 资产路径缺失 | AABB 非有限/反向 |
|---|---:|---:|---|---:|---:|---:|
| HKUST | 18,831 | 3,273 / 3,273 | 是 | 0 个不一致 | 0 | 0 / 0 |
| Metropolis | 41,298 | 3,669 / 3,669 | 是 | 0 个不一致 | 0 | 0 / 0 |

HKUST 有 `158` 个 component bounds 和 `95` 个 global AABB 为零尺寸，来自点云缓存标记的 `3` 个 `emptyGeometry` GLB（global ID `1260、1490、3036`）。这不是非有限或反向 AABB，但最终报告应把它作为真实数据的退化资源记录，不能宣称所有原型都有非零体积几何。Metropolis 没有 empty geometry。

### GLB 原型点云缓存

证据路径：

- `neural_instance_culling/dataset/out/glb_points_v3_formal_hkust_fov66_meta.json`
- `neural_instance_culling/dataset/out/glb_points_v3_formal_metropolis_fov66_meta.json`
- 对应 `.bin` 文件以及场景的 `glbIndex.json`

| 场景 | cache rows | points/GLB | decoded | failed | fallback | hash match |
|---|---:|---:|---:|---:|---:|---:|
| HKUST | 3,273 | 1,024 | 3,273 | 0 | 0 | 3,273 / 3,273 |
| Metropolis | 3,669 | 1,024 | 3,669 | 0 | 0 | 3,669 / 3,669 |

二进制文件使用 16-byte header，实际文件大小比纯 payload 多 16 bytes，符合
`neural_instance_culling/model/common/glb_points.py` 的读取格式。当前 M1 数据资源门通过；固定实例运行特征的行数和 hash 需要等训练导出后由 M0 `prepare` 再验证，当前尚未证明。

## M2：空间隔离四路 split

当前 authoritative manifest：

- HKUST：`neural_instance_culling/dataset/out/spatial_split_hkust_v3_raw_subpose_aabb_fov66_20260801/manifest.json`
- Metropolis：`neural_instance_culling/dataset/out/spatial_split_metropolis_dense_subpose_union_fov66_20260801/manifest.json`

两个 manifest 的 `sourceDatasetMetaSha256`、`sourcePosesSha256` 均与实际源文件匹配；应用到 formal CSR 后，`spatialSplitManifestSha256` 和 `spatialSplitPoseLabelsSha256` 也与 manifest 目录实际文件匹配。

| 场景 | block / guard | support points | train / val / cal / test / guard pose | group 数 | 非单一 split group | 最小跨 split support 距离 |
|---|---:|---:|---|---:|---:|---:|
| HKUST | 64 m / 16 m | 279,008 | 5,563 / 664 / 690 / 722 / 360 | 7,999 | 0 | 17.5287 m |
| Metropolis | 40 m / 10 m | 109,008 | 19,296 / 2,088 / 2,376 / 2,340 / 1,152 | 757 | 0 | 10.9366 m |

实际重算得到：

- HKUST `viewcell` group 7,999 个，每个 group 只有一个 split 标签；最近跨集合支撑距离 `17.5287 m > 16 m`。
- Metropolis `xz` group 757 个，每个 group 只有一个 split 标签；最近跨集合支撑距离 `10.9366 m > 10 m`。
- 所有 dense subpose 都继承其 view-cell/group 标签，没有 unknown label；guard group 数分别为 360 和 32。

因此 M2 的核心空间泄漏门已证明。类别覆盖不是均衡的：HKUST test 没有 sky 类；Metropolis 的 validation/calibration 没有 sky，且 plaza/perimeter 在该 formal 数据中没有样本。M2 不能据此支持类别均衡或逐类别泛化结论，后续 M11 必须单独报告这一限制。

## 本次审计使用的证据与风险边界

- 读取了计划、M0-M2 既有文档、formal `dataset_meta.json`、`resource_audit.json`、manifest、runtime metadata、GLB index、点云 metadata、训练 metadata、checkpoint 文件清单和 stdout/stderr。
- 独立读取并核对了候选/可见 offset、原始候选包含关系、AABB/mapping 反向关系、manifest hash、group 标签一致性和 guard 距离。
- 没有运行任何 test 集模型推理；没有生成或覆盖 benchmark 输出；没有重建候选、采样或 split；没有停止训练。
- 当前 `best.pt` 只是训练中间 checkpoint。中间 validation/calibration 数字不能替代最终 bootstrap 校准和冻结 test。
