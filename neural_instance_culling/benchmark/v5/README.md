# GCOF-PVS V5 Benchmark

V5 的正式 score bundle 是 **manifest + 列式 sidecar**。这里不接受 pose 内嵌候选数组的 JSON；候选实例、可见实例和 `visible_weights` 始终从冻结的 PoseCSR 读取，sidecar 只保存与 PoseCSR 候选顺序严格对齐的 `float32` 模型分数。

## 文件契约

根清单使用 `gcof-pvs-v5-columnar-score-bundle-v1`：

```text
manifest.json
geometry/<scene>/manifest.json
scores/<scene>/calibration/manifest.json
scores/<scene>/validation/manifest.json
```

每个 split sidecar 使用 `gcof-pvs-v5-columnar-score-sidecar-v1`，包含：

```text
pose_indices_int64.bin       # 完整 split 的 PoseCSR 行号
score_offsets_uint64.bin     # 每个 pose 在 scores 中的范围
scores_float32.bin           # 与 PoseCSR candidate_ids 同序的 logits
manifest.json
```

sidecar 不复制 `candidate_ids`、`visible_ids` 或权重。打开 sidecar 时会验证：

- PoseCSR schema、场景和实例数量；
- pose 索引完整覆盖声明 split，且 split 字段逐行一致；
- 每个 pose 的 candidate 数与 offset 完全一致；
- candidate ID 唯一且在实例范围内；
- visible ID 是 candidate 子集，visible weight 长度、有限性和非负性正确；
- score dtype、shape、字节长度和有限性正确。

surface manifest 中的非渲染退化 ID 写入 sidecar 的 `excludedUnitIds`。score offset、PoseCSR
候选校验和指标流统一排除这些 ID；若被排除 ID 出现在 visible GT，导出会失败，不能把占位
实例作为 TN 抬高 CNOR 或 Useful Cull。

## Checkpoint 推理

`inference.py` 完成两步：

1. 使用 checkpoint 的 V5 模型，在 GPU 上分块编译每个场景的 `z_i` 和 field/latent；
2. 逐 split、逐 pose 分块前向，只向 sidecar 写入 logits。

几何编译输出保存为 `geometry_fp32.bin` 与 `field_fp32.bin` 或
`generic_latent_fp32.bin`。浏览器不参与 benchmark 导出，也不会在推理时重新执行关系图传播。

默认只读取 `calibration` 和 `validation`。生成 test sidecar 必须显式加入
`--final-test`；含 test 的 bundle 标记 `testRead=true`，评估 test 也必须显式
`--final-test`。校准和模型选择只读 calibration，绝不读取 test。

LOSO held-out 场景只允许几何编译和显式评价 split。推理器不调用训练 split、训练 probe 或
任何参数更新；checkpoint manifest 会记录 `protocol`、`variant`、`seed`、`heldOutScene`
和 `sourceScenes`。

## 命令

在仓库根目录、`slm_pvs` 环境中运行。正式 GPU 运行使用 `--device cuda`，下面的
`--device cpu` 仅适合小 fixture 验证。

导出 calibration/validation：

```bash
conda run --no-capture-output -n slm_pvs \
  python -m neural_instance_culling.benchmark.v5.cli export \
  --checkpoint <checkpoint_last.pt> \
  --output-dir <columnar-score-bundle> \
  --registry neural_instance_culling/config/pvs_v5_scene_registry.json \
  --device cuda \
  > export_stdout.log 2> export_stderr.log
```

最终冻结 test replay：

```bash
conda run --no-capture-output -n slm_pvs \
  python -m neural_instance_culling.benchmark.v5.cli export \
  --checkpoint <checkpoint_last.pt> \
  --output-dir <columnar-score-bundle-final-test> \
  --final-test --device cuda \
  > final_test_export_stdout.log 2> final_test_export_stderr.log
```

评估 calibration-frozen validation：

```bash
conda run --no-capture-output -n slm_pvs \
  python -m neural_instance_culling.benchmark.v5.cli evaluate \
  --protocol shared \
  --bundle <full-seed01/manifest.json> \
  --bundle <full-seed02/manifest.json> \
  --bundle <full-seed03/manifest.json> \
  --split validation \
  --output <shared-validation-summary.json>
```

最终 test 只能使用显式 final-test bundle 和开关：

```bash
conda run --no-capture-output -n slm_pvs \
  python -m neural_instance_culling.benchmark.v5.cli evaluate \
  --protocol shared \
  --bundle <final-test-seed01/manifest.json> \
  --split test --final-test \
  --output <shared-test-summary.json>
```

单个 12k/36k 参数扫描成员使用独立入口，不要求凑齐正文 12 模型矩阵：

```bash
python -m neural_instance_culling.benchmark.v5.cli evaluate-run \
  --bundle <pilot-columnar-bundle/manifest.json> \
  --output <pilot-validation.json>

python -m neural_instance_culling.benchmark.v5.cli select-scan \
  --scan-matrix <scan_matrix.json> \
  --result model_lr1e-04_dual_lr1e-03=<result.json> \
  --result model_lr1e-04_dual_lr3e-03=<result.json> \
  --result model_lr2e-04_dual_lr1e-03=<result.json> \
  --result model_lr2e-04_dual_lr3e-03=<result.json> \
  --result model_lr4e-04_dual_lr1e-03=<result.json> \
  --result model_lr4e-04_dual_lr3e-03=<result.json> \
  --top-k 2 --output <pilot-selection.json>
```

selector 严格按 strict-LCB scene 数、mean-target scene 数、五场景等权 WR LCB、CNOR、
Useful Cull、较低 predicted/GT 的登记顺序排序，只接受 calibration-selected validation 结果。

## 测试

```bash
conda run --no-capture-output -n slm_pvs \
  python -m unittest discover \
  -s neural_instance_culling/benchmark/v5/tests -p 'test_*.py' -v
```

当前 19 项 fixture 覆盖 checkpoint 加载、GPU/CPU 分块几何编译接口、sidecar 读取、PoseCSR
候选/visible/weight 严格校验、非渲染 ID 排除、test 权限、参数扫描排序、指标分母、校准层级、
bootstrap 复杂度以及 shared/LOSO 矩阵契约。
