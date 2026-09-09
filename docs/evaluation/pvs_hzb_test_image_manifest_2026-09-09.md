# HZB 正式 test 图像 manifest 转换

日期：2026-09-09

`build_hzb_image_manifest.py` 将同场景的完整 formal-v2 test manifest 作为图像
相机、实例绑定和原始 GLB reference 模板，把正式 HZB Region66 test result 的
实例预测写回 `predictionComponentIdsByKey`。转换器只做 schema 和集合契约检查，
不读取指标、不运行模型、不启动 GPU；renderer 继续使用实例级通用 visibility mask。

## 输入契约

- base 必须是 `local-true-component-id-formal-render-manifest-v2` 的完整 test
  manifest，并通过 frozen-test 字段和 formal renderer validator；
- result 必须是 `geometry-shell-hzb-browser-result-v1`、`mode=Region66`，其
  `workload.schema` 为 `geometry-shell-hzb-browser-workload-v1`、`workload.split=test`；
- `samples[].poseId` 必须唯一，且与 base 去重后的 `samples[].viewcellRow` 完全覆盖；
- 每个 `visibleInstanceIds` 必须是显式 CSR candidate 行的子集，并且属于 base
  的完整 `componentToBinding` 实例范围。CSR 只读取
  `candidate_offsets.bin` 和 `candidate_ids.bin`；
- CLI 必须显式提供 `--candidate-dataset-dir` 和 `--asset-variant`。候选目录与
  result 的 poseId 使用同一 Pose CSR 行号。

## 输出契约

输出保留 base 的真实 `60°` 相机、dense subpose、GLB reference、完整实例绑定和
`predictionKey`；同一 view-cell 的多个 sample 继续共享一个预测 key。输出删除
神经选择字段 `threshold`、`thresholdSelection`、`thresholdProvenance`，写入：

```json
{
  "method": "geometry-shell-hzb",
  "selectionSplit": "calibration",
  "testRead": false,
  "assetVariant": "equal-asset",
  "resolution": [480, 270],
  "depthBiasM": 0.001,
  "regionSampleCount": 0,
  "sourceResult": "region66-test.json"
}
```

字段来源固定为当前 runner 的记录：`resolution` 读取
`result.workload.width/height`，`depthBiasM` 读取全部
`result.samples[].timings.depthBiasM` 并要求一致，`regionSampleCount` 读取
`result.regionSampling.requestedCount`。`assetVariant` 来自显式 CLI 参数；
`sourceResult` 为 result 文件名。`selectionSplit=calibration` 表示配置在 test
前冻结，不参与 test 配置选择。

## 转换与 schema 检查

```bash
conda run --no-capture-output -n slm_pvs \
  python neural_instance_culling/benchmark/build_hzb_image_manifest.py \
  --base-manifest <scene-formal-v2-test-manifest.json> \
  --hzb-result <scene-region66-test.json> \
  --candidate-dataset-dir <pose-csr-test-dataset> \
  --asset-variant equal-asset \
  --output <scene-hzb-formal-v2-test-manifest.json>
```

转换输出可交给薄 runner 做 schema-only 检查；该路径调用现有 Node renderer 的
`--validate-only`，不启动浏览器：

```bash
CUDA_VISIBLE_DEVICES= conda run --no-capture-output -n slm_pvs \
  python neural_instance_culling/benchmark/run_test_image_evaluation.py \
  --manifest <scene-hzb-formal-v2-test-manifest.json> \
  --output-dir <schema-output> --require-hardware-gpu --render-schema-only
```

薄 runner 对正式 test manifest 接受 calibration-frozen
`thresholdSelection` 或 `baselineSelection` 之一；HZB 输出使用后者。正式图像
评价仍需通过同一 runner 的硬件 GPU 入口单独执行。

## 验证记录

本次追加精简了转换器和契约测试，覆盖 formal base、Region66/test、pose 覆盖与
重复、candidate/实例范围、baselineSelection 和 schema-only renderer。验证命令为：

```bash
conda run --no-capture-output -n slm_pvs \
  python -m unittest discover -s neural_instance_culling/benchmark/tests -p 'test_*.py'
```

结果：`122 tests`、`2 skipped`、`OK`。未启动 GPU、Chrome 或正式图像评价，也没有
新的指标结论。该入口保留为 Geometry-shell HZB 正式 test 图像基线，不改变神经
PVS 主线。
