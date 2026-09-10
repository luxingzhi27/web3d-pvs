# HZB 正式 test 图像 manifest 转换

日期：2026-09-09

`build_hzb_image_manifest.py` 将同场景的完整 formal-v2 test manifest 作为图像
相机、实例绑定和原始 GLB reference 模板，把正式 HZB Region66 test result 的
实例预测写回 `predictionComponentIdsByKey`。转换器只做 schema 和集合契约检查，
不读取指标、不运行模型、不启动 GPU；renderer 继续使用实例级通用 visibility mask。
转换前会核对场景、60° 渲染/66° 查询 FOV、逐 pose aspect、完整 test pose 覆盖和
候选数据 provenance；输入 HZB result 必须是当前正式 v2 schema。

## 输入契约

- base 必须是 `local-true-component-id-formal-render-manifest-v2` 的完整 test
  manifest，并通过 frozen-test 字段和 formal renderer validator；
- result 必须是 `geometry-shell-hzb-browser-result-v2`、`mode=Region66`，其
  `workload.schema` 为 `geometry-shell-hzb-browser-workload-v1`、`workload.split=test`，
  并声明 `formalReady=true`、`executionClass=formal-hardware-gpu` 及通过的硬件门；
- result 若提供 `shellDir`，其 `shell_meta.json` 必须是
  `geometry-shell-hzb-v2`；
- result 的 `workload.scene` 必须与 base 的场景和完整实例 runtime metadata 一致，
  Region66 查询 FOV 必须为 `66°`，其 provenance configuration 同时记录 `60°`
  渲染 FOV 和 `66°` 区域 FOV；
- `samples[].poseId` 必须唯一，且与 base 去重后的 `samples[].viewcellRow` 完全覆盖；
  workload 的 `poseSelection.selectedPoseIndices` 和逐 pose `aspect` 也必须覆盖同一集合，
  不能使用 limit 或 warmup 截断；
- base 每个 sample 必须显式给出 `aspect`，同一 view-cell 的 dense subpose aspect
  必须一致，并与 HZB workload 的对应 pose 一致；
- 每个 `visibleInstanceIds` 必须是显式 CSR candidate 行的子集，并且属于 base
  的完整 `componentToBinding` 实例范围。CSR 只读取
  `candidate_offsets.bin` 和 `candidate_ids.bin`；
- CLI 必须显式提供 `--candidate-dataset-dir` 和 `--asset-variant`。候选目录必须有
  `dataset_meta.json`、`poses.bin` 和 CSR 文件；其 runtime metadata、实例数量、
  model/frontend FOV、candidate semantics、test split 行和 base manifest 一致。候选
  目录与 result 的 poseId 使用同一 Pose CSR 行号，result 搬迁时仍以目录 basename
  和 metadata 做校验。

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
  "sourceResult": "/absolute/path/to/region66-test.json"
}
```

字段来源固定为当前 runner 的记录：`resolution` 读取
`result.workload.width/height`，`depthBiasM` 读取全部
`result.samples[].timings.depthBiasM` 并要求一致，`regionSampleCount` 读取
`result.regionSampling.requestedCount`。`assetVariant` 来自显式 CLI 参数；
`sourceResult` 为存在且可解析的 HZB result 绝对路径，并且必须与传入 result 内容
一致。`selectionSplit=calibration` 表示配置在 test 前冻结，不参与 test 配置选择。

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

本次追加强化了转换器和契约测试，覆盖 formal base、同场景/FOV/aspect、Region66/test、
pose 覆盖与重复、candidate dataset/实例范围、绝对 sourceResult、baselineSelection
和 schema-only renderer。验证命令为：

```bash
conda run --no-capture-output -n slm_pvs \
  python -m unittest discover -s neural_instance_culling/benchmark/tests -p 'test_*.py'
```

结果：`122 tests`、`2 skipped`、`OK`。未启动 GPU、Chrome 或正式图像评价，也没有
新的指标结论。该入口保留为 Geometry-shell HZB 正式 test 图像基线，不改变神经
PVS 主线。

## Point60 Color-ID GT 计划

日期：2026-09-09。目的：为 HZB 的 `Point60` 模式生成真实 60° 相机的 canonical
Color-ID GT 计划，避免把 Pose CSR 的 66° 后退相机或 view-cell 可见并集当作同点
真实相机。计划构建只读 CPU 数据；它不启动浏览器、不采样 GPU，也不产生指标。

`build_hzb_point60_gt_plan.py` 要求 representative plan 与 directional Pose CSR
逐行对齐，固定读取 Pose CSR 的 `test` split。每个输出行都使用 representative
的 `camera_pos`、`camera_forward`、`aspect`、`width` 和 `height`，写入
`fov_y=60`、`subpose_id=0`，并把原 Pose CSR 行号同时保留为 `pose_index` 和
`viewcell_id`。它不会使用 Pose CSR 的 `camera_world`，因为该字段是 66° 后退候选
相机。IFCBench 的每个 `(width,height,aspect)` 会写入独立 JSONL，不能用一个默认
viewport 覆盖全场景。

两场景计划命令：

```bash
conda run --no-capture-output -n slm_pvs \
  python neural_instance_culling/benchmark/build_hzb_point60_gt_plan.py \
  --dataset-dir neural_instance_culling/dataset/out/pose_csr_hkust_v3_main_stratified_calibration_fov66_v1 \
  --representative-plan neural_instance_culling/sampler/out/hkust_v3_viewcell_fov66/representatives.jsonl \
  --output-dir <point60-plan-root>/hkust_v3 --scene hkust_v3

conda run --no-capture-output -n slm_pvs \
  python neural_instance_culling/benchmark/build_hzb_point60_gt_plan.py \
  --dataset-dir neural_instance_culling/dataset/out/pose_csr_ifcbench_fantasy_metropolis_main_stratified_calibration_fov66_v1 \
  --representative-plan neural_instance_culling/sampler/out/ifcbench_fantasy_metropolis_instanced_v2/pose_plan_fov66.jsonl \
  --output-dir <point60-plan-root>/ifcbench_fantasy_metropolis --scene ifcbench_fantasy_metropolis
```

本次 CPU smoke 的计划统计为：HKUST `684` 个 test pose、`1` 个
`512×288, aspect=1.7777777777777777` 分组；IFCBench `2710` 个 test pose、`8`
个分组，分别保留 `aspect=0.5625, 0.75, 1, 1.3333333333, 1.6,
1.7777777778, 2.1666666667, 2.3333333333` 及其原始宽高。每个计划目录的
`point60_gt_plan_summary.json` 记录分组文件和原始 pose 行号。

计划文件本身不含可见 ID；先对每个分组运行 Point60 sampler，sampler 产生的
顶层 JSONL 才是 evaluator 的 `--point-gt-raw-dir` 输入。每个计划分组必须单独
运行，以保持 IFCBench 的真实 viewport：

```bash
mkdir -p <point60-raw-root>/ifcbench_fantasy_metropolis
for plan in <point60-plan-root>/ifcbench_fantasy_metropolis/*.jsonl; do
  name=$(basename "$plan")
  node neural_instance_culling/sampler/run_sampler.mjs \
    --assets-dir <ifcbench-scene-root>/assets \
    --pose-plan "$plan" \
    --output <point60-raw-root>/ifcbench_fantasy_metropolis/"$name" \
    --point60-gt --require-hardware-gpu \
    > <point60-raw-root>/ifcbench_fantasy_metropolis/"${name%.jsonl}"_stdout.log \
    2> <point60-raw-root>/ifcbench_fantasy_metropolis/"${name%.jsonl}"_stderr.log
done
```

`run_sampler.mjs` 默认仍只接受 66°；只有显式 `--point60-gt` 才接受 60°，并且
该模式拒绝软件 GPU、空 fallback plan、非 canonical subpose、混合 viewport 和
非原始 `viewcell_id`。正式硬件运行会在同一个执行窗口保存
`hostGpuBefore`、`hostGpuDuring`、`hostGpuAfter` 三段 `nvidia-smi`/`pmon` 证据；
任一段不可用时 `formalReady=false`。`hostGpuAfter` 在浏览器和 sampler HTTP 服务
关闭后采集。

生成 raw 目录后，Point60 evaluator 直接读取它，不读取 Region66 union：

```bash
conda run --no-capture-output -n slm_pvs \
  python neural_instance_culling/benchmark/evaluate_geometry_shell_hzb.py \
  --result <point60-test-result.json> \
  --dataset-dir <pose-csr-test-dataset> \
  --runtime-meta <scene-root>/assets/runtimeVisibilityMeta.json \
  --point-gt-raw-dir <point60-raw-root>/<scene> \
  --output <point60-metrics.json>
```

本次验证运行了 `test_build_hzb_point60_gt_plan.py`（4 tests）、现有
`test_evaluate_geometry_shell_hzb.py`（8 tests）和 `test_run_sampler_flags.mjs`
（6 tests），以及 Python/Node 语法检查；Python `12 tests`、Node `6 tests` 均
通过。未运行正式 GPU 采样。Point60 的 image PER、miss pixel、wrong-ID pixel、
集合准确率和 GLB 资源指标在 raw GT 与 HZB result 齐备后由 evaluator 计算，本次
未实现、未报告，也不改变神经 PVS 主线。

## 本次 provenance 修复

日期：2026-09-09。正式 HZB 图像 manifest 现在拒绝场景、60°/66° FOV、逐 pose
aspect、test pose 覆盖、candidate CSR 或 base runtime metadata 不一致的输入；只
接受 `formalReady=true` 且通过硬件门的 Region66 result。`baselineSelection.sourceResult`
写入可解析、与传入 result 内容一致的绝对路径。HKUST 既有 formal manifest 的
schema validator 未被改写，也没有重新生成或读取正式 test 结果。

本次相关测试：`test_run_test_image_evaluation.py` 与
`test_build_hzb_image_manifest.py` 共 `16 tests`、`OK`；fake renderer/schema-only
测试没有启动 Chrome 或正式 GPU 实验。
