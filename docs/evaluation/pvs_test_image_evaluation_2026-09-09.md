# Frozen Test 图像评价入口

日期：2026-09-09

状态：入口为单场景薄 runner。正式 test 图像渲染须在选择配置和候选语义冻结后执行一次；本 worktree 本次只做 schema/选择测试，不启动 GPU 或 Chrome。

## 输入契约

输入是现有 `local-true-component-id-formal-render-manifest-v2`。manifest 必须声明：

- `split=test`、`testRead=true`、`testEvaluationCount=1`；
- calibration-frozen `thresholdSelection` 或 `baselineSelection` 二选一，且
  `selectionSplit=calibration`、`testRead=false`；HZB 基线使用
  `method=geometry-shell-hzb` 以及 `assetVariant`、`resolution`、`depthBiasM`、
  `regionSampleCount`、`sourceResult`；
- `testCoverage.selection=all_unique_test_viewcells`、`maxViewcells=0`、`sampledWithReplacement=false`；
- `testCoverage.subposesPerViewcell=0`，`subposeSelection.mode=all` 且 `requestedPerViewcell=0`。

这些字段只说明 test 已冻结以及覆盖范围。HZB 的 Region66 转换、pose 覆盖、显式
Pose CSR 候选和基线配置见 [`pvs_hzb_test_image_manifest_2026-09-09.md`](pvs_hzb_test_image_manifest_2026-09-09.md)。
完整 formal-v2 schema、实例绑定、`predictionKey` 复用和 sample FOV 由现有 renderer
validator 负责。

## 单场景 Runner

入口：`neural_instance_culling/benchmark/run_test_image_evaluation.py`。

Runner 不加载模型、不选择阈值、不重算预测，也不复制 manifest validator。它完成最小字段门控后调用现有正式 renderer；实际渲染时由 runner 在同一个 Node renderer 执行窗口采集宿主机 `nvidia-smi`/`pmon` 的 before/during/after 证据，并写入 `<output-dir>/gpu_evidence/`、`gpu_evidence.json` 和 `render_summary.json`。选择门接受校准冻结的神经阈值或 HZB 基线二选一：

```bash
conda run --no-capture-output -n slm_pvs \
  python neural_instance_culling/benchmark/run_test_image_evaluation.py \
  --manifest <hkust-or-ifcbench-formal-v2-test-manifest.json> \
  --output-dir <scene-render-output> \
  --require-hardware-gpu
```

当前 renderer 的参数和浏览器证据门必须保持唯一来源：正式调用使用
`--require-hardware-gpu`，由 `render_local_glb_color_id_browser.mjs` 负责 Chrome
Vulkan/ANGLE 参数、页面 `gpuBackend`/`gpuGate` 和浏览器日志；runner 负责同一
Node 执行窗口的宿主机证据。正式运行只有在 WebGL 后端不是软件实现、三阶段
`nvidia-smi`/`pmon` 均可解析且 during 阶段确实处于 renderer 进程存活期间、并且
summary 报告 `formalImageEvaluationReady=true` 时才成功。HKUST 与 IFCBench 分别
执行一次，不能把两个场景的 component ID 或资产清单合并。

不启动浏览器的 schema 检查：

```bash
CUDA_VISIBLE_DEVICES= conda run --no-capture-output -n slm_pvs \
  python neural_instance_culling/benchmark/run_test_image_evaluation.py \
  --manifest <formal-v2-test-manifest.json> \
  --output-dir <schema-output> \
  --require-hardware-gpu \
  --render-schema-only
```

`--render-schema-only` 只调用现有 Node renderer 的 `--validate-only`，输出由 renderer
写入 `render_summary.json`；它不会启动 Chrome、不会生成 GPU evidence，也不是正式
图像结果。正式运行缺少任一阶段证据、后端被判定为软件或 summary 的
`formalImageEvaluationReady` 为 false 时都会失败。

## 定性样本清单

入口：`neural_instance_culling/benchmark/select_qualitative_image_samples.py`。

它读取 renderer 生成的 `sample_image_metrics.json`，按每个 dense subpose 的 PER 选择：

| 角色 | 规则 |
|---|---|
| `median` | 线性 percentile 中位数的最近 sample |
| `p95` | 线性 percentile p95 的最近 sample |
| `max_error` | PER 最大 sample，按 sample ID 稳定平局 |
| `fine_component` | test 前登记的细构件 pose |
| `occlusion_boundary` | test 前登记的遮挡边界 pose |

注册文件使用 `pvs-qualitative-pose-registration-v1`，声明 `registeredBeforeTest=true`、`selectionSplit=pre_test`、`testRead=false`、`testEvaluationCount=0`，并为两个角色各提供 `sampleId` 或 `poseIndex`。选择器不会因为 raw buffer 尚未保存而拒绝或伪造结果；清单直接给出：
`samples/<sampleId>_reference_u32.bin`、`samples/<sampleId>_test_u32.bin` 和 `samples/<sampleId>_diff_u8.bin`，分别对应 `Reference`、`Prediction`、`Difference`。

```bash
conda run --no-capture-output -n slm_pvs \
  python neural_instance_culling/benchmark/select_qualitative_image_samples.py \
  --render-summary <scene-render-output>/render_summary.json \
  --sample-metrics <scene-render-output>/sample_image_metrics.json \
  --manifest <formal-v2-test-manifest.json> \
  --registration <pre-test-registration.json> \
  --output <scene-render-output>/qualitative_image_selection.json
```

选择器只使用 test manifest 的 sample 集合和 renderer 的逐 sample PER；它不参与模型、checkpoint 或阈值选择。普通集合指标、weighted recall、useful/bad cull 和图像 PER 的解释仍遵循现有评价文档，图像指标不能替代剔除效率指标。

## 本次记录

本次追加提交收敛 HZB formal-v2 转换入口：

- `build_hzb_image_manifest.py`：将 Region66 实例预测转换为 formal-v2 keyed manifest；
- `run_test_image_evaluation.py`：正式浏览器执行的同窗口 host GPU evidence 和
  `formalImageEvaluationReady`/WebGL 硬件门；
- `test_build_hzb_image_manifest.py`：转换契约和 schema-only renderer；
- `test_run_test_image_evaluation.py`：host evidence 完整性、软件后端和 formal-ready
  失败路径；
- 本文档、HZB 转换契约文档和 `docs/README.md` 索引。

相关单测命令为：

```bash
python -m unittest \
  neural_instance_culling.benchmark.tests.test_run_test_image_evaluation \
  neural_instance_culling.benchmark.tests.test_build_hzb_image_manifest
```

结果：`16 tests`、`OK`。测试使用 schema-only Node renderer、fake renderer 和
fake `nvidia-smi`，不启动 GPU、Chrome 或正式 Color-ID 渲染。正式 test 图像数值
结果需在硬件空闲且所有输入冻结后，通过上述单场景入口执行。
