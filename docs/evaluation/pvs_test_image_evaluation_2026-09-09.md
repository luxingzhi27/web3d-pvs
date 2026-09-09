# Frozen Test 图像评价入口

日期：2026-09-09

状态：入口已精简为单场景薄 runner。正式 test 图像渲染仍须在模型、checkpoint、calibration 阈值和候选语义冻结后执行一次；本 worktree 本次只做 schema/选择测试，不启动 GPU 或 Chrome。

## 输入契约

输入是现有 `local-true-component-id-formal-render-manifest-v2`。manifest 必须声明：

- `split=test`、`testRead=true`、`testEvaluationCount=1`；
- `thresholdSelection.selectionSplit=calibration`、`thresholdSelection.testRead=false`；
- `thresholdProvenance` 同样指向 test 之前的 calibration；
- `testCoverage.selection=all_unique_test_viewcells`、`maxViewcells=0`、`sampledWithReplacement=false`；
- `testCoverage.subposesPerViewcell=0`，`subposeSelection.mode=all` 且 `requestedPerViewcell=0`。

这些字段只说明 test 已冻结以及覆盖范围。完整 formal-v2 schema、实例绑定、`predictionKey` 复用和 sample FOV 由现有 renderer validator 负责。

## 单场景 Runner

入口：`neural_instance_culling/benchmark/run_test_image_evaluation.py`。

Runner 不加载模型、不选择阈值、不重算预测，不复制 manifest validator，不抓取 `nvidia-smi`，也不写入 renderer 的 GPU evidence。它完成最小字段门控后直接调用现有正式 renderer：

```bash
conda run --no-capture-output -n slm_pvs \
  python neural_instance_culling/benchmark/run_test_image_evaluation.py \
  --manifest <hkust-or-ifcbench-formal-v2-test-manifest.json> \
  --output-dir <scene-render-output> \
  --require-hardware-gpu
```

当前 renderer 的参数和证据门必须保持唯一来源：正式调用使用 `--require-hardware-gpu`，由 `render_local_glb_color_id_browser.mjs` 负责 Chrome Vulkan/ANGLE 参数、页面 `gpuBackend`/`gpuGate`、浏览器日志和同一执行窗口的主机证据。HKUST 与 IFCBench 分别执行一次，不能把两个场景的 component ID 或资产清单合并。

不启动浏览器的 schema 检查：

```bash
CUDA_VISIBLE_DEVICES= conda run --no-capture-output -n slm_pvs \
  python neural_instance_culling/benchmark/run_test_image_evaluation.py \
  --manifest <formal-v2-test-manifest.json> \
  --output-dir <schema-output> \
  --require-hardware-gpu \
  --render-schema-only
```

`--render-schema-only` 只调用现有 Node renderer 的 `--validate-only`，输出由 renderer 写入 `render_summary.json`；它不是正式图像结果。

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

## 修改与验证

本次追加提交精简了：

- `run_test_image_evaluation.py`：单场景最小字段门控和现有 renderer 调用；
- `select_qualitative_image_samples.py`：PER 分位数、最大误差和两个预登记角色映射；
- 两个 benchmark 测试：test 不参与选择、`predictionKey` 复用和预登记时序；
- `evaluate_viewcell_image_per.py` 与 `render_local_glb_color_id_browser.mjs`：保留现有 formal manifest 及 renderer 入口契约；
- 本文档和 `docs/README.md` 索引。

验证使用 `CUDA_VISIBLE_DEVICES=` 的 `slm_pvs` conda 环境，未启动 GPU、Chrome 或正式 Color-ID 渲染。正式 test 图像数值结果需在硬件空闲且所有输入冻结后，通过上述单场景入口执行。
