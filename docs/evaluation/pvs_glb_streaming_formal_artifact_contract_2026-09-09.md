# GLB Streaming Formal Artifact Contract

日期：2026-09-09；2026-09-10 补充成本感知神经排序规则

## 目的

论文 streaming 模拟器现在只把正式 test 产物接入排序链路。AABB MLP 使用冻结 test replay 写出的连续候选分数；HZB visible-first 使用正式 Region66 test 结果中的实例集合。两者都在同一份 CSR candidate 映射出的 GLB 集合上运行，阈值过滤仍然单独输出，不参与排序曲线。

本次只修改 streaming benchmark 的输入、排序和输出生成入口，没有运行正式 test、浏览器或 GPU，也没有改变模型训练、HZB runner 或生产调度器。

## 变更文件

- `neural_instance_culling/benchmark/export_glb_streaming_scores.py`：读取并校验 Full 与 AABB formal test sidecar，复用同一次 frozen-test 连续分数；
- `neural_instance_culling/benchmark/simulate_glb_streaming.py`：接入 Region66 实例集合、稳定 visible-first 排列和 unavailable 状态；
- `neural_instance_culling/benchmark/build_real_scheduler_plan.py`：让 12-pose 计划复用正式输入；
- `neural_instance_culling/benchmark/generate_streaming_paper_outputs.py`：阻止非正式 AABB/HZB 行进入表格和曲线；
- `neural_instance_culling/benchmark/tests/test_glb_streaming.py`：fixture 契约覆盖。

依赖资源是同一场景的 Pose CSR test 数据集、`runtimeVisibilityMeta.json`、`glbIndex.json`、GLB 文件、AABB formal test sidecar 和 Region66 formal test JSON。模型权重和 HZB runner 只负责在上游生成这些输入，不由 streaming 模拟器重新推理。

## 输入契约

### AABB MLP

`--aabb-test-sidecar` 接受 AABB formal test replay 的 sidecar 目录、`manifest.json`，或带有 `scoreSidecar` 的 formal test evaluation JSON。sidecar 必须满足：

- schema 为 `pvs-typed-score-sidecar-v1`；
- `split=test` 且 `testRead=true`；
- 覆盖数据集完整 test split；
- `candidateIds` 按 pose 与当前 CSR candidate 行逐项相等；
- `scores` 是有限的 `[0, 1]` 连续分数；
- `predictedIds`、阈值和 GT 标签不参与 threshold-free 排序。

exporter 将该 `scores` 数组写入 streaming sidecar 的 `aabb` 字段，并在 `scoreSources.aabb` 登记 `kind=formal_aabb_test_sidecar`。没有该输入时，`aabb` 不会调用 runner fallback，后续 summary 和 paper table 将明确标为 unavailable。

### 成本感知神经排序与启发式

2026-09-10 冻结以下论文计算规则，并已接入正式模拟器；现有旧 test 探索值仍不能替代按新协议重跑的正式产物。

- 先对同一 GLB 的候选实例可见性取 `p_g=max_i(p_i)`。
- 纯神经排序为 `p_g`；成本感知排序为 `p_g / Bytes_g^alpha`。
- `alpha` 只允许从预先登记的候选集合在 calibration/validation 上选择，选择完成后写入方法元数据并冻结到 test。test 不得重新选择指数或公式。
- AABB 面积不是成本感知神经排序的必需输入。带面积的神经变体必须单列为 `max_i(p_i * projectedArea_i) / Bytes_g^alpha`。
- `Distance`、`Projected AABB area` 和 `Projected AABB area / byte` 必须从 candidate instance AABB 分别计算，再以最小距离或最大面积聚合到 GLB；不再接受空间合并 GLB AABB 的投影作为正式启发式结果。
- Region66 streaming 使用 view-cell 中心的 66 度投影并评价 region-union `visible_weights`；真实 60 度当前画面属于独立 Point60/像素评价。
- threshold-free ranking 仍不读取安全阈值。冻结阈值分层加组内成本排序只进入单独的真实 scheduler replay，不能混入连续分数排序曲线。

正式 `ranking_summary.json` 的方法元数据必须记录实例到 GLB 的聚合方式、成本指数、是否使用 AABB 面积、字节来源和参数选择 split。论文正文至少并列纯神经分数与成本感知神经分数，证明收益来自遮挡可见性还是单纯资源大小修正。

### HZB visible-first

`--hzb-region66-result` 只接受正式 `geometry-shell-hzb-browser-result-v2` JSON。它必须声明 `mode=Region66`、test workload、完整 representative pose selection、`formalReady=true`、`executionClass=formal-hardware-gpu`，并通过 `gpuGate`。结果中的 pose 集合必须与冻结 CSR test split 完全相等；`candidateFile` 中每个 pose 的 uint32 实例编号和 `candidateCount` 都必须逐项等于同 pose 的 CSR candidate 行。`visibleInstanceIds` 再映射到该候选行和 GLB。任何 pose、数量、实例或候选不一致都会使该方法 unavailable。结果若提供 `shellDir`，其 `shell_meta.json` 必须声明 `geometry-shell-hzb-v2`。

HZB 不生成连续分数。排序规则是：

1. Region66 正式 visible 实例集合映射出的 GLB 排在前面；
2. visible 组内按已登记的 `projected_area_per_byte` 降序排序，分数相同时按 `globalGlbId` 升序；没有该字段时使用已登记的 `projected_area` 规则；
3. 非 visible 组按 GLB index 中的 `originalRank` 升序，再按 `globalGlbId` 升序。

产物只登记 visible instance set 和上述排序规则，`continuousScore=false`。缺少 Region66 结果、结果为 smoke/非硬件/非完整 test，都会产生 `hzb_visible_first` 的 unavailable 原因。

## 产物与过滤边界

- `export_glb_streaming_scores.py` 生成 `pvs-glb-streaming-score-sidecar-v1`，在 `score_manifest.json` 中保存 `scoreSources` 和 `unavailableSources`。
- `simulate_glb_streaming.py` 生成独立的 `ranking_summary.json`、`filtering_summary.json`、逐 pose JSONL 和 `streaming_manifest.json`。ranking 不读取阈值；filtering 只使用 checkpoint calibration 阈值，并且不把预测子集写入 ranking 曲线。
- `generate_streaming_paper_outputs.py` 只接受 `split=test`、`testRead=true`、完整 pose 数且 decision mode 正确的正式 ranking/filtering summary。缺少 formal source 的历史 AABB/HZB 行会在 CSV/Markdown 中保留为 `status=unavailable`，不会被绘图；validation、部分 pose 和旧占位 summary 会直接拒绝。
- `build_real_scheduler_plan.py` 固定从 test split 选择 12 个 pose；它复用同一个 score sidecar、CSR candidate GLB 集合和 HZB 排列。计划仍显式关闭 startup-100，并只写 `urgent/warm/speculative` 三层。

本次 fixture 验证已覆盖 AABB 连续分数对齐、Region66 pose/candidate 对齐、HZB visible-first 排序以及旧占位拒绝。正式两场景的 AABB 数值、HZB 数值、网络时间和 GPU 性能尚未生成，不能从 fixture 或历史输出推断。

当前指标状态：

| 指标 | 状态 | 中文含义 |
|---|---|---|
| `Bytes@95/99/99.9/100` | 代码完成，正式值待重跑 | 排序前缀达到对应 visible-weight coverage 所需的完整下载字节 |
| `coverage ceiling` / `unreachable` | 代码完成，正式值待重跑 | 固定 candidate GLB 集合可达到的 visible-weight coverage 上限及目标不可达比例 |
| `first-frame`、带宽时间、解析/挂载延迟 | 未实现正式值 | 真实调度计划中的首帧加载和运行成本 |

这套输入契约保留为当前 streaming 评价主线；正式 artifact 到位后仍需重新生成两场景全 test 表格，并单独完成 12-pose 真实调度 replay。主要风险是上游 sidecar、Region66 workload 和 Pose CSR 的 pose/candidate 语义不一致，入口会将其拒绝或标记 unavailable，不应人工修补后继续汇总。

## 正式产物就绪后的命令

以下命令使用 Linux shell。将尖括号替换为对应 formal AABB checkpoint/evaluation sidecar、Region66 result 和场景资产路径；不加 `--pose-limit` 才是完整 test。

### HKUST 与 IFCBench 分数 sidecar

```bash
mkdir -p neural_instance_culling/benchmark/out/paper_results/streaming_formal

conda run -n slm_pvs python neural_instance_culling/benchmark/export_glb_streaming_scores.py \
  --dataset-dir <hkust-test-dataset> \
  --runtime-meta <hkust-runtimeVisibilityMeta.json> \
  --full-test-sidecar <hkust-full-test-sidecar-or-test-json> \
  --aabb-test-sidecar <hkust-aabb-ray-seed-sidecar-or-test-json> \
  --result-dir neural_instance_culling/benchmark/out/paper_results/streaming_formal/hkust_scores \
  --split test --device cuda --poses-per-batch 4 \
  > neural_instance_culling/benchmark/out/paper_results/streaming_formal/hkust_scores_stdout.log \
  2> neural_instance_culling/benchmark/out/paper_results/streaming_formal/hkust_scores_stderr.log

conda run -n slm_pvs python neural_instance_culling/benchmark/export_glb_streaming_scores.py \
  --dataset-dir <ifcbench-test-dataset> \
  --runtime-meta <ifcbench-runtimeVisibilityMeta.json> \
  --full-test-sidecar <ifcbench-full-test-sidecar-or-test-json> \
  --aabb-test-sidecar <ifcbench-aabb-ray-seed-sidecar-or-test-json> \
  --result-dir neural_instance_culling/benchmark/out/paper_results/streaming_formal/ifcbench_scores \
  --split test --device cuda --poses-per-batch 4 \
  > neural_instance_culling/benchmark/out/paper_results/streaming_formal/ifcbench_scores_stdout.log \
  2> neural_instance_culling/benchmark/out/paper_results/streaming_formal/ifcbench_scores_stderr.log
```

### 先在 validation 冻结成本指数

每个场景先导出最终 Full 的 validation 连续分数，再运行一次 validation 模拟。该阶段只比较 `alpha=0/0.5/1`，将选择结果写入 `neural_cost_ranking_manifest.json`，并保持 `testRead=false`。

```bash
conda run -n slm_pvs python neural_instance_culling/benchmark/simulate_glb_streaming.py \
  --dataset-dir <scene-dataset> \
  --runtime-meta <runtimeVisibilityMeta.json> \
  --glb-index <glbIndex.json> --glb-root <asset-root> \
  --result-dir <scene-validation-score-sidecar> \
  --output-dir <scene-validation-simulation> \
  --split validation --utility-source visible_weights \
  --methods full,neural_cost
```

### 两场景完整 test 模拟

```bash
conda run -n slm_pvs python neural_instance_culling/benchmark/simulate_glb_streaming.py \
  --dataset-dir <hkust-test-dataset> \
  --runtime-meta <hkust-runtimeVisibilityMeta.json> \
  --glb-index <hkust-glbIndex.json> --glb-root <hkust-asset-root> \
  --result-dir neural_instance_culling/benchmark/out/paper_results/streaming_formal/hkust_scores \
  --hzb-region66-result <hkust-region66-formal-test.json> \
  --neural-cost-manifest <hkust-validation-simulation/neural_cost_ranking_manifest.json> \
  --output-dir neural_instance_culling/benchmark/out/paper_results/streaming_formal/hkust_sim \
  --split test --utility-source visible_weights --log-every 100 \
  > neural_instance_culling/benchmark/out/paper_results/streaming_formal/hkust_sim_stdout.log \
  2> neural_instance_culling/benchmark/out/paper_results/streaming_formal/hkust_sim_stderr.log

conda run -n slm_pvs python neural_instance_culling/benchmark/simulate_glb_streaming.py \
  --dataset-dir <ifcbench-test-dataset> \
  --runtime-meta <ifcbench-runtimeVisibilityMeta.json> \
  --glb-index <ifcbench-glbIndex.json> --glb-root <ifcbench-asset-root> \
  --result-dir neural_instance_culling/benchmark/out/paper_results/streaming_formal/ifcbench_scores \
  --hzb-region66-result <ifcbench-region66-formal-test.json> \
  --neural-cost-manifest <ifcbench-validation-simulation/neural_cost_ranking_manifest.json> \
  --output-dir neural_instance_culling/benchmark/out/paper_results/streaming_formal/ifcbench_sim \
  --split test --utility-source visible_weights --log-every 100 \
  > neural_instance_culling/benchmark/out/paper_results/streaming_formal/ifcbench_sim_stdout.log \
  2> neural_instance_culling/benchmark/out/paper_results/streaming_formal/ifcbench_sim_stderr.log

conda run -n slm_pvs python neural_instance_culling/benchmark/generate_streaming_paper_outputs.py \
  --summary neural_instance_culling/benchmark/out/paper_results/streaming_formal/hkust_sim/ranking_summary.json \
  --summary neural_instance_culling/benchmark/out/paper_results/streaming_formal/ifcbench_sim/ranking_summary.json \
  --filter-summary neural_instance_culling/benchmark/out/paper_results/streaming_formal/hkust_sim/filtering_summary.json \
  --filter-summary neural_instance_culling/benchmark/out/paper_results/streaming_formal/ifcbench_sim/filtering_summary.json \
  --output-dir neural_instance_culling/benchmark/out/paper_results/streaming_formal/figures
```

### 两场景 12-pose 真实调度计划

```bash
conda run -n slm_pvs python neural_instance_culling/benchmark/build_real_scheduler_plan.py \
  --dataset-dir <hkust-test-dataset> \
  --runtime-meta <hkust-runtimeVisibilityMeta.json> \
  --glb-index <hkust-glbIndex.json> --glb-root <hkust-asset-root> \
  --result-dir neural_instance_culling/benchmark/out/paper_results/streaming_formal/hkust_scores \
  --hzb-region66-result <hkust-region66-formal-test.json> \
  --output-dir neural_instance_culling/benchmark/out/paper_results/streaming_formal/hkust_scheduler_plan \
  --split test --pose-count 12 --methods full,aabb,hzb_visible_first

conda run -n slm_pvs python neural_instance_culling/benchmark/build_real_scheduler_plan.py \
  --dataset-dir <ifcbench-test-dataset> \
  --runtime-meta <ifcbench-runtimeVisibilityMeta.json> \
  --glb-index <ifcbench-glbIndex.json> --glb-root <ifcbench-asset-root> \
  --result-dir neural_instance_culling/benchmark/out/paper_results/streaming_formal/ifcbench_scores \
  --hzb-region66-result <ifcbench-region66-formal-test.json> \
  --output-dir neural_instance_culling/benchmark/out/paper_results/streaming_formal/ifcbench_scheduler_plan \
  --split test --pose-count 12 --methods full,aabb,hzb_visible_first
```

计划生成后，现有 Node replay 可显式读取每个计划的 `full,aabb,hzb_visible_first` 方法；该 replay 的网络/解析计时仍须单独记录日志，不能替代 HZB 浏览器硬件证据。
