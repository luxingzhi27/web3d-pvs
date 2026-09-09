# Strict Cold-Cache GLB Streaming

日期：2026-09-09

## 目的

本轮实现论文计划中的 GLB 渐进下载评价。每个 pose 从空缓存开始，先把存储 CSR candidate instance 映射成一个固定 candidate GLB 集合；所有排名方法共享这个集合。只有完整 GLB HTTP 资源到达后，才累计下载字节和 GT coverage。

阈值过滤和连续排名是两个独立决策模式：

- `threshold_filtering` 读取 checkpoint calibration 冻结阈值，输出预测 GLB 子集及其覆盖上限。
- `threshold_free_ranking` 不应用可见性阈值，给完整 candidate GLB 集合排序并计算 `Bytes@95/99/99.9/100`。

本次 coverage source 是 `gt_glb_presence`，即 GT 中出现的 GLB 等权存在性，不是像素覆盖率。AABB 行使用现有的确定性 `aabb_ray` 几何基线；没有可用的独立 AABB MLP checkpoint，因此不能将该行解释为训练得到的 AABB MLP。

## 输入与运行

输入只来自主仓库，输出写入当前 worktree：

- HKUST dataset：`/mnt/sda/rhyang/slm/neural_instance_culling/dataset/out/pose_csr_hkust_v3_main_stratified_calibration_fov66_v1`
- IFCBench dataset：`/mnt/sda/rhyang/slm/neural_instance_culling/dataset/out/pose_csr_ifcbench_fantasy_metropolis_main_stratified_calibration_fov66_v1`
- scene runtime metadata、`glbIndex.json` 和 GLB root 使用各自场景的显式路径。
- score sidecar：`neural_instance_culling/benchmark/out/paper_results/streaming/{hkust_scores,ifcbench_scores}`

主要入口：

```bash
conda run -n slm_pvs python neural_instance_culling/benchmark/export_glb_streaming_scores.py \
  --dataset-dir <dataset> --runtime-meta <runtimeVisibilityMeta.json> \
  --model-spec '<name>|<kind>|<checkpoint>|<runtime_features>|<calibration_summary>' \
  --model-spec 'aabb|aabb_ray|-|-|-' --result-dir <score-result> --split test \
  --device cuda

conda run -n slm_pvs python neural_instance_culling/benchmark/simulate_glb_streaming.py \
  --dataset-dir <dataset> --runtime-meta <runtimeVisibilityMeta.json> \
  --glb-index <glbIndex.json> --glb-root <asset-root> \
  --result-dir <score-result> --output-dir <simulation-output> --split test \
  --utility-source binary_gt

conda run -n slm_pvs python neural_instance_culling/benchmark/generate_streaming_paper_outputs.py \
  --summary <hkust>/ranking_summary.json --summary <ifcbench>/ranking_summary.json \
  --filter-summary <hkust>/filtering_summary.json \
  --filter-summary <ifcbench>/filtering_summary.json \
  --output-dir neural_instance_culling/benchmark/out/paper_results/figures
```

实际 test 读取规模为 HKUST `684` pose、IFCBench `2710` pose；score sidecar 分别包含 `3,174,148` 和 `27,061,482` candidate instance rows。输出目录还保存 score alignment manifest、per-pose JSONL、ranking/filtering summary、CSV、Markdown 以及 PNG/PDF/SVG 曲线。

## 已有结果

下表为 threshold-free ranking 的每 pose 平均值。字节单位为 MiB；coverage 是 GT GLB presence coverage。

| 场景 | 方法 | Bytes@95 | Bytes@99 | Bytes@99.9 | Bytes@100 | waste-before-99 | required rank |
|---|---|---:|---:|---:|---:|---:|---:|
| HKUST | Full | 32.048 | 39.129 | 40.075 | 40.076 | 23.357 | 179.6 |
| HKUST | AABB + ray | 117.762 | 121.284 | 121.725 | 121.726 | 105.484 | 477.6 |
| HKUST | Projected area / byte | 49.198 | 60.440 | 64.334 | 64.395 | 46.191 | 608.3 |
| HKUST | GT utility / byte oracle | 6.344 | 11.828 | 15.723 | 15.806 | 0 | 50.7 |
| IFCBench | Full | 9.705 | 12.150 | 15.561 | 15.561 | 8.131 | 671.2 |
| IFCBench | AABB + ray | 40.945 | 47.088 | 48.653 | 48.653 | 43.005 | 1288.8 |
| IFCBench | Projected area / byte | 29.436 | 39.055 | 43.762 | 43.762 | 35.029 | 1299.4 |
| IFCBench | GT utility / byte oracle | 2.536 | 3.710 | 4.101 | 4.101 | 0 | 240.9 |

完整方法行（Full、AABB、original、distance、projected area、projected area/byte、20 个 fixed random、HZB visible-first、GT utility/byte oracle）以及 10/25/50/100 Mbps 时间、coverage ceiling 和每个 target 的 unreachable ratio 在 `table5_streaming_ranking.csv` 中。两场景所有 ranking 方法的 coverage ceiling 均为 100%，因为正式 candidate 集合包含对应 GT GLB；每个 pose 的完整缺失资源诊断仍保存在 per-pose 输出中。

独立 threshold filtering 的 Full 结果为：HKUST 平均预测 `161.8` 个 GLB、`36.843` MiB、coverage ceiling `97.651%`，`99%` unreachable ratio `12.57%`；IFCBench 平均预测 `679.3` 个 GLB、`15.486` MiB、coverage ceiling `99.744%`，`99%` unreachable ratio `5.02%`。这些 predicted bytes 没有写入 ranking 的 `Bytes@...` 字段。

## 指标口径

- `Bytes@x`：按方法顺序下载，累计到完整 GLB 到达后首次达到 x coverage 的字节数；传输中的部分 GLB不贡献 coverage。
- `waste-before-99`：达到 99% coverage 的完整 GLB 前缀中，不属于 GT GLB 的字节。
- `required rank`：最后一个 GT GLB 在该完整 candidate 排名中的 1-based rank；它和 utility coverage target 分开报告。
- `coverage ceiling`：完整 candidate 集合可达到的 GT utility coverage；`unreachable` 是 target 高于该上限的 pose 比例。
- 时间：按 `bytes * 8 / (Mbps * 1,000,000)` 换算，四个带宽均输出，正文表可展示 25/50 Mbps。
- `GT utility / byte oracle` 是按 GT utility/byte 的贪心顺序，不称为所有目标的全局最优；summary 另有 fractional utility/byte byte lower bound。

## Reference-frontmost 语义

`build_reference_frontmost_histogram.py` 输出的 schema 名称是 `reference-frontmost-pixel-histogram-v1`。它把 Color-ID buffer 中的 `componentGlobalId + 1` 映射到 GLB，并统计每个 pose 的最前表面像素。

该直方图只表示 front-most surface proxy：看不到隐藏面，不等于完整可见性计数，也没有表示解码成本、交互重要性或下载后新暴露表面。分多个 subpose 求和时，它是 view-cell utility proxy，不是一个物理图像的像素总数。本轮主 test streaming 没有完整对应的 test reference buffer，因此没有把 `visible_weights` 或 GT 结果改名成像素 coverage；图轴明确使用 `GT GLB presence coverage (%)`。

## 真实调度状态

`slm2viewer/scripts/run_real_scheduler_streaming.mjs` 和 `build_real_scheduler_plan.py` 已实现并通过 synthetic fixture。计划固定每场景 12 个 test pose，支持 25/50 Mbps、每项 3 次，调用现有 `classifyGlbSchedule` 和 `GlbResourceScheduler` 的 `urgent/warm/speculative` 状态机，且计划和驱动都显式记录 `startup100Enabled=false`、不使用 startup tier。

正式真实调度 replay 尚未运行，等待主线程安排以避免和 IFCBench 训练/扫描争用；本轮不把 synthetic/local-file smoke 当作桌面或移动端性能结论。驱动使用真实 GLB 响应和空应用缓存，但 Node 阶段只做 GLB container parse 与 mount phase 计时，不构造 Three.js 场景。

## 文件与测试

实现文件包括 `glb_streaming.py`、`glb_streaming_io.py`、`export_glb_streaming_scores.py`、`simulate_glb_streaming.py`、`build_reference_frontmost_histogram.py`、`build_real_scheduler_plan.py`、`generate_streaming_paper_outputs.py`，以及对应 Python/Node fixture tests。

已通过：

- `conda run -n slm_pvs python -m unittest neural_instance_culling/benchmark/tests/test_glb_streaming.py -v`：8 个 fixture/契约测试。
- `conda run -n slm_pvs python -m py_compile`：全部新增 Python 入口。
- `node scripts/test_glb_resource_scheduler.mjs`：现有 scheduler 契约。
- `node scripts/test_real_scheduler_streaming.mjs`：12 pose synthetic scheduler replay。

本实现不修改训练入口、`evaluate_pvs.py` 或 HZB 核心。
