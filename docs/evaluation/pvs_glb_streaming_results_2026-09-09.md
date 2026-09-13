# Strict Cold-Cache GLB Streaming

日期：2026-09-09；2026-09-13 更新

## 目的

本轮实现论文计划中的 GLB 渐进下载评价。每个 pose 从空缓存开始，先把存储 CSR candidate instance 映射成一个固定 candidate GLB 集合；所有排名方法共享这个集合。只有完整 GLB HTTP 资源到达后，才累计下载字节和 GT coverage。

阈值过滤和连续排名是两个独立决策模式：

- `threshold_filtering` 读取 checkpoint calibration 冻结阈值，输出预测 GLB 子集及其覆盖上限。
- `threshold_free_ranking` 不应用可见性阈值，给完整 candidate GLB 集合排序并计算 `Bytes@95/99/99.9/100`。

正式 coverage source 是 Pose CSR 的 `visible_weights`。Color-ID 场景中它表示每个实例在 view-cell subpose 上最大池化的前景屏幕覆盖权重，不是隐藏表面覆盖或单张图像像素并集。AABB 行读取正式 AABB MLP frozen-test sidecar。

## 输入与运行

输入只来自主仓库，输出写入当前 worktree：

- HKUST dataset：`/mnt/sda/rhyang/slm/neural_instance_culling/dataset/out/pose_csr_hkust_v3_main_stratified_calibration_fov66_v1`
- IFCBench dataset：`/mnt/sda/rhyang/slm/neural_instance_culling/dataset/out/pose_csr_ifcbench_fantasy_metropolis_main_stratified_calibration_fov66_v1`
- scene runtime metadata、`glbIndex.json` 和 GLB root 使用各自场景的显式路径。
- score sidecar：`neural_instance_culling/benchmark/out/paper_results/streaming_formal/{hkust_scores,ifcbench_scores}`

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
  --utility-source visible_weights

conda run -n slm_pvs python neural_instance_culling/benchmark/generate_streaming_paper_outputs.py \
  --summary <hkust>/ranking_summary.json --summary <ifcbench>/ranking_summary.json \
  --filter-summary <hkust>/filtering_summary.json \
  --filter-summary <ifcbench>/filtering_summary.json \
  --output-dir neural_instance_culling/benchmark/out/paper_results/streaming_formal/figures
```

实际 test 读取规模为 HKUST `684` pose、IFCBench `2710` pose；score sidecar 分别包含 `3,174,148` 和 `27,061,482` 个 candidate instance rows。IFCBench 指标对应其冻结的四点 `camera_aligned_box` view-cell 协议。输出目录保存 score alignment manifest、per-pose JSONL、ranking/filtering summary、CSV、Markdown 以及 PNG/PDF/SVG 曲线。

## 正式 Test 结果

下表为 threshold-free ranking 的每 pose 平均值，单位 MiB。神经成本指数只在 validation 从 `{0,0.5,1}` 选择，HKUST 冻结为 `1.0`，IFCBench 冻结为 `0.5`。距离和面积先逐 candidate instance AABB 计算，再按 GLB 聚合；没有使用合并 GLB AABB。

| 场景 | 方法 | Bytes@95 | Bytes@99 | Bytes@99.9 | Bytes@100 | Waste@99 |
|---|---|---:|---:|---:|---:|---:|
| HKUST | Neural `p_g` | 8.842 | 18.031 | 26.187 | 40.076 | 6.240 |
| HKUST | Neural `p_g/B^alpha` | 6.108 | 10.819 | 18.052 | 35.240 | 4.907 |
| HKUST | AABB MLP | 12.292 | 23.727 | 38.174 | 89.664 | 11.708 |
| HKUST | Projected area/byte | 8.161 | 12.881 | 25.186 | 65.033 | 6.271 |
| HKUST | HZB visible-first | 10.976 | 17.498 | 40.952 | 82.051 | 8.779 |
| HKUST | GT utility/byte oracle | 1.191 | 3.543 | 8.337 | 15.806 | 0 |
| IFCBench | Neural `p_g` | 4.948 | 8.078 | 11.678 | 15.980 | 4.620 |
| IFCBench | Neural `p_g/B^alpha` | 3.616 | 7.119 | 11.370 | 13.764 | 4.638 |
| IFCBench | AABB MLP | 13.097 | 24.639 | 36.623 | 44.304 | 21.405 |
| IFCBench | Projected area/byte | 5.875 | 13.426 | 25.326 | 37.264 | 10.580 |
| IFCBench | GT utility/byte oracle | 0.689 | 1.616 | 3.087 | 4.101 | 0 |

HKUST 和 IFCBench 各自的 `ranking_summary.json` 保存完整方法行、20 个 fixed-random
seeds、10/25/50/100 Mbps 换算、coverage ceiling 和不可达率。当前论文 figures 目录只
汇总了 HKUST，须在 IFCBench 正式 HZB 完成后重新生成两场景统一表图。所有可用的
threshold-free 方法共享完整 candidate GLB 集合，因此 coverage ceiling 为 1；
IFCBench 的 `hzb_visible_first` 当前明确为 unavailable，不能进入表格。

冻结阈值过滤单独报告：HKUST Full 平均保留 `161.76` GLB、`36.84 MiB`，平均 coverage ceiling `0.996914`；IFCBench Full 平均保留 `561.25` GLB、`11.66 MiB`，coverage ceiling `0.997467`。过滤集合不参与上表的 threshold-free Bytes@x 排名。

## 指标口径

- `Bytes@x`：按方法顺序下载，累计到完整 GLB 到达后首次达到 x coverage 的字节数；传输中的部分 GLB不贡献 coverage。
- `waste-before-99`：达到 99% coverage 的完整 GLB 前缀中，不属于 GT GLB 的字节。
- `required rank`：最后一个 GT GLB 在该完整 candidate 排名中的 1-based rank；它和 utility coverage target 分开报告。
- `coverage ceiling`：完整 candidate 集合可达到的 GT utility coverage；`unreachable` 是 target 高于该上限的 pose 比例。
- 时间：按 `bytes * 8 / (Mbps * 1,000,000)` 换算，四个带宽均输出，正文表可展示 25/50 Mbps。
- `GT utility / byte oracle` 是按 GT utility/byte 的贪心顺序，不称为所有目标的全局最优；summary 另有 fractional utility/byte byte lower bound。

## Reference-frontmost 语义

`build_reference_frontmost_histogram.py` 输出的 schema 名称是 `reference-frontmost-pixel-histogram-v1`。它把 Color-ID buffer 中的 `componentGlobalId + 1` 映射到 GLB，并统计每个 pose 的最前表面像素。

该直方图只表示 front-most surface proxy：看不到隐藏面，不等于完整可见性计数，也没有表示解码成本、交互重要性或下载后新暴露表面。正式主图使用 `Visible-weight coverage (%)`；reference-frontmost 只作为补充核验，不能替代它。

## 真实调度结果

`slm2viewer/scripts/run_real_scheduler_streaming.mjs` 和 `build_real_scheduler_plan.py` 固定
12 个 HKUST test pose，运行 25/50 Mbps、每项 3 次，调用现有 `classifyGlbSchedule`
和 `GlbResourceScheduler` 的 `urgent/warm/speculative` 状态机，且显式记录
`startup100Enabled=false`、不使用 startup tier。HKUST 共 `216` 次 replay，均完成且
无下载失败。IFCBench scheduler plan/replay 尚未执行。

25 Mbps 结果如下；GLB 与 waste 是达到完整 GT GLB 集合时的逐 run 平均值：

| 场景 | 方法 | 调度阶段 median | p95 | GLB MiB | Waste MiB | 启动资产 MiB | 含启动传输 median 下界 | p95 下界 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| HKUST | Full | 0.126 s | 116.208 s | 49.703 | 28.062 | 5.225 | 1.879 s | 117.961 s |
| HKUST | AABB MLP | 0.800 s | 136.141 s | 99.127 | 77.486 | unavailable | unavailable | unavailable |
| HKUST | HZB visible-first | 27.317 s | 153.304 s | 151.488 | 129.847 | 380.814 | 155.097 s | 281.084 s |

调度阶段使用真实 GLB 响应和空应用缓存，但 Node 阶段只校验 GLB container 并执行生产调度状态机，不构造 Three.js 场景，因此不冒充完整浏览器首帧渲染时间。`含启动传输下界` 在调度时间上加对应可见性资产按相同聚合带宽的传输时间，不包含资产解码、模型初始化、HZB 构建或最终绘制。AABB MLP 尚无正式部署 bundle，不能把 replay 中的空 asset list 解释成零字节，故该列保持 unavailable。完整 25/50 Mbps 数据在 `table5_scheduler_replay.csv`。

IFCBench replay 必须使用同一 frozen test split 和现有四点 box GT，并在正式 Region66
HZB 结果可用后一次性生成 Full、AABB MLP 和 HZB visible-first 三种方法的计划与 replay。

## 文件与测试

实现文件包括 `glb_streaming.py`、`glb_streaming_io.py`、`export_glb_streaming_scores.py`、`simulate_glb_streaming.py`、`build_reference_frontmost_histogram.py`、`build_real_scheduler_plan.py`、`generate_streaming_paper_outputs.py`，以及对应 Python/Node fixture tests。

已通过：

- `conda run -n slm_pvs python -m unittest neural_instance_culling/benchmark/tests/test_glb_streaming.py -v`：27 个 fixture/契约测试。
- `conda run -n slm_pvs python -m py_compile`：全部新增 Python 入口。
- `node scripts/test_glb_resource_scheduler.mjs`：现有 scheduler 契约。
- `node scripts/test_real_scheduler_streaming.mjs`：12 pose synthetic scheduler replay。

本实现不修改训练入口、`evaluate_pvs.py` 或 HZB 核心。
