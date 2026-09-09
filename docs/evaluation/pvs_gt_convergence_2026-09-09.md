# View-cell GT 128 点收敛评估

日期：2026-09-09

## 变更目的与正式口径

本次修复针对新的 128 点嵌套计划。正式 evaluator 只读取一份计划 JSONL 和对应的 Color-ID raw JSONL（或其 shard 目录），不从数据集二进制重新选择 view-cell，也不重新排列 subpose。计划必须严格包含 `viewcell_id=0..99`、每个 cell 的 `subpose_id=0..127` 和每行的 `source_pose_index`，共 `100 x 128 = 12,800` 行。

raw 行通过 `(viewcell_id, subpose_id, source_pose_index)` 三元组与计划逐行对齐；raw 行缺失、重复、额外或三元组错配都会失败。`pose_index` 如果同时存在，只用于检查计划与 raw 是否一致，不能替代三元组连接。前缀顺序就是计划中的 `subpose_id` 升序，因此 `G_N` 是每个 cell 前 N 个 subpose 的 visible component ID 并集，`G_128` 是最终参考并集。

本评估不读取 test。`component_weights` 按采样器声明的可见重要性权重解释，不称为真实像素覆盖率；每个 component 在一个 cell 内取所有已观测权重的最大值，再计算前缀权重质量。

## 输出指标

`per_viewcell.csv` 和 `summary.csv` 默认均输出 `1/2/4/8/16/32/64/128` 八个前缀。主要字段含义如下：

| 字段 | 含义 | 类型 |
|---|---|---|
| `visibleCoverage` | `|G_N| / |G_128|`，零 GT cell 定义为 1 | 画面/集合覆盖诊断 |
| `newInstanceRate` | 本前缀相对上一个二分前缀新增的 ID 数，即 `|G_N - G_(N/2)| / |G_128|`；N=1 的上一个前缀为空 | 新发现比例 |
| `remainingInstanceRate` | `|G_128 - G_N| / |G_128|`，显示仍会在后续 128 点中出现的实例比例 | 收敛残差 |
| `weightedConvergence` | 前缀已取得的最大权重总和除以最终 128 点最大权重总和 | 加权收敛 |

`weightedCoverage` 保留为 `weightedConvergence` 的同值诊断列，避免历史表读取时丢失语义；新的正式结论使用 `weightedConvergence`。汇总行同时给出均值和 5% 分位，不能只看普通实例覆盖。

## 计划生成与正式采样入口

计划生成命令保持如下，生成结果分别是 100 个 validation cell、每 cell 128 个嵌套 subpose：

```bash
conda run -n slm_pvs python \
  neural_instance_culling/benchmark/build_gt_convergence_pose_plan.py \
  --dataset-dir neural_instance_culling/dataset/out/pose_csr_hkust_v3_main_stratified_calibration_fov66_v1 \
  --representative-plan neural_instance_culling/sampler/out/hkust_v3_viewcell_fov66/representatives.jsonl \
  --output neural_instance_culling/benchmark/out/paper_results/gt_convergence/hkust_128_plan.jsonl \
  --scene hkust_v3 --viewcell-shape horizontal_disk --radius 2 --half-up 0

conda run -n slm_pvs python \
  neural_instance_culling/benchmark/build_gt_convergence_pose_plan.py \
  --dataset-dir neural_instance_culling/dataset/out/pose_csr_ifcbench_fantasy_metropolis_main_stratified_calibration_fov66_v1 \
  --representative-plan neural_instance_culling/sampler/out/ifcbench_fantasy_metropolis_instanced_v2/pose_plan_fov66.jsonl \
  --output neural_instance_culling/benchmark/out/paper_results/gt_convergence/ifcbench_128_plan.jsonl \
  --scene ifcbench_fantasy_metropolis --viewcell-shape horizontal_disk --radius 2.5 --half-up 0
```

IFCBench 必须使用半径 `2.5 m` 的世界 XZ 水平圆盘，与 V4 前端一次区域查询契约一致；
带垂直位移的 camera-aligned box 计划已作废，不能用于证明当前模型的区域 GT 收敛。

正式硬件采样由唯一的 view-cell wrapper 调度。wrapper 对每个 shard 强制传递硬件 GPU 要求；这里仅列出后续正式运行命令，本次修复没有启动它：

```bash
conda run -n slm_pvs node \
  neural_instance_culling/sampler/run_scene_viewcell_colorid_sampling.mjs \
  --scene <scene> --assets-dir <scene>/assets \
  --pose-plan neural_instance_culling/benchmark/out/paper_results/gt_convergence/<scene>_128_plan.jsonl \
  --output-dir neural_instance_culling/benchmark/out/paper_results/gt_convergence/<scene>_128_raw \
  --subposes-per-viewcell 128 --shards 16 --parallel 4
```

每个 raw shard 旁的 `*.jsonl.gpu_evidence.json` 必须包含 `hostGpuBefore`、`hostGpuDuring`、`hostGpuAfter`。wrapper 的 `gpu_execution_summary.json` 会在 `jobs` 中原样汇总这三个对象；任一 shard 缺少任一阶段、硬件门失败或有错误时，汇总写成 `formalReady=false` 并以非零状态退出。汇总 schema 为 `viewcell-color-id-sampler-gpu-execution-v2`。

采样完成后，新的正式 evaluator 入口只有以下形式：

```bash
conda run -n slm_pvs python \
  neural_instance_culling/benchmark/evaluate_gt_convergence.py \
  --scene <scene> \
  --plan neural_instance_culling/benchmark/out/paper_results/gt_convergence/<scene>_128_plan.jsonl \
  --raw neural_instance_culling/benchmark/out/paper_results/gt_convergence/<scene>_128_raw \
  --output-dir neural_instance_culling/benchmark/out/paper_results/gt_convergence/<scene>_128_eval
```

## 历史结果与当前状态

`hkust_existing/` 和 `ifcbench_existing/` 下的旧 CSV/JSON 仍保留可读，用于复现上一轮有限 subpose 初检；它们不是新的 128 点结果，也不作为正式入口或论文表格输入。旧初检的代表数字为：HKUST 在 16/32 点相对旧参考并集的实例覆盖均值为 98.401%/99.668%，IFCBench 在 1/2/4 点为 70.539%/84.382%/100%。这些数字不能替代最终 128 点参考。

新的 128 点 raw 硬件采样尚未在本次修复中执行，因此本文件不登记新的场景数值、硬件延迟或收敛结论。代码验证只使用合成 JSONL、CPU Python 单元测试和 Node wrapper 测试。

## 修改与验证记录

- 修改文件：`neural_instance_culling/benchmark/evaluate_gt_convergence.py`、对应 Python 测试、`neural_instance_culling/sampler/run_scene_viewcell_colorid_sampling.mjs`、wrapper Node 测试及本报告。
- 依赖资源：新的 100×128 plan JSONL 与其完整 raw JSONL；正式采样另需每 shard 的三阶段 GPU evidence。
- 主要修复：直接按三元组对齐；按最终 `G_128` 计算新增实例率与加权收敛；汇总保留每 shard 的 before/during/after host GPU 证据并拒绝缺失。
- 主线状态：保留为新的 GT convergence 正式入口；旧结果仅作历史读取。
- 未决风险：正式结论仍取决于后续独占硬件采样是否完整，以及 64 到 128 点的新增实例率和 weighted 收敛是否达到协议要求。
