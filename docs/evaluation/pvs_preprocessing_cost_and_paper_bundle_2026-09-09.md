# PVS 离线成本与结果包

日期：2026-09-09。目的：按论文计划登记 HKUST/IFCBench 的六阶段离线成本，并把 paper-results 产物统一登记到 bundle；只读取既有日志、meta 和输出文件。

## 口径

`collect_preprocessing_cost.py` 使用固定的两场景来源表，阶段为 `sampling`、`relation`、`fixed_geometry`、`training`、`calibration`、`export`。只读取显式 `elapsed`、设备名、峰值 VRAM/CPU RAM；输出文件或目录用实际 `stat` 字节递归求和。缺失值是字符串 `unavailable`，不推算；训练时间取每个 seed history 的最后累计值再求和。采样和 IFCBench 关系时间只代表登记到的 shard 工作量，未记录的 wall time 保持缺失。

观测成本（秒；缺失不等于零）：

| 阶段 | HKUST | IFCBench |
|---|---:|---:|
| sampling | 7111.0773 | 22975.1142 |
| relation | unavailable | 4428.8736 |
| fixed_geometry | unavailable | unavailable |
| training | 170971.0466 | 846192.9364 |
| calibration | unavailable | unavailable |
| export | unavailable | unavailable |

## Bundle

`build_paper_result_bundle.py` 保留 ablation/rank/runtime/image 汇总，并用固定 artifact spec 登记 `sceneStatistics`、`testMetrics`、`imageMetrics`、`ablation`、`rankSweep`、`runtime`、`hzb`、`thresholdCurves`、`streaming`、`gtConvergence`、`preprocessing`、`figures` 的 `path/status`。可选产物不解析 JSON；不存在的路径登记为 `unavailable`。当前 test image 没有 test 文件，registry 明确保留缺失状态。

`generate_paper_tables.py` 将同一 scene/method 的真实三 seed test 结果输出为 Table 2 的 mean +/- sample std，并把 manifest registry 展平成简洁 CSV/Markdown；只接收 `testRead=true` 的 test 结果。

## 验证与复现

已通过 `py_compile`、新增 7 项契约测试和 benchmark 全量测试（`89 passed, 2 skipped`）。bundle/table smoke 只做 CPU 文件读取，未运行 GPU、浏览器或哈希流程。代码保留为论文结果包主线；CPU RAM 和缺失阶段 wall time 待后续产生显式记录后再登记。

~~~bash
conda run -n slm_pvs python neural_instance_culling/benchmark/collect_preprocessing_cost.py \
  --source-root /mnt/sda/rhyang/slm --output-dir /tmp/pvs_preprocessing_cost_20260909
conda run -n slm_pvs python neural_instance_culling/benchmark/build_paper_result_bundle.py \
  --data-root /mnt/sda/rhyang/slm --output-dir /tmp/pvs_paper_results_20260909 \
  --hzb-dir /mnt/sda/rhyang/slm-wt-hzb/neural_instance_culling/benchmark/out/paper_results/hzb
conda run -n slm_pvs python neural_instance_culling/benchmark/generate_paper_tables.py \
  --scene-statistics /mnt/sda/rhyang/slm/neural_instance_culling/benchmark/out/paper_results/scene_statistics.csv \
  --test-metrics-dir /mnt/sda/rhyang/slm/neural_instance_culling/benchmark/out/paper_results/test_metrics \
  --output-dir /tmp/pvs_paper_results_20260909 \
  --bundle-manifest /tmp/pvs_paper_results_20260909/bundle_manifest.json
~~~
