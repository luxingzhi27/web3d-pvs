# V4 精确阈值重校准记录

日期：2026-09-15

## 目的与口径

在不重新训练 HKUST/IFCBench Full V4 的前提下，用 calibration 上全部实际 float32 分数
变化点替代固定步长阈值网格。预测规则固定为 `score >= threshold`。每个 checkpoint 只用
自己的 calibration 先选择 aggregate weighted recall 与固定 pose bootstrap 单侧 95% LCB
均大于 `0.99` 的最高阈值；若 LCB 目标无法达成，则退到平均 WR 大于 `0.99` 的最高阈值。
Validation 只重放冻结阈值，不参与阈值搜索；本次没有
读取 test。

## 代码与产物

- 通用阈值入口：`neural_instance_culling/benchmark/v4_exact_calibration.py`
- 六成员编排：`neural_instance_culling/benchmark/run_v4_exact_recalibration.py`
- 结果目录：`neural_instance_culling/benchmark/out/paper_results/exact_recalibration_v1`
- 汇总：`exact_recalibration_v1/summary.json`

正式命令：

```bash
conda run -n slm_pvs python \
  neural_instance_culling/benchmark/run_v4_exact_recalibration.py run \
  --gpu-ids 1 2 3

conda run -n slm_pvs python \
  neural_instance_culling/benchmark/run_v4_exact_recalibration.py summarize
```

依赖保持为现有两场景 Pose CSR、`runtimeVisibilityMeta.json`、冻结 96D 几何表和六个
`best_safe.pt`；checkpoint 和原有 test 产物均未改写。

## Validation 结果

| Scene | Seed | Threshold | WR | WR LCB | Precision | Balanced Acc. | Occlusion Recall | CNOR | Useful Cull | Bad Cull | Avg Pred | Qualification |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| HKUST | 20260801 | 0.424937 | 0.999078 | 0.998263 | 0.205984 | 0.948556 | 0.910632 | 0.894835 | 0.889721 | 0.000310 | 522.12 | LCB 目标达成 |
| HKUST | 20260802 | 0.683167 | 0.996847 | 0.994558 | 0.234272 | 0.945474 | 0.925866 | 0.915014 | 0.904606 | 0.000802 | 449.12 | LCB 目标达成 |
| HKUST | 20260803 | 0.586447 | 0.994957 | 0.991306 | 0.208884 | 0.946573 | 0.912734 | 0.905418 | 0.891776 | 0.000450 | 511.70 | LCB 目标达成 |
| IFCBench | 20260801 | 0.768287 | 0.991020 | 0.989853 | 0.195832 | 0.778592 | 0.574765 | 0.607678 | 0.519958 | 0.001677 | 4728.28 | 平均 WR 达标 |
| IFCBench | 20260802 | 0.774344 | 0.990323 | 0.989395 | 0.228476 | 0.804560 | 0.663369 | 0.671480 | 0.600113 | 0.005173 | 3901.45 | 平均 WR 达标 |
| IFCBench | 20260803 | 0.767973 | 0.991350 | 0.990509 | 0.209282 | 0.788408 | 0.618249 | 0.648690 | 0.559296 | 0.003951 | 4316.98 | LCB 目标达成 |

指标含义：WR 是 `0.99` 保留条件，LCB 表示该估计的置信余量；Occlusion Recall 是不可见候选中的正确剔除比例；
CNOR 是按候选规模归一化后的遮挡机会实现比例；Useful Cull/Bad Cull 分别是 TN/FN 占全部
候选的比例。Precision 与 Balanced Accuracy 用于诊断过量保留和正负分类，不替代安全门。

## 冻结选择

- HKUST：三个种子均达到 LCB 目标，按 validation Useful Cull 优先选择 seed20260802。
- IFCBench：三个种子的 validation 平均 WR 均大于 `0.99`，全部保留；seed03 达到 LCB
  目标，seed01/02 标注为平均 WR 达标但 LCB 目标未达。成员比较继续联合 Useful Cull、
  CNOR、balanced accuracy 和预测数量，不再因 LCB 略低于 `0.99` 删除 seed01/02。

该选择目前只更新 calibration/validation 决策。正式 test、图像、streaming 和前端资产
只有在采用新阈值重放时才更新对应派生结果，不能把本表写成 test 结果。
