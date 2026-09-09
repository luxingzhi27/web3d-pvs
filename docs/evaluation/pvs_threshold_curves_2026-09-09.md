# PVS 安全效率曲线

日期：2026-09-09

`generate_threshold_curves.py` 是当前论文 Fig. 4 的唯一曲线入口。它读取 `evaluate_pvs.py` 导出的 canonical `pvs-typed-score-sidecar-v1`：一个 calibration sidecar，以及可选的同 checkpoint validation sidecar。

## 口径

- 阈值候选只来自 calibration 的 float32 score change-points。默认最多确定性压缩为 256 个点；validation 只在这组 calibration 阈值上评估。
- 预测规则固定为 `score >= threshold`。相同分数作为一个事件处理，不受候选排列影响。
- 横轴是 aggregate weighted recall；效率轴是 `useful cull = TN / candidate` 和按 pose 去重后的平均预测 GLB 字节。
- 每个点保留 weighted recall、单侧 LCB、precision、recall、accuracy、balanced accuracy、specificity、avg pred、useful/bad cull、普通 pose 诊断、candidate/GT 数和 GLB 字节。
- 安全门是 calibration 上 weighted recall 与单侧 95% pose-bootstrap LCB 均严格大于 `0.99`。validation 的安全标记只继承 calibration，不能反向选阈值。`--bootstrap-replicates 0` 仅用于测试或轻量诊断。
- test split 或 `testRead=true` 会直接拒绝。GLB 字节来自 `runtimeVisibilityMeta.json` 的实例映射和 `glbIndex.json` 指向文件的实际大小，不做估算。

## 运行

```bash
conda run -n slm_pvs python neural_instance_culling/benchmark/generate_threshold_curves.py \
  --calibration-sidecar <calibration-sidecar-dir> \
  --validation-sidecar <validation-sidecar-dir> \
  --runtime-meta <scene-assets>/runtimeVisibilityMeta.json \
  --glb-index <scene-assets>/glbIndex.json \
  --glb-root <scene-assets> \
  --output-csv neural_instance_culling/benchmark/out/paper_results/threshold_curves/visibility_curve.csv \
  --output-json neural_instance_culling/benchmark/out/paper_results/threshold_curves/visibility_curve.json
```

同时生成 `<output-json stem>.plot.json`，其中只保留两组绘图点：`weighted_recall -> useful_cull` 和 `weighted_recall -> avg_pred_glb_bytes`。CSV/JSON/plot source 都明确写出 `testRead=false`；正式结果应使用默认 `10,000` 次 bootstrap，并保存于计划登记的 `threshold_curves/` 目录。

本次提交只完成 canonical 入口和契约测试；正式 validation 数值待对应模型的 calibration/validation typed sidecar 生成后运行，不在此处预填实验结果。
