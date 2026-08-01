# Test Split Benchmark Protocol

日期：2026-05-25

## 变更目的

修正历史固定数量 benchmark 的默认评价口径。旧脚本默认从 test split 中有放回抽样固定数量 pose/viewcell，容易被误写成唯一测试样本。从本次变更开始，当前入口默认遍历 test split 中全部唯一可见 pose/viewcell；如果需要复现旧的抽样口径，必须显式传入 `--sample-with-replacement` 和正数 `--max-eval-poses`。

## 修改文件

- `neural_instance_culling/benchmark/compare_test_split.py`
- `docs/evaluation/test_split_benchmark_protocol.md`

## 当前口径

- 默认模式：`all_unique_visible_poses`
- 含义：对指定 split 中每个有 GT 可见实例的 pose/viewcell 只评估一次，不重复抽样。
- 对当前 `pose_csr_hkust_v3_viewcell_colorid_fov66` 数据集，test split 的唯一可见 viewcell 数量为 `684`，因此严格 test 结果应写作 `684 unique test viewcells`，不能写作固定数量 test。

## 兼容旧口径

如需复现历史 sampled-with-replacement 结果，使用：

```bash
python compare_test_split.py --sample-with-replacement --max-eval-poses <positive_count>
```

该结果只能标注为 `sampled_with_replacement`，不能作为严格 test split 结论。

## 指标解释

- `pose precision/recall/F1/Jaccard`：每个 pose/viewcell 单独计算集合指标，再对所有样本求平均。
- `agg precision/recall/F1`：把全部样本的 TP/FP/FN 合并后再计算，容易受大候选或大 GT viewcell 支配。
- `weighted recall`：按 rvcServer `component_weights` 对 GT 命中加权，只说明重要 GT 是否被找回，不惩罚 false positive。
- `avg pred / avg GT / avg candidate`：分别表示平均预测实例数、平均 GT 实例数和平均候选实例数，用于判断模型是否靠大量误报换召回。

## 注意事项

- 不建议在严格 test split 评估时使用 `--max-candidates-per-pose` 做投影相关候选裁剪，因为该裁剪会在 CPU 上执行大量候选和 GT 的屏幕重叠比较，并且改变前端真实候选口径。
- 若需要裁剪候选进行消融，报告必须明确写出 `maxCandidatesPerPose`，不能与完整候选结果混用。
