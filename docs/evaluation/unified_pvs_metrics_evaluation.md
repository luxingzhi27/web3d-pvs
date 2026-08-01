# Unified PVS Metrics Evaluation

日期：2026-06-03；指标选择规则更新：2026-07-31

> 口径说明：本文是评价指标定义和历史 HKUST benchmark 记录。当前模型/数据集边界以 `docs/current/neuralstreamweb3_architecture_technical.md`、`docs/current/neuralstreamweb3_dataset_protocol.md` 和 `docs/current/current_instance_pvs_versions.md` 为准。本文提到的旧 scheduler、hash、teacher 或 fixed-geo 名称不代表当前默认模型；当前正式运行模型由 `benchmark/model_runners.py` 和当前版本清单决定。

## 统一阈值与模型选择规则

当前所有正式训练、checkpoint 保存、前端导出和图像级评价都使用同一条规则：

```text
1. 只保留 weighted recall > 0.99 的阈值工作点；严格大于，不包含等于 0.99。
2. 在合格工作点中选择 pose-level precision 最高者。
3. precision 相同才依次比较 pose F1、weighted recall，并优先选择平均预测数量较少者。
```

普通 pose recall `>=0.95` 不再作为第一道筛选条件。它仍然必须报告，用于说明画面安全的另一种集合口径；useful cull、bad cull、平均预测数、GLB 字节削减、图像 PER 和前端延迟继续用于判断工程代价。若没有任何工作点满足严格 weighted recall 条件，训练/导出流程不得用不合格阈值静默覆盖 `best.pt` 或前端资产。

该规则由 `model/common/threshold_selection.py` 统一实现。训练验证阶段只有满足约束的工作点才能参与 `best.pt` 比较；benchmark 和运行时阈值读取器在摘要存在阈值表但没有合格行时直接报错，不回退到 `bestF1` 或摘要中的任意 `best`。前端导出默认只接受 `primaryWeightedPrecision`，手动阈值必须能在阈值表中验证安全性；不安全阈值只能通过显式诊断开关导出，并不得作为默认资产。

## 目的

本报告修正当前 PVS 评价口径：不再把裸 pose precision 或裸 candidate reduction 作为唯一主结论，而是把任务定义为：

```text
在画面安全约束下，最大化有效实例剔除和 GLB 资源节省。
```

逐实例 accuracy 有必要报告，因为它能对齐 Neural Visibility of Point Sets 这类“可见性二分类”论文的基本准确性叙事。但本项目候选集合极大、不可见实例占多数，普通 accuracy 容易被大量 true negative 抬高，因此必须同时报告 balanced accuracy、recall、useful cull 和 bad cull。

## 候选集合口径

这里的 `full candidate` 不是全场景全部实例，而是前端真实链路中已经经过 AABB 视锥剔除后的完整神经网络输入集合：

```text
全场景实例
  -> 后退扩大视锥 / 当前 viewcell AABB frustum culling
  -> 神经网络遮挡剔除和 GLB 优先级排序
```

当前正式空间 CSR 数据集（HKUST `pose_csr_hkust_v3_spatial_raw_subpose_aabb_fov66_v1`、Metropolis `pose_csr_metropolis_spatial_dense_subpose_union_fov66_v2`）的候选语义是：

```text
union of full AABB candidates computed independently for each successful dense subpose
no GT visible-positive union or candidate repair
```

旧的 `pose_csr_*_viewcell_colorid_*` 目录可能记录过“候选集合加 dense-subpose 可见正例”的探索性语义，不能与当前正式 strict candidate 结果混用。评估器读取保存的 `frustum_ids.bin` 作为候选集合，并要求 `visible_ids ⊆ candidate_ids`；缺失时直接失败。

因此正式训练、阈值校准和测试都应默认使用 `--max-candidates-per-pose 0`，表示不再裁剪这个 post-frustum 候选集合。正数上限，例如旧实验中的 `8192`，只能作为显存受限消融或历史结果解释，不能与完整候选结果混成同一主结论。

当前数据集统计：

| Split | Viewcells | Avg Candidate | Max Candidate | Candidate > 8192 |
|---|---:|---:|---:|---:|
| train | 6585 | 6688.93 | 18831 | 2457 |
| val | 730 | 6269.97 | 18831 | 250 |
| test | 684 | 6272.94 | 18831 | 236 |

这说明 `8192` cap 会裁剪一部分大候选 viewcell。正式 benchmark 应使用完整 post-frustum test 候选；如果为了训练速度使用候选上限，报告中必须单独说明它不是完整候选口径。

## 修改文件

```text
neural_instance_culling/benchmark/evaluate_unified_pvs_metrics.py
neural_instance_culling/benchmark/model_runners.py
neural_instance_culling/model/train_directional_occlusion_proxy_encoder.py
AGENTS.md
docs/evaluation/unified_pvs_metrics_evaluation.md
docs/README.md
```

## 指标定义

```text
TP：预测保留，且 GT 可见。
FP：预测保留，但 GT 不可见。
FN：预测剔除，但 GT 可见。它是画面风险，不是剔除效率。
TN：预测剔除，且 GT 不可见。它才是有效剔除。
```

主指标：

```text
pose recall:
  每个 pose 的 TP / (TP + FN)，再平均。它是画面安全约束。

weighted recall:
  按 visible_weights 统计 GT 找回率。当前 visible_weights 是 rvcServer component_weights 弱权重，不是真实像素覆盖率。

useful cull:
  TN / candidate。表示正确剔除掉的不可见候选占全部候选的比例。

bad cull:
  FN / candidate。表示错误剔除掉的可见候选占全部候选的比例。

raw reduction:
  (TN + FN) / candidate = 1 - avg_pred / avg_candidate。
  这个指标会把 TN 和 FN 混在一起，只能作为辅助，不能单独排名。

instance accuracy:
  (TP + TN) / candidate。用于说明逐实例分类正确率，但会受候选不均衡影响。

balanced accuracy:
  (recall + specificity) / 2，其中 specificity = TN / (TN + FP)。
  用于避免普通 accuracy 被大量不可见候选抬高。
```

工作点选择：

```text
primary weighted-safe precision workpoint:
  满足 weighted recall > 0.99；
  在满足约束的阈值中选择 pose precision 最高的工作点。

set-safe useful-cull workpoint:
  满足 weighted recall > 0.99；
  仅作为剔除效率诊断，在满足约束的阈值中选择 useful cull 最高的工作点。

utility-safe workpoint:
  满足 weighted recall > 0.99 且 weak utility recall >= 0.98；
  仅作为资源效用诊断，在满足约束的阈值中选择 useful cull 最高的工作点。
```

## 运行命令

```bash
CUDA_VISIBLE_DEVICES=2 conda run --no-capture-output -n slm_pvs python -u \
  neural_instance_culling/benchmark/evaluate_unified_pvs_metrics.py \
  --models baseline_aabb_hzb,pvs_directional_occlusion_proxy_encoder_rvl_w042_full40_hkust_fov66_best \
  --dataset-dir neural_instance_culling/dataset/out/pose_csr_hkust_v3_viewcell_colorid_fov66 \
  --runtime-meta hkust-v3/assets/runtimeVisibilityMeta.json \
  --glb-index hkust-v3/assets/glbIndex.json \
  --glb-root hkust-v3/assets \
  --output-dir neural_instance_culling/benchmark/out/pvs_directional_occlusion_proxy_encoder_rvl_w042_full40_hkust_fov66_test_20260731 \
  --poses-per-batch 4 --max-candidates-per-pose 0 \
  --target-recall 0.95 --target-weighted-recall 0.99 --target-utility-recall 0.98 \
  --device cuda \
  > neural_instance_culling/benchmark/out/pvs_directional_occlusion_proxy_encoder_rvl_w042_full40_hkust_fov66_test_20260731/benchmark_stdout.log \
  2> neural_instance_culling/benchmark/out/pvs_directional_occlusion_proxy_encoder_rvl_w042_full40_hkust_fov66_test_20260731/benchmark_stderr.log
```

正式评估为 `684` unique test viewcells，完整候选集合，不是固定 1024。

输出：

```text
neural_instance_culling/benchmark/out/pvs_directional_occlusion_proxy_encoder_rvl_w042_full40_hkust_fov66_test_20260731/summary.json
neural_instance_culling/benchmark/out/pvs_directional_occlusion_proxy_encoder_rvl_w042_full40_hkust_fov66_test_20260731/summary.md
```

## 历史 Set-Safe 结果

下面表格保留 2026-06-03 的历史 useful-cull 主选结果，安全约束当时是 `pose recall >= 0.95` 且 `weighted recall >= 0.99`。它用于复现旧比较，不代表当前阈值选择规则；当前规则见本文“统一阈值与模型选择规则”。

| Model | Threshold | Accuracy | Balanced Acc | Recall | Weighted Recall | Useful Cull | Bad Cull | Raw Reduction | Avg Candidate | Avg GT | Avg Pred | Overfetch | Byte Reduction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| scheduler nohash | 0.100 | 0.913 | 0.921 | 0.955 | 0.995 | 0.850 | 0.00163 | 0.871 | 6272.94 | 158.08 | 809.85 | 5.12 | 0.842 |
| scheduler hash | 0.030 | 0.908 | 0.918 | 0.953 | 0.997 | 0.846 | 0.00155 | 0.874 | 6272.94 | 158.08 | 787.96 | 4.98 | 0.834 |
| occlusion teacher nohash | 0.150 | 0.923 | 0.925 | 0.951 | 0.996 | 0.861 | 0.00195 | 0.880 | 6272.94 | 158.08 | 751.03 | 4.75 | 0.852 |
| fixed PointNet++ | 0.010 | 0.904 | 0.918 | 0.960 | 0.997 | 0.841 | 0.00100 | 0.871 | 6272.94 | 158.08 | 808.64 | 5.12 | 0.830 |

解读：

```text
occlusion teacher nohash 在 set-safe 约束下 useful cull 最高，为 0.861；
scheduler nohash 为 0.850，高于 fixed PointNet++ 的 0.841；
fixed PointNet++ 的 bad cull 最低，为 0.00100，说明它更保守地保护 GT；
hash 消融没有带来更好的 useful cull 或 byte reduction。
```

这说明之前“precision 不够高所以模型不行”的判断过于粗糙。以 scheduler nohash 为例，虽然 pose precision 是 0.611，但在 684 test 上它平均把候选从 6272.94 降到 809.85，正确剔除不可见候选占全部候选 0.850，错误剔除可见候选只占全部候选 0.00163。

但也不能反过来说它已经足够好，因为：

```text
avg pred / avg GT 仍为 5.12；
平均 FP 仍有 653.62；
image PER 和真实像素级结果仍需一起看；
GLB utility 和字节排序仍不完全优于旧基线。
```

## GLB Budget 结果

| Model | Budget | Utility Recall | Required Recall | Avg Selected Bytes | Byte Reduction |
|---|---:|---:|---:|---:|---:|
| scheduler nohash | 50 | 0.890 | 0.870 | 8,312,570 | 0.671 |
| scheduler hash | 50 | 0.879 | 0.854 | 9,162,838 | 0.668 |
| occlusion teacher nohash | 50 | 0.893 | 0.872 | 9,519,444 | 0.649 |
| fixed PointNet++ | 50 | 0.900 | 0.892 | 12,375,593 | 0.629 |
| scheduler nohash | 100 | 0.910 | 0.892 | 11,729,183 | 0.626 |
| scheduler hash | 100 | 0.900 | 0.875 | 12,715,190 | 0.626 |
| occlusion teacher nohash | 100 | 0.915 | 0.896 | 14,528,557 | 0.594 |
| fixed PointNet++ | 100 | 0.924 | 0.920 | 21,756,467 | 0.562 |

解读：

```text
fixed PointNet++ 的 weak utility recall 更高；
scheduler nohash 在同样 GLB budget 下 selected bytes 更低；
这符合“download head 学到部分成本偏好，但效用排序仍不够强”的旧结论。
```

## 结论

本次统一指标复评后，评价口径应调整为：

```text
先满足 `weighted recall > 0.99`；
在合格工作点中选择最高 pose precision；
再报告 pose recall、useful cull、bad cull、avg pred、overfetch、GLB byte reduction、图像 PER 和 latency；
F1、accuracy 和普通 aggregate 指标作为诊断指标保留。
```

当前已有 checkpoint 可以重新进行阈值校准；后续训练必须在验证阶段按上述严格规则保存 `best.pt`，前端导出必须读取同一工作点。历史表格中的 `pose recall >= 0.95` 主筛选只代表旧口径，不能作为当前默认模型选择依据。

## 未实现项

```text
本脚本没有重新运行 true GLB image PER；
图像级 PER 仍引用已有 evaluate_viewcell_image_per.py 管线。
前端移动端 latency / Worker smoke 未在本次统一评测中执行。
```
