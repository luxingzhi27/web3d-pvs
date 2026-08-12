# 统一 PVS 评价指标

更新时间：2026-08-12

本文是当前实例级可见性预测的指标定义。模型、数据集和前端资产边界分别以
[`../current/neuralstreamweb3d_architecture_technical.md`](../current/neuralstreamweb3d_architecture_technical.md)、
[`../current/neuralstreamweb3d_dataset_protocol.md`](../current/neuralstreamweb3d_dataset_protocol.md) 和
[`../current/current_instance_pvs_versions.md`](../current/current_instance_pvs_versions.md) 为准。

## 评价目标

PVS 评价的目标是在画面安全约束下减少不必要的实例推理、显示和 GLB 下载。普通 precision、F1 和逐实例 accuracy 仍需报告，但不能单独代表方案质量；候选集合中不可见实例较多时，普通 accuracy 会被大量正确负样本抬高。

当前阶段的安全主门由每个 checkpoint 自己的 calibration split 冻结：

```text
weighted recall > 0.99
weighted recall 的单侧 95% 置信下界 > 0.99
```

满足安全门后，再比较 precision、balanced accuracy、useful cull、平均预测实例数、GLB 字节和前端延迟。普通 pose recall 必须报告，但当前不替代 weighted recall 作为安全主门。`bad cull` 仍是画面风险指标；本阶段暂不使用它的置信区间上界作为路线否决条件。

阈值不能在 validation/test 上重新选择。test 只能在模型、阈值、资产和候选协议冻结后全量执行一次。

## 候选集合

对每个 pose 定义：

```text
C：后退扩大视锥对实例 AABB 筛选出的候选集合
G：同一 pose 或 view-cell 的真实可见实例集合
P：模型阈值筛选后的预测可见实例集合
```

正式候选是后退相机对每个成功 subpose 独立执行 AABB 筛选后的并集，不能补入 GT 可见实例：

```text
TP = P ∩ G
FP = P - G
FN = G - P
TN = C - (P ∪ G)
```

必须验证 `G ⊆ C`。候选上限、候选哈希、相机 FOV 或 view-cell 语义改变后，结果不能与原协议混合。正式评测默认使用完整的 post-frustum 候选集合；为显存或速度使用上限时，必须作为独立消融报告。

## 指标定义

| 指标 | 定义 | 解释 |
|---|---|---|
| pose recall | 每个 pose 计算 `TP / (TP + FN)` 后取宏平均 | 每个视点是否保护了真实可见实例 |
| aggregate recall | 合并全部 pose 的 TP、FN 后计算 recall | 全部样本合并后的总体召回 |
| weighted recall | 按 `visible_weights` 加权统计 GT 找回比例 | 对重要可见实例更敏感，不惩罚 FP |
| pose precision | 每个 pose 计算 `TP / (TP + FP)` 后取宏平均 | 预测集合中可见实例的比例 |
| aggregate precision | 合并全部 pose 的 TP、FP 后计算 precision | 总体误报诊断 |
| F1 | `2 × precision × recall / (precision + recall)` | precision 与 recall 的综合诊断 |
| Jaccard | `TP / (TP + FP + FN)` | 预测集合与 GT 集合的交并比 |
| specificity | `TN / (TN + FP)` | 不可见候选被正确剔除的比例 |
| instance accuracy | `(TP + TN) / C` | 逐候选分类正确率，受类别不平衡影响 |
| balanced accuracy | `(recall + specificity) / 2` | 同时考虑正类召回和负类识别 |
| useful cull | `TN / C` | 正确剔除的不可见候选比例，是有效剔除效率 |
| bad cull | `FN / C` | 错误剔除的可见候选比例，是画面风险 |
| raw reduction | `(TN + FN) / C = 1 - P/C` | 把正确和错误剔除混在一起，只能辅助报告 |

其中 pose-level 指标先逐 pose 计算再平均；aggregate 指标先合并所有 pose 的计数再计算。`visible_weights` 必须注明来源：Color-ID 数据使用屏幕覆盖率 parts-per-million，历史 rvcServer 数据使用 `component_weights` 重要性权重，不能统一称为真实像素覆盖率。

Precision 还受候选集合中的正样本比例影响。令正样本比例为 $\pi$、真正率为 $TPR$、假正率为 $FPR$，则：

\[
\mathrm{precision}=\frac{TPR\,\pi}{TPR\,\pi+FPR(1-\pi)}.
\]

当候选集合加入更多不可见实例时，$\pi$ 会降低；即使模型的 $TPR$ 和 $FPR$ 不变，precision 仍可能下降。因此 precision 只应在候选口径一致时直接排名；跨场景或跨候选规模时必须同时报告平均候选数、GT/候选比例、specificity、accuracy 和 balanced accuracy。普通 accuracy 可能被大量 TN 抬高，balanced accuracy 对正负比例更稳定，是安全门之后判断分类能力的重要参考，但仍不能替代 weighted recall、useful cull、bad cull 和最终资源指标。

## 资源和运行指标

集合指标不能直接替代系统指标。满足安全门后还需报告：

- 平均预测实例数、预测数/候选数、预测数/GT 数；
- GLB 数量削减、GLB 字节削减、预算内可见效用；
- 模型特征表和权重大小；
- CUDA 前向时间，以及浏览器 Worker、WebGPU、主线程和完整调度延迟；
- 同位姿 Color-ID 的 miss-pixel rate、wrong-ID pixel rate、extra-pixel rate 和 PER。

`useful cull` 必须始终和 `bad cull`、recall、weighted recall 一起展示。一个模型可能因为预测数量过少而获得较高 useful cull，但同时漏掉大量可见实例；这种结果不能解释为模型更好。

## 工作点和报告

每个 checkpoint 的校准摘要必须保存阈值来源、weighted recall 点估计和置信下界。安全工作点在合格阈值中选择有效剔除效率或资源效率最高者；best-F1、最高 precision 和固定阈值只作为诊断工作点，不能绕过安全门成为默认资产。

正式报告至少包含：

1. pose-level 和 aggregate 的 precision、recall、weighted recall、F1、Jaccard、accuracy、balanced accuracy、specificity；
2. useful cull、bad cull、平均 TP/FP/FN/TN、平均预测数和资源削减；
3. 阈值的 calibration 来源、候选哈希、pose 数量和 test 是否读取；
4. 图像级和浏览器运行时指标，未实现的项目明确写为 `not_available`。

推荐入口：

```bash
conda run --no-capture-output -n slm_pvs python -u \
  neural_instance_culling/benchmark/evaluate_unified_pvs_metrics.py \
  --dataset-dir <frozen-pose-csr> \
  --runtime-meta <scene>/assets/runtimeVisibilityMeta.json \
  --glb-index <scene>/assets/glbIndex.json \
  --glb-root <scene>/assets \
  --max-candidates-per-pose 0 \
  --target-weighted-recall 0.99 \
  --device cuda
```

正式 test 的全量遍历规则见 [`test_split_benchmark_protocol.md`](test_split_benchmark_protocol.md)，图像级评价见 [`viewcell_image_per_evaluation.md`](viewcell_image_per_evaluation.md)，当前优化阶段的具体结果见 [`../current/optimization_restart_2026-08-10.md`](../current/optimization_restart_2026-08-10.md)。
