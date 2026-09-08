# IFC Metropolis Instanced V2 Test 评测

> 历史方向代理基线，不是当前 IFCBench V4 长训结果。保留用于实例化场景和旧 test 口径对照。

## 日期与口径

2026-07-15，2026-07-31 按统一规则补充阈值校准。评测遍历 test split 的 2684 个唯一 viewcell，候选集合、真实可见集合和权重口径与旧 metropolis 模型一致。当前工作点先满足严格 `weighted recall > 0.99`，再选择 `pose precision` 最高；普通 pose recall、F1 和平均预测数量继续报告。

## 安全工作点

当前 weighted-safe precision 工作点阈值为 `0.6399999856948853`,前端显示为 `0.640`。旧阈值 `0.220` 作为历史普通 recall 安全工作点保留。

| 指标 | 结果 | 含义 |
| --- | ---: | --- |
| pose precision | 0.5622 | 每个 viewcell 分别计算可见预测精确率后取平均 |
| pose recall | 0.9041 | 每个 viewcell 的真实可见实例召回率平均值 |
| weighted recall | 0.9901 | 按 color-ID 屏幕占比权重统计的重要可见实例召回率 |
| pose F1 | 0.6844 | pose precision 与 pose recall 的调和平均 |
| pose Jaccard | 0.5324 | 预测可见集合与真实可见集合的交并比 |
| aggregate precision | 0.5343 | 汇总所有 viewcell 的 TP/FP 后计算精确率 |
| aggregate recall | 0.8883 | 汇总所有 viewcell 的 TP/FN 后计算召回率 |
| avg candidate | 9991.98 | 后退扩展视锥平均候选实例数 |
| avg GT | 977.60 | 平均真实可见实例数 |
| avg pred | 1625.19 | 模型阈值化后的平均预测实例数 |

## 画面安全与剔除效率

| 指标 | 结果 | 含义 |
| --- | ---: | --- |
| useful cull | 待统一 benchmark | `TN / candidate`,候选中被正确剔除的不可见实例比例 |
| bad cull | 待统一 benchmark | `FN / candidate`,候选中被错误剔除的可见实例比例 |
| FN / GT | 待统一 benchmark | 真实可见实例中被漏掉的比例 |
| candidate reduction | 0.8374 | `1 - avgPred / avgCandidate`,包含正确剔除和错误剔除,不能单独作为效率结论 |
| instance accuracy | 待统一 benchmark | `(TP + TN) / candidate`,逐候选实例二分类正确率 |
| specificity | 待统一 benchmark | `TN / (TN + FP)`,不可见候选被正确识别的比例 |
| balanced accuracy | 待统一 benchmark | aggregate recall 与 specificity 的平均值 |

当前精度优先工作点把 weighted recall 保持在 `0.9901`，平均预测降至 `1625.19`；useful cull、bad cull 和 specificity 需要按新阈值运行统一 benchmark 后补齐。主要资产收益仍来自实例化组织：GLB 数从 41298 降到 3669，场景 GLB 字节下降 67.36%。

## 未完成指标

- 图像 PER:未实现本次重评。
- miss pixel rate:未实现本次重评。
- wrong-ID pixel rate:未实现本次重评。
- 预算内 GLB utility recall:未实现。
- 手机端 WebGPU 延迟:未测试。

本地 Linux Chrome WebGPU smoke 在后退视锥 10793 个模型构件上记录到约 3.9 秒推理时间,该数字受无头浏览器、Vulkan 后端和首次运行影响,只能作为当前桌面 smoke 风险记录,不能代表手机端性能已经满足目标。
