# 冻结 Test Split 评价协议

更新时间：2026-09-05

## Split 职责

| Split | 允许用途 | 禁止用途 |
|---|---|---|
| train | 参数训练和 train-only 离线关系编码 | 阈值安全结论 |
| calibration | 为每个 checkpoint 冻结安全阈值 | 比较论文模型优劣 |
| validation | 选择 checkpoint、配置和消融结论 | 重新选择阈值 |
| test | 对已冻结最终模型执行一次正式评价 | 选择阈值、模型、超参数或评价口径 |

主实验固定使用 `5926 train / 659 calibration / 730 validation / 684 test`。Test 的实际数量必须从数据集 split 读取并写入报告，不能在脚本或文档中假定所有场景都是 684。

## 正式遍历

正式 test 默认遍历 split 中全部唯一且有 GT 可见实例的 pose/view-cell，每个样本只评价一次，不做有放回抽样，不通过 `max-eval-poses` 截断，也不按结果补抽样本。

对于当前 HKUST 主数据集，严格 test 应写作：

```text
684 unique test view-cells
```

如果某场景存在零 GT pose，必须在评价前登记纳入或排除规则，并单独报告数量。

## 冻结内容

读取 test 前必须冻结：

- 模型架构、三个 seed 和每个 seed 的 checkpoint epoch；
- 每个 checkpoint 自己的 calibration 阈值；
- 候选生成、FOV、view-cell 和 subpose 并集语义；
- 实例 GT、`visible_weights` 来源和 GLB 映射；
- 图像评价分辨率、硬件后端和运行时测试条件；
- 论文主指标、统计方法和结果表结构。

Test 结果不能反向修改上述内容。发现实现错误时，应修复协议并将受影响的 test 结果作废，而不是在同一结果上继续调参。

## 必须报告

- 实际 test pose/view-cell 数和是否全部唯一遍历；
- pose-macro 与 aggregate 的 precision、recall、F1、Jaccard、accuracy、balanced accuracy 和 specificity；
- pose-macro 与 aggregate PR-AUC，以及各自同口径正样本比例；
- weighted recall、单侧置信下界、useful cull 和 bad cull；
- 平均候选、GT、预测实例和 GLB 数/字节；
- 同位姿硬件 Color-ID 图像指标；
- 浏览器运行资产、推理延迟和渲染成本；
- `testRead=true` 的明确 provenance。

完整指标定义见[统一 PVS 论文评价与指标报告协议](unified_pvs_metrics_evaluation.md)。
