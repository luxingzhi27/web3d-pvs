# M5 HKUST 正式 calibration 图像评价

日期：2026-08-01
状态：calibration 图像评价完成；均值 miss-pixel 门通过，按 view-cell 的 p95 门 No-Go；未读取 test split。

## 评价口径

本次遍历 HKUST calibration split 中全部 `168` 个 view-cell，每个 view-cell 取一个真实 subpose。模型输入和
后退候选相机使用 66°，真实 Color-ID 图像使用 60°垂直视场角。Chrome 页面一次加载全部 `3,273` 个本地 GLB，
以实例槽位绑定 `componentGlobalId`，先渲染完整 reference，再根据冻结 calibration 前的模型阈值渲染 prediction。
两次渲染使用同一页面、同一相机和同一实例级 ID 编码。

运行输出：

```text
neural_instance_culling/benchmark/out/m5_hkust_strong_v2_calibration_image_20260801_full/summary.json
neural_instance_culling/benchmark/out/m5_hkust_strong_v2_calibration_image_20260801_full/true_glb_render/render_summary.json
```

阈值为 `0.0199999996`，来源是 pre-test calibration summary；本次没有重新扫描阈值，也没有读取 test。
`visible_weights` 仍按当前数据语义解释为弱可见性权重，不宣称为真实像素覆盖率。

## 结果

| 指标 | 聚合值 | 中文含义 |
|---|---:|---|
| view-cell / subpose | 168 / 168 | 全部样本完成 reference/prediction 渲染 |
| 有效 reference 像素 | 5,295,716 | reference 中属于场景实例的像素 |
| miss pixels | 14,086 | 真实实例未被预测保留造成的漏像素 |
| wrong-instance pixels | 23,772 | prediction 实例 ID 与 reference 不一致的像素 |
| aggregate miss-pixel rate | 0.2660% | `missPixels / validReferencePixels` |
| aggregate wrong-ID rate | 0.4489% | `wrongInstancePixels / validReferencePixels` |
| aggregate PER | 0.7149% | miss 与 wrong-ID 的有效像素比例 |
| extra-pixel rate | 0 | reference 背景上没有额外预测像素 |
| self-consistency PER | 0 | reference 与自身比较的渲染一致性 |

按单 view-cell 的 miss-pixel rate 统计：

| 指标 | 数值 |
|---|---:|
| 均值 | 0.2935% |
| 中位数 | 0% |
| p95 | 1.2934% |
| 最大值 | 20.6621% |

模型集合和运行代价：

| 项目 | 数值 |
|---|---:|
| weighted recall | 0.995703 |
| 平均预测实例数 | 778.26 |
| 平均真实可见实例数 | 73.81 |
| 平均模型预测耗时 | 27.01 ms |
| GLB 加载耗时 | 14.23 s |
| 168 个 subpose 渲染耗时 | 1,076.10 s |
| 页面总耗时 | 1,090.42 s |

## 门控判断与诊断

计划门槛为按 view-cell 的 miss-pixel rate 均值 `<0.5%` 且 p95 `<1%`。本次均值为 `0.2935%`，满足均值门；
p95 为 `1.2934%`，未满足长尾门，因此 M5 calibration 图像门整体保持 No-Go。self-consistency PER 为 0，
说明实例 ID 绑定、深度测试和同页 reference/prediction 切换没有产生自洽性错误。

漏像素主要集中在以下构件：`componentGlobalId=15295`（7,729 pixels）、`15294`（6,519 pixels）、
`4265`（5,742 pixels）、`15221`（2,839 pixels）和 `15296`（2,683 pixels）。它们属于少数高视觉贡献
构件，不能简单归因于低权重细小实例。后续若要修复，应在 validation/calibration 上检查 view-cell union
监督、候选覆盖和高贡献实例的安全约束，不能使用 test 阈值扫描绕过 p95 门。

## 保留判断

本结果作为 calibration 阶段的正式图像质量证据保留，并用于阈值/模型改进诊断；不能进入通过图像安全门的论文
主表。test 图像评价仍保持封存，真实移动设备和硬件 WebGPU 性能也不由本报告替代。
