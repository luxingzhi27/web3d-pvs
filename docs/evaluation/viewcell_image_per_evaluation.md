# View-cell 图像级评价协议

更新时间：2026-09-05

图像级评价回答：模型在实例集合上满足安全约束后，使用前端真实相机渲染时仍会损失多少可见像素。它不能替代集合 precision、recall、useful cull、bad cull 或 GLB 资源指标。

## 输入契约

每次正式评价必须固定：

- 已冻结 checkpoint 及其 calibration 阈值；
- 与实例评价相同的 validation 或最终 test pose；
- 同一 view-cell 的真实 `60°` 显示相机和离线 subpose；
- 完整 reference 实例集合和模型预测实例集合；
- 实例级 `componentGlobalId`，不能先压缩为 GLB 可见集合；
- 同一版本的场景 GLB、实例变换和运行时元数据。

后退 `66°` 相机只用于生成模型候选，不能作为最终图像对比相机。Validation 图像结果用于模型比较；test 图像结果只能在 checkpoint、阈值和评价协议全部冻结后执行一次。

## 硬件要求

正式 Color-ID 图像必须通过 Playwright 无头 Chrome 的 NVIDIA Vulkan/ANGLE WebGL 硬件门，并保存页面后端字段、Chrome 参数及同一运行窗口的 `nvidia-smi` 和 `pmon` 证据。

SwiftShader、llvmpipe、softpipe、swrast 或无法确认后端的结果只能用于语义 smoke，不能进入论文图像指标。完整硬件规则见[硬件 GPU 执行政策](../current/hardware_gpu_execution_policy.md)。

## 评价流程

1. 使用模型 checkpoint 自己的 calibration 冻结阈值，在后退候选集合上得到实例级预测集合。
2. 对同一 view-cell 的每个真实 `60°` subpose 加载完整场景，渲染 reference Color-ID 图。
3. 在同一相机、分辨率、深度和场景版本下，只保留模型预测实例，渲染 prediction Color-ID 图。
4. 逐像素比较 reference 和 prediction 的实例 ID。
5. 先在每个 subpose 计算指标，再报告 pose/view-cell 宏平均、p95 和 aggregate 像素统计。
6. 同时关联该 pose 的实例集合、GLB 字节和运行时指标，不能只输出图像误差。

浏览器页面和完整 GLB 场景应在一个批次内复用，避免把重复页面初始化或完整场景重载误计为每 pose 推理成本。页面复用范围、GLB 加载次数和 reference/prediction 渲染时间必须记录。

## 图像指标

| 指标 | 定义 | 作用 |
|---|---|---|
| Image PER | reference 非背景像素中 prediction 实例 ID 不一致的比例 | 总体可见像素错误 |
| Miss-pixel rate | reference 为实例、prediction 为背景的比例 | 模型漏掉实例造成的空洞 |
| Wrong-ID pixel rate | 两张图均非背景但实例 ID 不同的比例 | 遮挡次序或实例选择错误 |
| Extra-pixel rate | reference 为背景、prediction 出现实例的比例 | 额外显示实例造成的画面差异 |
| p95 miss-pixel rate | 各 pose 漏检像素率的 95 分位数 | 困难视点尾部风险 |

报告必须注明分母。Image PER、miss 和 wrong-ID 通常以 reference 非背景像素为主要分母；extra-pixel 使用 reference 背景区域或全图像素时必须明确写出。

## 论文报告

图像表必须同时给出：

- mean、median、p95 的 PER、miss-pixel 和 wrong-ID pixel；
- extra-pixel rate 及其分母；
- 实际评价 pose、view-cell 和 subpose 数；
- 相同工作点的 pose/aggregate precision、recall 和 weighted recall；
- useful cull、bad cull、平均预测实例数；
- 预测 GLB 数、预测 GLB 字节和相对完整候选的削减；
- 浏览器后端、GPU、分辨率、页面复用和运行时间。

当前 V4 主线的 validation 图像指标尚未完成正式回填，应标记为 `not_available`。历史方向代理和 2026-08-11/12 矩阵的图像结果只能作为历史基线，不能改写成当前 V4 Full 的图像结论。
