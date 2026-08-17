# 视线关系场与生存场补充矩阵结论

日期：2026-08-12

## 评价范围

本补充实验用于拆分当前轻量可见性模型中的四个实现因素：上下文表容量、相机查询编码、遮挡生存参数化和离线遮挡证据来源。它与核心 2×2×2 formal matrix 使用独立名称和独立输出目录，不能把补充对照解释为核心因子主效应。

- 5 个预注册变体 × 3 个随机种子 × 40 epoch，共 15 个成员；
- 213 个 validation pose；所有成员使用相同的后退相机候选集合、实例 GT 和 pose 顺序；
- 每个 checkpoint 使用自己的 calibration split 冻结阈值，安全门为 weighted recall `>0.99` 及其单侧置信下界 `>0.99`；
- validation 只比较冻结工作点，test split 未读取；
- 使用 10,000 次按 seed 聚类、seed 内按 pose 重采样的 paired bootstrap；
- 15 个成员均完成 Playwright 无头 Chrome 的 Color-ID 图像评价，并通过 NVIDIA Vulkan/ANGLE 硬件 WebGL 门。

逐成员指标、全部成对差值和图像指标见[正式 validation 明细](pvs_ray_context_survival_owrb_supplement_v1_formal40_20260812_formal_validation.md)。

## 成员概览

下表是三个 seed 的均值，用于概览；安全判定和统计结论以逐成员结果及 paired bootstrap 为准。`平均预测数`和`绝对预测 GLB 字节`越低通常越有利，但必须与 weighted recall、bad cull 和图像漏检一起解释。

| 变体 | 主要变化 | pose precision | pose weighted recall | aggregate weighted recall | pose useful cull | pose bad cull | 平均预测数 | 预测 GLB 字节 | 前向 p95 (ms) | 固定表 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 三角形关系 + 32 维上下文 + 9 维 ray + 单调生存 | 补充参考 | 0.4308 | 0.9900 | 0.9908 | 0.6650 | 0.0289 | 342.0 | 41.47 MB | 2.44 | 5.88 MB |
| 三角形关系 + 64 维上下文 + 9 维 ray + 单调生存 | 仅扩大上下文表 | 0.4246 | 0.9909 | 0.9916 | 0.6658 | 0.0297 | 326.5 | 41.05 MB | 2.52 | 7.08 MB |
| 三角形关系 + 32 维上下文 + 117 维 Fourier ray + 单调生存 | 恢复历史 ray 编码 | 0.6472 | 0.9959 | 0.9963 | 0.7202 | 0.0110 | 366.9 | 57.05 MB | 3.45 | 5.88 MB |
| 三角形关系 + 32 维上下文 + 9 维 ray + 无单调 28 维参数 | 取消生存场单调构造 | 0.4294 | 0.9906 | 0.9912 | 0.6668 | 0.0304 | 324.2 | 40.57 MB | 2.14 | 5.88 MB |
| AABB 关系 + 32 维上下文 + 9 维 ray + 单调生存 | 更换离线关系证据 | 0.4305 | 0.9906 | 0.9911 | 0.6737 | 0.0341 | 265.7 | 37.30 MB | 2.65 | 5.88 MB |

validation 中同时满足 pose-level 和 aggregate weighted recall `>0.99` 的 seed 数分别为：32 维参考 `2/3`、64 维上下文 `1/3`、Fourier ray `3/3`、无单调参数 `2/3`、AABB 关系 `2/3`。该统计用于揭示跨 seed 稳定性，不改变 calibration 的阈值冻结规则。

## 配对差值

差值均定义为左侧变体减右侧变体，区间为 95% paired bootstrap 置信区间。

| 比较 | 指标 | 差值 | 95% 区间 | 解释 |
|---|---|---:|---:|---|
| 64 维上下文 - 32 维上下文 | pose precision | -0.0062 | [-0.0188, 0.0083] | 跨零，无稳定改善 |
| 64 维上下文 - 32 维上下文 | pose weighted recall | +0.0009 | [-0.0015, 0.0034] | 跨零，安全差异不稳定 |
| 64 维上下文 - 32 维上下文 | pose useful cull | +0.0009 | [-0.0133, 0.0106] | 跨零 |
| 64 维上下文 - 32 维上下文 | pose bad cull | +0.0008 | [-0.0071, 0.0072] | 跨零 |
| 117 维 Fourier ray - 9 维 ray | pose precision | +0.2164 | [0.1961, 0.2372] | 稳定提高 |
| 117 维 Fourier ray - 9 维 ray | pose balanced accuracy | +0.0555 | [0.0468, 0.0646] | 稳定提高 |
| 117 维 Fourier ray - 9 维 ray | pose useful cull | +0.0552 | [0.0420, 0.0681] | 稳定提高 |
| 117 维 Fourier ray - 9 维 ray | pose bad cull | -0.0179 | [-0.0249, -0.0123] | 稳定降低 |
| 117 维 Fourier ray - 9 维 ray | pose miss-pixel rate | -0.0035 | [-0.0070, -0.0009] | 图像漏检稳定降低 |
| 117 维 Fourier ray - 9 维 ray | 绝对预测 GLB 字节 | +15.58 MB | [10.27, 21.35] MB | 资源负担稳定增加 |
| 无单调参数 - 单调生存 | pose precision | -0.0013 | [-0.0090, 0.0080] | 跨零 |
| 无单调参数 - 单调生存 | pose useful cull | +0.0018 | [-0.0031, 0.0055] | 跨零 |
| 无单调参数 - 单调生存 | pose bad cull | +0.0015 | [-0.0002, 0.0037] | 跨零，点估计偏向恶化 |
| AABB 关系 - 三角形关系 | aggregate precision | +0.0584 | [0.0175, 0.1173] | 稳定提高 |
| AABB 关系 - 三角形关系 | aggregate balanced accuracy | -0.0422 | [-0.0799, -0.0121] | 稳定降低 |
| AABB 关系 - 三角形关系 | aggregate useful cull | +0.0105 | [0.0037, 0.0213] | 稳定提高 |
| AABB 关系 - 三角形关系 | aggregate bad cull | +0.0042 | [0.0013, 0.0084] | 稳定增加，存在画面风险 |
| AABB 关系 - 三角形关系 | 平均预测数 | -76.27 | [-151.86, -27.61] | 预测量减少，但伴随 bad cull 上升 |

Fourier ray 的 `GLB 字节削减率`比 9 维 ray 高，并不意味着它降低了实际下载量：其绝对预测 GLB 字节稳定增加。该现象说明比例型资源指标不能脱离绝对字节和视觉效用单独解释；本报告以绝对预测字节、图像漏检和 weighted recall 共同判断运行代价。

## 机制结论

### 上下文特征

把固定上下文表从 32 维扩大到 64 维没有产生稳定的 precision、balanced accuracy、useful cull 或 weighted recall 增益，同时固定表从 5.88 MB 增至 7.08 MB。当前证据不支持“简单增加上下文容量”作为创新点。它也不能证明上下文表征完全无效，因为本对照只改变容量，没有改变关系选择、监督质量或聚合结构；后续应优先改进方向关系监督和上下文聚合语义，而非继续无条件扩维。

### 遮挡生存场

无单调 28 维参数与单调生存场的主要差值区间跨零。由此可以确认，单调约束在本数据和训练设置下没有被证明带来独立的可测收益，但该结果不能否定生存场本身在核心矩阵中表现出的诊断价值。当前应保留单调形式作为物理语义更清楚、前端输入不增加的表示，并把“单调参数化优于无约束参数化”降级为未证实假设。

### 相机查询编码

117 维 Fourier ray 显著改善 pose-level 分类和图像漏检，说明更丰富的视角编码确实能提升可见性判别。然而它增加了 visibility head 输入维度（177→285），平均前向 p95 增加约 1.0 ms，并使绝对预测 GLB 字节增加约 15.6 MB。对于移动端预算，这是一项可解释的精度/成本折中对照，不应取代 9 维直接 ray 主线，也不能把它包装成离线遮挡关系创新。

### 遮挡证据来源

AABB 关系证据在 aggregate precision、useful cull、平均预测数和部分绝对字节指标上有改善，但 aggregate balanced accuracy 下降、bad cull 稳定增加。它更像是较粗关系证据造成的效率与安全权衡，不能替代三角形深度关系作为正式遮挡监督。

## 最终处理

本补充矩阵的路线状态为“机制诊断完成，默认路线不变”：

- 不修改默认 checkpoint、默认阈值、前端固定特征或部署资产；
- 32 维上下文、9 维直接 ray、单调生存场和三角形深度关系继续作为轻量主线候选；
- 64 维上下文、无单调参数和 AABB 关系作为失败/未证实消融保留；
- Fourier ray 作为精度上界与运行成本对照保留，不进入移动端默认路径；
- 后续创新优先改进上下文关系监督与预算感知训练目标，同时维持 Playwright 无头 Chrome 的硬件 WebGL 采样和图像评价证据。

补充汇总和校验命令：

```bash
conda run -n slm_pvs python \
  neural_instance_culling/benchmark/summarize_ray_context_survival_owrb.py \
  --manifest neural_instance_culling/benchmark/out/pvs_ray_context_survival_owrb_supplement_v1_formal40_20260812/pvs_ray_context_survival_owrb_supplement_v1_formal40_20260812_formal_matrix_manifest.json \
  --output neural_instance_culling/benchmark/out/pvs_ray_context_survival_owrb_supplement_v1_formal40_20260812/pvs_ray_context_survival_owrb_supplement_v1_formal40_20260812_formal_summary.json \
  --report docs/evaluation/pvs_ray_context_survival_owrb_supplement_v1_formal40_20260812_formal_validation.md \
  --bootstrap-replicates 10000

conda run -n slm_pvs python \
  neural_instance_culling/benchmark/validate_ray_context_survival_owrb_supplement.py \
  --summary neural_instance_culling/benchmark/out/pvs_ray_context_survival_owrb_supplement_v1_formal40_20260812/pvs_ray_context_survival_owrb_supplement_v1_formal40_20260812_formal_summary.json \
  --output neural_instance_culling/benchmark/out/pvs_ray_context_survival_owrb_supplement_v1_formal40_20260812/formal_supplement_schema_validation.json \
  --require-artifact-files
```
