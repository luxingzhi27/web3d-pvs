# 当前模型优化阶段

日期：2026-08-13
状态：进行中；不修改当前默认前端资产

## 阶段定位

2026-08-11 已完成修正 pooled-context 定义后的方向深度关系、生存场和损失三因素正式矩阵；2026-08-12 已完成上下文容量、Fourier 视角编码、单调参数化和证据来源补充矩阵。两组结果取代 2026-08-09 的旧汇总，后者使用了已修正的上下文定义，不再作为当前证据。本文件标志新的优化阶段，后续实验不再沿用关系残差、固定 `0.5` 锚点或选择性纠错路线。

本阶段暂时不使用 `bad cull` 置信区间上界作为路线否决条件。`bad cull` 仍必须计算、报告和分析；它不能被删除，也不能用“预测更少”掩盖可见实例漏检。当前安全主门仍是校准集冻结的 `weighted recall > 0.99` 及其单侧 95% 置信下界大于 `0.99`。

## 当前有效版本

| 用途 | 模型/资产 | 状态 |
|---|---|---|
| HKUST 前端默认运行 | `pvs_directional_occlusion_proxy_encoder_rvl_strong_v2_full40_best` | 保留，不因优化实验自动替换 |
| IFCBench Metropolis 前端运行 | `pvs_directional_occlusion_proxy_ifcbench_fantasy_metropolis_instanced_v2_k4_full40_best` | 保留，不因优化实验自动替换 |
| 正式机制参考 | 生存场 + Fourier117 + `rvl_strong_v2` | 只作新计划的控制组，不导入默认前端 |

前端仍使用真实渲染垂直视场角 `60°`，采样、后退相机和模型查询使用 `66°`。浏览器只读取离线固定实例特征和轻量查询头，不运行点云编码、图传播、动态邻居查询或 HZB。

## 最新正式证据

两组正式实验均使用 213 个相同 validation pose、相同候选集合、实例 GT 和候选哈希；每个 checkpoint 在自己的 calibration split 冻结阈值，统计采用按 seed 聚类、seed 内按 pose 重采样的 10,000 次 paired bootstrap。

| 机制 | 当前证据 | 当前判定 |
|---|---|---|
| 普通方向上下文 | 在 RVL + 生存场条件下，pose precision 差值 `+0.0001`、useful cull 差值 `-0.0011`，区间均跨零；32 维扩为 64 维也无稳定增益 | 不继续扩维；改为由真实遮挡边直接生成生存场 |
| 遮挡生存场 | RVL 条件下两组比较的 pose precision 提升 `8.31` 和 `9.50` 个百分点，balanced accuracy 提升 `5.54` 和 `5.30` 个百分点；weighted recall 差值区间跨零，部分条件 bad cull 上升 | 保留为候选创新基础，但必须重新证明安全贡献和真实关系来源作用 |
| 旧安全约束损失 | weighted recall 提高，但 precision、balanced accuracy、useful cull、平均预测数和下载资源明显恶化 | 只作失败控制，不进入新主线 |
| Fourier117 查询 | 相比 direct9，pose precision `+21.64` 个百分点、balanced accuracy `+5.55` 个百分点、useful cull `+5.52` 个百分点、bad cull `-1.79` 个百分点；绝对预测 GLB 字节增加约 `15.58 MB`，CUDA p95 增加约 `1 ms` | 作为精度强参考；研究更紧凑的视点区域积分频谱 |

完整结果见[修正正式矩阵](../evaluation/pvs_direction_depth_relation_survival_constrained_v1_surface_fallback_owrb_formal40_pooled_context_20260811_formal_validation.md)、[正式结论](../evaluation/pvs_direction_depth_relation_survival_constrained_v1_surface_fallback_owrb_formal40_pooled_context_20260811_conclusion.md)、[补充机制矩阵](../evaluation/pvs_ray_context_survival_owrb_supplement_v1_formal40_20260812_formal_validation.md)和[补充结论](../evaluation/pvs_ray_context_survival_owrb_supplement_v1_formal40_20260812_conclusion.md)。阈值位置只作分布健康诊断；不能用 bias 或温度缩放移动阈值后宣称模型排序能力提高。

## 本阶段优化目标

1. 在 weighted recall 及其置信下界保持合格的前提下，提高 precision、balanced accuracy、F1 和 useful cull。
2. 降低平均预测实例数、GLB 下载字节和前端候选推理成本。
3. 继续使用实例级输出和 GLB 级下载排序，不能把显示过滤退化成 GLB 级开关。
4. 用同一候选集合和同一 pose 协议比较所有新实验；不补入 GT 可见实例，不使用前端白名单修复结果。
5. 在模型效果稳定后，再进行同位姿 Color-ID 图像评价和浏览器 WebGPU/移动设备测量；服务器 CUDA 时间不能替代浏览器结果。

## 后续实验登记规则

2026-08-13 起，下一阶段只以[分层遮挡关系生存网络、视点区域积分频谱查询与质量风险损失实施计划](../experiments/pvs_hierarchical_occlusion_survival_viewcell_spectral_query_2026-08-13.md)为实施入口。实验统一使用前缀 `pvs_hierarchical_relation_survival_integrated_spectral_quality_rvl_v1`，先单种子 8 epoch 快速淘汰，再双种子 16 epoch 复验；冻结模块后执行 80 epoch、三个种子、三个留一消融和两个机制鉴别对照。

- 新实验只使用唯一计划登记的目录和成员名，不再创建旧路线别名。
- pilot 只用于排查数值稳定性和粗选参数；正式比较使用三个固定 seed，从头训练 80 epoch。
- 每个 checkpoint 的阈值只由自己的 calibration split 冻结；validation 用于比较，test 只在模型、阈值、资产和协议全部冻结后执行一次。
- 每个模型只使用自己的 calibration split 冻结安全阈值。固定 `0.5`、安全阈值位置和校准误差作为分布诊断，不是额外安全硬门；不允许用 bias 或 temperature 后处理移动阈值后冒充模型效果。模型排序能力以同一 weighted recall 安全门下的 precision、accuracy、balanced accuracy、specificity、useful cull、q-tail 间隔、图像和资源指标判断。
- 每个结果至少报告 weighted recall、普通 recall、precision、F1、balanced accuracy、useful cull、bad cull、平均预测数、GLB 字节和前向延迟。bad cull 不是当前路线硬门，但必须作为画面风险诊断保留。
- 正式采样、Color-ID 图像评价和三角形深度采集必须通过 Chrome NVIDIA Vulkan/ANGLE 硬件 GPU 门，并保存页面后端、Chrome 日志及 `nvidia-smi`/`pmon` 证据。

## 保留边界

当前默认模型、数据集、前端资产、8 月 11 日正式矩阵、8 月 12 日正式补充矩阵，以及 `pvs_ray_context_survival_owrb_v1_subpose5_20260811_directchrome` 的三角形深度层硬件证据缓存均不得删除。已经被正式结果覆盖的 pilot、使用错误定义的旧汇总、未通过安全门的关系残差/选择性纠错输出和重复过程报告应清理。
