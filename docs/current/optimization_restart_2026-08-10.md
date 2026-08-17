# 当前模型优化阶段

日期：2026-08-15
状态：进行中；不修改当前默认前端资产

## 阶段定位

2026-08-11 已完成修正 pooled-context 定义后的方向深度关系、生存场和损失三因素正式矩阵；2026-08-12 已完成上下文容量、Fourier 视角编码、单调参数化和证据来源补充矩阵。两组结果取代 2026-08-09 的旧汇总，后者不再作为当前证据。2026-08-14 至 2026-08-15 又完成了有界真实关系、视点矩包络、安全裕度效用损失及逐实例校准残差的代码纠错。当前只执行 v4 路线；旧的无界关系残差、固定 `0.5` 锚点和选择性纠错实现不再使用。

本阶段暂时不使用 `bad cull` 置信区间上界作为路线否决条件。`bad cull` 仍必须计算、报告和分析；它不能被删除，也不能用“预测更少”掩盖可见实例漏检。当前安全主门仍是校准集冻结的 `weighted recall > 0.99` 及其单侧 95% 置信下界大于 `0.99`。

## 当前有效版本

| 用途 | 模型/资产 | 状态 |
|---|---|---|
| HKUST 前端默认运行 | `pvs_directional_occlusion_proxy_encoder_rvl_strong_v2_full40_best` | 保留，不因优化实验自动替换 |
| IFCBench Metropolis 前端运行 | `pvs_directional_occlusion_proxy_ifcbench_fantasy_metropolis_instanced_v2_k4_full40_best` | 保留，不因优化实验自动替换 |
| 正式机制参考 | 生存场 + Fourier117 + `rvl_strong_v2` | 只作新计划的控制组，不导入默认前端 |
| 当前实验入口 | `pvs_bounded_relation_prior_instance_calibrated_moment_safety_reserve_v4` | 实现与 CUDA smoke 已通过，尚未形成效果结论 |

前端仍使用真实渲染垂直视场角 `60°`，采样、后退相机和模型查询使用 `66°`。浏览器只读取离线固定实例特征和轻量查询头，不运行点云编码、图传播、动态邻居查询或 HZB。

## 最新正式证据

两组正式实验均使用 213 个相同 validation pose、相同候选集合和实例 GT；每个 checkpoint 在自己的 calibration split 冻结阈值，统计采用按 seed 聚类、seed 内按 pose 重采样的 10,000 次 paired bootstrap。

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

2026-08-13 的架构、pilot 和 R0/R2 双 seed 32 epoch 学习曲线已经完成，记录见[分层遮挡关系生存网络、视点区域积分频谱查询与质量风险损失实施记录](../experiments/pvs_hierarchical_occlusion_survival_viewcell_spectral_query_2026-08-13.md)及[32 epoch 学习曲线评价](../evaluation/pvs_hierarchical_relation_survival_learning_curve_e32_2026-08-13.md)。R2 虽通过 weighted recall 安全门，但没有稳定提高 precision、balanced accuracy、useful cull 或预测资源效率。随后完成的 v2 `formal80` 暴露了共享生存生成器缺少逐实例表达能力等实现问题，旧矩阵只保留为诊断证据。

后续只以[v4 实施记录](pvs_bounded_relation_survival_moment_v4_implementation_2026-08-15.md)和[v4 四卡运行与消融手册](../experiments/pvs_bounded_relation_survival_moment_v4_run_and_ablation_2026-08-15.md)为执行入口。v4 使用共享分层关系先验加有界逐实例校准残差；残差只在离线训练中存在，导出时融合进每实例 28 维生存系数，因此浏览器仍读取 `96 + 28 = 124` 维 FP16 固定表。

- 新实验只使用唯一登记的 v4 模型、checkpoint、运行包和输出 schema。冻结的 PoseCSR 与关系数据继续使用经校验的 v3 数据 schema，不重命名、不重建。
- 快速扫描使用八组完整模型配置、单个固定 seed、8 epoch 和每 epoch 50 step。八组完成后无条件冻结相对最优配置，并用四张 GPU 动态队列从头完成 5 个变体、3 个固定 seed、80 epoch 的全部 15 个成员；扫描或中间指标不能取消长训。
- weighted recall 安全门及 precision、accuracy、balanced accuracy、资源和运行成本只决定结果能否晋级论文主模型、默认资产或前端部署，不决定是否训练。没有合格安全工作点的成员保存非安全诊断工作点并如实标记。
- 每个 checkpoint 的阈值只由自己的 calibration split 冻结；validation 用于比较，test 只在模型、阈值、资产和协议全部冻结后执行一次。
- 每个模型只使用自己的 calibration split 冻结安全阈值。固定 `0.5`、安全阈值位置和校准误差作为分布诊断，不是额外安全硬门；不允许用 bias 或 temperature 后处理移动阈值后冒充模型效果。模型排序能力以同一 weighted recall 安全门下的 precision、accuracy、balanced accuracy、specificity、useful cull、q-tail 间隔、图像和资源指标判断。
- 每个结果至少报告 weighted recall、普通 recall、precision、F1、balanced accuracy、useful cull、bad cull、平均预测数、GLB 字节和前向延迟。bad cull 不是当前路线硬门，但必须作为画面风险诊断保留。
- 正式采样、Color-ID 图像评价和三角形深度采集必须通过 Chrome NVIDIA Vulkan/ANGLE 硬件 GPU 门，并保存页面后端、Chrome 日志及 `nvidia-smi`/`pmon` 证据。

## 保留边界

当前默认模型、数据集、前端资产、8 月 11 日正式矩阵、8 月 12 日正式补充矩阵，以及 `pvs_ray_context_survival_owrb_v1_subpose5_20260811_directchrome` 的三角形深度层硬件证据缓存均不得删除。已经被正式结果覆盖的 pilot、使用错误定义的旧汇总、未通过安全门的关系残差/选择性纠错输出和重复过程报告应清理。
