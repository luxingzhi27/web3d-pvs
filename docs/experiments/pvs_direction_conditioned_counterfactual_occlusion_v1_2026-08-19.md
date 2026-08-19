# 方向条件化反事实遮挡表征快速迭代计划

- 日期：2026-08-19
- 实验前缀：`pvs_direction_conditioned_counterfactual_occlusion_v1`
- 场景：HKUST v3
- 状态：实现及单种子 8 epoch 快速矩阵已完成；反事实损失保留，当前门控关系特征不保留
- 基线：`pvs_cross_pose_operating_exposure_representation_v1` 的 S4/D 组合
- test split：不读取

## 问题证据

当前 D 组合在 validation 的普通 ROC-AUC 为 `0.91999`，加权 ROC-AUC 为 `0.99809`，说明主体排序已经具有有效信息，主要误差集中在安全阈值附近的困难尾部。其加权正例 `q0.5%` 为 `0.10467`，负例 `q99.5%` 为 `0.39821`，二者仍明显交叠。

对全部 validation pose 按实例重组后，D 的假阳性中有 `88.2%` 来自“在其他视点真实可见”的实例，只有 `11.8%` 来自 validation 中从未可见的实例。同一实例同时具有正负观察时，D 的跨视点成对 AUC 为 `0.89106`，仍有 `6.20%` 的实例出现平均正例分数不高于平均负例分数。旧 A 组的同口径 AUC 为 `0.82597`。因此，继续增加静态实例类别先验不能解决主要误差；模型需要更直接地表达“同一实例在当前方向和 view-cell 区域内为何暴露或受遮挡”。

当前区域暴露辅助头接在 64 维共享隐藏层之后。该隐藏层同时包含固定几何、ray、区域矩和生存语义，辅助任务可以绕过 28 维关系生存场完成拟合，不能保证导出的遮挡表示真正携带区域暴露信息。当前训练器虽然保留同实例跨视点排序函数，但在 `pose_balanced_frontier` 和 `cross_pose_operating` 两条主损失路径中没有把该项加入最终优化目标，因此打开参数也不会生效。本实验不把这一行为当作可继续沿用的结果。

## 联合研究假设

1. 困难正负实例的关键差异是实例关系条件与观察方向之间的乘性交互，而非额外的静态实例类别特征。
2. 将区域暴露监督限制在一个低维关系瓶颈上，可以迫使离线生存场编码当前方向下的遮挡边界信息，避免辅助头只从通用隐藏层旁路拟合。
3. 比较同一实例在可见和不可见视点的分数，可以消除静态实例偏置，直接优化条件化查询；使用视觉权重的软正尾和软负尾可以避免由单个低重要度正例支配训练。

## 方向条件化低秩关系特征

现有查询已经得到：

- 四维方向基 `b_i(v)`：由 view-cell 区域频谱、实例关系条件和观察方向生成；
- 八维实例关系条件 `g_i`：由实例的 `4×7` 生存系数生成；
- 八维单调生存语义：描述当前方向和深度下的生存概率、首遮挡深度和不确定度。

新增四维低秩乘性交互：

```text
c_i(v) = b_i(v) * (1 + tanh(g_i[0:4]))
```

它显式建模“实例关系状态 × 当前方向”，取值缩放保持在 `[0,2]`。主查询输入中用 `c_i(v)` 等量替换原来的四维 `b_i(v)`，其余输入不变。生存函数仍使用原始方向基查询，不改变单调遮挡语义。

训练期区域暴露头从四维 `c_i(v)` 预测 subpose 可见命中率，结构为 `4→8→1`。该窄瓶颈不能直接读取 96 维静态几何或完整 ray，因此只有关系系数和方向查询真正表达区域暴露时才能降低损失。辅助头只在训练模式执行，不导出。

该设计属于低秩双线性关系查询：重型真实遮挡关系和逐实例校准仍在离线阶段完成，浏览器只增加四次 `tanh` 和四次逐元素乘法。它不新增固定实例维度、在线邻居或第二推理分支。

## 视觉加权同实例反事实排序

对一个多 pose 训练批次内同时出现正负标签的实例，设正例 logits 为 `p_j`，视觉权重为 `w_j`，负例 logits 为 `n_k`。构造视觉权重软正尾和软负尾：

```text
positive_anchor = -T log sum_j(alpha_j exp(-p_j/T))
negative_anchor =  T log mean_k(exp(n_k/T))
alpha_j = sqrt(w_j) / sum sqrt(w_j)
```

反事实排序损失为：

```text
L_cf = softplus((margin + negative_anchor - positive_anchor) / T) * T
```

每个实例的固定偏置同时出现在正负 anchor 中并相互抵消，因此该损失只能通过视角条件化特征改善。正例和负例梯度分别记录，但两者都属于分类排序目标，不再错误地把负例一侧当作可被梯度上限清零的资源效率项。训练继续保留 S4 的跨 pose 工作目标和 calibration 冻结协议；反事实损失不读取 calibration、validation 或 test。

## 快速矩阵

四组均使用 seed `20260801`、从随机初始化训练 `8 epoch × 100 step`，四张 GPU 并行。基础超参数冻结为现有 S4：正例 BCE 份额 `0.30`，软 weighted recall 目标 `0.995`，区域暴露权重 `0.05`。反事实损失初始权重 `0.05`、margin `0.50`、temperature `0.25`。

| 变体 | 查询关系特征 | 暴露监督位置 | 同实例反事实排序 |
|---|---|---|---|
| A | 原始四维方向基 | 64 维共享隐藏层 | 关闭 |
| B | 四维低秩乘性交互 | 四维关系瓶颈 | 关闭 |
| C | 原始四维方向基 | 64 维共享隐藏层 | 开启 |
| D | 四维低秩乘性交互 | 四维关系瓶颈 | 开启 |

矩阵必须全部完成，不因中间安全门提前取消。若反事实损失量级超过主可见性损失的 `25%` 或梯度非有限，只允许按预先定义的 `{0.02, 0.05, 0.10}` 做一次小扫描，不追加结果驱动的新结构。

## 评价与保留条件

每个 checkpoint 只使用自己的 calibration split 冻结阈值，回放相同 validation pose、候选和 GT。首先要求 weighted recall `>0.99` 且单侧 95% 下界 `>0.99`；普通 recall 和 pose recall用于覆盖诊断。

必须报告：

- aggregate/pose precision、recall、weighted recall、accuracy、balanced accuracy、specificity 和 F1；
- useful cull、bad cull、平均预测数和 GLB 数量/字节削减；
- 普通/加权 ROC-AUC、AP、正例低尾、负例高尾；
- 同实例跨视点成对 AUC、平均正负分差、顺序错误实例比例；
- 区域暴露 MAE 和反事实成对数量/间隔；
- 固定表字节、查询输入维度、导出总资产与 CUDA 回放。

只有当 B 或 D 相对对应对照提高同实例跨视点 AUC，并在安全工作点改善 precision、balanced accuracy 或 useful cull，才能认为新表征有效。只降低辅助 MAE 不算有效。只有当 C 或 D 的反事实排序改善安全工作点且没有通过增加预测数量换取 weighted recall，才保留新损失。

## 前端硬约束

- 固定实例表保持 `96D geometry + 28D survival = 124D FP16`；
- HKUST 固定表保持 `4,670,088 bytes`；
- 主查询输入保持 `130D`；
- 总神经资产必须小于 `7 MiB`；
- 每个 view-cell 仍只执行一次候选批量查询；
- 不新增在线邻居查询、图传播、subpose、HZB、八角点投影或第二推理分支；
- 四维暴露解码头和反事实训练状态不得导出；
- 新乘性交互必须完成 CUDA 回放，若进入后续长训再执行硬件 WebGPU 和移动端 `10k` 候选 p95 测试。

正式结论产生前不修改默认 checkpoint、前端阈值、默认资产或部署目录。

## 执行结果

四组均按预登记配置完成 `8 epoch × 100 step` 从头训练和完整 validation 回放。C、D 在各自 calibration 冻结阈值下通过 validation weighted recall 安全门；A、B 的 calibration 正式安全阈值未通过 validation，因此没有合格正式 checkpoint。C 的 validation aggregate precision 为 `0.34395`，balanced accuracy 为 `0.75332`，useful cull 为 `0.90881`，平均预测 `370.10` 个实例，GLB 字节削减 `40.79%`。其同实例跨视角 AUC 为 `0.86142`，高于 A 的 `0.84476`。

当前四维门控关系特征没有通过保留条件。B 相对 A 只产生很小的排序变化；D 相对 C 的 precision 降至 `0.10673`，useful cull 降至 `0.64621`，平均预测增至 `1792.93`，同实例跨视角 AUC 也降至 `0.84562`。因此后续快速改进以 C 的原始 4 维方向基加反事实损失为起点，不把 `gated_contrast` 进入默认模型或前端。

运行契约已通过实际导出检查：固定实例表 `4,670,088 bytes`，查询输入 `130D`，全部神经资产 `4,748,630 bytes`（约 `4.53 MiB`）；计入实例 AABB、实例到 GLB 映射和元数据后的完整运行包为 `5,480,464 bytes`（约 `5.23 MiB`），仍低于 `7 MiB` 上限。训练期暴露头和反事实状态未导出，C 不增加浏览器在线算子。安全成员按 useful cull 和 GLB 字节削减优先排序，自动选择 C，避免用微小的 balanced accuracy 增益换取更大的前端预测与下载负担。详细指标见 [`../evaluation/pvs_direction_conditioned_counterfactual_occlusion_v1_quick8_2026-08-19.md`](../evaluation/pvs_direction_conditioned_counterfactual_occlusion_v1_quick8_2026-08-19.md)。
