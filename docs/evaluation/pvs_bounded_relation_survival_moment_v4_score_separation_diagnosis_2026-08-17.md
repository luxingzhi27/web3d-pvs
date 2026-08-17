# v4 可见性分数分离问题诊断

- 日期：2026-08-17
- 实验：`pvs_bounded_relation_prior_instance_calibrated_moment_safety_reserve_v4_formal80`
- 场景：HKUST
- 目的：记录当前完整模型的实际结果，以及安全阈值下误报较多、best-F1 阈值下召回骤降的原因
- 本文性质：结果与问题诊断，不修改默认 checkpoint、前端资产或训练代码

## 评价口径

本轮使用完整模型的 3 个随机种子，每个成员训练 80 epoch。阈值只从该成员自己的 calibration split 冻结，再回放 validation split；validation 包含 213 个 pose，平均每个 pose 有 5,174.4 个后退相机候选和 229.1 个真实可见实例。test split 未用于阈值选择，也未在本次诊断中读取。

安全工作点表中的 `recall`、`precision`、`accuracy`、`balanced accuracy`、`specificity`、`useful cull` 和 `bad cull` 均为 validation 上的 aggregate 指标。`weighted recall` 使用可见性权重衡量重要实例是否被保留；`useful cull` 是正确剔除不可见候选的比例，`bad cull` 是错误剔除真实可见候选的比例。

当前 validation JSON 没有生成 weighted recall 的置信下界，图像级 Color-ID 指标也尚未回填。因此下面的 weighted recall 是点估计，不能写成已经通过正式置信安全门的结论。

## 安全工作点结果

这些阈值由各自 calibration split 冻结，代表当前协议下优先保证画面安全的工作点。

| seed | checkpoint epoch | 阈值 | aggregate recall | weighted recall | precision | accuracy | balanced accuracy | specificity | useful cull | bad cull | 平均预测实例数 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 20260801 | 24 | 0.00048697 | 74.81% | 99.711% | 31.28% | 91.61% | 83.60% | 92.39% | 88.30% | 1.115% | 547.7 |
| 20260802 | 32 | 0.00023714 | 99.775% | 99.988% | 7.80% | 47.76% | 72.56% | 45.35% | 43.35% | 0.010% | 2,931.0 |
| 20260803 | 24 | 0.00100000 | 90.231% | 99.872% | 18.17% | 81.58% | 85.70% | 81.17% | 77.58% | 0.432% | 1,137.7 |
| 三 seed 平均 | - | - | 88.271% | 99.857% | 19.08% | 73.65% | 80.62% | 72.97% | 69.74% | 0.519% | 1,538.8 |

平均候选数为 5,174.4，平均 GT 数为 229.1，正例比例只有 4.43%。安全工作点平均预测比例为 29.74%，约为 GT 数量的 6.72 倍。seed 20260802 尤其明显：它的 weighted recall 为 99.988%，但 specificity 只有 45.35%，平均预测 2,931 个实例，accuracy 下降到 47.76%。

这说明当前模型可以通过降低阈值保护重要可见实例，却不能稳定地把不可见候选压到低分。weighted recall 达标本身不能证明正负实例已经分离。

## best-F1 诊断工作点

best-F1 阈值只用于观察分数分布和分类权衡，不是当前安全工作点，也不能直接用于前端。已有 validation 回放如下：

| seed | best-F1 阈值 | aggregate recall | weighted recall | precision | accuracy | balanced accuracy | useful cull | 平均预测实例数 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 20260801 | 0.86 | 2.71% | 78.91% | 58.98% | 95.61% | 51.31% | 95.49% | 10.5 |
| 20260802 | 0.42 | 7.74% | 85.37% | 61.14% | 95.70% | 53.76% | 95.36% | 29.0 |
| 20260803 | 0.80 | 1.81% | 72.85% | 73.04% | 95.62% | 50.89% | 95.54% | 5.7 |
| 平均 | - | 4.09% | 79.04% | 64.39% | 95.64% | 51.99% | 95.46% | 15.1 |

best-F1 点的 pose 宏平均 recall 约为 51.57%，pose 宏平均 weighted recall 约为 83.52%。从安全阈值切换到 best-F1 阈值后，precision 和普通 accuracy 看起来变好，主要原因是预测数量骤降；同时真实可见实例大量被漏掉，weighted recall 从约 99.86% 降到约 79.04%。因此这不是模型分离能力已经良好的证据，而是阈值移动带来的漏检换取误报下降。

## 问题判定

当前问题是正负实例的分数分离能力不足，并伴随明显的分数尺度和随机种子不稳定性。它不只是“安全阈值选得太低”。如果只是统一的概率偏置，三个 seed 在同一协议下应主要表现为阈值平移；实际结果同时出现了预测规模、specificity、accuracy 和 recall 的大幅变化，说明正负分数尾部存在较强重叠，且模型输出尺度没有稳定校准。

### 1. 安全目标明显强于负例抑制

当前 v4 的安全项沿用了 RVL 强召回控制，并加入了正例低分尾部保护。训练配置中：

- 正例二元交叉熵权重为 14；
- Tversky 项的漏检权重为 7；
- 正例边界尾部保护继续提高低分正例的代价；
- 负例阈值带权重为 0.06，GLB 无需求资源项权重为 0.03；
- 效率项在前 10% optimizer step 内关闭，之后还要经过安全裕度门控，并受效率梯度上限 0.25 限制。

这种组合优先回答“不能漏掉可见实例”。当正例只占候选的约 4.43% 时，负例方向的梯度相对不足，模型容易把大量负例保留在中低分区域。阈值越低，更多负例被判为可见，最终表现为 precision、specificity 和 useful cull 同时下降。

### 2. RVL 的 pose 计数约束不等于逐实例分类分离

RVL 中的 `lossCount` 约束一个 pose 内所有候选概率之和接近 GT 数量。它约束的是总概率质量，不强制每个正例得到高分、每个负例得到低分。

例如一个 pose 有约 5,174 个候选和 229 个 GT。若大量负例各自输出很小但非零的概率，所有候选概率之和仍可能接近 229。连续损失可以认为这种结果满足数量约束，但在离散阈值判断中，阈值若落到 `0.0002--0.001`，这些负例会大批转化为 FP。当前实现因此可能出现“总数损失正常、weighted recall 很高、逐实例 accuracy 和 balanced accuracy 很低”的情况。

### 3. 输出分数不是自然概率边界

加权 BCE、Tversky、RVL 漏检保护和 pose 计数项共同训练出的 sigmoid 输出是代价敏感分数，不是经过概率校准的后验概率。`0.5` 没有天然分类含义，安全阈值也不能要求接近 `0.5`。

当前安全阈值分布在 `0.000237--0.001`，best-F1 阈值却在 `0.42--0.86`。这组差异说明工作点附近的分数分布很陡，或者正负尾部重叠且种子间发生了明显漂移。后续必须直接查看正负分数分位数、logit 间隔、阈值扰动曲线和校准误差，不能只看单个阈值下的结果。

### 4. view-cell 并集标签扩大了正例边界

view-cell 的标签语义是保守并集：只要一个合法 subpose 能看到实例，该实例在整个 view-cell 查询中就标为正例。边缘位置才可见的实例因此与大量“当前中心位置不可见”的候选共享同一个正例标签。

这个定义符合前端一次查询覆盖整个 view-cell 的安全要求，但会让实例级分类边界变宽。模型若只看到 view-cell 中心和压缩后的区域特征，就难以从固定特征中恢复每个边缘实例的最坏位置可见性。结果会表现为正负分数重叠，以及为了覆盖边缘正例而保留更多负例。

### 5. 生存关系和可见性头之间没有形成强制分离路径

当前模型离线生成共享关系先验并叠加逐实例校准残差，生存损失和关系一致性损失作为辅助目标进入共享训练过程。最终实例可见性仍由共享 trunk 后的可见性头输出。

因此，生存场学得好并不自动意味着可见性分数会形成更清晰的正负边界。可见性头可以主要依靠几何特征、视角特征和偏置完成高召回，而弱化关系表示；辅助损失的数值下降也不能直接证明遮挡关系改善了二分类分离。

### 6. 数据关系监督存在可表达性限制

当前关系层级和遮挡观察经过离线压缩，真实关系边仍然存在长尾置信度和方向覆盖不均衡。遮挡事件、右删失可见观察以及 view-cell 可见并集共同作用时，生存监督与最终 visibility 标签不是完全同一个目标。

这会使生存头倾向于学习“整体遮挡趋势”，却不能为每个候选实例提供足够强的逐实例负例证据。逐实例校准残差改善了共享先验的表达能力，但它没有解决可见性损失对负例尾部约束不足的问题。

## 当前结论

1. v4 在三 seed 的 validation 点估计上能维持约 99.857% weighted recall，但 seed 间效率和分类质量差异很大；由于 weighted recall 置信下界为空，当前结果没有正式安全认证。
2. 在安全阈值下，平均 precision 19.08%、accuracy 73.65%、balanced accuracy 80.62%、useful cull 69.74%，说明高召回是以大量 FP 和较高前端预测量换来的。
3. best-F1 阈值能显著减少预测实例并提高 precision，但 weighted recall 只有约 79.04%，不能作为画面安全工作点。
4. 当前首要缺陷是正负分数分离和校准稳定性，而不是单纯阈值数值。损失函数对漏检的强不对称、pose 级总概率约束、较晚且较弱的负例/资源抑制、view-cell 并集标签以及共享可见性头共同造成了这一结果。
5. 当前结果可以用于定位问题，不能据此修改默认前端模型，也不能把生存场、视点积分或安全裕度损失写成已经被正式验证的论文贡献。

## 证据位置

- 训练配置与成员摘要：`neural_instance_culling/model/out/pvs_bounded_relation_prior_instance_calibrated_moment_safety_reserve_v4_formal80/`
- validation 回放：`neural_instance_culling/benchmark/out/pvs_bounded_relation_prior_instance_calibrated_moment_safety_reserve_v4_formal_validation/`
- 安全裕度损失：`neural_instance_culling/model/common/safety_reserve_operating_utility_loss.py`
- RVL 可见性损失：`neural_instance_culling/model/common/safety_constraint_utility_loss.py`
- v4 训练入口：`neural_instance_culling/model/train_bounded_relation_survival_moment_safety.py`
- view-cell 并集标签构建：`neural_instance_culling/model/pose_csr_dataset.py`

