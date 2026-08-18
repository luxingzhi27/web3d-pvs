# 108 维视点区域尾部分离探针与长训消融计划

日期：2026-08-19

## 研究目的

当前模型在总体排序上已经具有较好的区分能力，但在安全工作点所依赖的极端尾部仍存在交叠：少量重要可见实例分数很低，同时少量不可见实例分数很高。为判断该问题是否来自线性表达能力不足，本轮固定使用前端查询路径已经具备的 108 维视点区域特征，对比线性双探针、分段线性探针和单隐藏层小型多层感知机。目标是在 calibration 的 `weighted recall > 0.99` 且单侧 95% 置信下界大于 `0.99` 的前提下，提高 validation precision，并同步考察 accuracy、balanced accuracy、specificity、useful cull、bad cull、平均预测实例数和 GLB 字节。

本轮不修改候选集合、真实可见集合、view-cell 语义、默认 checkpoint、前端阈值或前端资产。所有探针参数和风险证书只由 train split 拟合；calibration 只冻结当前 checkpoint 的运行阈值；validation 只用于最终评价；test split 不读取。

## 108 维输入与运行边界

探针输入由四部分组成：9 维中心视线查询、64 维视点区域 Fourier 统计、8 维旧边界摘要，以及 27 维区域上下界和跨度，共 108 维。这些量已经存在于当前视点区域查询链路中，不增加逐实例特征表，也不要求前端执行邻居搜索、图传播、多 subpose 查询或几何光栅化。

三个候选探针为：

| 方案 | 形式 | 共享参数量 | 作用方式 |
|---|---:|---:|---|
| 固定线性双探针 | 标准化后的岭线性模型 | 每支 109 | 两支 train-only 探针分别保护高视觉效用低分正例和普通低分正例 |
| 分段线性探针 | 每维加入 `-1/0/1` 三个 ReLU 折点的加性岭模型 | 每支 433 | 表达单变量非线性尾部，不建模维度间交互 |
| 小型多层感知机 | `108 -> 8 -> 1`、ReLU | 每支 881 | 用低容量共享网络表达特征交互和非线性尾部边界 |

探针仅签发非负分数救援，不压低候选分数。两支救援取逐候选最大值，保持“先保护困难正例，再由较高 calibration 阈值剔除更多负例”的语义。该后验增加的是不到两千个共享参数，不增加随场景实例数量增长的资产。

## 已完成的快速迭代

快速迭代使用 seed `20260801` 的第 24 epoch 主模型快照，新增有界尾部残差分支并训练 4 epoch、每 epoch 100 step。该 checkpoint 只用于机制筛选，不作为最终论文结果。

困难尾部 validation ROC-AUC 为：

| 探针 | 参数量 | ROC-AUC |
|---|---:|---:|
| 现有 108 维线性探针 | 109 | 约 0.878 |
| 分段线性、普通正例覆盖 | 433 | 0.887878 |
| 分段线性、高视觉效用保护 | 433 | 0.884028 |
| 小型多层感知机、普通正例覆盖 | 881 | 0.894244 |
| 小型多层感知机、高视觉效用保护 | 881 | 0.899990 |

完整工作点扫描显示，单支困难尾部 ROC-AUC 最高的组合并不等于最终安全工作点最优组合。当前最优方案使用：

- 高视觉效用正例保护：分段线性探针，视觉权重幂为 `0.5`；
- 普通正例覆盖：8 隐层单元的小型多层感知机，视觉权重幂为 `0`；
- 只允许非负救援，不进行分数抑制；
- 最大救援 logit 为 `0.5`，温度为 `0.05`，覆盖分支权重为 `1.0`；
- 两支 train-only 风险上限均为 `0.05`。

该方案在 10,000 次 bootstrap 的单种子 validation 回放中得到：

| 指标 | 固定线性双探针 | 分段线性 + 小型 MLP |
|---|---:|---:|
| calibration weighted recall / LCB | 未改变既有安全结论 | 0.993638 / 0.990267 |
| validation weighted recall / LCB | 0.993611 / 0.991295 | 0.992864 / 0.990396 |
| aggregate precision | 0.463048 | 0.489821 |
| aggregate accuracy | 0.951316 | 0.954591 |
| aggregate balanced accuracy | 0.795792 | 0.795228 |
| useful cull | 0.923642 | 0.927129 |
| bad cull | 0.016593 | 0.016804 |
| 平均预测实例数 | 309.25 | 290.11 |
| GLB 字节削减 | 0.443554 | 0.466992 |

非线性组合相对固定线性双探针将 aggregate precision 提高约 2.68 个百分点，同时改善 accuracy、useful cull、预测数量和 GLB 字节。balanced accuracy 基本持平，weighted recall 和 bad cull 略有退化，但仍通过已登记安全门。因此它被选为长训后验候选；当前结果不能替代三种子长训结论。

## 长训设计

### 基础模型训练

对 `20260801`、`20260802`、`20260803` 三个随机种子分别使用其自身完整模型的第 24 epoch checkpoint 作为起点。基础模型和离线生存场保持冻结，只训练 4,889 参数的有界尾部残差分支。每个成员执行 80 epoch、每 epoch 100 step，学习率 `5e-4`，每 4 epoch 完整评价一次；不得因中间 precision、recall 或安全门未通过而提前取消。

训练配置固定为快速迭代当前相对最优值：按 pose 零均值残差、平方根视觉权重、跨 pose 配对权重 `0.4`、批次全局尾部配对权重 `0.25`、最大绝对残差 `0.5`。三个种子分别使用 GPU 0、1、2；GPU 3 在 checkpoint 完成后执行 score capture、探针拟合和回放。

这三次基础训练是唯一需要重复的神经网络训练。后验探针不属于基础优化器，因此不为“无探针、线性探针、非线性探针”重复训练相同的基础模型。

### 同 checkpoint 后验消融

每个种子的最终选定 checkpoint 均生成 train、calibration 和 validation score capture，然后在完全相同的候选、GT 和基础分数上评价：

| 消融成员 | 高视觉效用保护 | 普通正例覆盖 |
|---|---|---|
| 无后验 | 关闭 | 关闭 |
| 固定线性双探针 | 线性 | 线性 |
| 最优非线性组合 | 分段线性 | 8 单元小型多层感知机 |

每个 checkpoint 的探针只在自身 train capture 上拟合；风险证书只读 train；运行阈值只由自身 calibration capture 冻结。validation 不参与探针选择或阈值选择。三个后验成员均使用同一个 checkpoint，因此差异可归因于尾部分离后验，而非基础训练随机差异。

## 评价与统计

每个成员必须报告 pose 宏平均和 aggregate 指标：普通 recall、weighted recall 及其单侧 95% 下界、precision、F1、accuracy、balanced accuracy、specificity、useful cull、bad cull、平均预测实例数、预测/候选比例、预测/GT 比例、GLB 数量削减和 GLB 字节削减。图像 Color-ID 评价可用时补充 miss-pixel、wrong-ID pixel 和 extra-pixel；缺失时必须明确标记未回填。

正式安全工作点要求 calibration weighted recall 严格大于 `0.99` 且单侧 95% 下界严格大于 `0.99`。普通 pose recall 用于诊断，不作为替代安全门。满足安全门后，主要比较 precision、accuracy、balanced accuracy、useful cull、bad cull、平均预测数量和 GLB 字节。

最终比较按 seed 聚类，并在 seed 内对相同 pose 重采样，执行 10,000 次 paired bootstrap。至少报告非线性组合减无后验、非线性组合减线性双探针的指标差值、95% 置信区间、方向及是否跨零。

## 完成条件

1. 三个随机种子的 80 epoch 训练全部完成并保留完整日志和训练曲线。
2. 每个种子的 train、calibration、validation capture 完成，且 `testRead=false`。
3. 每个 checkpoint 完成无后验、固定线性双探针和最优非线性组合回放。
4. 完成三种子 10,000 次 paired bootstrap 与结果 schema 校验。
5. 更新正式评价报告和文档索引，并提交代码、测试、计划和可跟踪的小型汇总结果。

在上述条件完成前，不修改默认模型、默认前端阈值或前端资产。

## 执行结果

截至 2026-08-19，本计划已经完整执行：三个随机种子均完成 80 epoch，每个种子均生成 train、calibration、validation capture，三个后验成员使用同 checkpoint 回放，validation 每种子 213 个 pose，并完成 10,000 次按 seed 聚类、seed 内同 pose 配对 bootstrap。所有输出均标记 `testRead=false`，候选集合和真实可见集合未改变。

快速迭代选出的分段线性与小型多层感知机组合在三种子合并结果中优于固定线性双探针，但没有稳定优于无后验长训基础模型。非线性组合相对无后验的 aggregate precision、accuracy、balanced accuracy 和 useful cull 分别变化 `-0.714`、`-1.213`、`-0.517` 和 `-1.224` 个百分点，95% 置信区间均跨零。固定线性双探针相对无后验使 balanced accuracy 显著下降 `2.178` 个百分点，95% 置信区间为 `[-4.967, -0.505]` 个百分点。

因此本轮路线决定为：不晋级线性或非线性尾部分离后验，当前相对最优成员为无后验长训基础模型；默认 checkpoint、阈值和前端资产保持不变。完整结果、逐种子差异、统计区间及未完成的图像/浏览器评价见 [`../evaluation/pvs_train_owned_nonlinear_tail_posterior_longtrain_v1_2026-08-19.md`](../evaluation/pvs_train_owned_nonlinear_tail_posterior_longtrain_v1_2026-08-19.md)。
