# 跨视点安全工作边界与区域暴露表征快速验证计划

- 日期：2026-08-19
- 实验前缀：`pvs_cross_pose_operating_exposure_representation_v1`
- 场景：HKUST v3
- 状态：快速扫描和单种子 2x2 因子实验已完成，等待三种子正式验证
- 目的：在不增加浏览器固定表、在线查询次数或动态图计算的前提下，改善安全工作点附近困难正负实例的分离

## 当前证据与问题定义

逐位姿平衡与动态前沿损失的三种子 40 epoch 结果已经说明：模型能够把安全阈值恢复到 `0.20--0.24`，并将 aggregate weighted recall 保持在 `99.873%`，但 aggregate precision 只有 `12.04%`，平均预测 `1,921.4` 个实例。训练后期从 epoch 24 到 epoch 28 时，weighted recall 从 `99.959%` 上升到 `100.000%`，precision 从 `10.35%` 降至 `7.16%`，平均预测数从 `2,406` 增至 `3,395`。现有目标继续奖励召回超额，未在安全要求满足后把优化能力转向负例。

当前动态前沿还在每个 pose 内分别选择低分正例和高分负例，而 calibration 与前端使用一个跨 pose 统一阈值。训练前沿扩大不能证明统一边界附近的正负尾部已经分开。正式结果中正例加权 q0.5% 仍低于负例 q99.5%，说明主要问题位于全局安全边界，而非普通样本的平均排序。

表示侧已经具有 `96` 维固定几何和 `28` 维方向深度生存场。生存场由 train-only 三角形深度事件和右删失观察训练，能够表达“给定方向和深度时仍未被遮挡的概率”，但没有直接要求共享查询表征解释同一 view-cell 内实例在多少个合法 subpose 可见。最终可见性是 subpose 可见集合的并集，连续命中率能够区分稳定可见、边界偶现和全程不可见三类样本，因此可作为比二元并集标签更细的 train-only 表征监督。

## 研究假设

1. 将训练目标改成跨 pose 的统一安全工作边界，并在 weighted recall 达到目标后停止正例安全梯度，可以减少后期通过过预测换取召回超额的行为。
2. 用真实 subpose 命中率监督共享查询表征，可以使现有几何、生存场和视点区域特征更明确地编码“区域暴露稳定度”，改善边界正例与困难负例的区分。
3. 两项改动分别作用于优化目标和表征学习，应在 2x2 因子实验中单独验证；不能只比较完整组合与旧模型。

## 跨视点安全工作边界损失

对一个包含多个 pose 的训练批次，使用一个 train-only 可学习 logit 边界 `b`。候选实例 `i` 的软保留概率为：

```text
q_i = sigmoid((z_i - b) / temperature)
```

其中 `z_i` 是可见性 logit。跨 pose 合并后的软 weighted recall 和软 false-positive rate 为：

```text
R_w = sum(w_i * y_i * q_i) / sum(w_i * y_i)
F_mean = mean_pose(mean(q_i | y_i = 0, pose))
F_tail = mean(top-q(q_i | y_i = 0))
```

总可见性损失定义为：

```text
L = L_pose_asymmetric_bce
  + operating_weight * ((1-tail_mix) * F_mean + tail_mix * F_tail)
  + dual * relu(recall_target - R_w)
  + 0.5 * penalty * relu(recall_target - R_w)^2
```

基础 BCE 仍按 pose 分别归一化正负类，但正例份额是可调的 `positive_class_fraction`，不再固定为 `0.5`。对偶变量根据未截断的 `recall_target - R_w` 做投影上升；模型的正例安全惩罚只使用正部违反量。当召回高于目标时，安全项严格为零，不再因负的拉格朗日项继续奖励把召回推向 `1.0`。

`F_mean` 先在每个 pose 内计算负例软保留率再宏平均，避免候选很多的 pose 独占梯度；`F_tail` 聚焦整个多 pose 批次中统一边界附近的高分困难负例。边界只存在于训练状态，不替代 calibration 阈值，不导出到浏览器，也不读取 calibration、validation 或 test 数据。

## 区域暴露表征监督

在现有共享查询表征后增加一个仅训练时启用的小型解码头：

```text
64 维共享查询表征 -> 16 维隐藏层 -> 1 个区域暴露 logit
```

监督目标是 train split 中每个候选实例的：

```text
subpose 可见命中率 = 可见 subpose 数 / 成功 subpose 总数
```

损失使用允许 `[0,1]` 软目标的二元交叉熵，并在每个 pose 内分别归一化并集正例和全程不可见负例。正例份额可配置，默认与可见性基础损失相同。该监督要求共享表征同时区分：

- 命中率接近 `1` 的稳定可见实例；
- 只在 view-cell 边缘少量 subpose 出现的边界正例；
- 命中率为 `0` 的候选负例。

辅助头不参与最终可见性分数的后处理，不作为前端第二次分类器。它只通过训练梯度改善现有共享表征；checkpoint evaluator 为加载训练状态而保留该头，但 `eval()` 和浏览器导出均不执行、不打包该头。暴露监督进入受保护的辅助梯度组；若其梯度与主可见性安全梯度冲突，沿安全梯度方向的冲突分量必须被投影掉，避免把低命中率但合法的并集正例压成不可见。

该方法与 Neyman--Pearson 受约束分类、负例尾部风险和多任务辅助监督有关，但不宣称新的 NP 最优检验、partial-AUC 理论或多任务学习理论。可验证的项目贡献限定为：针对保守 view-cell PVS，把统一工作点负例压制和连续区域暴露监督结合到固定运行表不变的训练流程中。现有 V4 已经用命中率加权保护稀有正例，因此只有独立暴露解码相对该机制产生稳定增益时，才能作为新的表征贡献。

## 前端资产和计算约束

本实验不改变运行时 schema：

- 固定实例表继续为 `96` 维几何加 `28` 维生存场，共 `124` 维 FP16；
- HKUST 固定表继续为 `4,670,088 bytes`；
- 单实例运行查询输入继续为 `130` 维；
- 浏览器继续对一个 view-cell 执行一次候选批量查询；
- 不新增在线邻居查询、图传播、subpose 展开、HZB、AABB 八角点投影或第二个推理分支；
- train-only 暴露解码头和 train-only 工作边界不得进入 `query_weights_fp16.bin`。

快速实验必须验证导出器的运行权重布局不包含暴露头。若后续任何表示改动需要新增固定维度，必须另立实验并保证神经资产总量不超过 `7 MiB`；本轮不允许通过扩维取得结果。

## 快速实验矩阵

所有成员从随机初始化训练，使用相同 train、calibration、validation split，相同后退相机候选、GT 和 subpose sidecar。test 不读取。旧 V4 和已经完成的逐 pose 前沿矩阵不续跑。

### 损失参数扫描

使用 seed `20260801`，四张 GPU 并行训练 8 epoch：

| 成员 | 正例 BCE 份额 | 软 weighted recall 目标 |
|---|---:|---:|
| S1 | 0.20 | 0.993 |
| S2 | 0.20 | 0.995 |
| S3 | 0.30 | 0.993 |
| S4 | 0.30 | 0.995 |

固定参数：工作边界温度 `0.25`、负例尾部比例 `0.01`、尾部混合权重 `0.5`、增广惩罚 `10`、对偶初值 `1`、对偶学习率 `0.05`、最大值 `20`。扫描始终选出相对最优配置，不因未达到正式安全门取消后续矩阵。

### 2x2 因子快速验证

冻结扫描中相对最优损失配置，使用 seed `20260801`、16 epoch、每 epoch 100 step，四张 GPU 并行：

| 变体 | 跨视点工作边界 | 区域暴露监督 |
|---|---|---|
| A | 关闭，使用当前逐 pose 前沿损失 | 关闭 |
| B | 开启 | 关闭 |
| C | 关闭 | 开启 |
| D | 开启 | 开启 |

逐实例生存残差采用固定绝对调度语义：前 `400` optimizer step 关闭，随后 `800` step 线性开启。16 epoch runner 通过相应 fraction 生成相同绝对 step，后续长训也保持 `400/800`，避免 pilot 与长训动力学不同。

## 评价和选择

每个 checkpoint 只在自己的 calibration split 冻结阈值，并回放全部 validation pose。安全主门为 calibration weighted recall 严格大于 `0.99` 且单侧 95% 下界严格大于 `0.99`；validation weighted recall 及下界必须同步报告。普通 pose recall 仅作诊断。

必须报告：

- pose 和 aggregate precision、recall、F1、accuracy、balanced accuracy、specificity；
- pose 和 aggregate weighted recall 及其单侧下界；
- useful cull、bad cull、平均预测实例数、预测/候选和预测/GT；
- 正例加权 q0.5%、q1%，负例 q99%、q99.5%，以及尾部间隔；
- 固定表字节、导出查询权重字节、CUDA 回放耗时；
- 暴露头训练误差及 stable/boundary/invisible 三组误差；
- 逐 pose FPR 宏平均、最差 5% pose FPR、训练边界与 calibration logit 阈值差。

快速阶段不做论文显著性结论。相对最优配置必须首先守住 weighted recall 安全门；安全成员内部依次比较 balanced accuracy、precision、useful cull、accuracy 和平均预测数。若没有安全成员，仍按 weighted recall 下界、weighted recall、balanced accuracy、precision 的顺序选择诊断最优，并如实标记不安全。

若完整组合相对 A 在相同安全口径下提高 precision 或 balanced accuracy，且预测数量下降，同时运行资产布局保持不变，则进入三种子长训；否则分别根据 B-A 和 C-A 判断是损失还是表征无效，不追加无依据的模型复杂度。

## 预期代码范围

- 新增跨视点安全工作边界损失及单元测试；
- 在模型中新增可选的 train-only 区域暴露解码头；
- 在训练器中接入新 loss variant、暴露软标签监督、对偶状态和指标；
- 在 evaluator 中按 checkpoint config 重建训练头，但评价前向不执行该头；
- 新增独立四 GPU runner 和 runner 测试；
- 更新本文件执行状态并新增正式快速验证报告。

本实验不得修改默认 checkpoint、前端阈值、前端资产或部署目录。

## 快速执行结果

四组 8 epoch 扫描完成后冻结 S4（正例 BCE 份额 `0.30`，软 weighted recall 目标 `0.995`）。S4 在 validation 的 aggregate weighted recall 为 `0.992100`，单侧 95% 下界为 `0.990093`，precision 为 `0.283261`，balanced accuracy 为 `0.740600`，平均预测 `440.77` 个实例。阈值仅由该 checkpoint 的 calibration split 冻结为 `0.15`。

单种子 16 epoch 因子实验显示，两项机制存在明显交互。跨视点边界损失单独使用时退化为接近全预测；区域暴露监督单独使用时也明显过预测。二者组合后，aggregate precision 为 `0.287144`，ordinary recall 为 `0.753387`，weighted recall 为 `0.996759`，其单侧下界为 `0.995717`，balanced accuracy 为 `0.833379`，useful cull 为 `0.872938`，平均预测 `600.98` 个实例。相对旧逐 pose 前沿且无暴露监督的 A 组，组合 D 的 precision 提高 `8.09` 个百分点，balanced accuracy 提高 `7.45` 个百分点，useful cull 提高 `2.46` 个百分点，平均预测数减少 `98.89`。

当前证据只能说明组合训练机制在单种子快速实验有效，不能证明两个模块分别具有稳定独立贡献。训练边界 logit 与 calibration 阈值 logit 仍相差约 `2.50`，因此也不能宣称工作边界已经与部署阈值对齐。普通 recall 与 pose recall 继续作为细小和低权重实例覆盖诊断报告，不能被较高 weighted recall 隐藏。

运行契约保持不变：四个变体均为 `124D` FP16 固定表、`4,670,088 bytes`、`130D` 查询输入。D 组全 validation CUDA 回放为 `2.014 s`，A 组为 `2.118 s`，未观察到推理退化。D 组实际导出总神经资产为 `4,748,630 bytes`（约 `4.53 MiB`），低于 `7 MiB` 上限；其中训练期暴露头和工作边界均未导出。图像级 Color-ID 指标和真实移动端 WebGPU p95 尚未执行，不能据此宣称移动端性能已经完成验证。
