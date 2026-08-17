# v4 可见性分数分离问题诊断

- 日期：2026-08-17
- 实验：`pvs_bounded_relation_prior_instance_calibrated_moment_safety_reserve_v4_formal80`
- 场景：HKUST
- 目的：记录当前完整模型的实际结果，并通过独立诊断分支定位安全阈值下误报较多、best-F1 阈值下召回骤降的原因
- 本文性质：正式结果审计与单种子短程原因诊断；不修改默认 checkpoint、前端资产或部署路径
- 诊断基线：Git 提交 `ca8bb33`；独立 worktree 分支 `research/v4-loss-diagnosis`

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

## 现有完整模型的分数尾部

在不重新训练的前提下，本次直接回放三个 80 epoch checkpoint，并补充普通接收者操作特征曲线下面积（Receiver Operating Characteristic Area Under Curve，ROC-AUC）、平均精确率（Average Precision，AP）及极端分位数。正例分位数按 `visible_weights` 加权，负例按实例等权。

| seed | ROC-AUC | AP | 正例加权 q1% | 负例 q99% | q1%-q99% | weighted recall | precision | balanced accuracy | 平均预测数 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 20260801 | 93.10% | 47.19% | 0.001852 | 0.005595 | -0.003743 | 99.711% | 31.28% | 83.60% | 547.7 |
| 20260802 | 95.28% | 49.93% | 0.002354 | 0.060815 | -0.058461 | 99.988% | 7.80% | 72.56% | 2,931.0 |
| 20260803 | 93.43% | 46.53% | 0.001721 | 0.003960 | -0.002239 | 99.872% | 18.17% | 85.70% | 1,137.7 |

三个成员的整体排序能力并不低，普通 ROC-AUC 均超过 0.93。真正的问题集中在安全工作点依赖的极端尾部：三个成员的正例 q1% 均低于负例 q99%，seed 20260802 的负例 q99% 达到 0.0608，q99.5% 达到 0.1457。该成员虽然具有最高的整体 ROC-AUC，却需要保留 2,931 个实例才能满足 weighted recall 点估计，说明整体排序指标不能替代安全尾部评价。

这一证据修正了“模型整体完全没有学到正负差异”的表述。模型已经学到大部分样本的排序，但少量重要低分正例和高分负例重叠；候选中负例超过一百万时，1% 的负例尾部已经足以产生大量 FP。

## 损失训练动力学审计

当前 RVL 核心在每个 pose 内同时计算正例加权二元交叉熵、Tversky、预测总量、排序及软 FP/FN。正式三 seed 的第 1 epoch 中：

- `lossCount` 为 109.8--127.7；
- 按 GT 数归一化的 `lossRvlFp` 为 110.7--128.6；
- 二者远高于同期约 0.96--1.08 的 BCE 和约 0.96 的 Tversky；
- 到第 4 epoch，数量项和 FP 项已经快速下降到 0.15--0.70，而 BCE 上升到约 2.95--3.21。

这表明训练首先压低全体候选的总概率质量，再尝试恢复正例分数。以平均 5,174 个候选、229 个 GT 为例，`sum(p)/GT=1` 对应候选平均分数约为 0.044。该数量约束控制全局概率预算，不保证低分质量集中在负例上。

正式训练前 800 个 optimizer step 完全关闭效率梯度。到第 80 epoch，安全梯度范数仍为 5.24--6.99，投影并截断后的效率梯度只有 0.232--0.302，即安全梯度的约 3.7%--5.0%；负例工作带损失仍为 0.625--0.660。安全门后期已经开启到 0.87--0.90，因而后期效率不足不能仅归因于门永久关闭。

### 负例工作带的梯度饱和

当前负例工作带把候选 logit 与采样阈值 logit 的差除以温度 0.1，再通过 sigmoid 得到软保留率。初始化时实例分数约为 0.5，即 logit 为 0。即使使用训练范围内最大的阈值 0.2，归一化差值也为：

```text
(logit(0.5) - logit(0.2)) / 0.1 = 13.86
```

此时 sigmoid 已约为 0.999999；更低阈值的饱和更严重。因此该项在训练初期呈现“损失接近 1、梯度接近 0”的状态。GLB 请求概率使用相同类型的低温 sigmoid，也存在相同风险。短训实测中，提前打开并提高三倍权重后的效率梯度仍约为 `10^-10`，证明权重数值并不是初始失效的主要原因。

诊断分支增加了不改变默认行为的 softplus 形式。它将第 1 epoch 的效率梯度从约 `10^-10` 恢复到 0.085；三倍尾部权重时达到 0.256。该结果确认了饱和问题，但 4 epoch 短训仍保留了约 5,130 个实例，普通 ROC-AUC 只有 0.636--0.646，尚未形成可用修复。恢复梯度是必要条件，不代表当前权重和主辅目标组合已经正确。

## 三项创新的整体作用

代码审计未发现关系系数、实例残差或视点区域矩包络在最终可见性头前被错误 detach、漏接或替换。固定的 96 维几何特征与 28 维逐实例生存系数共同进入运行时表；生存系数继续生成方向基、单调生存语义和关系条件，矩包络频谱进入边界摘要，最终共同输入可见性头。

已有三 seed 正式消融的均值如下。它们共享当前 RVL 核心，因此只能说明相对贡献，不能证明绝对方案已经合格。

| 变体 | weighted recall | precision | accuracy | balanced accuracy | useful cull | 平均预测数 |
|---|---:|---:|---:|---:|---:|---:|
| 完整组合 | 99.857% | 19.08% | 73.65% | 80.62% | 69.74% | 1,538.8 |
| 去除安全裕度效用项 | 99.832% | 13.25% | 68.86% | 75.47% | 65.20% | 1,761.4 |
| 去除分层遮挡关系 | 99.770% | 10.04% | 49.87% | 71.43% | 45.66% | 2,800.7 |
| 去除逐实例校准残差 | 99.934% | 13.11% | 56.68% | 73.26% | 52.63% | 2,431.5 |
| 去除视点区域矩包络 | 99.673% | 20.47% | 83.86% | 76.56% | 80.83% | 920.1 |

分层关系和逐实例残差明显减少预测量并改善分类指标；矩包络提高 weighted recall 和 balanced accuracy，但以更多预测为代价；新增安全裕度项相对关闭版本也有正向均值。因而当前证据不支持“三项创新都没有被模型使用”或“新增安全裕度项单独造成退化”。更合理的解释是：三个模块提供了相对增益，但最终共享的可见性损失没有稳定处理极端正负尾部，限制了全部模块的绝对效果。

## 单种子最小原因实验

所有快速实验固定 seed 20260801、相同训练/校准/validation pose、相同候选和 GT，test 未读取。第一轮为 8 epoch×50 step，第二轮为 6 epoch×50 step，第三轮为 4 epoch×50 step；训练时只用 64 个 calibration pose 加速 checkpoint 选择，结果再回放全部 213 个 validation pose。它们仅用于因果定位，不能视为正式安全实验。

### RVL 基础项

| 对照 | weighted recall | precision | accuracy | balanced accuracy | useful cull | 平均预测数 | ROC-AUC | q1%-q99% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 原损失对照 | 97.594% | 12.68% | 87.21% | 60.93% | 85.79% | 579.5 | 73.98% | -0.001790 |
| 去除数量项 | 99.172% | 7.22% | 59.53% | 63.92% | 56.48% | 2,180.2 | 68.66% | -0.002458 |
| FP 改为按负例数归一化 | 96.458% | 9.72% | 84.32% | 58.73% | 82.96% | 722.7 | 68.48% | -0.003453 |
| 去除数量项并提前效率项 | 99.378% | 7.26% | 55.83% | 65.59% | 52.45% | 2,406.1 | 69.95% | -0.002642 |

删除数量项会提高召回并显著增加预测数，同时降低整体排序。将 FP 改为按负例数归一化会把该项缩小约一个候选正负比例量级，同样没有改善排序。因此预测总量和原 FP 项确实造成低分尺度，但它们也是现有模型的主要剔除压力，不能直接删除。

### 尾部作用路径

| 对照 | weighted recall | precision | balanced accuracy | useful cull | 平均预测数 | ROC-AUC | q1%-q99% |
|---|---:|---:|---:|---:|---:|---:|---:|
| 三倍尾部权重、取消预热、梯度上限 1.0 | 99.319% | 12.75% | 62.67% | 84.34% | 666.3 | 70.79% | -0.002865 |
| 尾部项直接并入主目标 | 98.545% | 11.70% | 61.64% | 83.62% | 700.6 | 71.32% | -0.002777 |
| 三倍尾部项直接并入主目标 | 96.132% | 9.36% | 58.28% | 82.71% | 734.5 | 66.24% | -0.002115 |
| 排序聚焦最高 32 个负例并加倍权重 | 98.462% | 5.29% | 57.33% | 27.16% | 3,737.8 | 63.72% | -0.002612 |

单纯增加权重、取消预热、取消门控投影或缩小排序负例集合均未改善整体排序。聚焦 32 个负例尤其容易过拟合少量极端样本并破坏其余负例排序。结合第 1 epoch 的梯度记录，当前首要问题是工作带形状的饱和和主辅目标的尺度冲突，而不是少量系数没有调大。

## 原因分级

### 已确认

1. 没有发现标签反转、阈值/logit 单位混用、候选并集补 GT、关系特征漏接或矩包络漏接等确定性实现错误。
2. 当前模型具有较好的全局排序能力，但安全工作点依赖的正例最低尾部与负例最高尾部稳定交叠；seed 20260802 的极端负例尾部异常严重。
3. 数量项与按 GT 归一化 FP 项造成明显的早期全局降分，但直接删除会恶化排序和剔除效率。
4. 负例工作带和 GLB 请求 sigmoid 在初始分数范围内饱和，导致“高损失、近零梯度”；后期效率梯度仍只有安全梯度的约 4%。
5. 当前排名项优化平均 top-256 负例与全部正例的 softplus，能改善中央分布，却没有直接约束正例 q0.5%/q1% 与负例 q99%/q99.5% 的间隔。
6. 三个 seed 还使用各自的训练 seed 生成工作阈值序列，因此随机种子同时改变参数初始化和负例工作带轨迹，放大了成员间方差。

### 高可信但仍需正式实验确认

1. View-cell 合法 subpose 的可见并集扩大了正例边界，极少数边缘可见实例形成了低分正例尾部；该语义正确，不能改标签规避，只能改进表征和尾部目标。
2. 逐实例生存残差在训练 10% 后启用并在约 30% 时达到全幅，可能放大不同 seed 的分数尺度漂移，但现有正式消融显示残差总体有益，不能据此删除。
3. 关系和生存辅助损失改善了离线表示，却没有强制最终可见性 logit 服从遮挡生存语义；最终头仍可能用几何和视角捷径满足高召回。

### 已被本轮排除

1. 低安全阈值仅由统一 bias 或温度缩放造成；极端尾部间隔和平均预测量均显著随 seed 改变。
2. `lossCount` 是唯一根因；去除后全局排序和剔除效率更差。
3. 新增安全裕度效用项是唯一根因；正式关闭该项的模型均值更差。
4. 简单提高尾部权重、提前启用、移入主目标或缩小排序 top-k 即可修复。

## 后续修复方向

下一版损失不应继续围绕固定低阈值的饱和 sigmoid 调权重。应先采用与分数尺度无关的可微尾部间隔，直接约束每个 pose 或多 pose 批次中的“重要正例低分尾部”高于“负例高分尾部”，并分别按正负类归一化。候选形式为正例加权 q0.5%/q1% 与负例 q99%/q99.5% 的平滑条件风险价值差；其目标与安全工作点的实际失败模式一致。

原 RVL 的数量和 FP 项暂时保留为弱预算正则，并采用分阶段权重，避免初始化时主导全部梯度。待尾部间隔形成后，再启用非饱和的资源/工作区目标。工作阈值采样序列应与模型初始化 seed 解耦，使三 seed 只反映参数和批次随机性。训练日志必须继续记录普通 ROC-AUC、AP、正例 q0.5%/q1%、负例 q99%/q99.5%、尾部间隔、效率/安全梯度比及阈值扰动曲线。

上述修复必须先做同 seed 短训对照，再按完整三 seed 长训验证。当前 softplus 诊断只证明梯度饱和可以被消除，其短训分类结果没有改善，不能直接作为下一版默认损失。

## 当前结论

1. v4 在三 seed 的 validation 点估计上维持约 99.857% weighted recall，但效率和分类质量不稳定；validation weighted recall 置信下界与图像指标仍未回填，因此没有正式安全认证。
2. 模型的全局 ROC-AUC 为 0.931--0.953，说明整体表征并非完全失效；核心失败集中在安全工作点所依赖的极端正负尾部。
3. 分层关系、逐实例生存残差、视点区域矩包络和新增安全裕度项在现有消融中均表现出相对作用，但共享的 RVL 核心及尾部作用路径限制了绝对效果。
4. 损失函数是当前最有证据支持的主要瓶颈：早期概率预算项主导、固定低阈值 sigmoid 饱和、后期效率梯度过弱，以及平均排序项未对齐极端尾部共同造成低阈值和大量 FP。
5. 三轮最小实验没有产生可晋级模型。默认 checkpoint、前端阈值和资产保持不变；后续应重构尺度无关的尾部间隔目标，再进行正式长训。

## 本轮代码与复现入口

诊断分支修改了 RVL 参数化、负例工作带形状、分数尾部汇总和 v4 回放输出；所有新增参数均保持正式版本原默认值，未修改主分支和前端。涉及文件为：

- `neural_instance_culling/model/common/safety_constraint_utility_loss.py`
- `neural_instance_culling/model/common/safety_reserve_operating_utility_loss.py`
- `neural_instance_culling/model/train_bounded_relation_survival_moment_safety.py`
- `neural_instance_culling/model/current_pvs_utils.py`
- `neural_instance_culling/benchmark/evaluate_pvs_bounded_relation_survival_moment_v4.py`
- 对应的 common、trainer 和 evaluator unittest

训练入口统一采用下面的 Linux/CUDA 形式；三轮只替换文中列出的诊断参数、epoch 数和独立输出目录：

```bash
CUDA_VISIBLE_DEVICES=<gpu> conda run -n slm_pvs python \
  neural_instance_culling/model/train_bounded_relation_survival_moment_safety.py \
  --dataset-dir <pose-csr-v3> --relation-dir <bounded-relation-csr-v3> \
  --runtime-meta <runtimeVisibilityMeta.json> \
  --initial-geo-features <instance_geo_features_fp16.bin> \
  --glb-index <glbIndex.json> --glb-root <scene-assets> \
  --variant full --relation-source bounded_hierarchical \
  --spectral-mode moment_envelope --instance-calibration-mode residual \
  --loss-variant safety_reserve --steps-per-epoch 50 --poses-per-batch 4 \
  --max-eval-poses 64 --seed 20260801 --device cuda \
  --output-dir <independent-diagnostic-output>
```

Focused tests were executed with：

```bash
conda run -n slm_pvs python -m unittest \
  neural_instance_culling.model.common.tests.test_safety_reserve_operating_utility_loss \
  neural_instance_culling.model.tests.test_train_bounded_relation_survival_moment_safety \
  neural_instance_culling.benchmark.tests.test_evaluate_pvs_bounded_relation_survival_moment_v4
```

## 证据位置

- 训练配置与成员摘要：`neural_instance_culling/model/out/pvs_bounded_relation_prior_instance_calibrated_moment_safety_reserve_v4_formal80/`
- validation 回放：`neural_instance_culling/benchmark/out/pvs_bounded_relation_prior_instance_calibrated_moment_safety_reserve_v4_formal_validation/`
- 安全裕度损失：`neural_instance_culling/model/common/safety_reserve_operating_utility_loss.py`
- RVL 可见性损失：`neural_instance_culling/model/common/safety_constraint_utility_loss.py`
- v4 训练入口：`neural_instance_culling/model/train_bounded_relation_survival_moment_safety.py`
- view-cell 并集标签构建：`neural_instance_culling/model/pose_csr_dataset.py`
- 第一轮损失短训：`neural_instance_culling/model/out/pvs_v4_loss_miniscan_20260817/`
- 第二轮尾部路径短训：`neural_instance_culling/model/out/pvs_v4_loss_tail_miniscan_20260817/`
- 第三轮工作带形状短训：`neural_instance_culling/model/out/pvs_v4_loss_shape_miniscan_20260817/`
