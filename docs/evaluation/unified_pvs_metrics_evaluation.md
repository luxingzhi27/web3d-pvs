# 统一 PVS 论文评价与指标报告协议

更新时间：2026-09-10

本文是当前实例级 PVS 论文、正式实验报告和对外汇报的唯一指标口径。模型、数据集、图像评价和前端资产边界分别以
[架构技术文档](../current/neuralstreamweb3d_architecture_technical.md)、
[数据集协议](../current/neuralstreamweb3d_dataset_protocol.md)、
[图像评价协议](viewcell_image_per_evaluation.md) 和
[当前版本清单](../current/current_instance_pvs_versions.md) 为准。

本文解决五个经常混淆的问题：

- 一个指标是逐 pose 宏平均，还是把全部候选合并后的 aggregate；
- 一个指标是否依赖 calibration 冻结阈值；
- PR-AUC 对应的正样本比例和随机基线是什么；
- 哪些指标证明画面安全，哪些指标只说明分类、剔除、资源或运行效率。
- GLB 连续下载排序如何把神经可见性、资源字节和可选的投影效用分开计算。

## 1. 评价对象与基本集合

对每个 pose 或 view-cell `p` 定义：

```text
C_p：后退 66 度扩大视锥产生的候选实例集合
G_p：离线采样得到的真实可见实例集合
P_p：模型在冻结阈值下输出的预测可见实例集合
```

正式候选集合不能补入 GT，也不能在模型之间改变：

```text
TP_p = P_p ∩ G_p
FP_p = P_p - G_p
FN_p = G_p - P_p
TN_p = C_p - (P_p ∪ G_p)
```

必须验证 `G_p` 是 `C_p` 的子集。候选生成方式、FOV、view-cell 范围、subpose 并集语义或候选上限改变后，结果属于不同协议，不能放进同一张模型对比表。

## 2. 结果的四个维度

任何指标名称都必须同时说明以下维度，不能只写一个没有口径的 `precision`、`recall` 或 `PR-AUC`。

| 维度 | 可选值 | 含义 |
|---|---|---|
| 数据划分 | calibration、validation、test | calibration 冻结阈值，validation 选择模型，test 只做最终一次评价 |
| 汇总层级 | pose-macro、aggregate | pose-macro 让每个 pose 等权；aggregate 让每个候选实例等权 |
| 权重 | unweighted、visible-weighted | 普通实例等权，或只对 GT 正例按 `visible_weights` 加权 |
| 工作点 | threshold-free、frozen safe threshold | 排序指标不依赖阈值；集合和资源指标使用 checkpoint 自己的冻结阈值 |

规范写法示例：

```text
validation pose-macro precision at the calibration-frozen safe threshold
validation aggregate weighted recall at the calibration-frozen safe threshold
validation pose-macro PR-AUC (Average Precision), threshold-free
```

以下写法不完整，不能出现在论文表头或结论中：

```text
precision = 0.48
recall = 0.99
PR-AUC = 0.87
```

## 3. Pose-macro 与 Aggregate

### 3.1 Pose-macro

先在每个 pose 内计算指标，再对 `N` 个 pose 等权平均。例如：

\[
\mathrm{Precision}_{pose}
=\frac{1}{N}\sum_p\frac{TP_p}{TP_p+FP_p},
\]

\[
\mathrm{Recall}_{pose}
=\frac{1}{N}\sum_p\frac{TP_p}{TP_p+FN_p}.
\]

Pose-macro 回答“一个平均视点表现如何”，不会让候选数特别多的 pose 自动获得更大权重。论文中的普通分类结果应优先给出这一口径。

### 3.2 Aggregate

先跨 pose 合并 TP、FP、FN、TN，再计算指标。例如：

\[
\mathrm{Precision}_{agg}
=\frac{\sum_p TP_p}{\sum_p(TP_p+FP_p)},
\]

\[
\mathrm{Recall}_{agg}
=\frac{\sum_p TP_p}{\sum_p(TP_p+FN_p)}.
\]

Aggregate 回答“全部候选实例合起来表现如何”。候选数大的 pose 权重更高，因此它不能替代 pose-macro，也不能与 pose-macro 数值直接比较。

F1、Jaccard、accuracy、balanced accuracy、specificity、useful cull 和 bad cull 也必须分别按上述两种方式计算。Pose-macro F1 必须先算每个 pose 的 F1 再平均，不能由宏平均 precision 和宏平均 recall 二次计算。

## 4. 正样本比例与 PR-AUC

### 4.1 两种正样本比例

Aggregate 正样本比例为：

\[
\pi_{agg}=\frac{\sum_p|G_p|}{\sum_p|C_p|}.
\]

Pose-macro 正样本比例为：

\[
\pi_{pose}=\frac{1}{N}\sum_p\frac{|G_p|}{|C_p|}.
\]

两者通常不同。候选数很多且正样本稀疏的 pose 会显著降低 `pi_agg`，但在 `pi_pose` 中仍只占一个 pose 的权重。

### 4.2 项目中的 PR-AUC 定义

项目字段 `averagePrecision` 使用按预测分数从高到低排序、相同分数合并后的非插值 Average Precision：

\[
\mathrm{AP}=\sum_k(R_k-R_{k-1})P_k.
\]

论文统一写作 **PR-AUC (Average Precision, AP)**。不要把它与 PR 曲线梯形积分、固定阈值 precision 或 ROC-AUC 混为一谈。

- `pose-macro PR-AUC`：每个 pose 独立排序并计算 `AP_p`，再计算 `mean(AP_p)`；对应随机排序基线是 `pi_pose`。
- `aggregate PR-AUC`：拼接全部 pose 的候选分数和标签后计算一次 AP；对应随机排序基线是 `pi_agg`。
- 两种 AP 都与最终阈值无关，但必须使用同一个已冻结 checkpoint，不能在 test 上根据 AP 重新选 checkpoint。
- 当前 AP 是普通实例等权 AP。`weighted recall` 或 `weighted ROC-AUC` 不能改称 weighted PR-AUC。

随机排序的 AP 期望约等于同口径正样本比例。跨场景报告 AP 时必须同时报告正样本比例和提升倍数：

\[
\mathrm{AP\ lift}=\frac{\mathrm{AP}}{\pi}.
\]

可在附录提供归一化 AP：

\[
\mathrm{normalized\ AP}=\frac{\mathrm{AP}-\pi}{1-\pi},
\]

但它不是当前主指标，不能替代原始 AP 和正样本比例。

如果正式 split 含有零 GT pose，必须报告数量并预先声明处理规则；不能临时把其 AP 记为 0 或 1。当前正式评价默认遍历有真实可见实例的唯一 pose。

### 4.3 当前 HKUST 口径检查案例

以当前前端 checkpoint seed `20260802`、epoch `36` 的 `730` 个 validation pose 为例：

| 汇总层级 | 正样本比例 | PR-AUC (AP) |
|---|---:|---:|
| Pose-macro | 0.09319 | 0.87023 |
| Aggregate | 0.02296 | 0.34644 |

这两个结果都正确，但回答的问题不同。禁止用 pose-macro 的 `0.09319` 作为 aggregate AP 的随机基线，也禁止把 aggregate AP `0.34644` 写成 pose PR-AUC。

### 4.4 ROC-AUC

ROC-AUC 同样分别报告 pose-macro 和 aggregate。由于大量负例会让 ROC-AUC 在 precision 较低时仍接近 1，正文以 PR-AUC 为主要阈值无关排序指标，ROC-AUC 作为辅助指标。

`weighted ROC-AUC` 只对正例使用 `visible_weights`，用于检查重要可见实例的排序；它不惩罚安全阈值下的过量 FP，不能替代 precision、useful cull 或资源指标。

## 5. 冻结安全工作点指标

### 5.1 阈值和 split

每个 checkpoint 使用自己的 calibration split 冻结阈值。当前安全门为：

```text
calibration aggregate weighted recall > 0.99
calibration aggregate weighted recall 的单侧 95% 置信下界 > 0.99
```

Validation 使用同一冻结阈值报告 weighted recall，并用于比较 checkpoint 或配置；它不能重选阈值。Test 只能在模型、阈值、候选协议和资产全部冻结后读取一次。

固定阈值 `0.5`、best-F1 阈值和最高 precision 阈值只作为分布诊断，不是安全主工作点。低阈值必须结合分数分布解释，但不能通过 bias 或 temperature 把阈值移动到中间后宣称模型改善。

### 5.2 画面安全

| 指标 | Pose-macro 定义 | Aggregate 定义 | 作用 |
|---|---|---|---|
| 普通 recall | `mean(TP_p / |G_p|)` | `sum(TP) / sum(|G|)` | 诊断普通实例覆盖，必须报告 |
| weighted recall | `mean(wTP_p / wG_p)` | `sum(wTP) / sum(wG)` | 当前画面安全主指标 |
| weighted recall LCB | 对 pose 重采样后的单侧 95% 下界 | 对 pose 的加权 TP/GT 和重采样后的单侧 95% 下界 | 安全余量，不是双侧差值 CI |
| bad cull | `mean(FN_p / |C_p|)` | `sum(FN) / sum(|C|)` | 错误剔除占候选的比例 |
| FN / GT | `mean(FN_p / |G_p|)` | `sum(FN) / sum(|G|)` | 与普通 recall 等价的漏检风险表达 |

`visible_weights` 必须注明来源。Color-ID 屏幕覆盖和历史 rvcServer `component_weights` 是不同语义，不能都写成真实像素数。

### 5.3 分类诊断

| 指标 | 公式 | 解释 |
|---|---|---|
| precision | `TP / (TP + FP)` | 预测可见集合中真正可见的比例 |
| F1 | `2PR / (P + R)` | precision 与 recall 的联合诊断 |
| Jaccard | `TP / (TP + FP + FN)` | 预测集合和 GT 的交并比 |
| specificity | `TN / (TN + FP)` | 不可见候选被正确拒绝的比例 |
| instance accuracy | `(TP + TN) / |C|` | 逐候选正确率，容易受大量 TN 抬高 |
| balanced accuracy | `(recall + specificity) / 2` | 正负类等权，更适合类别不平衡诊断 |

上述指标必须同时给出 pose-macro 和 aggregate。Precision 受正样本比例影响。令正样本比例为 `pi`、真正率为 `TPR`、假正率为 `FPR`：

\[
\mathrm{precision}=\frac{TPR\,\pi}{TPR\,\pi+FPR(1-\pi)}.
\]

因此跨场景或跨候选策略比较时，不能只比较 precision；必须同时报告 `pi_pose`、`pi_agg`、平均候选数、specificity、accuracy 和 balanced accuracy。

### 5.4 有效剔除

| 指标 | 公式 | 解释 |
|---|---|---|
| useful cull | `TN / |C|` | 正确剔除不可见候选的比例 |
| bad cull | `FN / |C|` | 错误剔除可见候选的比例 |
| raw reduction | `(TN + FN) / |C| = 1 - |P|/|C|` | 混合正确和错误剔除，只能辅助报告 |
| 平均预测数 | `mean(|P_p|)` | 前端每个 pose 保留的实例数量 |
| Pose-macro 预测/候选 | `mean(|P_p| / |C_p|)` | 一个平均 pose 保留多少候选 |
| Aggregate 预测/候选 | `sum(|P_p|) / sum(|C_p|)` | 全部候选中的保留比例 |
| Pose-macro 预测/GT | `mean(|P_p| / |G_p|)` | 一个平均 pose 的过预测程度 |
| Aggregate 预测/GT | `sum(|P_p|) / sum(|G_p|)` | 全部预测数相对全部 GT 的比例 |

`useful cull` 必须和 bad cull、普通 recall、weighted recall 同表展示。预测集合过小可能同时得到较高 useful cull 和较高漏检，不能解释为模型更好。

### 5.5 面向图形学读者的遮挡术语

本项目以“可见”为正类：`target=True` 表示 GT 可见，`pred=True` 表示预测可见并保留。因此同一个混淆矩阵可以从可见性预测和遮挡剔除两个方向解释：

```text
GT visible  + Pred visible  = TP
GT occluded + Pred visible  = FP
GT visible  + Pred occluded = FN
GT occluded + Pred occluded = TN
```

论文面向图形学读者时使用以下显示名称，但机器结果字段不重命名：

| 论文显示名称 | 现有字段 | 公式 | 含义 |
|---|---|---|---|
| Visible Recall | `recall` | `TP / (TP + FN)` | 真正可见实例中被保留的比例 |
| Occlusion Recall | `specificity` | `TN / (TN + FP)` | 真正遮挡实例中被正确剔除的比例 |
| False Occlusion Rate | `1 - recall`，等价于 `FN / GT` | `FN / (TP + FN)` | 真正可见实例中被错误剔除的比例 |
| Useful Cull Ratio | `usefulCull` | `TN / |C|` | 所有候选中被正确剔除的比例 |
| Bad Cull Ratio | `badCull` | `FN / |C|` | 所有候选中属于错误剔除的比例 |

`Occlusion Recall` 与 `Useful Cull Ratio` 的分母不同；`False Occlusion Rate` 与 `Bad Cull Ratio` 的分母也不同。论文不得把较小的 `badCull` 解释为“可见实例误剔除率”。例如 `badCull=0.2%` 只说明 FN 占全部候选的 `0.2%`，实际可见实例误剔除率必须读取 `1-recall`。

上述等价关系必须分别在 pose-macro 和 aggregate 口径内成立：

\[
\mathrm{FOR}_{pose}=\frac{1}{N}\sum_p\frac{FN_p}{TP_p+FN_p}=1-\mathrm{Recall}_{pose},
\]

\[
\mathrm{FOR}_{agg}=\frac{\sum_p FN_p}{\sum_p(TP_p+FN_p)}=1-\mathrm{Recall}_{agg}.
\]

正文安全与剔除表推荐使用 `Visible Recall`、`Occlusion Recall`、`False Occlusion Rate`、`Weighted Recall`、`Useful Cull Ratio` 和 `Bad Cull Ratio`；表注给出与机器字段的映射。Precision、balanced accuracy 和正样本比例仍需保留，避免遮挡术语掩盖过量保留问题。

## 6. 图像、资源与运行指标

### 6.1 图像质量

图像评价必须使用与前端显示相同的真实 `60` 度相机，而不是后退 `66` 度候选相机，并通过硬件 Chrome Color-ID 路径执行。

正文至少报告：

| 指标 | 含义 |
|---|---|
| image PER | reference 非背景像素中实例 ID 不一致的比例 |
| miss-pixel rate | reference 有实例、预测画面为背景的像素比例 |
| wrong-ID pixel rate | 两张图均非背景但实例 ID 不同的比例 |
| extra-pixel rate | reference 为背景但预测画面出现实例的比例 |
| p95 miss-pixel rate | 困难 pose 的尾部画面漏检风险 |

图像指标主要证明漏检是否真的影响画面，不能单独证明剔除效率。

### 6.2 GLB 与下载资源

正文至少报告每 pose 的绝对量和相对量：

- 候选 GLB 数、预测 GLB 数及 GLB 数量削减率；
- 候选 GLB 字节、预测 GLB 字节及 GLB 字节削减率；
- 达到相同图像效用所需的 GLB 字节；
- 预算内 GLB utility recall；
- 冷启动首屏下载字节、可交互时间和完整加载时间。

实例级预测和 GLB 下载是不同粒度。GLB 指标不能反向替代实例 precision，也不能把同一 GLB 内未预测的实例自动当成可见实例。

### 6.3 Progressive streaming 排序

正式 streaming 在每个完整 test pose 上从空缓存开始。所有方法必须使用同一个由 CSR candidate instance 映射得到的完整 candidate GLB 集合；一个 GLB 只有在全部字节到达后才累计 coverage。主 utility 为该 test pose 的 `visible_weights` 按 GLB 求和，必须写作 **visible-weight coverage**，不能改称像素 coverage。

模型首先把实例连续可见性分数聚合为 GLB 可见性：

\[
p_g=\max_{i\in g}p_i.
\]

纯神经排序按 `p_g` 降序。成本感知神经排序定义为：

\[
S_g^{neural-cost}=\frac{p_g}{Bytes_g^{\alpha}}.
\]

其中 `Bytes_g` 是完整 GLB 文件字节，`alpha=0` 退化为纯神经排序，`alpha=1` 是可见性分数/字节，`alpha=0.5` 是对小文件偏好更温和的可见性分数/平方根字节。`alpha` 必须只在 calibration/validation 上按预先登记的候选集合选择；模型、聚合规则和 `alpha` 冻结后，test 只报告一次。不得根据 test 的 `Bytes@99` 为不同场景临时选择不同公式。

AABB 投影不是神经成本排序的必要输入。若额外评价视觉效用调制，必须单列为：

\[
S_g^{neural-area-cost}=\frac{\max_{i\in g}(p_i A_i)}{Bytes_g^{\alpha}},
\]

其中 `A_i` 是同一 pose 下实例 AABB 八角点投影得到的裁剪屏幕矩形面积。它不能与 `p_g / Bytes_g^alpha` 混名，也不能在一个场景使用面积、另一个场景不用面积后仍称为同一方法。

廉价启发式 baseline 必须从每个 candidate instance 的 AABB 计算，再按 GLB 聚合：

\[
A_g=\max_{i\in g}A_i,\qquad
D_g=\min_{i\in g}D_i.
\]

比较项至少包括 `Distance`、`Projected AABB area` 和 `Projected AABB area / byte`。不得先把空间分离的多个实例合成一个巨大 GLB AABB 再投影，因为空区域会抬高面积，并使相机到合并 AABB 的距离失真。Region66 visible-weight 实验使用 view-cell 中心的 `66` 度查询投影；真实当前画面实验必须另用 `60` 度相机和 Point60/像素 GT，二者不能混用。

冻结阈值过滤与连续排序继续分开：过滤只报告预测 GLB 子集、字节削减和 coverage ceiling；threshold-free ranking 对完整 candidate GLB 集合排序。真实前端的 `urgent/warm/speculative` 队列可以由冻结安全阈值分层，但必须另外标为 scheduler replay，并报告组内使用的固定排序分数。

正式排序至少报告：

- `Bytes@95/99/99.9/100`；
- `waste-before-99` 和达到目标时的 GLB rank；
- `10/25/50/100 Mbps` 下的换算时间，正文至少给出 25/50 Mbps；
- coverage ceiling 与目标不可达 pose 比例；
- 纯神经 `p_g`、成本感知神经 `p_g/Bytes_g^alpha`、AABB 启发式、AABB MLP、HZB visible-first、随机/原始顺序和 GT utility/byte oracle。

`p_g/Bytes_g^alpha` 衡量“预测需要程度相对下载成本”，仍不等于真实视觉效用。任何关于画面恢复速度的结论都必须同时由 visible-weight coverage 和固定子集的 reference-frontmost/真实重渲染检查支撑。

### 6.4 模型资产和运行成本

| 类别 | 必须报告 |
|---|---|
| 模型资产 | 固定实例特征表、网络权重、关系/索引运行资产及总下载大小 |
| 推理输入 | 每候选输入维度、平均及 p95 候选数 |
| CUDA | batch 条件下单 pose forward 的 mean、p50、p95 |
| 浏览器 | Worker 总调度、WebGPU 或 WASM SIMD 推理的 mean、p50、p95 |
| 主线程 | 每帧主线程时间、可见集合提交时间和长任务数量 |
| 内存 | 峰值 JS/WASM 内存、GPU 内存和常驻模型资产 |
| 渲染 | FPS、frame-time p50/p95、draw calls 和 drawn instances |

所有延迟必须注明硬件、浏览器版本、后端、冷启动或预热状态、候选数量和重复次数。WebGPU 性能只有 adapter 通过硬件门时才能写为硬件结果；软件 adapter 只能报告数值 parity。

## 7. 三种子与统计比较

每个正式模型使用固定三个种子。不能先把三个种子的候选分数平均后再计算指标；应先对每个 seed 独立选择 calibration 阈值并完成评价，再汇总指标。

单模型表格报告：

```text
三种子 mean ± sample standard deviation
每个 seed 的 checkpoint epoch 和冻结阈值
安全门通过种子数，例如 3/3
```

消融差值使用相同 seed、相同 pose 的 paired bootstrap：先按 seed 聚类，再在 seed 内重采样 pose，至少 `10,000` 次。差值报告：

```text
mean difference
95% bootstrap confidence interval
区间是否跨零
改善方向
```

安全工作点的 weighted recall LCB 是单侧 95% 下界；模型间差值 CI 是双侧 95% 区间。二者目的不同，不能互换。

## 8. 论文表格组织

### 8.1 数据与候选统计表

每个场景至少列出：

- train/calibration/validation/test pose 数；
- 场景实例数和 GLB 数；
- 平均候选数、平均 GT 数；
- pose-macro 正样本比例 `pi_pose`；
- aggregate 正样本比例 `pi_agg`；
- `visible_weights` 来源和真实渲染/候选 FOV。

### 8.2 主结果表

主结果表建议分成四组，避免在一张超宽表里混合所有语义。

| 表 | 必须包含 |
|---|---|
| 安全与剔除 | 阈值、weighted recall、单侧 LCB、普通 recall、useful cull、bad cull、平均预测数 |
| 分类与排序 | pose/aggregate precision、balanced accuracy、PR-AUC、正样本比例；ROC-AUC 作为辅助 |
| 图像与资源 | PER、mean/p95 miss-pixel、预测 GLB 数/字节、同视觉效用字节和首屏时间 |
| 部署成本 | 运行资产大小、输入维度、WebGPU/WASM p50/p95、内存、frame-time 和 draw calls |

Pose-macro 是论文叙述“平均视点”的主口径；aggregate 必须同时提供，用于揭示大候选 pose 的总体影响。任何 PR-AUC 都必须在同一列或表注中给出对应正样本比例。

### 8.3 消融表

每项创新至少报告：

- 安全工作点 weighted recall 及 LCB；
- pose-macro 和 aggregate PR-AUC；
- pose-macro 和 aggregate precision、recall、balanced accuracy；
- useful cull、bad cull、平均预测数和 GLB 字节；
- 图像 miss-pixel 与运行资产/延迟；
- 相对完整模型的 paired-bootstrap 差值和 95% CI。

安全门不合格的成员仍需报告，但标注“没有合格安全工作点”，不能降低安全要求后继续参与主排名。

### 8.4 附录

以下内容可放附录，但必须保留：

- 三个 seed 的逐成员结果、epoch 和阈值；
- F1、Jaccard、accuracy、specificity、TP/FP/FN/TN；
- PR 曲线、可靠性图、Brier、ECE 和分数尾部分位数；
- best-F1、固定 `0.5` 和最高 precision 等诊断工作点；
- 浏览器、CUDA 和图像评价的完整运行条件。

## 9. 最小报告清单

发布任何“模型更好”的结论前，至少检查：

- [ ] 明确 scene、split、pose 数、seed 数和 checkpoint epoch；
- [ ] 明确阈值来自该 checkpoint 的 calibration；
- [ ] 同时报 pose-macro 与 aggregate；
- [ ] 同时报普通 recall、weighted recall 及其单侧 LCB；
- [ ] 同时报 precision、specificity、accuracy 和 balanced accuracy；
- [ ] PR-AUC 标明 pose-macro 或 aggregate，并给出同口径正样本比例；
- [ ] 同时报 useful cull、bad cull 和平均预测数；
- [ ] 同时报图像漏检、GLB 数/字节和运行资产/延迟，缺失项写 `not_available`；
- [ ] Streaming 明确 utility 来源、完整 candidate GLB 集合、冷缓存、GLB 完整到达语义和排序公式；
- [ ] 成本指数及其他调度参数只由 calibration/validation 确定，test 不参与规则选择；
- [ ] 三种子报告 mean ± sample standard deviation；
- [ ] 消融使用相同 pose 的 paired bootstrap，而不是比较两个独立均值；
- [ ] test 未参与阈值、checkpoint 或配置选择。

## 10. 当前代码字段映射

正式评价入口为：

```text
neural_instance_culling/benchmark/evaluate_pvs.py
neural_instance_culling/benchmark/run_pvs.py summarize
neural_instance_culling/benchmark/summarize_core_ablation.py
```

当前 JSON 字段解释：

| JSON 字段 | 口径 |
|---|---|
| `poseMacro.precision` | 冻结阈值下 pose-macro precision |
| `aggregate.precision` | 冻结阈值下 aggregate precision |
| `poseMacro.weightedRecall` | 冻结阈值下 pose-macro weighted recall |
| `aggregate.weightedRecall` | 冻结阈值下 aggregate weighted recall |
| `scoreDistribution.averagePrecision` | threshold-free aggregate PR-AUC (AP) |
| `scoreDistribution.rocAuc` | threshold-free aggregate ROC-AUC |
| `scoreDistribution.weightedRocAuc` | threshold-free aggregate weighted ROC-AUC |

`scoreDistribution.averagePrecision` 当前只表示 aggregate AP，不能改名解释为 pose AP。Pose-macro AP 必须从每个 pose 的候选分数和标签独立计算后再平均，并在结果中明确命名为 `poseMacroAveragePrecision`。

正式 test 的全量遍历规则见 [test split 评价协议](test_split_benchmark_protocol.md)。任何未实现或未回填的正式指标必须写成 `not_available` 并说明原因，不能留空或用其他指标代替。
