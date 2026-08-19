# 逐位姿平衡与安全前沿可见性损失实验计划

日期：2026-08-19

> **执行状态**：两轮单种子参数扫描和冻结配置的三种子 40 epoch 从头训练均已完成。正式结果见[逐位姿平衡与安全前沿损失正式训练结果](../evaluation/pvs_pose_balanced_frontier_visibility_loss_v1_formal40_2026-08-19.md)。新损失改善了安全阈值尺度，但三种子平均 precision、accuracy、useful cull 和预测数量未超过旧 V4，因此不晋级默认模型；旧矩阵不再续跑。

## 目的

当前完整模型后期会通过持续抬高正例分数、容忍大量假正例来降低漏检损失。其直接表现是校准阈值跌到极低区间，平均预测实例数从数百增长到数千，precision、accuracy、balanced accuracy 和 useful cull 同时退化。主要原因是可见性主损失中正例 BCE、Tversky 和 FN 项远强于 FP、数量及排序约束，优化目标与“在 weighted recall 安全前提下分开正负实例”不一致。

本实验只重写可见性主损失，不改变候选、GT、split、模型结构、视角特征、分层关系先验或逐实例生存场。目标是在保持 weighted recall 的前提下改善正负分数分离，重点观察 precision、instance accuracy、balanced accuracy、specificity、平均预测数量和 useful cull。

实验统一前缀为 `pvs_pose_balanced_frontier_visibility_loss_v1`，全部输出写入独立目录，不修改当前默认模型和前端资产。

## 新损失

可见性主目标只保留两项：

\[
L_{visibility}=L_{pose-balanced}+\lambda L_{frontier}.
\]

### 逐位姿类别平衡分类损失

每个 pose 内分别计算正例和负例 BCE，两类各自归一化，再对 pose 做宏平均。正例视觉权重采用温和压缩：

\[
\tilde w_i=0.5+0.5\sqrt{w_i/\max_j w_j}.
\]

存在正负两类时，两类损失各占一半；只有单类时只使用存在的类别。该设计取消固定 `pos_weight=14`，避免候选规模和正负比例直接改变单个 pose 对梯度的贡献。

### 加权安全前沿分离损失

每个 pose 从当前最终可见性 logit 中动态选取：

- 按分数从低到高累计达到可见权重 `0.5%` 的重要正例，最多 64 个；
- 分数最高的 `1%` 负例，至少 1 个、最多 256 个。

对选出的正负前沿直接优化：

\[
L_{frontier}=\operatorname{mean}\operatorname{softplus}
\left((z_n-z_p+m)/T\right).
\]

前沿成员选择在 detached 分数上完成，损失梯度只进入选中成员的最终 logit。初始参数为正例权重质量 `0.005`、负例比例 `0.01`、间隔 `m=0.5`、温度 `T=0.25`、损失权重 `\lambda=0.2`。

旧可见性损失中的高权重 BCE、Tversky、count、独立 FP/FN RVL、boundary tail、negative band、query-tail 残差、GLB 资源项和 utility/download 对可见性主干的梯度不进入本实验。生存场删失监督与关系一致性监督保留，作为当前模型已有离线表征的训练信号；其权重在所有对照中保持一致。

## 快速参数扫描

所有 pilot 使用 seed `20260801`、从随机初始化开始、12 epoch、每 epoch 100 step。第一轮四卡并行扫描前沿权重：

| 配置 | 前沿权重 | 间隔 | 温度 |
|---|---:|---:|---:|
| balanced_only | 0.0 | 0.5 | 0.25 |
| frontier_w010 | 0.1 | 0.5 | 0.25 |
| frontier_w020 | 0.2 | 0.5 | 0.25 |
| frontier_w040 | 0.4 | 0.5 | 0.25 |

第二轮以第一轮相对最优非零权重为中心，四卡并行比较四组前沿形状：`(m,T)` 为 `(0.25,0.25)`、`(0.50,0.20)`、`(0.50,0.35)`、`(0.75,0.25)`。

每个 checkpoint 只用 calibration split 冻结阈值，再在 validation 上评价。test 不参与扫描、阈值选择或排序。calibration 必须满足 weighted recall 及其单侧置信下界安全门，validation 必须在冻结阈值下满足 weighted recall 点估计要求；当前 validation evaluator 不生成置信下界，因此该字段必须明确记录为空，不能伪造或借用 calibration 的下界。合格配置按 aggregate precision、balanced accuracy、instance accuracy、useful cull 和平均预测数量依次比较。如果没有配置达到安全门，仍按可用的 weighted recall 安全证据、balanced accuracy、precision 的顺序选择相对最优配置，不取消后续长训。

## 40 epoch 正式训练

冻结扫描选出的配置后，使用种子 `20260801`、`20260802`、`20260803` 从头训练 40 epoch，四张 GPU 并行调度。正式命令禁止包含 `--initial-checkpoint`；日志与 checkpoint 写入各自成员目录。

训练期间保留每次 evaluation 的 calibration 阈值和 validation 指标。最终按每个成员自己的 `best_safe.pt` 评价；没有安全 checkpoint 时使用 `best_diagnostic.pt` 并明确标记“没有合格安全工作点”。不能用最后一个 epoch 覆盖早期最佳结果，也不能因中间退化提前停止登记成员。

## 指标与保留规则

必须报告 pose-level 与 aggregate 的 precision、recall、F1、instance accuracy、balanced accuracy、specificity、weighted recall、useful cull、bad cull、平均候选、平均 GT、平均预测和冻结阈值。另记录正例低尾、负例高尾、前沿间隔及阈值变化，用于判断分数分布是否继续后期塌缩。

本实验的主要比较对象是同数据、同架构的旧损失成员。新损失只有在 weighted recall 安全口径下稳定提高 precision、balanced accuracy、accuracy 或 useful cull 时才晋级；否则保留为失败分析，不进入默认模型。无论结果是否晋级，既定三种子 40 epoch 训练与汇总都必须完成。

## 依赖与运行

依赖沿用当前 V4 数据集、train-only 分层关系 CSR、运行时实例元数据和 96 维离线几何特征。运行环境为 Linux conda `slm_pvs` 与 NVIDIA CUDA。正式 runner 应提供 preflight、scan、train、evaluate、summarize 和 all 入口，并将 stdout/stderr 写入明确日志。

计划运行命令：

```bash
conda run -n slm_pvs python \
  neural_instance_culling/benchmark/run_pvs_pose_balanced_frontier_visibility_loss_v1.py \
  all --root /mnt/sda/rhyang/slm --gpu-ids 0 1 2 3
```
