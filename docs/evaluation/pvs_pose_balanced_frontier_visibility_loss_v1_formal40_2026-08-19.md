# 逐位姿平衡与安全前沿损失正式训练结果

- 日期：2026-08-19
- 场景：HKUST v3
- 实验：`pvs_pose_balanced_frontier_visibility_loss_v1`
- 状态：两轮参数扫描和三种子 40 epoch 从头训练均已完成
- 结论：安全阈值尺度明显恢复，但平均分类和剔除效果未超过旧 V4，不晋级默认模型

## 实验口径

本实验只替换可见性主损失，模型结构、固定实例特征、分层关系、生存场、候选集合、GT、split 和视角输入均保持不变。新主损失由逐 pose 类别平衡 BCE 与动态安全前沿分离项组成；旧高权重 BCE、Tversky、预测数量、RVL 排序、边界尾部、负例带和 GLB 资源梯度均关闭，生存场删失监督与关系一致性监督保留。

所有 pilot 和正式成员均从随机初始化训练，runner 会拒绝包含 `--initial-checkpoint` 的命令。正式成员使用种子 `20260801`、`20260802` 和 `20260803`，每个成员训练 40 epoch、每 epoch 100 step；最终评价使用各成员自己的 `best_safe.pt`，没有使用第 40 epoch 或 `last.pt` 替换早期最佳 checkpoint。阈值只由对应 checkpoint 的 calibration split 冻结，test split 未读取。

Validation 共 213 个 pose，平均每个 pose 有 5,174.4 个后退相机候选和 229.1 个 GT 可见实例。表中的 validation weighted recall 置信下界来自训练期对完整 validation 回放执行的 10,000 次 pose bootstrap；独立 evaluator 只复核点估计，没有把该下界复制到自己的输出字段。

## 参数扫描

第一轮固定前沿间隔 `0.5`、温度 `0.25`，扫描前沿损失权重。单种子 12 epoch validation 结果如下：

| 前沿权重 | 阈值 | weighted recall | precision | accuracy | balanced accuracy | useful cull | 平均预测数 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.03 | 99.995% | 4.86% | 13.44% | 54.63% | 9.02% | 4,707 |
| 0.10 | 0.24 | 99.395% | 10.28% | 72.05% | 70.46% | 69.01% | 1,532 |
| **0.20** | **0.28** | **99.713%** | **16.58%** | **81.94%** | **79.29%** | **78.56%** | **1,055** |
| 0.40 | 0.32 | 99.979% | 4.80% | 14.60% | 53.95% | 10.30% | 4,635 |

前沿权重并非越大越好。`0.2` 在保留 weighted recall 的同时获得了最好的分类分离；`0.4` 重新退化为大量过预测。

第二轮固定权重 `0.2`，扫描前沿间隔和温度：

| 间隔 / 温度 | 阈值 | weighted recall | precision | accuracy | balanced accuracy | useful cull | 平均预测数 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0.25 / 0.25 | 0.34 | 99.917% | 5.98% | 40.40% | 61.54% | 36.65% | 3,243 |
| 0.50 / 0.20 | 0.30 | 99.667% | 11.22% | 68.86% | 77.63% | 65.00% | 1,782 |
| 0.50 / 0.35 | 0.24 | 99.806% | 6.92% | 46.59% | 66.74% | 42.65% | 2,942 |
| **0.75 / 0.25** | **0.28** | **99.347%** | **19.75%** | **85.58%** | **79.90%** | **82.32%** | **855** |

因此正式训练冻结为前沿权重 `0.2`、间隔 `0.75`、温度 `0.25`。扫描只用于选择配置，没有读取 test。

## 三种子正式结果

以下均为 calibration 冻结阈值后的 validation 结果。Pose 指标先逐 pose 计算再宏平均；aggregate 指标先合并全部 TP、FP、FN、TN 再计算。

| seed | 最佳 epoch | 阈值 | pose precision | pose recall | pose weighted recall | aggregate precision | aggregate recall | aggregate weighted recall / LCB | aggregate accuracy | aggregate balanced accuracy | useful cull | bad cull | 平均预测数 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 20260801 | 24 | 0.22 | 23.03% | 98.62% | 99.916% | 14.95% | 99.19% | 99.889% / 99.679% | 74.98% | 86.52% | 70.59% | 0.036% | 1,520.2 |
| 20260802 | 16 | 0.24 | 24.00% | 94.90% | 99.825% | 12.37% | 90.49% | 99.740% / 99.589% | 71.21% | 80.40% | 67.20% | 0.421% | 1,675.3 |
| 20260803 | 24 | 0.20 | 22.41% | 97.48% | 99.989% | 8.80% | 98.67% | 99.989% / 99.985% | 54.66% | 75.65% | 50.30% | 0.059% | 2,568.8 |
| 三 seed 平均 | 21.3 | 0.22 | 23.15% | 97.00% | 99.910% | 12.04% | 96.12% | 99.873% / 99.751% | 66.95% | 80.86% | 62.69% | 0.172% | 1,921.4 |

三个成员的 calibration weighted recall 及其单侧 95% 下界均超过 `0.99`；完整 validation 的 weighted recall 点估计和 bootstrap 下界也均超过 `0.99`。普通 aggregate recall 在 seed `20260802` 上只有 90.49%，说明视觉权重较高的 GT 得到保护，但等权正例覆盖仍存在明显种子差异。

完整分类诊断如下：

| seed | pose F1 | pose Jaccard | pose specificity | aggregate F1 | aggregate Jaccard | aggregate specificity | 预测/候选 | 预测/GT |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 20260801 | 31.94% | 22.85% | 52.12% | 25.98% | 14.93% | 73.86% | 29.38% | 6.64x |
| 20260802 | 31.13% | 22.02% | 50.75% | 21.77% | 12.21% | 70.31% | 32.38% | 7.31x |
| 20260803 | 29.86% | 21.63% | 43.73% | 16.16% | 8.79% | 52.63% | 49.64% | 11.21x |
| 三 seed 平均 | 30.98% | 22.17% | 48.87% | 21.30% | 11.98% | 65.60% | 37.13% | 8.39x |

每 pose 平均混淆数量为：TP `220.2`、FP `1,701.3`、FN `8.9`、TN `3,244.1`。其中大量 FP 是 precision 和 useful cull 未能超过旧 V4 的直接原因；较少的加权高价值 FN 则使 weighted recall 仍能通过安全门。

资源侧结果如下：

| seed | 平均预测 GLB 数 | pose 宏平均 GLB 数量削减 | 平均预测 GLB 字节 | pose 宏平均 GLB 字节削减 |
|---:|---:|---:|---:|---:|
| 20260801 | 555.3 | 32.75% | 118.00 MB | 22.30% |
| 20260802 | 626.4 | 29.46% | 154.40 MB | 17.85% |
| 20260803 | 733.0 | 24.24% | 156.16 MB | 12.88% |
| 三 seed 平均 | 638.3 | 28.82% | 142.85 MB | 17.68% |

候选 GLB 的平均基准为 963.8 个、183.62 MB。削减率按每个 pose 分别计算后取宏平均，因此不等于“平均预测量除以平均候选量”；GLB 字节以原始字节计数，表中 MB 采用十进制显示。

## 与旧 V4 的同口径比较

旧 V4 三种子 80 epoch 使用相同的 213 个 validation pose，平均候选数和 GT 数与本实验一致，因此可以作直接诊断比较。旧 V4 结果来自既有正式诊断，不需要继续运行旧矩阵。

| 三 seed aggregate 均值 | 旧 V4 损失 | 新前沿损失 | 新损失变化 |
|---|---:|---:|---:|
| weighted recall | 99.857% | 99.873% | +0.016 pp |
| precision | 19.08% | 12.04% | -7.04 pp |
| accuracy | 73.65% | 66.95% | -6.70 pp |
| balanced accuracy | 80.62% | 80.86% | +0.24 pp |
| useful cull | 69.74% | 62.69% | -7.05 pp |
| bad cull | 0.519% | 0.172% | -0.347 pp |
| 平均预测实例数 | 1,538.8 | 1,921.4 | +382.6 |
| 安全阈值范围 | 0.00024--0.00100 | 0.20--0.24 | 分数尺度明显恢复 |

新损失显著改善了阈值尺度和 bad cull，也使三个 seed 的阈值集中在正常中间区间；这说明去掉极强的正例代价与全局数量项后，输出分数不再被压缩到接近零。但阈值位置本身不是模型效果，三种子平均 precision、accuracy、useful cull 和预测数量均比旧 V4 更差。Balanced accuracy 仅提高 0.24 个百分点，无法抵消负例抑制能力的下降。

## 结论

1. 逐 pose 类别平衡和动态安全前沿的方向有效：12 epoch pilot 能把平均预测数降到 855，并将阈值保持在 `0.28`；它证明旧损失的极低阈值不是模型结构的必然结果。
2. Pilot 的优势没有稳定延续到三种子长训。最佳安全 checkpoint 多出现在第 16--24 epoch，之后继续训练没有形成更好的安全分类分离。
3. 三个正式成员均守住 weighted recall 安全口径，但 seed 间 aggregate precision 为 8.80%--14.95%，平均预测数为 1,520--2,569，稳定性仍不足。
4. 新损失不晋级论文主模型，不替换默认 checkpoint、阈值或前端资产。已有 checkpoint 和日志保留用于后续损失动力学分析；旧矩阵不再补跑。

本轮没有执行图像级 Color-ID 评价、WebGPU 延迟或移动端测试。原因是新损失没有超过同口径旧 V4 的主要分类与剔除结果，且模型结构与运行时资产布局未改变；这些指标不能写成已改善，也不影响本轮“损失未晋级”的结论。

独立 CUDA evaluator 记录的固定运行特征为 124 维、4,670,088 bytes，单实例推理输入为 130 维；三个成员回放全部 213 个 validation pose 的总耗时分别为 1.99、2.00 和 2.01 秒。该时间包含 Python 数据组织和指标汇总，不是隔离的单 pose 网络 forward 延迟，因此只作运行 smoke，不作为浏览器或移动端性能结论。

## 产物与复现

- 训练与扫描入口：`neural_instance_culling/benchmark/run_pvs_pose_balanced_frontier_visibility_loss_v1.py`
- 损失实现：`neural_instance_culling/model/common/pose_balanced_frontier_loss.py`
- 训练输出：`neural_instance_culling/model/out/pvs_pose_balanced_frontier_visibility_loss_v1_20260819/`
- 评价输出：`neural_instance_culling/benchmark/out/pvs_pose_balanced_frontier_visibility_loss_v1_20260819/`

完整执行命令：

```bash
conda run -n slm_pvs python \
  neural_instance_culling/benchmark/run_pvs_pose_balanced_frontier_visibility_loss_v1.py \
  all --root /mnt/sda/rhyang/slm --gpu-ids 0 1 2 3
```
