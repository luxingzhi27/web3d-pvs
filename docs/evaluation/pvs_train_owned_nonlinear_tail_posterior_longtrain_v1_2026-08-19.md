# 108 维视点区域非线性尾部分离长训消融

日期：2026-08-19

## 摘要

本实验研究现有 108 维视点区域特征能否通过更强的非线性判别器改善安全工作点附近的正负样本尾部分离。实验先在单种子短训上比较固定线性、加性分段线性和单隐藏层小型多层感知机，随后选取“分段线性高视觉效用探针 + 8 单元小型多层感知机覆盖探针”进行三个随机种子、每种子 80 epoch 的同 checkpoint 长训消融。所有探针只从 train split 拟合，运行阈值只由各自 calibration split 冻结，validation 仅用于评价，test split 未读取。

短训结果证明非线性探针能够比固定线性探针更好地分离困难尾部；三种子长训则表明，该优势主要体现为修复线性探针造成的退化，尚未稳定超过不使用后验探针的基础模型。非线性组合相对固定线性双探针将合并 precision 提高 2.438 个百分点、accuracy 提高 5.786 个百分点、useful cull 提高 5.913 个百分点，并平均少预测 312.53 个实例；但这些主要差值的 95% 置信区间仍跨零。相对无后验基础模型，非线性组合的 precision 下降 0.714 个百分点，accuracy 下降 1.213 个百分点，useful cull 下降 1.224 个百分点，均无统计显著性。因此，本轮不晋级尾部分离后验，也不修改默认模型或前端资产。

## 方法

### 视点区域输入

每个候选实例使用当前运行链路已有的 108 维视点区域特征：9 维中心视线查询、64 维视点区域 Fourier 统计、8 维边界摘要，以及 27 维区域上下界与跨度。这些特征描述当前视点区域相对实例的位置、方向和变化范围，不增加逐实例固定特征表，也不要求浏览器展开多个 subpose、搜索邻居或执行在线图传播。

### 尾部探针

固定线性探针在标准化特征上学习一个岭线性判别面，每支包含 109 个参数。分段线性探针在每个标准化维度上加入位于 `-1`、`0`、`1` 的三个 ReLU 折点，每支包含 433 个参数，用于表达各维特征与困难尾部之间的非线性关系。小型多层感知机采用 `108 -> 8 -> 1` 和 ReLU 激活，每支包含 881 个参数，用于表达不同视点区域特征之间的低容量交互。

探针只对基础分数提供非负救援，不降低任何候选分数。高视觉效用分支保护具有较大 `visible_weights` 的低分正例，覆盖分支保护普通低分正例，两支输出取最大值。该设计保持保守可见性语义：先救回困难正例，再由 calibration 上冻结的较高阈值剔除负例。

### 快速筛选与长训

快速筛选使用随机种子 `20260801` 的第 24 epoch 快照，额外训练有界尾部残差分支 4 epoch。分段线性探针的 validation 困难尾部 ROC-AUC 为 0.887878 和 0.884028；小型多层感知机分别达到 0.894244 和 0.899990。最终工作点扫描选择分段线性探针负责高视觉效用保护、小型多层感知机负责普通正例覆盖。

正式训练从三个随机种子各自的第 24 epoch checkpoint 出发，仅优化 4,889 参数的有界尾部残差分支。每个成员执行 80 epoch、每 epoch 100 step，学习率为 `5e-4`，每 4 epoch 评价并保存一次。三个种子的完整训练分别耗时 4,674.04、4,623.40 和 4,640.50 秒；最终由 calibration 协议选中的安全 checkpoint 分别位于本轮第 40、4 和 36 epoch。较早的最佳 checkpoint 不表示训练提前停止，三个成员均完整运行到第 80 epoch。

对每个选定 checkpoint 使用相同基础分数评价三种后验：无后验、固定线性双探针、分段线性与小型多层感知机组合。三者的候选集合、真实可见集合和 validation pose 完全一致。

## 评价协议

每个随机种子的 validation split 包含 213 个 pose，三种子合计评价 639 个 pose。每个 pose 平均包含 5,174.39 个后退相机候选和 229.06 个真实可见实例。正式安全工作点要求各 checkpoint 自身的 calibration weighted recall 严格大于 0.99，且其单侧 95% 置信下界严格大于 0.99。普通 pose recall 仅用于诊断，不参与安全工作点选择。

统计比较采用 10,000 次配对聚类 bootstrap：先对三个随机种子有放回重采样，再在每个采样种子内对完全对齐的 validation pose 有放回重采样。报告的差值均为左侧方案减右侧方案。实验未读取 test split，也未改变候选集合或真实可见集合。

本报告中的 pose 指标先对每个 pose 计算再取宏平均，aggregate 指标先合并全部 pose 的 TP、FP、FN 和 TN 再计算。Precision 表示预测为可见的实例中真实可见的比例；accuracy 表示全部候选中判断正确的比例；balanced accuracy 是 recall 与 specificity 的均值，用于降低正负样本比例对 accuracy 的影响；weighted recall 表示按历史 `visible_weights` 加权后找回的真实可见效用；useful cull 是正确剔除的不可见实例占候选的比例；bad cull 是错误剔除的可见实例占候选的比例。GLB 数量和字节削减描述下载资源效率，不能替代画面安全指标。

## 三种子合并结果

| 后验方案 | Pose precision | Pose recall | Aggregate precision | Aggregate recall | Weighted recall | Accuracy | Balanced accuracy | Specificity | Useful cull | Bad cull | 平均预测数 | GLB 数量削减 | GLB 字节削减 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 无后验 | 29.645% | 91.468% | **15.346%** | 82.398% | 99.665% | **79.100%** | **80.673%** | **78.947%** | **75.452%** | 0.779% | **1229.89** | **45.011%** | 29.569% |
| 固定线性双探针 | 27.713% | **93.260%** | 12.193% | **85.510%** | **99.770%** | 72.100% | 78.495% | 71.479% | 68.315% | **0.641%** | 1606.33 | 38.143% | 25.376% |
| 分段线性 + 小型 MLP | **30.195%** | 91.391% | 14.632% | 82.646% | 99.700% | 77.886% | 80.156% | 77.666% | 74.228% | 0.768% | 1293.81 | 44.794% | **30.516%** |

三种方案均在各自 calibration split 上通过 weighted recall 安全门。固定线性双探针通过预测更多实例换取更高 recall 和更低 bad cull，但明显降低 precision、specificity、accuracy、balanced accuracy、useful cull 及资源削减。非线性组合基本修复了这种过量救援，但平均仍比无后验基础模型多预测 63.91 个实例。

Weighted recall 接近 1 而 aggregate recall 仅约 0.82，并不矛盾。当前 `visible_weights` 使高视觉效用实例对 weighted recall 的贡献更大；漏掉一批低权重细小实例会明显降低普通 recall，但对 weighted recall 的影响较小。因而本实验仍同时报告普通 recall、weighted recall、bad cull 和图像指标状态，不能用 weighted recall 单独证明分类质量。

## 随机种子稳定性

| Seed | 方案 | Calibration 阈值 | Calibration WR / LCB | Validation precision | Accuracy | Balanced accuracy | Weighted recall | Useful cull | 平均预测数 | GLB 字节削减 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 20260801 | 无后验 | 0.000866 | 99.366% / 99.023% | 45.224% | 94.982% | 79.831% | 99.381% | 92.184% | 320.15 | 43.395% |
| 20260801 | 线性 | 0.001000 | 99.385% / 99.037% | 47.320% | 95.263% | 79.338% | 99.368% | 92.524% | 299.47 | 44.904% |
| 20260801 | 非线性 | 0.001155 | 99.394% / 99.064% | 47.641% | 95.299% | 79.636% | 99.364% | 92.535% | 300.26 | 46.589% |
| 20260802 | 无后验 | 0.000422 | 99.839% / 99.524% | **8.890%** | **54.833%** | **76.138%** | 99.987% | **50.427%** | **2563.96** | **7.140%** |
| 20260802 | 线性 | 0.000649 | 99.839% / 99.522% | 8.485% | 52.414% | 74.928% | 99.991% | 48.003% | 2689.66 | 6.548% |
| 20260802 | 非线性 | 0.000649 | 99.839% / 99.522% | 8.473% | 52.341% | 74.893% | 99.991% | 47.930% | 2693.46 | 6.533% |
| 20260803 | 无后验 | 0.001155 | 99.466% / 99.120% | **24.019%** | **87.484%** | **86.049%** | 99.627% | **83.744%** | **805.57** | 38.171% |
| 20260803 | 线性 | 0.001334 | 99.753% / 99.445% | 11.896% | 68.623% | 81.218% | **99.952%** | 64.416% | 1829.87 | 24.677% |
| 20260803 | 非线性 | 0.001778 | 99.549% / 99.229% | 22.153% | 86.019% | 85.939% | 99.745% | 82.218% | 887.69 | **38.427%** |

非线性组合在 seed `20260801` 上优于无后验基础模型，在 seed `20260803` 上大幅优于线性探针但仍略弱于无后验模型，在 seed `20260802` 上两种探针均使结果恶化。seed `20260802` 的安全阈值仅约 `6.49e-4`，平均预测超过 2,690 个实例，说明该 seed 的安全尾部分布发生坍缩；后验探针没有恢复其正负分离。这种随机种子差异是总体置信区间较宽的主要原因。

## 配对统计

| 比较与指标 | 差值 | 95% 置信区间 | 是否跨零 |
|---|---:|---:|---|
| 非线性 - 无后验：precision | -0.714 pp | [-2.225, 2.250] pp | 是 |
| 非线性 - 无后验：accuracy | -1.213 pp | [-2.487, 0.295] pp | 是 |
| 非线性 - 无后验：balanced accuracy | -0.517 pp | [-1.243, 0.285] pp | 是 |
| 非线性 - 无后验：weighted recall | +0.035 pp | [-0.021, 0.122] pp | 是 |
| 非线性 - 无后验：useful cull | -1.224 pp | [-2.500, 0.326] pp | 是 |
| 非线性 - 无后验：平均预测数 | +63.91 | [-18.27, 131.96] | 是 |
| 非线性 - 无后验：GLB 字节削减 | +0.948 pp | [-0.666, 3.093] pp | 是 |
| 非线性 - 线性：precision | +2.438 pp | [-0.012, 13.253] pp | 是 |
| 非线性 - 线性：accuracy | +5.786 pp | [-0.077, 16.700] pp | 是 |
| 非线性 - 线性：balanced accuracy | +1.661 pp | [-0.033, 4.612] pp | 是 |
| 非线性 - 线性：weighted recall | -0.070 pp | [-0.200, 0.009] pp | 是 |
| 非线性 - 线性：useful cull | +5.913 pp | [-0.085, 17.111] pp | 是 |
| 非线性 - 线性：平均预测数 | -312.53 | [-898.45, 5.08] | 是 |
| 非线性 - 线性：GLB 字节削减 | +5.140 pp | [-0.010, 13.491] pp | 是 |
| 线性 - 无后验：balanced accuracy | -2.178 pp | [-4.967, -0.505] pp | **否** |

非线性组合相对线性探针的 GLB 数量削减提升 6.651 个百分点，95% 置信区间为 `[0.0004, 18.527]` 个百分点，没有跨零；pose precision 与 pose F1 也有同方向显著改善。然而论文主张需要相对无后验基础模型成立，该比较中的主要分类和资源指标全部跨零，因此不能据此晋级后验模块。固定线性探针相对无后验模型使 balanced accuracy 显著下降，说明线性尾部救援会稳定增加负例误报。

## 运行与资产开销

两种后验均复用现有 108 维视点区域输入，不增加随场景实例数量增长的特征表。固定线性双探针包含 218 个共享参数；最终非线性组合包含 433 参数的分段线性探针和 881 参数的小型多层感知机，共 1,314 个共享参数。该参数规模很小，但浏览器 WebGPU 前向延迟、主线程时间、内存和移动端延迟尚未测量，因此本报告不把参数量等同于实际前端性能。

本轮未导出前端资产，也未实现该后验的 WebGPU 运行路径。默认 checkpoint、默认阈值和默认前端资产保持不变。

## 实现与依赖

实验依赖 HKUST 66 度 view-cell CSR 数据集、train-only 分层遮挡关系、96 维离线几何表、运行时实例与 GLB 映射，以及三个种子各自的 v4 第 24 epoch checkpoint。主要实现入口为 `fit_pvs_train_owned_tail_nonlinear_probe.py`、`scan_pvs_train_owned_tail_residual.py`、`run_pvs_train_owned_tail_nonlinear_posterior_longtrain.py` 和 `summarize_pvs_train_owned_tail_nonlinear_posterior_longtrain.py`。新增单元测试覆盖线性、分段线性与小型多层感知机的拟合和运行参数一致性、80 epoch runner 契约、同 pose 后验回放及三种子配对汇总。

## 图像评价状态

Color-ID 图像评价尚未回填，`miss-pixel rate`、`wrong-ID pixel rate`、`extra-pixel rate` 及其 p95 均标记为未实现。本轮只完成集合级、加权可见性和 GLB 资源代理评价，因此不能声明后验改善了最终画面质量。

## 结论

现有 108 维视点区域特征包含可被非线性探针利用的尾部信息，小型非线性组合也确实优于固定线性探针。然而，三种子长训没有证明后验探针能够稳定超过不加后验的基础模型。当前相对最优方案是无后验长训基础模型；非线性后验保留为失败消融和诊断工具，不进入默认模型。

后续若继续研究，应优先解决基础训练在不同随机种子下的安全尾部分布坍缩，而不是继续增加只做正例救援的后验容量。只有当改进在相同 calibration 安全门下稳定提高 precision、accuracy、balanced accuracy 或资源削减，并完成图像及浏览器评价后，才具备晋级价值。

## 复现入口

训练、捕获、探针拟合、阈值冻结和正式汇总由以下入口统一执行：

```bash
conda run -n slm_pvs python \
  neural_instance_culling/benchmark/run_pvs_train_owned_tail_nonlinear_posterior_longtrain.py \
  --shared-root /mnt/sda/rhyang/slm \
  --model-output-root /mnt/sda/rhyang/slm/neural_instance_culling/model/out/pvs_train_owned_nonlinear_tail_posterior_longtrain_v1_20260819 \
  --benchmark-output-root /mnt/sda/rhyang/slm/neural_instance_culling/benchmark/out/pvs_train_owned_nonlinear_tail_posterior_longtrain_v1_20260819
```

正式机器可读汇总为 `neural_instance_culling/benchmark/out/pvs_train_owned_nonlinear_tail_posterior_longtrain_v1_20260819/formal_summary.json`。训练日志、checkpoint 和 score capture 分别保存在同名 model/out 与 benchmark/out 目录中。

聚焦验证命令如下，当前结果为 27 项测试全部通过：

```bash
conda run -n slm_pvs python -m unittest \
  neural_instance_culling.benchmark.tests.test_fit_pvs_train_owned_tail_nonlinear_probe \
  neural_instance_culling.benchmark.tests.test_scan_pvs_train_owned_tail_residual \
  neural_instance_culling.benchmark.tests.test_run_pvs_train_owned_tail_nonlinear_posterior_longtrain \
  neural_instance_culling.benchmark.tests.test_summarize_pvs_train_owned_tail_nonlinear_posterior_longtrain
```

正式汇总另行重跑并通过 schema、三种子、三变体、213 pose、10,000 次 bootstrap、协议标志和数值有限性检查。
