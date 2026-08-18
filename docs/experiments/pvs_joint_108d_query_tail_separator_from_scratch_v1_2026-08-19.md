# 108 维视角尾部分离器从头联合长训与消融计划

日期：2026-08-19

## 目的与问题界定

此前记录的三种子 80 epoch 实验并非完整模型从头训练。该实验分别加载旧 v4 模型的 `checkpoint_epoch_024.pt`，冻结基础可见性网络与遮挡生存场，只更新边界尾部残差分支，随后又在训练结果上拟合线性、分段线性或小型 MLP 探针。因此，该结果只能回答“固定旧模型能否被后验分离器修复”，不能回答非线性视角尾部分离是否能在端到端训练中改善模型，也不能作为完整模型三种子结果。

本实验从 epoch 1 统一训练完整模型。108 维视角区域查询的分离分支直接属于模型，参与可见性、视觉效用和 GLB 调度的联合优化；不再训练完成后拟合额外探针。实验使用三个随机种子和四个结构变体完成 80 epoch 长训，不因 pilot 或中间安全指标提前取消成员。

实验前缀固定为：

```text
pvs_joint_108d_query_tail_separator_from_scratch_v1
```

## 固定模型基础

所有变体保留当前完整 v4 数据流：

- 后退扩展相机候选；
- 96 维离线实例几何特征；
- 共享分层遮挡关系先验与逐实例校准生存场；
- 视点区域矩包络频谱查询；
- 安全裕度工作区效用损失；
- 实例可见性、视觉效用和 GLB 下载优先级联合输出。

108 维查询由同一视点区域的一次批量输入生成，包含 9 维中心视线描述、64 维频谱统计、8 维边界摘要和 27 维区域极值/跨度。浏览器仍只执行一次候选批量查询，不展开离线 subpose，也不执行在线邻居图传播。

## 从头训练契约

每个正式成员必须满足以下条件：

```text
initialCheckpoint: null
refinementScope: all
startEpoch: 1
epochs: 80
```

训练入口不得生成或接受旧 checkpoint 作为本实验的初始化参数。基础可见性网络、关系生存场、逐实例校准、查询分离器、视觉效用头和 GLB 调度头全部可训练。checkpoint 必须记录各模块可训练参数数量和总可训练参数数量；runner 测试必须证明正式命令中不存在 `--initial-checkpoint`，且 `refinement_scope` 为 `all`。

## 消融矩阵

| 变体 | 108 维查询分离器 | 目的 |
|---|---|---|
| `without_query_tail_separator` | 关闭 | 完整 v4 从头训练基线 |
| `linear_query_tail_separator` | `108 -> 1` | 判断线性重排是否足够 |
| `hinge_query_tail_separator` | 原特征及在 -1、0、1 处折叠的 ReLU 特征，`432 -> 1` | 判断分段线性边界是否能分离尾部 |
| `mlp8_query_tail_separator` | `108 -> 8 -> 1`，SiLU | 判断低容量非线性交互是否有效 |

三个启用分支均输出有界的有符号 logit 修正，最大绝对值固定为 `0.5`，在每个 pose 内减去均值，避免通过整体抬高或压低分数代替实例分离。输出层使用零初始化，使启用分支的模型从与基线相同的初始可见性行为开始。修正后的可见性表征同时供实例可见性、视觉效用和 GLB 调度使用。

所有启用分支使用相同的训练尾部损失设置。分离损失只在 train split 上读取 subpose 监督：在每个 pose 内选择低分的重要正例和高分负例，以进入分离器前的 logit 作为停止梯度的样本选择依据，用可训练的有符号修正拉开两类尾部。该损失只改变训练目标，不改变候选、GT 或 calibration/validation 口径。

固定损失参数为：总权重 `0.30`、正例加权尾部质量 `0.005`、每 pose 正例上限 `64`、负例高分尾部比例 `0.04`、负例数量范围 `64` 至 `256`、温度 `0.25`、pose CVaR 比例/权重 `0.25/0.50`、跨 pose 配对权重 `0.40`、全批次尾部配对权重 `0.25`、残差正则 `0.02`。无分离器基线的该项权重严格为零。尾部分离项并入安全目标，完整 RVL、安全裕度、关系生存、视觉效用和下载损失均继续参与联合训练。

## 数据与训练预算

| 项目 | 固定值 |
|---|---|
| PoseCSR | `neural_instance_culling/dataset/out/pose_csr_hkust_v3_bounded_relation_moment_fov66_v3` |
| 关系 CSR | `neural_instance_culling/dataset/out/pvs_bounded_relation_survival_moment_envelope_safety_reserve_v3/bounded_relation_csr_v3` |
| 固定几何表 | `neural_instance_culling/model/out/pvs_directional_occlusion_proxy_encoder_rvl_w042_full40/instance_geo_features_fp16.bin` |
| train-only subpose 监督 | `neural_instance_culling/dataset/out/pvs_v4_viewcell_extreme_support_scan_20260818/subpose_supervision_sidecar` |
| split | train 2,772；calibration 168；validation 213；test 不读取 |
| 随机种子 | `20260801`、`20260802`、`20260803` |
| 正式预算 | 4 变体 × 3 seed × 80 epoch |
| GPU | RTX A6000 × 4，每卡一个训练进程，动态任务队列 |

预计单成员约 3.7 小时，四卡三轮总墙钟时间约 11 至 13 小时。所有成员写入独立 stdout/stderr 日志。单个成员失败不取消其余队列；只有代码错误、非有限 loss、CUDA OOM 或硬件故障可以中断该成员，修复后从未完成成员继续。

## 阈值与评价

每个 checkpoint 只能使用自己的 calibration split 冻结阈值。validation 不重新选阈值，test 在模型与方案冻结前保持未读。安全工作点要求 calibration weighted recall 严格大于 `0.99`，且其单侧 95% 置信下界严格大于 `0.99`；没有合格工作点的成员仍完成 80 epoch 和全部诊断评价，并明确标记为“不具备合格安全工作点”。

validation 至少输出：

- pose 与 aggregate precision、recall、F1；
- weighted recall 及单侧置信下界；
- instance accuracy、balanced accuracy、specificity；
- useful cull、bad cull、平均预测实例数；
- GLB 数量和字节削减；
- 每个 pose 的 TP、FP、FN、TN 和预测集合；
- checkpoint、固定运行特征表和评价模型的一致性字段。

正式比较使用相同 validation pose、候选集合、实例 GT 和 split 语义。统计阶段执行 10,000 次聚类配对 bootstrap：先对 seed 重采样，再在对应 seed 内对相同 pose 成对重采样。报告变体相对无分离器基线的差值、95% 置信区间、方向及是否跨零。weighted recall 是画面安全主门；precision、accuracy、balanced accuracy 和 specificity共同判断分离质量，不能只用平均预测数量或 useful cull 得出结论。

## 实施顺序与完成定义

1. 将 108 维分离器加入完整模型，统一配置、checkpoint 和评价重建 schema。
2. 在 `refinement_scope=all` 下启用 train-only 尾部分离监督，保留完整 v4 损失；删除本路线对初始 checkpoint 的依赖。
3. 新增独立 runner、汇总入口和对应单元测试，不覆盖旧 v4 与冻结主干 refinement 结果。
4. 执行四个变体的单 optimizer-step CUDA smoke，检查 loss、梯度、输出和 checkpoint 数值有限。
5. 使用四张 GPU 启动 12 个正式成员并排队运行到 80 epoch；不根据中间指标取消矩阵。
6. 对每个成员执行 calibration 阈值冻结、完整 validation 回放和资源评价。
7. 完成三种子汇总和 10,000 次配对 bootstrap，形成正式评价报告。

本实验只有在 12/12 成员均达到 80 epoch、每个成员完成独立 calibration 和 validation、汇总 schema 校验通过并生成正式报告后才算完成。正式结论产生前，不修改当前默认 checkpoint、默认阈值、前端资产和部署包。

## 当前执行状态

截至 2026-08-19，模型、训练器、评价器、独立 runner、汇总器和测试已经实现。正式 dry-run 得到 `4 × 3 = 12` 个成员，全部命令均为 `initialCheckpoint: null`、`refinementScope: all` 和 `80 epoch`。相关单元测试通过 `132/132`，汇总器 self-test 通过。

四个变体均已完成真实 RTX A6000 单 optimizer-step CUDA smoke。所有 loss、梯度和 checkpoint 数值有限；训练元数据记录初始化模式为 `from-scratch`。同一种子下四个变体的共享模型、视觉效用头和下载头初值一致，三个启用分支的尾部分离损失初值也完全一致。MLP smoke checkpoint 已由正式 evaluator 严格重建并完成全部 213 个 validation pose 回放，证明新增 schema 可从 checkpoint 复现。上述 smoke 指标不参与模型选择。

下一步是启动四卡 `all` 模式；该模式会无条件完成 12 个 80 epoch 成员，随后自动执行各自 calibration 冻结阈值下的完整 validation 回放和 10,000 次配对聚类 bootstrap。
