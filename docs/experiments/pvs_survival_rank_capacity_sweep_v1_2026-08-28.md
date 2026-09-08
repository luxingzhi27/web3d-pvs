# 遮挡生存场方向秩容量实验

日期：2026-08-28

状态：四种方向秩、三种子、每成员 `40 epoch × 900 step` 已全部完成。

## 实验目的

当前完整模型为每个实例保存 `4 x 7 = 28` 个遮挡生存场系数。本实验固定七个参数的物理语义和全部主线模块，只改变方向低秩基数量，判断方向表达容量、分类效果、画面安全、固定资产大小和推理成本之间的关系。

实验名称统一为 `pvs_survival_rank_capacity_sweep_v1`。模型输出写入 `neural_instance_culling/model/out/pvs_survival_rank_capacity_sweep_v1/`，评价输出写入 `neural_instance_culling/benchmark/out/pvs_survival_rank_capacity_sweep_v1/`，不得覆盖当前主线和既有核心消融结果。

## 唯一实验变量

对每个实例保存系数矩阵 `A_i in R^(R x 7)`。七个参数始终表示无阻挡概率、两个首遮挡深度分量的混合权重、两个深度位置和两个尺度；查询时由当前视点区域条件产生 `R` 个方向基响应，再与 `A_i` 相乘得到当前方向的七个参数。

| 方向秩 R | 每实例生存场维度 | 固定运行特征维度 |
|---:|---:|---:|
| 2 | 14 | 110 |
| 4 | 28 | 124 |
| 8 | 56 | 152 |
| 12 | 84 | 180 |

`R=4` 必须从头重训，作为当前完整模型在同一代码和调度窗口下的复现实验。不得通过截断或补零修改现有 28 维表，也不得从已有 checkpoint 继续训练。

## 固定主线契约

- 数据划分固定为 `5926 train / 659 calibration / 730 validation / 684 test`。
- 三个随机种子固定为 `20260801/20260802/20260803`。
- 每个成员从头训练 `40 epoch x 900 step/epoch`，`poses-per-batch=4`。
- 保留共享分层遮挡关系先验、逐实例校准残差和视点区域矩包络频谱查询。
- 损失固定为当前无对比学习的完整模型：逐 pose 平衡分类、RVL 加权召回保护和困难边界间隔；`integrated_contrastive_mix=0.0`。
- 不启用视觉效用、下载优先级或 GLB 字节训练目标。
- 每个 checkpoint 只能使用自己的 calibration split 冻结阈值；test 不参与阈值选择，也不在容量选择阶段读取。
- 画面安全主门为 validation weighted recall 点估计和单侧 95% 置信下界均大于 `0.99`。

## 运行与评价

四张 GPU 并行运行四种 rank；每种 rank 的三个 seed 均完整训练，任何成员未通过安全门也不能提前取消。正式比较使用相同 validation pose、候选集合、GT 和可见权重。

正式运行命令：

```bash
conda run -n slm_pvs python \
  neural_instance_culling/benchmark/run_survival_rank_capacity.py run \
  --gpu-ids 0 1 2 3
```

runner 同时运行四个成员，每张 GPU 保持一个训练进程；一个成员完成后自动领取下一 seed。训练结束后自动使用各 checkpoint 自己的 calibration 阈值评价 validation，并生成 `capacity_summary.json`。

必须报告 pose-level 与 aggregate 的 precision、recall、F1、accuracy、balanced accuracy 和 specificity，并同时报告 weighted recall、置信下界、useful cull、bad cull、平均预测实例数、预测/候选比例、固定特征表大小、模型参数量和 CUDA 前向延迟。

容量结论按以下顺序判断：先检查 weighted recall 安全门，再比较 balanced accuracy、precision、accuracy、useful cull 和平均预测数量；最后结合每增加 7 维的边际收益、固定资产增量和 CUDA 延迟选择容量。低 rank 若性能接近，应优先保留更小资产；高 rank 只有产生稳定分类或剔除收益时才值得进入前端验证。

## 代码与前端边界

训练模型、checkpoint 配置和评价器必须支持 rank 自描述；当前默认 rank 仍为 4。只有选择非 4 的 rank 时，才需要单独更新导出器并完成 WebGPU/WASM parity 和移动端性能测试。

## 正式结果

所有 12 个成员均完成训练，并使用各 checkpoint 自己的 calibration 阈值回放相同的 730 个 validation pose；三个种子全部通过 weighted-recall 安全门，test 未读取。

| 方向秩 | 生存维度 | 固定表 | Pose precision | Pose recall | Aggregate precision | Aggregate recall | Aggregate weighted recall | Aggregate accuracy | Aggregate balanced accuracy | Aggregate useful cull | Aggregate bad cull | 平均预测数 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 14 | 3.95 MiB | 0.4621 | 0.9802 | 0.1998 | 0.9801 | 0.99848 | 0.9081 | 0.9432 | 0.8856 | 0.000458 | 540.9 |
| **4** | **28** | **4.45 MiB** | **0.5101** | 0.9691 | **0.2163** | 0.9725 | 0.99772 | **0.9177** | 0.9445 | **0.8954** | 0.000631 | **493.7** |
| 8 | 56 | 5.46 MiB | 0.4729 | 0.9802 | 0.1891 | 0.9849 | 0.99794 | 0.9024 | 0.9427 | 0.8798 | **0.000347** | 569.2 |
| 12 | 84 | 6.47 MiB | 0.4905 | 0.9809 | 0.2119 | 0.9829 | 0.99678 | 0.9156 | **0.9485** | 0.8931 | 0.000392 | 505.9 |

表中分类和剔除指标为三个种子的 validation 均值。所有 rank 均满足安全门，因此可以在相同安全条件下比较效率。

Rank 4 在 pose/aggregate precision、aggregate accuracy、useful cull 和平均预测数上最好。Rank 12 的 balanced accuracy 略高，但固定表增加到 6.47 MiB，且没有带来对应的 precision 或剔除收益；rank 2 虽节省约 0.50 MiB，但分类和有效剔除下降；rank 8 同时增加资产并降低主要效率指标。因此当前 28 维 rank-4 生存场继续作为论文和前端默认容量。

机器可读结果：

```text
neural_instance_culling/benchmark/out/pvs_survival_rank_capacity_sweep_v1/capacity_summary.json
```
