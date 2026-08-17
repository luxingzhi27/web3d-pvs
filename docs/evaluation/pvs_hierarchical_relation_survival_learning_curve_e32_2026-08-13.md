# 分层关系生存模型 32 Epoch 学习曲线评价

日期：2026-08-13  
状态：阶段 B 完成；未通过进入模块复验的效率门  
实验：`pvs_hierarchical_relation_survival_integrated_spectral_quality_rvl_v1`

## 结论

本次从头训练了自由生存场控制组和分层遮挡关系生存模型，各使用两个随机种子，训练 32 epoch。随后对 epoch `4/8/12/16/20/24/28/32` 的保存 checkpoint 在相同的 213 个 validation pose 上进行冻结阈值 replay，并使用 10,000 次 pose bootstrap 计算 aggregate weighted recall 下界。

分层关系模型 R2 的两个 seed 在 32 epoch 都通过画面安全门：aggregate weighted recall 及其单侧 95% 下界均大于 `0.99`。但它没有通过计划规定的效率晋级门：相对同一 32 epoch 学习曲线中的 `checkpoint_epoch_016.pt`，两个 seed 的 precision 平均只提升约 `0.0012`，平均预测实例数反而变化约 `+2.3%`；其中一个 seed 的 precision 和 balanced accuracy 下降。因此，训练时长本身没有稳定解决过预测问题，不能据此启动 `module_recheck16` 或 `formal80`。

本报告不读取 test split，不修改默认模型、默认阈值或前端资产。图像级 Color-ID 和浏览器 WebGPU 性能本轮未执行，相关字段明确标记为不可用。

## 数据与协议

固定数据集为 `pose_csr_hkust_v3_spatial_fov66_v1`，pose 数为 train `2772`、calibration `168`、validation `213`。所有结果均使用原始后退相机候选集合，未补入 GT 可见实例、未改变候选集合、未使用前端白名单，阈值只从每个 checkpoint 自身的 calibration 记录读取。审计版输出为：

```text
neural_instance_culling/benchmark/out/pvs_hierarchical_relation_survival_integrated_spectral_quality_rvl_v1_learning_curve_snapshot_replay_audited_v2_20260813/
```

其中 8 个 epoch、32 个成员均有独立 validation summary，顶层和 epoch manifest 均为 `testRead=false`。汇总器还核验了 checkpoint schema、calibration summary、阈值来源、pose 数和候选语义。

## 32 Epoch 结果

以下为 32 epoch snapshot 的 aggregate validation 指标。`weighted recall / LCB` 是跨所有 validation pose 合并的加权召回及单侧 95% 下界；`useful cull` 是正确剔除不可见候选的比例，`bad cull` 是错误剔除真实可见候选的比例。

| 变体 | seed | weighted recall / LCB | precision | accuracy | balanced accuracy | useful cull | bad cull | 平均预测实例数 | GLB 数量削减 | 预测 GLB 字节 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| R0 自由生存场 | 20260801 | 0.993755 / 0.990775 | 0.199721 | 0.827726 | 0.891587 | 0.785156 | 0.001698 | 1102.9 | 约 65.3% | 67.22 MB |
| R0 自由生存场 | 20260802 | 0.994810 / 0.992509 | 0.218454 | 0.847292 | 0.896438 | 0.805223 | 0.002198 | 996.5 | 约 65.3% | 65.29 MB |
| R2 分层关系 | 20260801 | 0.998073 / 0.996767 | 0.125719 | 0.694403 | 0.836058 | 0.650513 | 0.000377 | 1806.4 | 约 58.0% | 121.71 MB |
| R2 分层关系 | 20260802 | 0.993332 / 0.990348 | 0.213977 | 0.847169 | 0.880638 | 0.806560 | 0.003658 | 982.0 | 约 57.9% | 68.92 MB |

R2 的安全性成立，但 seed 间效率差异较大；seed 20260801 在高召回阈值下预测了更多实例，precision、accuracy、balanced accuracy 和 useful cull 均明显较弱。这说明当前关系生存模型的分数分布仍存在跨 seed 不稳定性。

## 16 到 32 Epoch 的晋级复核

计划要求 R2 两个 seed 同时满足：安全门通过；balanced accuracy 不下降超过 `0.01`；平均 precision 提高至少 `0.03`，或平均预测实例数下降至少 `15%`；正负分数间隔不恶化。

| seed | precision 变化 | balanced accuracy 变化 | 平均预测数变化 | useful cull 变化 | 结果 |
|---:|---:|---:|---:|---:|---|
| 20260801 | -0.009637 | -0.011243 | +8.21% | -0.026270 | 不通过 |
| 20260802 | +0.012080 | -0.000054 | -7.13% | +0.013923 | 不通过 |

两个 seed 的平均 precision 变化为 `+0.0012`，按两个 seed 的平均预测数计算，平均预测数量变化约 `+2.3%`，距离计划门槛 `+0.03` 或 `-15%` 均有明显差距。seed 20260801 的 balanced accuracy 下降超过允许的 `0.01`。因此不能把 R2 解释为“训练更久后效率改善”，也不能用 seed 20260802 的局部改善替代跨 seed 稳定性要求。

## 学习曲线趋势

- epoch 4 和 8 的多数工作点尚未达到 validation 安全门；R2 在 epoch 8 已有安全工作点，但预测数量仍高。
- epoch 12 以后安全门普遍可通过，但 R2 seed 20260801 的预测数量持续偏高，precision 没有改善。
- epoch 16 到 32，R2 seed 20260801 的平均预测数从约 `1669` 增至约 `1806`，precision 从约 `0.1354` 降至约 `0.1257`；seed 20260802 的平均预测数从约 `1057` 降至约 `982`，precision 从约 `0.2019` 升至约 `0.2140`，但改善不足以通过正式门。
- R0 控制组在 32 epoch 的 balanced accuracy 略高于 R2 两个 seed，说明 R2 的 weighted recall 优势没有转化为安全工作点下的分类和剔除优势。

## 阶段判定

| 阶段 | 判定 |
|---|---|
| 32 epoch 训练 | 完成，4 个成员均正常结束 |
| validation replay | 完成，8 个 epoch、32 个成员、213 pose |
| aggregate weighted recall 安全门 | R2 两个 seed 均通过 32 epoch 工作点 |
| 训练时长改善效率门 | 未通过 |
| `module_recheck16` | 不启动 |
| `formal80` | 不启动 |
| 默认模型/前端资产 | 保持不变 |
| test split | 未读取 |

后续若继续优化，应先针对阈值下的分数分离、损失目标或校准策略进行新的、独立命名的快速实验；不能直接用 80 epoch 长训掩盖当前过预测，也不能把未通过门控的模块写成正式论文贡献。

## 证据

- 32 epoch 训练矩阵：`neural_instance_culling/model/out/pvs_hierarchical_relation_survival_integrated_spectral_quality_rvl_v1_learning_curve_e32_20260813/`
- 审计版快照与汇总：`neural_instance_culling/benchmark/out/pvs_hierarchical_relation_survival_integrated_spectral_quality_rvl_v1_learning_curve_snapshot_replay_audited_v2_20260813/`
- e16 与 e32 的直接比较均使用上述审计目录中的 `e016/` 和 `e032/`，对应同一训练矩阵的 `checkpoint_epoch_016.pt` 与 `checkpoint_epoch_032.pt`；根目录 `best.pt` replay 作为单独的 best-checkpoint 结果保留，不与 e32 snapshot 混称。
- 训练日志：`neural_instance_culling/benchmark/out/pvs_hierarchical_relation_survival_integrated_spectral_quality_rvl_v1_learning_curve_e32_20260813_logs/`
- 后续唯一计划：`docs/experiments/pvs_full_innovation_hyperparameter_scan_longtrain_ablation_2026-08-13.md`
