# HKUST 主实验划分与 RVL 基线恢复实验

- 日期：2026-08-20
- 状态：执行中
- 目的：恢复旧 684 个固定 test view-cell，在不读取 test 选择阈值的前提下，验证训练覆盖率和数据划分是否是近期 precision 下降的主要原因。
- 实验前缀：`pvs_hkust_main_stratified_calibration_rvl_recovery_v1`

## 问题与假设

旧 HKUST 数据集包含 6,585 个 train、730 个 validation 和 684 个 test view-cell。近期空间四路划分把 4,608 个 view-cell 放入 guard，只留下 2,772 个 train、168 个 calibration 和 213 个 validation；同时 calibration 与 validation 的平均 GT 数量明显失衡。

本实验检验以下假设：

1. calibration split 的概念本身不会降低模型能力；主要影响来自训练样本大幅减少和空间 split 分布偏移。
2. 从旧 train 内分层抽取少量 calibration，可以保留绝大多数训练覆盖，并阻止 test 阈值泄漏。
3. 使用相同 RVL 架构和损失重新训练后，安全工作点的 pose precision、balanced accuracy、specificity 和 useful cull 应明显接近旧 684 test 结果。

## 主实验划分

固定旧 split 边界，只拆分旧 train：

| Split | 数量 | 用途 |
|---|---:|---|
| train | 5,926 | 参数更新和 train-only 遮挡证据构建 |
| calibration | 659 | 冻结每个 checkpoint 自己的安全阈值 |
| validation | 730 | checkpoint 和实验配置比较，不选择阈值 |
| test | 684 | 模型、阈值和配置全部冻结后执行一次 |

calibration 从旧 6,585 个 train view-cell 中确定性选取 10%。选择过程同时匹配以下边际分布：

- 场景采样类别；
- 相机水平观察方向；
- 相机俯仰方向；
- 后退相机候选数量；
- view-cell GT 可见实例数量；
- visible weight 总量。

每个 view-cell 的全部离线 subpose 已经聚合为同一 CSR pose，因此不会跨 split 拆分。旧 validation 和 test 的 pose ID、候选集合、GT、权重和顺序均保持不变。

## 数据和监督边界

- 基础数据：`pose_csr_hkust_v3_bounded_relation_moment_fov66_v3`，其候选、GT 和 visible weight 与旧 7,999 view-cell 数据一致。
- 旧 split 来源：`hkust_v3_viewcell_colorid_fov66_source/viewcell_split_ids.bin`。
- 新数据视图：`pose_csr_hkust_v3_main_stratified_calibration_fov66_v1`。
- 新 split manifest：`main_split_hkust_v3_stratified_calibration_20260820`。
- 方向遮挡证据必须从新 train 的 5,926 个 pose 重建；不得复用由旧 6,585 train 构建、包含新 calibration pose 的证据。
- test 不参与训练、离线证据、checkpoint 选择、参数扫描或阈值选择。

## 首轮模型实验

先复现历史方向遮挡代理 RVL 模型，不引入近期 v4 复杂损失，以隔离数据划分影响：

- 固定实例特征：96 维几何、64 维上下文、192 维方向代理；
- 视角查询：117 维 Fourier ray；
- 损失：历史 `w042` RVL 参数和当前 `rvl_strong_v2` 各运行一个单种子短训成员；
- 从头训练，不从旧 checkpoint 续训；
- 快速阶段使用 12 epoch、每 epoch 300 step；
- 两个成员并行使用不同 GPU；
- calibration 和 validation 均完整遍历，不进行 pose 截断。

快速阶段始终选择相对更好的成员进入正式训练，不设置提前取消门。选择顺序为：先比较 validation weighted recall 是否达到 0.99，再比较 balanced accuracy、useful cull、pose precision、specificity和平均预测数量。

正式成员从头训练 40 epoch、每 epoch 900 step。最终阈值只由 659 个 calibration view-cell 冻结；完成后在 730 validation 上回放。如果 calibration 的 weighted recall 点估计及单侧 95% 下界均大于 0.99，再对固定 684 test 执行一次评价；不在 test 上重新扫描。

## 主要比较

必须统一报告：

- pose precision、pose recall、pose weighted recall；
- aggregate precision、recall、accuracy、balanced accuracy 和 specificity；
- useful cull、bad cull、平均候选、平均 GT 和平均预测数量；
- calibration 与 validation 的分布差异；
- 固定 684 test 上相对旧 `0.64` 工作点的变化。

空间隔离 split 继续作为“未见区域泛化”辅助实验，不再替代同场景主实验。后续单独构建分布平衡的大空间块划分，并将 guard 距离与 view-cell 实际半径绑定，避免随机小块边界使多数样本进入 guard。

## 运行记录

待实现和执行后回填实际命令、耗时、checkpoint、阈值及完整指标。正式结果产生前不修改默认前端 checkpoint、阈值或资产。
