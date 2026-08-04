# M5 视觉贡献安全损失 Pilot

日期：2026-08-04  
状态：运行中，诊断性 pilot，不是正式质量门结果  
对应阶段：投稿计划 M5

## 动机与假设

上一轮 dense 图像评价显示，漏像素集中在少数空间 view-cell，并且一个高视觉贡献实例可以在同一方向的全部
subpose 中持续漏检。当前 `m5_subpose_robust_v1` 已加入 subpose 出现频率风险，但没有启用显式视觉安全损失，
因此本 pilot 检验一个独立机制假设：在保持候选集合、真实可见集合和后退相机不变的前提下，对每个 pose 的
高屏幕贡献正例施加保守的正向覆盖边界，能否减少高价值漏检，而不是仅增加平均正例召回。

该损失只使用已有 `visible_weights` 作为视觉重要性代理，并不把它称为真实像素覆盖率；它不使用 dynamic-pool
teacher、不补入 GT、不修改候选集合，也不读取 test。若 pilot 只提高预测数量而不改善高贡献漏像素，将不登记为
正式方案。

## 独立配置

输出目录：

```text
neural_instance_culling/model/out/pvs_m5_subpose_visual_safety_robust_v2_pilot_hkust_spatial_fov66_seed20260804_e10
```

配置为：

- 基础损失：`m5_subpose_robust_v1`；
- `visual-safety-loss-weight=0.35`；
- 视觉权重幂：`1.0`；
- 高贡献正例数量：`8`；
- 高贡献正例目标 margin：`2.0`；
- margin 相对权重：`0.75`；
- `10` epoch、`300` steps/epoch、seed `20260804`；
- CUDA 物理 GPU `3`，`slm_pvs` conda 环境；
- 训练若 CUDA 初始化失败立即退出，不切换到 CPU；后续浏览器图像评价必须使用 Chrome Vulkan/ANGLE 硬件 WebGL，
  并通过 `gpuGate.hardware=true`，同时保存 Chrome 日志及 `nvidia-smi`/`nvidia-smi pmon` 证据；SwiftShader
  等软件后端只允许单独的语义调试，不能进入正式 M5 结果；
- 模型输入/后退相机 FOV `66°`，数据集和 spatial split 与正式 M5-v2 相同；
- 训练阶段不读取 test，pilot 的 `1,000` 次 bootstrap 仅用于诊断，不可替代正式 `10,000` 次校准。

运行会话：`tmux` 的 `m5_visual_pilot_20260804`。stdout/stderr 保存在输出目录中。

## 验收边界

pilot 只用于比较以下中间证据：validation/calibration 的高贡献正例 logit、固定 calibration 阈值下的普通 recall、
weighted recall、平均预测数量，以及已有 dense 图像结果中困难 view-cell 的代表性漏像素。它不能选择正式阈值，不能
进入 M5 主表，也不能改变默认模型或前端资产。

只有在 pilot 对高贡献漏检有明确改善且没有明显安全/资源退化后，才登记独立三 seed 完整训练；完整版本仍需通过
普通 pose recall、weighted recall 点估计和置信下界、硬件 GPU dense 图像 mean/p95 门，最后才考虑 one-shot test。

## Pilot 结果与决定

pilot 已完成 `10/10` epoch，使用独立 calibration 摘要，`testEvaluationCount=0`。校准选择阈值为
`0.03`，校准集上的 pose recall 为 `0.9511`，weighted recall 为 `0.99315`，1,000 次 bootstrap 的
weighted-recall 下置信界为 `0.99141`，平均预测实例数为 `164.10`。但在同一冻结阈值下，validation 的
pose recall 为 `0.94397`，低于普通安全门 `0.95`，平均预测实例数为 `189.85`，因此不能把 calibration
结果外推成 validation 安全通过。

本 pilot 没有输出高视觉贡献正例长尾召回、dense view-cell miss-pixel rate 的独立改善证据；与已有视觉质量
修复实验相比，也没有证明新增视觉安全项能在相同安全约束下减少预测数量或漏像素。结论为“假设未被 pilot
确认”，不登记新的三 seed 正式架构，不修改默认模型、前端资产或 M5 主线结果。该 pilot 只保留作参数和失败
分析记录；后续优先等待 `m5_subpose_robust_v1` 三 seed 的正式 calibration 与硬件 dense 图像门。
