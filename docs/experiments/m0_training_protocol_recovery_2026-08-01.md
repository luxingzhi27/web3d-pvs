# M0/M1 训练协议恢复记录

日期：2026-08-01
状态：协议修复已实现；HKUST 新的 40 epoch 正式训练正在运行，尚未形成准入结论。

## 变更目的

第一条空间正式训练在 HKUST 第 21 个 epoch 的中间校准阶段停止。审计确认原因不是
非有限损失、候选语义错误或显存错误，而是当轮验证模型暂时没有任何阈值达到预注册的
`weighted recall > 0.99`。训练主程序把中间监控失败误当成最终安全门失败，提前退出，导致
后续 epoch 无法继续优化。

中间 checkpoint 不能决定论文准入：它只用于观察训练是否趋于安全工作点。最终 calibration
仍必须严格满足点估计和单侧置信下界规则；如果最终没有安全工作点，训练必须失败并降级，
不能用诊断阈值生成正式模型或 test 结果。

## 修改内容

修改文件：

- `neural_instance_culling/model/train_directional_occlusion_proxy_encoder.py`
- `neural_instance_culling/benchmark/tests/test_training_calibration_control.py`

实现细节：

1. `evaluate_calibration_and_validation()` 增加 `require_safe_workpoint` 参数，默认值为 `True`。
2. 最终 calibration、checkpoint 导出和正式 test 路径保持严格模式；没有安全工作点仍直接抛错。
3. 中间 epoch 评估使用严格模式关闭：若当轮没有安全阈值，选择 weighted recall 最高的诊断行仅用于
   validation telemetry，并在摘要中写入 `selectionStatus=unsafe_diagnostic_fallback` 和
   `diagnosticFallback`。
4. 不安全诊断行不能产生 `best.pt`，也不会写入 `frozenThreshold`；只有安全行才参与 checkpoint
   比较和 `best.pt` 选择。
5. 添加空校准表和不安全诊断行的单元测试，防止以后把监控阈值误称为冻结阈值。

## 首次运行证据

输出目录：

`neural_instance_culling/model/out/pvs_directional_occlusion_proxy_encoder_rvl_strong_v2_full40_hkust_spatial_fov66_seed20260801`

命令使用了默认实际 seed `20260610`，目录中的 `seed20260801` 只是历史启动标签。该运行在
epoch 21 终止，stderr 的致命信息为：

```text
RuntimeError: No calibration threshold satisfies the strict weighted-recall rule
pose_weighted_recall > 0.990.
```

当轮日志中没有 `trainSkippedNonFiniteLoss` 或 `trainSkippedNonFiniteGrad`，因此该目录只能作为
“中间安全门过严导致提前退出”的失败记录，不能作为模型结果。其 validation 诊断工作点的
weighted recall 约为 `0.9882`，低于安全目标，未被转化为正式阈值。

## 修复后正式运行

输出目录：

`neural_instance_culling/model/out/pvs_directional_occlusion_proxy_encoder_rvl_strong_v2_full40_hkust_spatial_fov66_seed20260801_protocolfix`

运行配置：

- 数据：`pose_csr_hkust_v3_spatial_raw_subpose_aabb_fov66_v1`；
- 证据：`directional_occlusion_evidence_hkust_v3_spatial_raw_fov66_v1`；
- FOV：模型/后退相机 66°，真实显示 60°；
- `rvl_strong_v2`，RVL 默认开启；
- 40 epoch，900 steps/epoch，pose batch size 2；
- `seed=20260801`，FP32，无 AMP，GPU 0；
- `target-weighted-recall=0.99`；
- calibration point floor `0.9925`，最终 view-cell bootstrap 单侧下界 floor `0.99`，10,000 次；
- `--skip-final-test`，训练只生成 calibration-ready 资产，正式 test 由冻结入口执行一次；
- 候选集合读取原始 CSR，不补入 GT，不裁剪候选。

启动命令已记录在该输出目录的日志对应进程命令行中，stdout/stderr 分别为
`train_stdout.log` 和 `train_stderr.log`。

## 准入判断

- 协议修复：通过，已有 Python 编译和 2 个单元测试；
- 旧中断运行：失败记录，降级，不进入主表；
- 新训练 checkpoint：等待 40 epoch 完成；
- calibration：必须重新执行并检查点估计、10,000 次 bootstrap 和单侧下界；
- one-shot test：在 calibration-ready 后使用 `evaluate_frozen_test.py prepare/evaluate`，完成前不运行；
- 主线保留：只有完整训练、冻结校准和 test 证据都通过后，才将新目录提升为主线候选。

## 后续风险

如果修复后完整 40 epoch 仍无法在 calibration 达到安全门，下一步优先检查 RVL 与可见性集合
损失的梯度比例、空间 split 的长尾构件和数据候选语义；不直接降低目标或在 test 上重新扫阈值。
只有在预注册实验修改达到规定次数且有论文合理性说明时，才讨论放宽门槛，并将理由写入新的
实验记录。
