# M0：评测语义回归覆盖

日期：2026-08-01  
状态：协议回归子门通过；正式空间 checkpoint 的 calibration/frozen test 仍待训练收尾。

## 变更目的

M0 的冻结测试协议已经阻止 test 阈值扫描和候选集合修复，但评测器还需要证明两类边界不会污染后续指标：非法的 `visible_weights` 不能被静默修复，零弱效用姿态不能被报告为有效的视觉效用召回；M3 干预必须优先使用训练结束后、尚未读取 test 的 calibration-ready 阈值。

## 修改内容

- 新增 `neural_instance_culling/benchmark/tests/test_evaluation_protocol_semantics.py`。
- `evaluate_visual_utility_metrics.py` 的 pose 聚合器保留 `utilityStatus` 和 `requiredStatus`；多个 pose 状态不一致时报告 `mixed`，不再只保留数值平均值。
- 回归测试覆盖：
  - 负值和非有限 `visible_weights` 直接失败；
  - 所有弱效用为零时，视觉效用召回为 `null`，并记录 `not_applicable` 及有效/不适用 pose 数；
  - M3 同时存在正式 test 摘要和 pre-test 摘要时优先读取 `calibration_ready_summary.json`；只有显式诊断开关才允许读取 one-shot test 摘要。

## 验证命令与结果

```bash
PYTHONDONTWRITEBYTECODE=1 conda run --no-capture-output -n slm_pvs \
  python -m unittest discover \
  -s neural_instance_culling/benchmark/tests -p 'test_*.py' -v

PYTHONDONTWRITEBYTECODE=1 conda run --no-capture-output -n slm_pvs \
  python -m py_compile \
  neural_instance_culling/benchmark/evaluate_visual_utility_metrics.py \
  neural_instance_culling/benchmark/evaluate_proxy_interventions.py \
  neural_instance_culling/benchmark/tests/test_evaluation_protocol_semantics.py

PYTHONDONTWRITEBYTECODE=1 conda run --no-capture-output -n slm_pvs \
  python neural_instance_culling/benchmark/evaluate_visual_utility_metrics.py --self-test

PYTHONDONTWRITEBYTECODE=1 conda run --no-capture-output -n slm_pvs \
  python neural_instance_culling/benchmark/evaluate_proxy_interventions.py --self-test
```

结果：新增回归测试 3 项通过，全套 benchmark 测试 26 项通过；两个 model-free self-test 通过；Python 编译通过。测试未读取正式 test pose，也没有改变候选集合、阈值或模型权重。

## 指标语义影响

`visible_weights` 仍只表示采样阶段产生的弱重要性权重，不能解释为真实像素覆盖率。若某个 pose 的可见实例权重总和为零，视觉效用召回不参与平均，并在 JSON/表格中标记为 `not_applicable`；普通可见性 recall、weighted recall、剔除指标仍照常计算。这样避免把没有效用监督的 pose 伪装成零损失或满召回。

## 质量门判断

该修正提高了 M0/M7 评测协议的可审计性，但不改变正式准入标准。M0 总门仍未通过，原因是当前空间训练尚未产出带最终 bootstrap 校准、固定特征导出和独立 one-shot test manifest 的完整 artifact。训练完成后仍必须先执行 `evaluate_frozen_test.py prepare`，再在独占目录中执行一次 test。

## 保留决定与风险

保留为当前主线协议代码和回归测试。剩余风险是正式训练进程启动时可能使用旧版源码；其输出必须按实际日志和 artifact provenance 审计，不能把本次新增测试追溯应用到已经运行的进程。
