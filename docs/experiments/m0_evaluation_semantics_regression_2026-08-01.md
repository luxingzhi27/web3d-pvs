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

## 冻结测试入口回归（2026-08-01）

HKUST 训练完成后首次执行冻结测试时，入口在真正遍历 test pose 之前失败：
`evaluate_frozen_test.py` 错误导入了 `evaluate_unified_pvs_metrics.evaluate_runner`，但调用处使用的是
带实例效用、GLB 字节预算和多个调度模式的 `evaluate_visual_utility_metrics.evaluate_runner`
契约，因此出现 `evaluate_runner() got multiple values for argument 'poses_per_batch'`。
该失败 claim 的 `testEvaluationCount` 保持为 `0`，没有产生 test 指标，不能被当作一次成功的测试运行。

修复内容：

- 冻结入口改为导入 `evaluate_visual_utility_metrics` 中与实际调用参数一致的评测器；
- 新增回归测试，检查冻结入口包含 `count_budgets`、`target_utility_recall` 参数且不误用统一可见性评测器的 `target_recall` 参数；
- 保留失败输出目录作为审计证据，修复后的真正 test 运行使用新鲜输出目录，避免覆盖失败 claim。

该问题属于评测入口缺陷，不是模型、数据或阈值问题；修复后必须重新执行完整唯一 test split，且仍禁止重新扫描阈值或修复候选集合。

## HKUST 正式冻结 test 结果（修复后）

修复后复用同一份 immutable `frozen_manifest.json`，将首次失败目录保留为
`test_failed_evaluator_contract_20260801`，在新的规范 `test/` 目录执行完整 test。
本次真正的推理遍历 `722` 个唯一 test pose，`testEvaluationCount=1`，设备为 CUDA；没有阈值扫描、候选 GT 补入或候选数量截断。

| 指标 | 结果 | 指标含义 |
|---|---:|---|
| 冻结阈值 | `0.020000` | 仅来自 calibration 安全工作点 |
| Pose precision | `0.462652` | 逐 pose 计算后平均的预测可见集合精确率 |
| Pose recall | `0.946084` | 逐 pose 找回真实可见实例的比例 |
| Weighted recall | `0.997047` | 按 `visible_weights` 加权的可见实例找回率，不是像素覆盖率 |
| Pose accuracy | `0.878341` | 候选集合内 TP 与 TN 的总体比例 |
| Balanced accuracy | `0.879442` | 可见召回与不可见 specificity 的平均 |
| Useful cull | `0.759446` | `TN / candidate`，只统计正确剔除 |
| Bad cull | `0.004613` | `FN / candidate`，表示错误剔除风险 |
| 平均候选实例 | `5037.85` | 后退相机保存的候选集合大小 |
| 平均预测实例 | `457.82` | 冻结阈值下模型保留的实例数 |
| 视觉效用召回 | `0.956932` | 按弱重要性权重统计的效用找回率 |

测试集合共包含 `3,637,329` 个候选引用和 `32,211` 个可见引用。GLB 层面的平均候选
数量为 `934.38`、平均预测数量为 `149.28`；平均候选 GLB 字节约 `163.58 MB`，平均
预测 GLB 字节约 `34.28 MB`。这些 GLB 数值描述当前字节索引下的调度结果，不把固定
特征页和模型权重成本隐藏在“节省”中；时间效用因没有 GLB 解码时间表而未实现。

该结果通过 HKUST 的 M0 冻结测试门，但不能代表 Metropolis 或跨场景泛化结论；后续仍需
完成 Metropolis 的同口径冻结 test、M4 多种子消融和 M5 完整图像评价。

## 收尾脚本可靠性修复

首次自动跟进过程中，HKUST 的冻结入口错误会使带 `set -e` 的并行脚本提前退出，导致 Metropolis
虽然仍在训练，却没有继续进入 M0。现已修复 `run_formal_training_followup.sh`：已完成且包含
`summary.json` 的场景会被明确跳过；不完整输出会拒绝复用；两个场景的后台任务都会等待并汇总
退出状态，不再因单个场景失败而静默跳过另一个场景。Metropolis 的冻结评测槽改用 GPU 3，避免
与 M3/M4 的已登记 GPU 槽冲突。修复后的持久会话为 `formal_m0_followup_retry`，其日志持续写入
`neural_instance_culling/benchmark/out/formal_m0_followup_retry.log`。

## 保留决定与风险

保留为当前主线协议代码和回归测试。剩余风险是正式训练进程启动时可能使用旧版源码；其输出必须按实际日志和 artifact provenance 审计，不能把本次新增测试追溯应用到已经运行的进程。
