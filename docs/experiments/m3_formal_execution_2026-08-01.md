# M3 正式代理干预执行记录（2026-08-01）

## 目的

M3 用同一个正式 checkpoint 做推理期干预，判断固定方向遮挡代理和上下文特征是否对可见性输出具有独立贡献。该实验不更新模型参数、不重选阈值，也不打开封存的 test split。

## 执行方式

新增 `neural_instance_culling/benchmark/run_formal_m3_interventions.sh` 作为收尾入口。脚本等待对应训练目录写入 `calibration_ready_summary.json`，再等待 M0 one-shot test 完成，随后在完整 validation split 上调用 `evaluate_proxy_interventions.py`。两个场景分别占用 GPU 0 和 GPU 2，输出目录按正式实验名独立保存，已有结果不会被覆盖。

每个场景固定以下干预并保存逐 pose 结果和 10,000 次配对 bootstrap：原始模型、代理清零、同分布随机代理、方向平均、方向循环移位、跨实例代理置换、上下文清零、跨实例上下文置换、上下文与代理同时清零。候选集合来自数据集中保存的后退相机候选，禁止 GT 补候选；阈值来自 calibration 记录，不能使用测试集扫描结果。

## 结果状态

本记录创建时两个正式训练仍未完成，尚无正式 M3 数值。脚本运行日志和 JSON 指标将在以下目录生成：

```text
neural_instance_culling/benchmark/out/m3_formal_<experiment>_validation/
```

当前只能引用旧模型的探索性干预结果，不能作为正式模型证据。正式路线判断仍遵循投稿计划：相对于 `geometry + context + ray`，只有在安全工作点下 useful cull 提升至少 2 个百分点，或同一图像效用下字节减少至少 10%，且三种子 paired bootstrap 不跨零时，才保留“方向代理具有独立贡献”的主张；否则转为路线 B 并删除该主张。

## 可复现命令

```bash
cd /mnt/sda/rhyang/slm
tmux new-session -d -s formal_m3_followup \
  'bash neural_instance_culling/benchmark/run_formal_m3_interventions.sh 2>&1 | tee neural_instance_culling/benchmark/out/formal_m3_followup.log'
```

## 当前结论

- 目标：已登记。
- 代码：已实现并通过 shell 静态检查和 Python 入口自检；`formal_m3_followup` tmux 会话已启动，当前等待正式 checkpoint 和 M0 one-shot 结果。
- 指标：未生成，不能宣称代理有效或无效。
- 是否保留为主线：待正式干预和三种子重训练结果决定。
