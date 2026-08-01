# M3 正式代理干预执行记录（2026-08-01）

## 目的

M3 用同一个正式 checkpoint 做推理期干预，判断固定方向遮挡代理和上下文特征是否对可见性输出具有独立贡献。该实验不更新模型参数、不重选阈值，也不打开封存的 test split。

## 执行方式

新增 `neural_instance_culling/benchmark/run_formal_m3_interventions.sh` 作为收尾入口。脚本等待对应训练目录写入 `calibration_ready_summary.json`，再等待 M0 one-shot test 完成，随后在完整 validation split 上调用 `evaluate_proxy_interventions.py`。两个场景分别占用 GPU 0 和 GPU 2，输出目录按正式实验名独立保存，已有结果不会被覆盖。

每个场景固定以下干预并保存逐 pose 结果和 10,000 次配对 bootstrap：原始模型、代理清零、同分布随机代理、方向平均、方向循环移位、跨实例代理置换、上下文清零、跨实例上下文置换、上下文与代理同时清零。候选集合来自数据集中保存的后退相机候选，禁止 GT 补候选；阈值来自 calibration 记录，不能使用测试集扫描结果。

## 结果状态

HKUST 正式 checkpoint 已完成 M3；Metropolis 仍等待正式 calibration-ready checkpoint。HKUST
输出目录为：

```text
neural_instance_culling/benchmark/out/m3_formal_pvs_directional_occlusion_proxy_encoder_rvl_strong_v2_full40_hkust_spatial_fov66_seed20260801_protocolfix_retry2_validation/
```

本次使用 calibration 冻结阈值 `0.02`，在完整 validation 的 `664` 个 pose 上执行九种干预，
每个变体共享同一候选集合，并运行 `10,000` 次 paired bootstrap。代表性逐 pose 结果如下：

| 变体 | Pose precision | Pose recall | Weighted recall | Useful cull | Bad cull | 平均预测 |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 0.446219 | 0.904137 | 0.991802 | 0.877723 | 0.009547 | 221.73 |
| proxy zero | 0.440359 | 0.897835 | 0.991601 | 0.876430 | 0.010033 | 247.27 |
| proxy random | 0.492361 | 0.869254 | 0.988547 | 0.887352 | 0.011891 | 223.14 |
| direction mean | 0.491488 | 0.879813 | 0.990159 | 0.885368 | 0.010646 | 202.82 |
| direction roll | 0.454790 | 0.903501 | 0.991779 | 0.878660 | 0.009524 | 212.11 |
| proxy cross-instance permutation | 0.470260 | 0.870182 | 0.988557 | 0.885671 | 0.011796 | 221.45 |
| context zero | 0.170041 | 0.934004 | 0.987467 | 0.729636 | 0.003626 | 286.07 |
| context cross-instance permutation | 0.512473 | 0.556397 | 0.763175 | 0.896647 | 0.024688 | 208.66 |
| context + proxy zero | 0.158226 | 0.951072 | 0.991636 | 0.720109 | 0.003419 | 331.30 |

相对于 `proxy_zero`，baseline 的 useful-cull 配对差为 `-0.001293`（95% CI
`[-0.002912, 0.000235]`），weighted-recall 差为 `-0.000202`（95% CI
`[-0.000988, 0.001003]`）；但 baseline precision、recall 和 F1 均有统计上的改善。
这说明代理分支在当前模型中被实际读取，清零会改变输出，但尚未证明它在安全工作点上带来
稳定的 useful-cull 增益，也没有达到预注册的 2 个百分点门槛。随机代理和跨实例置换的
weighted recall 降到约 `0.98855`，因此不能视作保持安全约束的等价替代。

当前只能引用旧模型的探索性干预结果，不能作为正式模型证据。正式路线判断仍遵循投稿计划：相对于 `geometry + context + ray`，只有在安全工作点下 useful cull 提升至少 2 个百分点，或同一图像效用下字节减少至少 10%，且三种子 paired bootstrap 不跨零时，才保留“方向代理具有独立贡献”的主张；否则转为路线 B 并删除该主张。

## 可复现命令

```bash
cd /mnt/sda/rhyang/slm
tmux new-session -d -s formal_m3_followup \
  'bash neural_instance_culling/benchmark/run_formal_m3_interventions.sh 2>&1 | tee neural_instance_culling/benchmark/out/formal_m3_followup.log'
```

## 当前结论

- 目标：已登记并完成 HKUST 单种子正式干预；Metropolis 和三种子重训练仍待完成。
- 代码：正式 runner、候选配对校验和 bootstrap 结果均已生成。
- 指标：HKUST 已生成；方向代理存在可观测影响，但当前单种子结果未满足路线 A 的独立 useful-cull 准入条件。
- 是否保留为主线：暂不把“方向代理独立贡献”写入主张；等待 M4 三种子结果，若仍不达门槛则转路线 B。
