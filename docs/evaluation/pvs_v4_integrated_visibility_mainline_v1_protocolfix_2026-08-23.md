# V4 综合可见性主线正式评价协议修正

日期：2026-08-23

## 修正目的

三种子 Full 长训在 checkpoint 内冻结 calibration 阈值时使用了 `2,000` 次 bootstrap，四组消融使用了 `10,000` 次。两者不能进入同一正式比较。本次不重训网络，不读取 test，也不改变候选、GT、split、实例特征或前端资产；只对 Full 保留的每四个 epoch 快照重新执行 calibration 和 validation，并统一使用 `10,000` 次 bootstrap。

每个种子重审计 epoch `4, 8, ..., 40` 共十个 checkpoint。阈值只由该 checkpoint 自己的 `659` 个 calibration pose 冻结；validation 只在冻结阈值上回放。checkpoint 先满足 calibration 和 validation 的 aggregate weighted recall 点估计及单侧 95% 下界均大于 `0.99`，再按 validation balanced accuracy、precision、accuracy、useful cull 和平均预测数排序。最后把修正后的 Full 与四组消融按相同 `730` 个 validation pose 做三种子聚类、种子内 pose 重采样的 `10,000` 次 paired bootstrap。

运行命令：

```bash
conda run -n slm_pvs python \
  neural_instance_culling/benchmark/reaudit_pvs_v4_integrated_visibility_mainline_v1.py \
  run --data-root /mnt/sda/rhyang/slm --bootstrap-replicates 10000 \
  --gpu-ids 0 1 2 3
```

实现变更：

- `evaluate_pvs_bounded_relation_survival_moment_v4.py`：诊断重校准显式记录 calibration 与 validation 的独立 bootstrap seed；
- `reaudit_pvs_v4_integrated_visibility_mainline_v1.py`：新增正式快照重审计、checkpoint 元数据冻结、逐 pose 回放和 paired bootstrap 入口；
- `run_pvs_v4_integrated_visibility_mainline_v1.py`：新正式训练固定 `10,000` 次 bootstrap，并拒绝把历史 `2,000` 次成员重新汇总为正式结果；
- 对应 unittest 覆盖完整快照集合、`10,000` 次下限、validation 安全优先、三种子矩阵和 test 未读取约束。

依赖资源保持为当前 HKUST 主数据集、固定运行元数据、96 维几何表、分层关系 CSR、GLB 索引，以及 Full 三种子的 epoch `4, 8, ..., 40` checkpoint。没有重新生成数据或特征表。

## Full 冻结结果

提高 bootstrap 次数后，三个种子的最佳 checkpoint 和阈值均未改变。

| 种子 | epoch | calibration 阈值 | validation weighted recall | 单侧 95% 下界 | 安全门 |
|---:|---:|---:|---:|---:|---|
| 20260801 | 40 | 0.46 | 0.99851 | 0.99746 | 通过 |
| 20260802 | 24 | 0.60 | 0.99795 | 0.99636 | 通过 |
| 20260803 | 32 | 0.56 | 0.99744 | 0.99593 | 通过 |

## 五变体三种子均值

pose-level 指标先对每个 pose 独立计算，再对 `730` 个 pose 和三个种子平均，每个视点权重相同。Full 的 pose precision 为 `0.48291`，与协议修正前的同 checkpoint 结果一致；模型没有因为重审计而退化。

| 变体 | Pose precision | Pose recall | Pose accuracy | Pose balanced accuracy |
|---|---:|---:|---:|---:|
| Full | 0.48291 | 0.97543 | 0.90419 | 0.92104 |
| 去除分层关系先验 | 0.48211 | 0.97100 | 0.88807 | 0.90947 |
| 去除视点区域矩包络 | 0.45464 | 0.97552 | 0.90145 | 0.92019 |
| 去除 RVL 召回保护 | 0.41756 | 0.98274 | 0.89774 | 0.92165 |
| 去除对比分离 | 0.48463 | 0.97722 | 0.91022 | 0.92663 |

aggregate 指标先合并所有 pose 的 TP、FP、FN、TN，再计算比例。由于各 pose 的候选数和预测数差异很大，大规模 pose 在该口径中权重更高，因此 Full aggregate precision 为 `0.19799`，不能与 pose precision `0.48291` 直接互换。`precision` 衡量预测可见实例中真实可见的比例；`accuracy` 是全部候选实例的逐实例正确率；`balanced accuracy` 对可见与不可见两类等权；`useful cull` 是正确剔除的不可见实例占候选比例；`bad cull` 是错误剔除的可见实例占候选比例。

| 变体 | Aggregate precision | Aggregate recall | Weighted recall | LCB | Aggregate accuracy | Aggregate balanced accuracy | Useful cull | Bad cull | 平均预测数 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Full | 0.19799 | 0.98063 | 0.99797 | 0.99658 | 0.90827 | 0.94360 | 0.88575 | 0.000445 | 540.34 |
| 去除分层关系先验 | 0.16042 | 0.97824 | 0.99719 | 0.99501 | 0.88036 | 0.92815 | 0.85790 | 0.000500 | 672.32 |
| 去除视点区域矩包络 | 0.20832 | 0.97495 | 0.99790 | 0.99654 | 0.91269 | 0.94308 | 0.89030 | 0.000575 | 518.12 |
| 去除 RVL 召回保护 | 0.19546 | 0.98157 | 0.99721 | 0.99548 | 0.90606 | 0.94293 | 0.88352 | 0.000423 | 551.04 |
| 去除对比分离 | 0.21353 | 0.97934 | 0.99729 | 0.99535 | 0.91646 | 0.94716 | 0.89397 | 0.000474 | 501.17 |

五个变体的三个种子均通过修正后的 validation weighted recall 安全门。普通 recall、precision、accuracy 和 balanced accuracy 仍同时报告，不能仅凭 weighted recall 判定分类质量。

## 配对比较结论

- 分层关系先验具有稳定分类与剔除贡献。Full 相对去除关系先验的 aggregate precision 提高 `0.03757`，95% CI `[0.02132, 0.05168]`；accuracy 提高 `0.02791`，CI `[0.01245, 0.04419]`；balanced accuracy 提高 `0.01545`，CI `[0.00958, 0.02193]`；useful cull 提高 `0.02785`，CI `[0.01204, 0.04430]`；平均预测数减少 `131.98`，CI `[-211.60, -55.11]`。weighted recall 差值区间跨零，没有稳定安全性恶化。
- 视点区域矩包络在当前训练和阈值协议下没有形成稳定收益。Full 与去除矩包络之间的 precision、recall、weighted recall、accuracy、balanced accuracy、useful cull、平均预测数和 GLB 字节差值均跨零。
- RVL 召回保护对实例分类指标的差值均跨零，但 Full 的平均预测 GLB 字节比去除 RVL 少约 `12.23 MB/pose`，95% CI `[-17.08, -5.82] MB`。该资源差异需要后续图像评价确认是否处于相同视觉效用，暂不单独解释为分类贡献。
- 当前对比分离项对分类和剔除有稳定负面影响。Full 相对去除对比分离的 aggregate precision 下降 `0.01554`，accuracy 下降 `0.00819`，balanced accuracy 下降 `0.00356`，useful cull 下降 `0.00822`，并多预测约 `39.17` 个实例；这些区间均不跨零。weighted recall 差值跨零。因此对比分离项不能按当前实现包装为有效创新。

## 输出与限制

正式修正输出：

- `formal40_s02_full_protocolfix_10000_summary.json`：三种子 Full 的 checkpoint、阈值和完整 validation 指标；
- `formal40_s02_paired_bootstrap_10000_summary.json`：五变体三种子的逐 pose 汇总与配对置信区间；
- `formal40_s02_protocolfix_10000/`：30 个 checkpoint 重审计、三个冻结成员及日志。

本次只修正实例级 validation 协议。硬件 Color-ID 图像指标尚未回填，test split 保持未读取，浏览器 WebGPU 和移动端延迟也未在本次重审计中重复测量。结果不修改当前默认 checkpoint、阈值和前端资产。
