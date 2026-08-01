# M0：固定验证、独立校准与一次性测试协议

日期：2026-08-01  
状态：协议代码与正式候选语义门控已实现并通过静态/资源 smoke；正式空间四路训练与投稿级 one-shot test 尚未通过准入。

## 变更目的

旧训练流程在每轮验证时以随机有放回的姿态子集比较 checkpoint，训练结束后又在测试集扫描阈值并选择工作点。这样会使 checkpoint 与阈值间接适配测试集，不能作为论文主结果协议。

本次先修复最小实验协议：完整且固定的验证集合负责 checkpoint 选择；从原训练集合中固定留出、且不参与参数更新的校准集合负责阈值选择；冻结阈值后，测试集合只运行一次。

## 当前实现

修改文件：

- `neural_instance_culling/model/pose_csr_dataset.py`
  - 增加共享底层 CSR 的显式姿态子集视图，不复制原始采样数据。
- `neural_instance_culling/model/train_directional_occlusion_proxy_encoder.py`
  - 增加基于固定 seed 的训练/验证/校准/测试协议切分。
  - 原 `val` 全量作为固定验证集合，不再使用 `--eval-pose-steps` 截断。
  - 原 `train` 的 10% 姿态固定留出为 calibration，其余用于参数更新。
  - 每轮只在 calibration 扫描阈值，再以该阈值评估完整 validation 并选择 checkpoint。
  - 训练结束和 `--export-eval-checkpoint` 路径均先重新校准，再以冻结阈值单次评估 test。
  - 输出 `protocol_split.json`、`frozenThreshold`、校准表和 `testEvaluationCount=1`。
- `neural_instance_culling/model/current_pvs_utils.py`
  - 正式评估支持收集每个 view-cell 的 weighted recall，用于校准置信下界。
  - 前端阈值优先读取冻结摘要中的单一 `frozenThreshold`，不再从测试阈值表重新推导。
- `neural_instance_culling/model/common/threshold_selection.py`
  - 增加校准点估计下限和 view-cell bootstrap 单侧下置信界过滤。
- `neural_instance_culling/model/pose_csr_dataset.py`
  - 正式 pose-set 构造不再自动把 GT 可见实例并入候选；存储候选漏正样本或候选上限会直接失败。
  - 旧数据的候选重算/正样本补入只能通过显式探索性参数开启。
- `neural_instance_culling/benchmark/evaluate_unified_pvs_metrics.py`
  - `--split test` 默认要求每个模型提供冻结阈值 JSON，并遍历完整唯一 test pose；历史 test 阈值扫描改为显式 `--exploratory-test-threshold-scan`。
  - 输出记录冻结阈值来源、候选语义模式和 one-shot 标记。
- `neural_instance_culling/benchmark/build_frozen_threshold_manifest.py`
  - 只接受带有 `frozen_calibration_one_shot_test`、校准/test digest 和
    `testEvaluationCount=1` 的训练摘要，生成正式 benchmark 使用的模型阈值清单。
  - 旧的 test 扫描摘要会被拒绝，不会被转换成正式阈值。
- `neural_instance_culling/dataset/build_directional_occlusion_evidence.py`
  - 正式遮挡证据构建复用存储候选并拒绝候选漏正样本，避免证据阶段再次隐藏候选错误。

## 阈值规则

校准候选必须同时满足：

1. `pose_weighted_recall > 0.99`；
2. 校准 weighted recall 点估计 `>= 0.9925`；
3. 正式最终校准启用 view-cell 聚类 bootstrap 后，单侧 95% 下置信界 `> 0.99`。

在安全候选中按 pose precision 最大化，其次比较 pose F1、weighted recall，并优先选择平均预测数量较少的工作点。测试集不参与任何阈值扫描、回退或重选。

这里的 weighted recall 使用采样数据记录的 `visible_weights`。HKUST 当前语义是 Three.js Color-ID 覆盖权重的 view-cell 内最大值，不是真实像素数；正式报告必须继续明确这一点。

## 验证结果

使用当前 HKUST CSR：

```text
原始 train / val / test：6585 / 730 / 684
协议 train_fit / validation / calibration / test：5927 / 730 / 658 / 684
selection seed：20260610
validation digest：526bc2ffd5582145
calibration digest：5ffe8c318242b541
```

已验证：

- train-fit 与 calibration 无交集；
- calibration、validation、test 互不重叠；
- 相同 seed 重建得到相同集合和摘要哈希；
- bootstrap 工作点过滤和 one-sided lower-bound 字段可运行；
- 相关 Python 文件通过 `py_compile`。
- 正式 HKUST 和 Metropolis 空间数据在 `strict_semantics=True` 下资源检查通过；抽样 pose 的批构造结果与存储候选逐项一致，未新增 GT 实例。
- 没有冻结阈值文件时，正式 test benchmark 会在参数解析阶段拒绝运行；这防止旧脚本无意间重新扫描 test。

验证命令：

```bash
conda run -n slm_pvs python -c '... build_protocol_splits(...) ...'
conda run -n slm_pvs python -m py_compile \
  neural_instance_culling/model/pose_csr_dataset.py \
  neural_instance_culling/model/common/threshold_selection.py \
  neural_instance_culling/model/current_pvs_utils.py \
  neural_instance_culling/model/train_directional_occlusion_proxy_encoder.py
```

## 重要边界

本轮 calibration 是从现有随机 train split 中留出的协议性独立集合，解决了阈值与 test 污染，但没有解决原始 view-cell 划分的空间泄漏。正式主表仍必须等待 M2 的空间 block 四路 split：同一物理 view-cell 及其所有 subpose 必须归入同一集合，并保存不可变 manifest 与 SHA-256。

Metropolis 还存在点云缓存与实例化 GLB 原型索引不一致，以及原始 AABB 候选漏正样本被构建器补入的问题，因此本轮协议不能使其结果自动获得论文准入资格。

## 准入判断

当前 M0 子门通过：代码不再在 formal test 扫描阈值，验证集合固定且完整，校准与测试路径已分离，候选集合不会在评测器内被 GT 补入。M0 总门仍待完成：

- 使用空间隔离四路数据集重建协议；
- 至少完成一次真实 checkpoint 的 calibration 与 frozen one-shot test；
- 输出按 view-cell 的逐样本明细、置信下界和完整日志；
- 复核前端只读取冻结阈值。

在这些条件完成前，旧的 `0.64` 仍只能标记为 `exploratory_test_calibrated`，不能写入投稿主表。

## 2026-08-01 协议审计修正：分离 checkpoint 选择与最终阈值冻结

审计发现训练结束时存在一个元数据一致性缺陷：训练期间的 `best.pt` 使用每轮 calibration 的点估计工作点选择 checkpoint，而最终导出阶段会在同一 checkpoint 上重新运行带 view-cell bootstrap 的 calibration。最终阈值可能因此不同，但旧代码没有把最终 calibration 记录回写到 checkpoint，导致严格冻结清单无法区分“哪一个 epoch 被 validation 选中”和“最终 test 使用哪一个冻结阈值”。

修正后的 checkpoint 同时保存：

- `checkpointSelection`：由完整 validation 历史选择的 epoch、阈值和指标，负责审计 checkpoint 选择没有使用 test；
- `best` 与 `workpoints`：最终独立 calibration（含 bootstrap 下置信界）和在冻结阈值下重新评估的完整 validation 记录，负责审计 test 阈值来源。

`evaluate_frozen_test.py prepare` 现在分别核验这两条证据，并要求最终 calibration 阈值满足点估计和单侧下置信界安全门；它不因此放宽任何安全要求。当前正在运行的训练进程使用修改前已加载的 Python 代码，完成后将通过导出/整理步骤补齐相同元数据，再建立不可变冻结清单。测试集合仍禁止重新扫描阈值。

## 2026-08-01 运行中审计快照

截至本次检查，正式空间训练仍在自然运行：HKUST 已完成约第 14 个 epoch，Metropolis 已完成约第 7 个 epoch。两个进程均使用 CUDA、FP32（未开启 AMP）、每 epoch 900 steps、`rvl_strong_v2`、候选集合不裁剪且不补入 GT 可见实例；日志中的非有限 loss 和非有限梯度跳过计数均为零。

本次核对还发现两个输出目录名带有 `seed20260801`，但命令没有显式传入 `--seed`，代码实际使用的选择种子是脚本默认值 `20260610`。这不改变当前训练的数值，但会造成复现实验歧义；训练结束后的 artifact manifest 必须同时记录“目录标签”和“实际 seed”，后续正式多种子实验必须显式传入 seed，不能继续依赖默认值。

当前输出尚未出现 `instance_runtime_features_fp16.bin`、`eval_summary.json` 或 frozen-test 字段，因此 M0 总门仍未通过。训练自然结束后按以下顺序处理：

1. 只读确认是否已经写出 `protocol=frozen_calibration_one_shot_test`、`testEvaluationCount=1`、完整 calibration bootstrap 和严格安全门；
2. 如果 summary 已包含 one-shot test，只用 `finalize_frozen_checkpoint.py` 将最终 calibration 元数据写入新的 checkpoint 副本，不覆盖原始 `best.pt`，也不重新评测 test；
3. 如果训练进程异常退出且没有 test summary，先确认 test 从未运行，再使用独占的 frozen manifest 和 `evaluate_frozen_test.py evaluate` 完成一次正式 test。

## 2026-08-01 后续训练的 test 解耦入口

为避免训练器自动读取 test 造成“训练收尾”和“正式冻结评测”边界不清，
`train_directional_occlusion_proxy_encoder.py` 新增 `--skip-final-test`。启用后训练器只完成：

- 最终 calibration（含注册的 view-cell bootstrap 安全门）；
- 完整 validation 在冻结阈值下的记录；
- 固定实例运行特征导出；
- `checkpointSelection`、`workpoints` 和 `finalCalibration` 元数据固化。

它写出 `calibration_ready_summary.json`，明确标记 `testEvaluationCount=0`，然后由
`evaluate_frozen_test.py prepare` 生成不可变清单，`evaluate_frozen_test.py evaluate` 在独占目录中
执行唯一一次完整 test。`--skip-final-test` 不允许与 `--export-eval-checkpoint` 同时使用。

本选项只用于当前正在运行的训练结束后的新实验和后续多种子实验；已经启动的旧进程不受源码修改影响，
其收尾结果仍按“内置 one-shot test 或失败”分支审计，不能用新选项追溯改变其 test 计数。
