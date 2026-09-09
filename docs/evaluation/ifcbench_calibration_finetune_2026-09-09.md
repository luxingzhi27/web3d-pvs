# IFCBench Calibration and Fine-Tune

日期：2026-09-09

状态：执行中。本文记录 IFCBench 精确校准、warm-start 微调扫描和确认实验；在长任务完成后回填实际指标，不把未完成任务写成模型结论。

## 目的与边界

本轮只验证现有 V4 可见性模型在 IFCBench/Fantasy Metropolis 上的精确 calibration 和从已有模型继续训练是否改善安全工作点。视觉效用、GLB 下载排序、HZB、streaming、前端默认资产和 HKUST 模型均不在本分支范围内。

共享源数据和原始 checkpoint 显式来自 `/mnt/sda/rhyang/slm`；新模型、sidecar、bootstrap 和 benchmark 日志写入本 worktree 的 `neural_instance_culling/model/out/` 与 `neural_instance_culling/benchmark/out/`。这些大型目录被 gitignore，提交不包含软链接或数据副本。

IFCBench 数据契约为 `41,298` 个实例、`3,669` 个 GLB、`19,647 / 2,183 / 2,712 / 2,710` 个 train/calibration/validation/test pose。正式训练和校准只解析 train、calibration、validation；test 只在 metadata 中核对 split 计数，不读取 test 样本，也不运行 test evaluator。

## 实现

提交 `27fbe56` 增加：

- `train_pvs.py` 的显式 `--init-checkpoint`。它严格验证当前 V4 checkpoint/runtime schema 和架构配置，加载模型状态后重新创建 fresh AdamW，不读取来源优化器状态；记录来源 seed、epoch、global step、继承的实例校准 blend 和当前额外更新数。warm-start 从第一步保持完整困难边界项。
- `ifcbench_exact_calibration.py`。`score` 子命令按 split 的固定 pose 顺序写入 `scores_f32.bin`、`labels_f32.bin`、`weights_f32.bin`，三者与 `pose_offsets_u64.bin` 一一对齐，并保存 `pose_indices_i64.bin`。标签是候选实例是否在 view-cell 可见实例并集中，权重是 `visible_weights`，不解释为真实像素覆盖率。
- 精确校准使用 score 的实际 float32 change-points，预测规则固定为 `score >= threshold`。安全工作点是最高 change-point，且 aggregate weighted recall 和固定 pose bootstrap 的单侧 95% 下界均严格大于 `0.99`。固定 bootstrap 索引写入公共文件并跨配置复用，正式数量为 `10,000` 组；零 GT/零权重 pose 不进入 aggregate bootstrap 分母，但仍保留在 offsets 和普通 pose 统计中。
- `run_ifcbench_calibration_finetune.py`。它从源数据 root 组装五组扫描、公共 calibration/validation bootstrap、validation 冻结阈值和三 seed confirmation 命令；每个长任务独立保存 stdout/stderr。

提交 `1c1d99d` 为训练日志补充 `stepSeconds`、`stepsPerSecond` 和 `etaSeconds`。本次已启动的 scan 进程在该提交前启动，因此其日志保留 `epoch/step/loss/elapsedSeconds`；后续 confirmation 使用完整吞吐和 ETA 字段。

## 代码契约与 smoke

运行命令：

```bash
conda run --no-capture-output -n slm_pvs \
  python neural_instance_culling/benchmark/run_ifcbench_calibration_finetune.py \
  preflight \
  --data-root /mnt/sda/rhyang/slm \
  --model-root /mnt/sda/rhyang/slm-wt-ifc/neural_instance_culling/model/out/pvs_ifcbench_v4_calibration_finetune_v1 \
  --benchmark-root /mnt/sda/rhyang/slm-wt-ifc/neural_instance_culling/benchmark/out/pvs_ifcbench_v4_calibration_finetune_v1
```

preflight 已通过，核对了上述 split、V4 relation CSR、96 维几何表、124 维运行 schema 和三份原始 `best_safe.pt`。

真实 GPU smoke 使用 1 epoch、2 updates、2 个评估 pose；输出为 `model/out/pvs_ifcbench_v4_calibration_finetune_v1/smoke/warm_start/seed20260802_e1`。实际结果：训练完成，来源 `seed=20260802, epoch=40, globalStep=36000`，继承 `instanceCalibrationBlend=1.0`，fresh AdamW 的 state step 为 `1..2` 且 `stateLoaded=false`，首个训练 epoch 的 `sharedTailSeparationScale=1.0`，`testRead=false`。新增和原有相关单测共 10 项通过，Python 编译和 diff 检查通过。

## 扫描协议

扫描从 IFCBench 原始 `seed20260802` 的 `best_safe.pt` 开始，五组均为 `4 x 900` updates、四 pose batch、8192 observation batch、学习率和困难边界设置如下：

| 配置 | RVL | 边界权重 | margin / temperature | LR |
|---|---:|---:|---:|---:|
| `original_loss` | 0.30 | 0.20 | 0.50 / 0.25 | 2e-5 |
| `boundary_half` | 0.30 | 0.10 | 0.50 / 0.25 | 2e-5 |
| `boundary_removed` | 0.30 | 0 | 0.50 / 0.25 | 2e-5 |
| `soft_boundary` | 0.30 | 0.20 | 0.25 / 0.50 | 2e-5 |
| `higher_lr` | 0.30 | 0.20 | 0.50 / 0.25 | 5e-5 |

训练器内部的旧网格诊断使用 1 个 bootstrap draw 以避免重复昂贵校准；正式阈值只使用 sidecar 的固定 10,000 pose bootstrap，不使用训练器内部阈值作为结论。

扫描命令：

```bash
conda run --no-capture-output -n slm_pvs \
  python neural_instance_culling/benchmark/run_ifcbench_calibration_finetune.py \
  scan \
  --data-root /mnt/sda/rhyang/slm \
  --model-root /mnt/sda/rhyang/slm-wt-ifc/neural_instance_culling/model/out/pvs_ifcbench_v4_calibration_finetune_v1 \
  --benchmark-root /mnt/sda/rhyang/slm-wt-ifc/neural_instance_culling/benchmark/out/pvs_ifcbench_v4_calibration_finetune_v1 \
  --gpu-ids 0 1 2 3
```

截至本记录写入时，scan runner PID 为 `1715972`，四个首批配置已到 `epoch 1 / step 180`，每组约 `559 s`；四张 RTX A6000 显存约 `18.6 GiB`，GPU 利用率处于计算状态，stderr 为空。第五组会在首批任务完成后启动。日志目录为 `benchmark/out/pvs_ifcbench_v4_calibration_finetune_v1/logs/scan/`，模型目录为 `model/out/pvs_ifcbench_v4_calibration_finetune_v1/scan/`。

## 未完成任务

以下任务尚未产生可报告的正式结果：

1. 五组 `seed20260802` 扫描完成后的 calibration score sidecar、实际 change-point 阈值和 validation 冻结指标。
2. 五组相对选择表。安全成员优先按 validation weighted recall 及其下界过滤，再比较 useful cull、balanced accuracy、precision 和预测数量；若没有安全成员，按 validation weighted recall 下界最高的成员继续。
3. 选中配置从三个原始 checkpoint 各自重新开始的 `8 x 900`、三 seed confirmation，以及每 1800 updates 的评价日志。
4. 本分支不实现图像 PER、移动端延迟、GLB 字节削减和 test 结果；这些指标在其他实验职责中单独完成，不能从本轮 calibration 结果推断。

每一项完成后必须补写实际 pose 数、阈值、普通集合指标、weighted recall/下界、useful cull、bad cull、balanced accuracy、specificity、AP、sidecar 路径、日志路径和是否满足安全门。

`testRead` 在当前 smoke、训练 manifest、checkpoint、sidecar schema 和本报告的已完成阶段均为 `false`。
