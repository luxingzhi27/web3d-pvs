# GCOF-PVS V5 Formal Trainer

`runner.py` 是 V5 正式训练的唯一编排入口。`train.py` 只负责一个 source
scene 的前向、约束损失、对偶更新和精确几何梯度回传；runner 负责场景权限、固定
轮转、优化器、日志和恢复。

## 训练权限

训练器只从每个 source scene 的 `train` pose 取标签和 pose probe。它不会读取
`calibration`、`validation` 或 `test` pose 行。LOSO 只把 held-out scene 的几何
资产列入候选目录，不实例化其标签或 probe。阈值校准和 validation 比较由
`benchmark/v5` 在训练完成后独立执行。

## 入口

生成六个 12k pilot 的参数矩阵，不启动训练：

```bash
conda run --no-capture-output -n slm_pvs \
  python -m neural_instance_culling.model.v5.runner scan \
  --phase pilot --protocol shared --output-dir <scan-dir>
```

确认阶段必须显式传入 pilot 后由 validation 选出的两个配置名：

```bash
conda run --no-capture-output -n slm_pvs \
  python -m neural_instance_culling.model.v5.runner scan \
  --phase confirmation --protocol shared --seed 0 \
  --selected-config model_lr1e-04_dual_lr1e-03 \
  --selected-config model_lr2e-04_dual_lr3e-03 \
  --output-dir <confirmation-dir>
```

上面两个命令默认只写 `scan_matrix.json`。只有增加 `--execute` 才会执行训练。

训练一个正式 shared 成员：

```bash
conda run --no-capture-output -n slm_pvs \
  python -m neural_instance_culling.model.v5.runner train \
  --protocol shared --variant FULL --seed 0 \
  --real-updates-per-scene 36000 \
  --output-dir <run-dir> --device cuda \
  > train_stdout.log 2> train_stderr.log
```

LOSO 使用 `--protocol loso --held-out-scene <scene-id>`。LOSO 不接受
`PBCE_OBJECTIVE`，因为该项只属于 shared 消融。

## 固定训练矩阵

可用变体为 `FULL`、`GEOMETRY_FIELD`、`GENERIC_RELATION_28` 和
`PBCE_OBJECTIVE`。最后一项使用 Full 表示架构，但把约束任务目标换成注册的
pose-balanced BCE；它不是另一套网络。

每两个 real scene step 插入一个 synthetic scene step，real scene 按 registry
顺序轮转，synthetic scene 由固定 schedule seed 均匀轮转。正式 shared 成员使用每个
real scene 36,000 次更新，再使用其一半数量的 synthetic 更新，共 270,000 次；四
source LOSO fold 共 216,000 次。参数扫描的 pilot 和 confirmation 分别是总计
12,000 与 36,000 次更新。

## 恢复与输出

每个 run 目录写入：

```text
run_config.json
train.log
train_metrics.jsonl
checkpoint_last.pt
checkpoint_step_<global-step>.pt
training_summary.json
```

checkpoint 包含模型、AdamW、CosineAnnealingLR、每个 source scene 的两个对偶乘子、
global step、scene update counts、Python/NumPy/PyTorch/CUDA/data RNG。`--resume`
会校验 schema、超参数、source scene 集合和更新计数，不接受旧 V4 checkpoint 或
test-tainted 状态。

正式训练依赖五个真实场景的 V5 surface/relation/probe 资产以及已生成的 96 个
synthetic train scene；资产不完整时 runner 会在进入训练前失败，不会静默跳过场景。
