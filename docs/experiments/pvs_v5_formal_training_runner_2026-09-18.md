# V5 正式训练编排记录

日期：2026-09-18  
状态：训练入口与扫描计划已实现，正式长训尚未启动。

## 目的

依据 `pvs_v5_generalizable_unified_architecture_2026-09-18.md` 第 3.3、6、6.1、7、8、10 节，
把 V5 的 source-only 训练、shared/LOSO 权限、固定场景调度、参数扫描和可恢复执行固定成
一个入口。训练器不读取 test，也不在训练过程中用 calibration/validation 选择 epoch。

## 实现范围

修改或新增：

| 文件 | 内容 |
|---|---|
| `neural_instance_culling/model/v5/runner.py` | 正式 CLI、shared/LOSO source 目录、固定 schedule、AdamW+cosine、dual LR、日志、checkpoint、pilot/confirmation 矩阵 |
| `neural_instance_culling/model/v5/train.py` | `PBCE_OBJECTIVE` 不再推进约束目标的对偶变量；它只保留 Full 表示和 field supervision |
| `neural_instance_culling/model/v5/training_data.py` | 支持合成场景 manifest 中逐 pose 的 disk/oriented-box 九点区域，保持 16D query 布局一致 |
| `neural_instance_culling/model/v5/tests/test_v5_runner.py` | LOSO 权限、扫描矩阵、更新预算和 RNG 恢复测试 |
| `neural_instance_culling/model/v5/README.md` | 当前入口和恢复协议 |
| `neural_instance_culling/dataset/v5/generate_synthetic_datasets.py` | 将合成 probe manifest 固定写为 `external_hit_probe_manifest.json`，与唯一 loader schema 对齐 |
| `neural_instance_culling/dataset/v5/tests/test_generate_synthetic_datasets.py` | 更新合成 probe manifest 断言 |

没有修改表面采样算法，也没有重建或修改 HKUST/IFCBench 数据。

## 训练权限与场景协议

shared 使用五个真实场景和 96 个 synthetic train scene。LOSO 排除一个真实 held-out scene，
只使用其余四个真实 source scene 和 96 个 synthetic train scene。held-out 场景不会被
`V5SceneTrainingData` 实例化，因此不会读取其 train/calibration/validation/test 标签或 probe。

source scene 的 pose batch 只调用 `split("train")`，field probe 由 source-train 权限 manifest
授权。训练结果中显式记录：

```text
labelSplitsRead: ["train"]
selectionSplitsRead: []
testRead: false
```

训练目录可以物理上包含四路 split 的 CSR 文件，但 runner 不访问 test 行；阈值和指标由
`benchmark/v5` 在训练后按 calibration/validation/test 的冻结顺序处理。

## 固定调度和优化器

每个 optimizer step 只处理一个场景。real scene 按 registry 顺序轮转；每两个 real step 插入
一个 synthetic step，synthetic scene 由固定 seed 洗牌后均匀轮转。全局 yaw 旋转沿用
`train_step` 的四个固定 quarter-turn。

优化器为 AdamW，默认：

```text
model learning rate = 2e-4
dual learning rate  = 3e-3
weight decay        = 1e-5
CosineAnnealingLR   = optimizer step 总数到 0
gradient norm       = 5
```

正式 shared 为每个 real scene 36,000 更新，加一半数量的 synthetic 更新，共 270,000 更新；
四 source LOSO 为 216,000 更新。pilot 使用总计 12,000 更新，confirmation 使用总计 36,000
更新。参数扫描只改变 model LR `{1e-4, 2e-4, 4e-4}` 和 dual LR `{1e-3, 3e-3}`，weight decay、
field coefficient、架构和数据权限固定。

## 参数扫描输出

```bash
conda run --no-capture-output -n slm_pvs \
  python -m neural_instance_culling.model.v5.runner scan \
  --phase pilot --protocol shared --output-dir <scan-dir>
```

该命令产生六行 pilot 矩阵。pilot 完成后只能用 calibration 固定阈值、validation 比较配置，
按规范的词典序选出两个配置，再显式传给 confirmation 命令。runner 不自动读取 test，也不
允许用 test 结果填写“最佳配置”。矩阵每行记录 `selectionSplit=validation`、
`thresholdSplit=calibration` 和 `testRead=false`。

每个成员训练完成后由 `benchmark/v5/cli.py export` 生成列式 calibration/validation 分数，再用
`evaluate-run` 生成五场景结果，最后由 `select-scan` 执行登记的词典序。pilot 固定选两名进入
36k confirmation；confirmation 用相同 selector 选出一组优化器参数。该链路没有 test 入口。

## Checkpoint 与日志

`checkpoint_last.pt` 及 step checkpoint 保存：

```text
model state
AdamW state
Cosine scheduler state
per-source lambda_count/lambda_visual
global step
per-source scene update counts
Python/NumPy/data/PyTorch/CUDA RNG state
run contract and testRead=false
```

恢复时重新生成相同 schedule，从 checkpoint 的 global step 跳过已完成 assignment，并检查当前
scene update count 与 assignment 一致。数据 RNG 从 checkpoint 恢复，因此 pose/probe 采样不会
因恢复改变后续序列。恢复前 `train_metrics.jsonl` 会回退到 checkpoint 的准确 global step，
删除崩溃后尚未入 checkpoint 的重复尾行；不带 `--resume` 时拒绝覆盖已有运行目录。

## 验证结果

本轮没有启动任何长训练。已通过：

```text
V5 model tests       27 passed
V5 benchmark tests   19 passed
registry tests        2 passed
dataset tests         18 passed
runner tests           6 passed
```

另外，`scan --phase pilot` 已实际生成六配置 `scan_matrix.json`，仅执行计划生成，不读取任何
场景数据。正式训练前仍需完成五个真实场景和 96 个 synthetic train scene 的 V5 资产检查；runner
发现缺失 probe、surface、relation 或逐 pose region manifest 时会停止并报告缺失路径。
