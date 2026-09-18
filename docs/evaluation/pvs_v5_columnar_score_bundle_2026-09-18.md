# V5 列式 Score Bundle 与可扩展推理

日期：2026-09-18  
状态：已实现并通过 fixture 验证，尚未生成五个真实场景的正式 V5 数值。

## 目的

旧的 V5 JSON 在每个 pose 内重复保存 candidate、score、target 和 weight。IFCBench
calibration+validation 接近 5000 万候选时，这种表示会生成不可接受的巨大 JSON，并且容易把
冻结 PoseCSR 的标签口径复制错。当前主线改为：PoseCSR 作为唯一的 candidate/visible/weight
来源，模型 score 以列式 `float32` sidecar 保存。

该改动只属于 benchmark 层，不改变 V5 模型、训练 runner、surface/probe/synthetic 生成器。

## 当前数据流

```text
V5 checkpoint
  -> GPU 分块编译 z_i + field/latent
  -> calibration/validation PoseCSR 候选逐 pose 分块前向
  -> scores_float32.bin + pose_indices + score_offsets
  -> manifest loader
  -> calibration threshold
  -> validation/test streaming metrics
```

sidecar 不保存候选 ID、可见 ID 或可见权重。打开 sidecar 时重新读取冻结 PoseCSR，并验证：

| 校验 | 含义 |
|---|---|
| split 完整覆盖 | sidecar 的 pose index 恰好等于该 split 的全部 PoseCSR 行 |
| candidate 对齐 | 每个 score offset 的长度等于对应 PoseCSR candidate 数 |
| visible 子集 | 每个 visible ID 都存在于同 pose candidate 集合 |
| weight 对齐 | visible weight 数量、有限性和非负性正确 |
| shape/dtype | score 只允许 little-endian `float32`，二进制长度与 manifest 一致 |

模型输出保存为 logit，不在 benchmark 导出阶段做 sigmoid 或固定 `0.5` 阈值。校准仍只在
calibration 选冻结阈值，validation 只回放该阈值。

## Test 权限

默认导出 split 为：

```text
calibration + validation
```

只有显式 `--final-test` 才允许写 test sidecar，并将根 manifest 标为 `testRead=true`。
读取 test sidecar 和运行 test evaluation 也都需要显式 `--final-test`。任何默认 calibration
或模型选择路径都不打开 test split。

LOSO held-out 场景只进入几何编译和显式评价 split。推理代码不调用 held-out train split、不读取
held-out external-hit probe，也不更新任何参数；source_global 的阈值仍只来自四个 source
scene 的 calibration sidecar。

## 文件与代码

- `neural_instance_culling/benchmark/v5/score_bundle.py`：列式 writer/reader、PoseCSR 严格校验和 lazy pose stream。
- `neural_instance_culling/benchmark/v5/contracts.py`：columnar bundle、shared/LOSO、test 权限契约。
- `neural_instance_culling/benchmark/v5/inference.py`：checkpoint 加载、GPU 分块几何编译和逐 split score 导出。
- `neural_instance_culling/benchmark/v5/metrics.py`：逐 pose 流式评估；不生成 `per_pose` 大 JSON。
- `neural_instance_culling/benchmark/v5/calibration.py`：正样本分数索引和 calibration-only 阈值选择。
- `neural_instance_culling/benchmark/v5/cli.py`：`export` 与 `evaluate` 命令。
- `neural_instance_culling/benchmark/v5/tests/test_columnar_score_bundle.py`：小 PoseCSR、checkpoint、几何编译、sidecar、权限和矩阵 fixture。

当前 `from_manifest` 只接受 columnar manifest；pose 内嵌数组会被直接拒绝。

## 评价指标

score loader 只负责提供 pose 对齐的连续分数，最终报告仍按论文协议计算：

- **weighted recall / LCB**：按 `visible_weights` 衡量真实可见内容是否被保留，LCB 是校准不确定性下界，属于画面安全指标。
- **普通 recall、FN/GT、bad cull**：衡量可见实例漏保留以及其候选归一化风险。
- **CNOR**：按 pose 候选数归一化的正确遮挡剔除机会比例，衡量跨候选规模的剔除效率。
- **Useful Cull**：`TN / candidate`，只统计正确剔除，不把漏检混入效率。
- **pose PR-AUC**：每个有正样本 pose 单独计算 AP 后取平均，作为论文主 PR-AUC 口径。
- **precision、specificity、balanced accuracy、accuracy、F1、Jaccard**：分类诊断，不单独替代画面安全或剔除效率结论。
- **aggregate AP**：小 split 在受控内存上限内精确计算；超大 split 明确标为 `omitted_large_split`，不为生成诊断指标重新分配巨量内存。

所有指标都在 sidecar loader 层复用同一 PoseCSR candidate/visible/weight 语义，不从 score
JSON 复制标签。

## 验证结果

运行：

```bash
conda run --no-capture-output -n slm_pvs python -m unittest discover \
  -s neural_instance_culling/benchmark/v5/tests -p 'test_*.py' -v
```

结果：`7 tests passed`。fixture 已验证：

- V5 checkpoint 加载和 CPU 分块几何编译接口；
- field 编译、逐 pose score 导出和 lazy metrics/calibration；
- sidecar 只含 `pose_indices_int64.bin`、`score_offsets_uint64.bin`、`scores_float32.bin`；
- PoseCSR candidate 重复或 split/offset 不一致会拒绝；
- test sidecar 没有显式权限时拒绝；
- shared 12-run 和 LOSO 45-fold 矩阵契约。

模型与数据集现有测试也保持通过：model/v5 `27 tests`，dataset/v5 `18 tests`。

## 保留状态与后续命令

该实现保留为 V5 benchmark 主线。下一步是在五个注册真实场景的正式 checkpoint 生成
calibration/validation bundle，然后运行参数扫描的 calibration-only selection 和 validation
比较；模型/阈值冻结后，单独用 `--final-test` 生成一次 test bundle 并报告最终结果。

正式导出命令见：

```text
neural_instance_culling/benchmark/v5/README.md
```

当前未完成项：五场景真实 V5 checkpoint 尚未训练完成，故本文档不填入伪造的正式精度、CNOR
或 PR-AUC 数值。
