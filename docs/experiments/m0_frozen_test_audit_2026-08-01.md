# M0 Formal Protocol 独立审计与冻结测试入口

日期：2026-08-01  
状态：冻结测试入口与 fixture 已完成；正式空间数据的 M0 总门仍未通过。  
范围：本次新增 `neural_instance_culling/benchmark/evaluate_frozen_test.py`，并在后续协议审计中同步修正训练 checkpoint 的最终 calibration 元数据回写；未修改 `model_runners.py`、M5 图像渲染文件或 M6 baseline 文件。

## 审计结论

当前训练实现已经具备 M0 协议的主要数据流。只读检查确认：

1. `build_protocol_splits` 在存在原生四路划分时使用完整的 validation、独立 calibration 和冻结 test；没有把 calibration 或 test 用于参数更新。
2. 每次 checkpoint 评估通过完整 validation 记录工作点，训练历史中的 `val.eval_pose_count` 可用于检查是否覆盖全部 validation，而不是随机截断的子集。
3. 阈值在 calibration 的阈值表中选择，再在冻结阈值下评估 validation；正式 test 路径传入单一阈值，不重新扫描。

但原有通用入口 `evaluate_unified_pvs_metrics.py` 仍保留显式的历史 test 扫描开关，并把 `testEvaluationCount=1` 作为摘要字段记录，不能阻止同一 test 输出被重复运行。因此它不适合作为投稿主结果的唯一入口。旧的 `build_frozen_threshold_manifest.py` 还要求读取已经包含 test 结果的摘要，无法独立证明“test 尚未运行”。此外，训练中间 checkpoint 的 calibration 记录没有最终 bootstrap，不能直接作为冻结清单证据；这已在训练脚本中修正为 `checkpointSelection` 与最终 `best/workpoints` 分离保存。

## 新入口

新增 `evaluate_frozen_test.py`，分为两个相互隔离的命令。

### `prepare`

该阶段只读取 checkpoint、训练历史和运行时固定特征，不读取 test CSR 行。它要求：

- checkpoint 具有完整四路 split provenance；
- calibration 的阈值表和 selected row 存在，且 calibration pose 数量完整；
- validation 在每个记录的 epoch 都是完整集合；
- `best.pt` 的 epoch 等于完整 validation 历史中安全工作点的最高选择键；
- selected threshold 同时出现在 calibration 记录、validation 冻结记录和 checkpoint best 记录；
- calibration 点估计、view-cell bootstrap 单侧下置信界和预注册安全下限均通过；
- 训练/评估没有启用候选裁剪或 exploratory resource semantics；
- checkpoint 与固定实例特征表存在，并写入 SHA-256。

它生成 `neuralstreamweb3d-frozen-test-manifest-v2`，清单中只保存一个固定阈值和其来源，不保存 test 结果，`preTestEvaluationCount` 必须为 `0`。

示例：

```bash
conda run --no-capture-output -n slm_pvs python \
  neural_instance_culling/benchmark/evaluate_frozen_test.py prepare \
  --model-checkpoint pvs_directional_occlusion_proxy_encoder_rvl_strong_v2_full40_hkust_spatial_fov66_seed20260801=/path/to/best.pt \
  --runtime-features pvs_directional_occlusion_proxy_encoder_rvl_strong_v2_full40_hkust_spatial_fov66_seed20260801=/path/to/instance_runtime_features_fp16.bin \
  --dataset-dir neural_instance_culling/dataset/out/pose_csr_hkust_v3_spatial_raw_subpose_aabb_fov66_v1 \
  --output neural_instance_culling/benchmark/out/m0_frozen_manifest_hkust.json
```

### `evaluate`

该阶段只接受 v2 清单，不接受裸阈值。入口的参数接口中没有 test 阈值扫描、GT 并集、候选上限、放回采样或 test pose 截断选项，并且在推理前执行：

- 清单 checkpoint 和固定特征 SHA-256 校验；
- validation/calibration/test digest 与当前数据集逐项比对；
- 四路 split 互斥检查；
- test 所有唯一 pose 的候选集合预检；
- `visible_ids ⊆ 存储 candidate_ids` 检查；
- candidate ID 去重、范围和权重有限性检查。

候选不满足包含关系时直接失败，不补入 GT；候选数量始终使用 CSR 原始集合，不裁剪。推理调用只传递一个阈值，并要求最终结果的 `eval_pose_count` 等于完整 test pose 数量。

正式输出目录使用原子独占创建。目录已经存在时，即使上一次运行失败，也拒绝再次执行；成功输出的 `testEvaluationCount` 为 `1`，失败输出保留 `failed` claim，避免通过覆盖结果重新运行 test。

示例：

```bash
conda run --no-capture-output -n slm_pvs python \
  neural_instance_culling/benchmark/evaluate_frozen_test.py evaluate \
  --models pvs_directional_occlusion_proxy_encoder_rvl_strong_v2_full40_hkust_spatial_fov66_best \
  --manifest neural_instance_culling/benchmark/out/m0_frozen_manifest_hkust.json \
  --dataset-dir neural_instance_culling/dataset/out/pose_csr_hkust_v3_spatial_raw_subpose_aabb_fov66_v1 \
  --runtime-meta hkust-v3/assets/runtimeVisibilityMeta.json \
  --glb-index hkust-v3/assets/glbIndex.json \
  --glb-root hkust-v3/assets \
  --output-dir neural_instance_culling/benchmark/out/m0_frozen_test_hkust \
  --device cuda
```

该命令只能在新输出目录上执行一次。`summary.md` 只报告冻结阈值下的结果，不产生 workpoint 选择结论。

## 验证命令与结果

静态编译：

```bash
PYTHONDONTWRITEBYTECODE=1 conda run --no-capture-output -n slm_pvs \
  python -m py_compile neural_instance_culling/benchmark/evaluate_frozen_test.py
```

结果：通过。

小型 fixture 使用合成 split provenance 和假 runner 数据验证了：

- 完整 validation 历史选择 checkpoint；
- calibration 缺少 bootstrap 时拒绝；
- test 可见实例不在存储候选中时拒绝；
- 第二次 claim 同一输出目录时拒绝；
- v2 清单只接受单一冻结阈值；
- 入口帮助信息不暴露阈值扫描参数。

fixture 结果：`M0 frozen-test fixture: PASS`。本次没有启动长训练，也没有读取正式 test 数据执行模型推理。

## 当前限制与准入判断

M0 的协议代码子门已补齐，但总门仍未通过，原因是当前正在运行的正式训练输出还没有可供 `prepare` 使用的完整冻结资产：

- 当前 `best.pt` 所在目录尚未导出 `instance_runtime_features_fp16.bin`；
- 训练过程中保存的 epoch checkpoint 的 calibration 记录尚未包含最终的 view-cell bootstrap 下置信界，严格入口会拒绝它；
- 因此尚未生成 v2 冻结清单，也没有执行正式 one-shot test。

这不是通过放宽规则解决的问题。待正式训练完成并生成带 bootstrap 证据的校准记录后，应先运行 `prepare`，人工核对清单，再运行 `evaluate`。旧的 test 扫描摘要和旧 v1 清单不能转换为正式 M0 结果。正式 test 完成后，不得再修改阈值、候选语义或后处理；任何变化必须建立新的实验和新的冻结清单。

## 2026-08-01 冻结入口调用契约修正

### 变更目的

审计发现 `evaluate_frozen_test.py evaluate` 已实现严格的冻结清单和一次性目录声明，但内部仍使用旧版 `evaluate_runner` 的位置参数契约。当前统一评估器已经增加严格字节/时间预算、评分模式和 GLB 聚合参数；静态编译不会暴露问题，而实际执行会因缺少参数或传入旧参数而失败。这会在训练完成后直接阻断正式 M0 评测。

### 修改内容

- `neural_instance_culling/benchmark/evaluate_frozen_test.py`
  - 按当前 `evaluate_runner` 签名传入空的时间/字节预算（M0 只需要可见性结果，不伪造下载预算）；
  - 将评分模式固定为 `visibility-only`，GLB 聚合固定为 `max`，避免冻结 test 引入未注册的调度选择；
  - 保持 `max_eval_poses=0`、`max_candidates_per_pose=0`、无放回遍历和单一阈值。
- `neural_instance_culling/benchmark/tests/test_frozen_test_entrypoint.py`
  - 新增运行期契约 fixture，使用严格签名的替身评估器验证实际调用参数，而不是只检查 `py_compile`。

### 验证命令与结果

```bash
PYTHONDONTWRITEBYTECODE=1 conda run --no-capture-output -n slm_pvs \
  python -m unittest neural_instance_culling.benchmark.tests.test_frozen_test_entrypoint -v

PYTHONDONTWRITEBYTECODE=1 conda run --no-capture-output -n slm_pvs \
  python -m unittest discover -s neural_instance_culling/benchmark/tests -p 'test_*.py' -v
```

新增 fixture 和现有图像/基线测试均通过。该修复没有读取正式 test 数据、没有生成阈值、没有改变模型、候选集合或已有实验输出；M0 总门仍等待正式训练收尾及其 one-shot 证据。

## 2026-08-01 正式阈值读取一致性修正

训练收尾摘要的正式 schema 使用根级 `frozenThreshold`，而旧的模型 runner 和组件图像评价器只会查找历史 `thresholdRows`；在正式摘要上会静默回退到 runner 默认值，造成前端、benchmark 和训练摘要使用不同阈值。

修改如下：

- `neural_instance_culling/benchmark/model_runners.py`：正式摘要必须满足 `protocol=frozen_calibration_one_shot_test` 且 `testEvaluationCount=1`，随后只返回 `frozenThreshold`；缺失或越界直接失败。
- `neural_instance_culling/benchmark/evaluate_viewcell_image_per.py`：组件级图像评价采用同一正式摘要规则，并明确记录“无阈值扫描”的来源。
- `neural_instance_culling/benchmark/tests/test_frozen_test_entrypoint.py`：加入模型 runner 和图像评价器的正式摘要读取测试。

验证命令：

```bash
PYTHONDONTWRITEBYTECODE=1 conda run --no-capture-output -n slm_pvs \
  python -m unittest discover -s neural_instance_culling/benchmark/tests -p 'test_*.py' -v
```

结果：`12 tests` 全部通过。该改动只影响正式摘要的阈值读取，不改变已冻结结果，也没有触碰正式 test 数据。

## 2026-08-01 实验名与冻结入口解耦

正式多种子实验名必须保持独立，例如 `pvs_directional_occlusion_proxy_encoder_rvl_strong_v2_full40_hkust_spatial_fov66_seed20260801`，但不应为了使用冻结入口而把每个新名字永久写入全局默认模型列表。`evaluate_frozen_test.py` 现在对未预注册的实验名建立显式 learned-runner 规格，并只从冻结清单读取 checkpoint、固定特征和阈值路径；非 learned runner 仍会被 `prepare` 拒绝。新增 fixture 覆盖了该动态实验名路径。

这项修改只解决实验注册和复现入口的职责边界，不改变模型结构、阈值选择或候选语义。
