# M11 航向泛化与跨场景适配协议

日期：2026-08-01
状态：航向数据划分与方向遮挡证据已生成；Metropolis 少样本 1% `retry3` 已完成冻结 test，5%/10% 已按同一协议排队，正式泛化汇总尚未完成。

## 目的

现有四路空间划分把物理位置隔离开，可以检查相邻视点泄漏，但不能单独回答模型是否能处理训练时没有出现过的观察方向。本实验增加一个正交的航向留出协议：整段相机航向扇区只属于一个 split，保持原始候选、可见实例、可见权重、相机 FOV 和实例特征维度不变。

当前模型的固定实例表是按场景离线生成的，包含场景实例的几何、上下文和方向遮挡代理；它不是一个可以直接跨场景复用的固定 token 表。因此跨场景实验分成两种明确情况：

- **零样本查询头迁移**：只迁移学习到的点云编码器和视线查询头，在目标场景重新计算目标场景的固定实例表，不更新参数；目标 calibration 只用于选择安全阈值，不用于训练。
- **少样本适配**：在目标场景的训练视点中只开放预注册的 1%、5% 和 10% 子集，重新训练或微调同一结构；validation/calibration/test 保持完整且不参与参数更新。

如果目标场景的输入维度、实例映射或资源语义不一致，实验必须失败并记录 schema 不兼容，不能通过补零、复制实例或把一个场景的实例编号强行映射到另一个场景来制造“迁移结果”。

## 航向划分

脚本：`neural_instance_culling/dataset/build_directional_viewcell_split.py`。航向定义为：`yaw=0°` 指向世界坐标 `-Z`，正方向朝 `+X`。默认将 360° 划为 20 个 18° 扇区，整扇区分配为：

| split | 扇区 | 说明 |
|---|---:|---|
| train | 其余 16 个扇区 | 用于参数更新 |
| validation | 14 | 用于 checkpoint 诊断 |
| calibration | 16 | 只用于冻结安全阈值 |
| test | 18 | 冻结后一次评测 |
| guard | 19 | 不参与训练和工作点选择 |

该协议是方向外推测试，不是空间隔离测试；同一物理位置可能在不同航向 split 出现，这是有意保留的变量。俯仰范围和类别分布单独写入 manifest，不能将 yaw 留出结果解释成完整球面方向泛化。

## 已生成资源

| 场景 | 方向 manifest | pose view | 计数（train/validation/calibration/test/guard） |
|---|---|---|---:|
| HKUST v3 | `dataset/out/directional_split_hkust_v3_yaw20_20260801/manifest.json` | `dataset/out/pose_csr_hkust_v3_directional_yaw20_fov66_v1` | 6406/393/420/378/402 |
| Metropolis | `dataset/out/directional_split_metropolis_yaw20_20260801/manifest.json` | `dataset/out/pose_csr_metropolis_directional_yaw20_fov66_v1` | 20454/1149/2271/2271/1107 |

两个 PoseCSR view 通过硬链接复用原始二进制。候选语义仍为“每个成功子姿态独立计算后退 AABB 候选并取并集，不补入 GT 可见实例”；两个新目录的 `candidateMissVisible=0`、`candidateVisibleUnionAdded=0`，FOV 仍为模型 `66°`、真实渲染 `60°`。

## 方向证据与运行命令

方向遮挡代理的离线监督只能使用方向训练 split。若继续复用空间训练证据，会把方向 test 的可见源信息带入固定代理表，因而不具备泛化解释。目前以下两条命令正在独立日志中运行：

```bash
conda run --no-capture-output -n slm_pvs python -u neural_instance_culling/dataset/build_directional_occlusion_evidence.py \
  --dataset-dir neural_instance_culling/dataset/out/pose_csr_hkust_v3_directional_yaw20_fov66_v1 \
  --runtime-meta hkust-v3/assets/runtimeVisibilityMeta.json \
  --output-dir neural_instance_culling/dataset/out/directional_occlusion_evidence_hkust_v3_directional_yaw20_fov66_v1 \
  --splits train --direction-bins 8 --depth-shells 3 --source-k 8

conda run --no-capture-output -n slm_pvs python -u neural_instance_culling/dataset/build_directional_occlusion_evidence.py \
  --dataset-dir neural_instance_culling/dataset/out/pose_csr_metropolis_directional_yaw20_fov66_v1 \
  --runtime-meta ifcbench_fantasy_metropolis_source/assets/runtimeVisibilityMeta.json \
  --output-dir neural_instance_culling/dataset/out/directional_occlusion_evidence_metropolis_directional_yaw20_fov66_v1 \
  --splits train --direction-bins 8 --depth-shells 3 --source-k 8
```

证据完成后，正式方向模型至少需要一个固定 checkpoint、独立 calibration 和冻结 test。报告必须同时给出普通 pose 指标、weighted recall、有效剔除、错误剔除和按航向扇区的长尾结果；方向 test 不能再扫描阈值。

## 跨场景边界

HKUST 和 Metropolis 的实例数、GLB 映射、场景边界和离线特征表不同。当前工程上可行的迁移路径是共享同维度的学习参数，并在目标场景重新运行离线编码器；浏览器仍只接收目标场景自己的固定特征表。若零样本迁移不能达到 weighted-recall 安全门，应报告“场景特定离线预处理”这一真实边界，再比较 1%/5%/10% 少样本适配的收益和适配成本，不把失败包装成通用模型。

## 当前质量门

- 方向划分：通过，manifest、哈希和 split 计数已生成。
- 候选语义：通过，两个方向 PoseCSR view 未发生 GT 补入。
- 方向遮挡证据：进行中，完成前不训练方向模型。
- 方向泛化指标：未执行。
- 零/少样本跨场景迁移：未执行。

本记录只描述协议和已核验资源，不把正在运行的证据生成或后续计划当成结果。

## 跨场景零样本执行入口（2026-08-02）

为避免把源场景的实例编号、场景缓冲区或固定特征误当成可迁移知识，新增
`neural_instance_culling/benchmark/evaluate_cross_scene_transfer.py`。该入口的迁移边界固定为：

1. 只复制源 checkpoint 中形状完全一致的可学习参数；`AABB`、实例到 GLB 映射、方向证据、场景范围和所有场景规模缓冲区均在目标场景重新建立；
2. 使用源点云编码器和查询头，对目标场景自己的 GLB 点云、实例 AABB 和训练方向证据离线生成固定实例特征表；浏览器和评测运行时不执行点云编码或图传播；
3. 只在目标场景 calibration split 选择阈值，并记录点估计和 view-cell bootstrap 下界；目标 test 不参与阈值选择；
4. 只有存在满足安全规则的目标 calibration 工作点时，才写入不可变冻结清单并对完整 test split 执行一次评测；没有安全工作点时只输出 `no_safe_target_calibration_workpoint`，不产生伪造 test 结果。

推荐运行形式为：

```bash
conda run --no-capture-output -n slm_pvs python -u \
  neural_instance_culling/benchmark/evaluate_cross_scene_transfer.py \
  --source-checkpoint <source-best.pt> \
  --target-dataset-dir <target-pose-csr> \
  --target-evidence-dir <target-directional-evidence> \
  --target-glb-points <target-glb-points.bin> \
  --target-runtime-meta <target-runtimeVisibilityMeta.json> \
  --output-dir neural_instance_culling/model/out/m11_transfer_<source>_to_<target> \
  > transfer_stdout.log 2> transfer_stderr.log
```

输出目录至少包含目标固定特征表、`instance_features_meta.json`、
`calibration_ready_summary.json`；若安全工作点存在，还包含
`frozen_test_manifest.json` 和 `summary.json`。入口已通过静态编译和一实例/两实例参数形状迁移
合成 smoke；截至本记录更新时尚未运行实际跨场景 test，不能把入口存在解释为跨场景泛化成立。少样本
适配仍需在该零样本执行完成后按预注册的 1%、5% 和 10% 训练视点另行实现和评测。

少样本训练使用同一训练器的两个显式选项：`--init-checkpoint <source-best.pt> --allow-scene-transfer`
只迁移可学习参数，`--train-pose-fraction 0.01|0.05|0.10` 使用按 seed 固定抽取的目标场景训练
view-cell 子集。选中的训练 pose 数量和摘要会写入 `protocolSplit`；验证、校准和测试 pose 不会因
该选项被重采样或缩减。默认 `--train-pose-fraction 1.0` 和不带 `--allow-scene-transfer` 的行为
保持原有同场景训练语义。

已登记 Metropolis 少样本队列 `tmux m11_metropolis_fewshot_retry3`，入口为
`neural_instance_culling/benchmark/run_m11_metropolis_fewshot.sh`。队列等待 HKUST 方向模型完成最终
calibration 后，在 GPU 3 上依次运行 1%、5%、10% 三个独立输出目录，并在每个模型自身的 calibration
安全时生成 frozen manifest 和一次完整 test；不安全的比例只保留训练/校准失败证据并继续后续比例。
该队列尚未产生正式指标，运行日志写入 `neural_instance_culling/benchmark/out/m11_metropolis_fewshot_queue.log`
及各实验目录，不能把队列启动状态当成泛化结果。

截至 2026-08-02，1% 适配使用单 pose 训练批次、AMP 和较小的离线特征导出批次，训练记录中的非有限 loss/gradient
跳过计数均为 `0`；只有该比例完成自身 calibration 后，队列才会顺序进入 5% 和 10%。此前资源缓存粒度错误和
`retry2` 显存不足的失败目录均保留，不纳入正式结果。

## 2026-08-02 资源语义修正与零样本迁移结果

### 资源校验失败与修正

首次启动少样本队列时，训练器在资源校验阶段拒绝运行：方向 Metropolis 数据集的
`dataset_meta.json` 使用 `ifcbench_fantasy_metropolis_source` 的 41,298 个原始构件记录，
但脚本误用了只有 3,669 行的实例化原型点云缓存。该失败目录和 stderr 保留为输入语义错误证据，
没有把不匹配的缓存补零或截断。

脚本 `neural_instance_culling/benchmark/run_m11_metropolis_fewshot.sh` 已改为使用
`neural_instance_culling/dataset/out/ifcbench_fantasy_metropolis_instanced_v2_glb_points_v3.bin`，
该文件的元数据报告 `41,298` 行、每个 GLB `1,024` 个点，与方向数据集的原始构件粒度和
`ifcbench_fantasy_metropolis_source/assets/runtimeVisibilityMeta.json` 一致。第一次修正后的
`retry2` 训练又在 1% 适配的第 1 个 epoch、约第 266 个 step 触发 CUDA out-of-memory；该目录和
stderr 保留为显存边界证据，没有复用其不完整输出。

为解决这个与场景规模相关的运行资源问题，队列默认追加 `retry3` 后缀，使用单 pose 训练批次、
`feature-export-batch-size=128`、冻结 test 默认单 pose、可选 AMP（默认开启）和 `expandable_segments:True`。这些参数只改变
训练的显存占用和批处理方式，不截断候选、不补入 GT、也不改变 validation/calibration/test 的数据
语义；retry3 完成前不记录任何少样本指标。

### 零样本迁移

使用 HKUST 方向模型的可学习参数，在 Metropolis 重新生成目标场景固定特征；实例 AABB、实例到 GLB
映射、方向证据和场景缓冲区均来自目标场景。输出为
`neural_instance_culling/model/out/m11_transfer_hkust_directional_to_metropolis_yaw20_seed20260801_retry2/`，
目标校准和 test 均遵循一次冻结规则。

| 指标 | calibration 冻结点 | Metropolis test |
|---|---:|---:|
| 阈值 | `1.7782794e-07` | 同左 |
| pose weighted recall | `0.9999959` | `0.9999826` |
| pose recall | `0.9999927` | `0.9999957` |
| agg precision | `0.0937433` | `0.0911073` |
| useful cull | 近似 `0` | 近似 `0` |
| 平均预测 / 候选 | `10520.06 / 10520.07` | `11481.48 / 11481.53` |

结论是：共享查询参数本身可以维持高召回，但不能在未重建目标场景表征并适配参数的情况下维持有效剔除。
因此论文不使用“零样本跨场景通用模型”表述，M11 的跨场景结论限定为“目标场景离线特征重建加少样本
适配”，1%、5%、10% 适配结果待 `retry3` 队列完成后按相同安全规则评估。

## 2026-08-02 少样本冻结测试与协议修复

### 修复内容

少样本 checkpoint 的原生数据集仍包含完整 `train` 划分，但 checkpoint 的
`protocolSplit.trainFitCount` 记录的是从该划分中按固定种子抽取的适配子集。冻结测试入口此前把完整
训练 pose 数量直接与子集数量比较，导致 1% 模型在推理前被错误拒绝。修复位于
`neural_instance_culling/benchmark/evaluate_frozen_test.py`，并新增
`neural_instance_culling/benchmark/tests/test_frozen_test_entrypoint.py` 回归用例。

修复后的校验规则为：先验证 `originalTrainCount` 与完整原生训练划分一致；若 protocol 记录
`trainFitSelectionFraction` 和 `trainFitSelectionSeed`，则用与训练器相同的确定性抽样重建适配子集，再校验
子集数量和 digest；validation、calibration、test 仍必须与数据集完整划分的数量和 digest 严格一致。
该修复不读取 test 来选择阈值，也不改变候选集合或一次性 frozen test 规则。原失败 manifest/输出目录保留，修复后使用
带 `protocolfix` 后缀的新证据目录。

验证命令：

```bash
conda run --no-capture-output -n slm_pvs python -m unittest \
  neural_instance_culling.benchmark.tests.test_frozen_test_entrypoint -v
```

结果：`6 tests, OK`；真实 Metropolis 数据集重建出完整训练划分 `20,454` 个 pose 和 1% 适配子集 `205`
个 pose，记录的 digest 为 `5e10771d52c5c45d`，与 checkpoint 一致。

### 1% 适配 frozen test

manifest：
`neural_instance_culling/benchmark/out/m11_pvs_m11_fewshot_1pct_metropolis_yaw20_rvl_strong_v2_full40_seed20260801_retry3_protocolfix_frozen_manifest.json`

test 输出：
`neural_instance_culling/benchmark/out/m11_formal_metropolis_fewshot_1pct_frozen_test_20260802_protocolfix/`

评测使用完整且唯一的 Metropolis directional test split（`2,271` poses），模型查询 FOV 为 `66°`，真实渲染
FOV 为 `60°`，候选集合采用严格存储候选语义，冻结阈值来自独立 calibration：`0.05000000074505806`。

| 指标 | 1% 适配 frozen test |
|---|---:|
| pose recall | 0.945077 |
| weighted recall | 0.993608 |
| pose precision | 0.156026 |
| instance accuracy | 0.395861 |
| balanced accuracy | 0.628111 |
| useful cull = TN / candidate | 0.279348 |
| bad cull = FN / candidate | 0.004880 |
| 平均 candidate / GT / prediction | 11,481.53 / 1,046.06 / 6,908.04 |

该结果满足 weighted-recall 安全约束，但普通 recall 略低于 `0.95` 且 useful cull 明显弱于场景内主线，因而只能说明
1% 少样本适配具备有限的安全召回能力，不能作为跨场景高效泛化结论。没有用 test 后调阈值修复这一结果。

### 5%/10% 适配执行状态

在 1% frozen test 完成后，队列脚本支持通过 `SLM_M11_FEWSHOT_LABELS=5pct,10pct` 从指定比例继续执行，避免
重复训练已完成的 1%。5% 和 10% 使用独立输出目录、同一目标场景、同一 `rvl_strong_v2`、40 epoch、相同
calibration 安全规则和完整 test 一次性规则；当前队列会话为 `m11_metropolis_fewshot_5_10`，日志为
`neural_instance_culling/benchmark/out/m11_metropolis_fewshot_5_10_queue.log`，完成前不记录为正式泛化结果。
