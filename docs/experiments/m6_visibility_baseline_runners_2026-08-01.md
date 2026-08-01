# M6 第一批可见性 Baseline Runner

日期：2026-08-01  
状态：L0 runner、合成测试以及 HKUST/Metropolis 的 validation/calibration 诊断已完成；确定性 `baseline_aabb_ray` 和学习型 AABB+ray MLP 均已完成。真实 HZB 和 NeuralPVS 适配仍未完成，M6 总质量门未通过。
目的：为 M6 建立候选集合严格一致的可见性对照，并区分冷启动几何规则、训练集统计先验和视点查表先验。

## 实现范围

本次保留并注册以下七个可执行名称：

| 名称 | 类型 | 输入 | 输出 |
|---|---|---|---|
| `baseline_keep_all` | 固定安全规则 | 当前 pose 的原始 `candidate_ids` | 对每个候选输出 1 |
| `baseline_static_frequency_train` | 训练集统计规则 | train CSR 的候选/可见计数、当前 `candidate_ids` | 对每个实例查表得到可见频率 |
| `baseline_camera_distance` | 几何启发式 | 相机位置、实例 AABB | 到 AABB 距离的逆函数分数 |
| `baseline_projected_aabb_area` | 几何启发式 | 实例 AABB、当前 MVP | AABB 投影矩形面积分数 |
| `baseline_aabb_ray` | 几何启发式 | 实例 AABB、当前相机位置/视线/FOV、当前 MVP | AABB 角向贴合、投影面积和近距离的连续分数 |
| `baseline_aabb_hzb`（显示名 `baseline_aabb_depth_proxy`） | 深度代理规则 | 实例 AABB、当前 MVP | 两级 CPU AABB 深度代理分数，不是真实 HZB |
| `baseline_viewcell_bitset_train` | 训练集视点查表 | train pose 的位置、朝向和可见集合 bitset | 最近 train pose 的 bitset 成员分数 |

七个 runner 都返回一个与输入候选数组等长的 `PredictionResult.scores`，并保持
`candidate_ids` 原有顺序。runner 不读取 batch 中的 GT 来补入实例，也不修改候选数组。
bitset runner 遵守同一契约：bitset 只用于给当前候选行打分，不把查表 pose 的实例集合
并入当前候选集合。

## 统计规则

静态频率只读取 `dataset.split("train")`：

```text
frequency(i) =
  train 中 i 为可见实例的 pose 数
  --------------------------------
  train 中 i 出现在原始候选集合的 pose 数
```

分母使用候选条件频率，避免把一个从未进入候选的实例与当前推理候选混为一谈。
如果 train 中存在 `visible_id` 不属于同一 pose 的保存候选，runner 直接报错，
不会使用 GT 并集修复数据。验证集和测试集不会参与统计。

距离规则使用相机到 AABB 的欧氏距离 `d`，分数为：

```text
1 / (1 + d)
```

投影面积规则复用统一 AABB 八角点投影函数，先得到裁剪后的屏幕矩形，再使用
矩形面积作为分数。它使用包围盒近似，不代表三角形实际覆盖率。

### `baseline_aabb_ray` 规则

该 runner 是可复现的确定性几何基线，不是 AABB+ray 神经网络。对一个 pose 和候选实例，先
用当前相机位置 `camera_world`、视线方向及水平/垂直视场切线
`camera_view=[forward, tan_x, tan_y]` 建立相机坐标，再把 AABB 中心和半尺寸投影到相机的
右、上、前方向。角向项采用 AABB 在视线方向上的保守角向矩形：中心射线落入该矩形时残差为零，
落在外部时用固定指数衰减得到连续分数。随后使用当前 MVP 的 AABB 裁剪投影面积平方根和
以场景包围盒对角线归一化的近距离项：

```text
score = ray_angular_proximity
        × sqrt(clipped_mvp_aabb_area)
        × 1 / (1 + near_depth / scene_diagonal)
```

若 AABB 没有有效的 MVP 投影或完全位于近裁剪面之后，分数为零。这个定义只使用实例 AABB、
当前相机/视线/FOV 和 MVP；它不读取 GLB、三角形、材质、深度图或 GT，也不修改输入候选。
评估器读取 `glbIndex.json` 和文件字节数仅用于报告 GLB 资源指标，不参与 runner 打分。
因此该规则属于 L0 元数据冷启动基线，不能解释为遮挡推理、HZB 或 NeuralPVS。

视点 bitset 规则只读取 train split。每个 train pose 将其 GT 可见实例编号压成按实例编号寻址的
packed bitset；查询 pose 以相机位置平方距离加方向不一致惩罚选择最近的 train pose，然后仅查询
当前 `candidate_ids` 对应的 bit。当前默认方向惩罚为 `25 m²`，bitset 存储量为
`train_pose_count * ceil(num_instances / 8)` 字节。它是“最近训练视点可见集合”的查表基线，不是
连续的视点插值，也不等价于遮挡缓冲。

## 信息预算与公平边界

前六个几何/固定规则 runner 属于投稿计划中的 L0 元数据冷启动层；bitset runner 属于只使用 train 统计的
L1 视点查表层：

- 允许：实例 AABB、实例编号、实例到 GLB 映射、相机信息和 MVP；
- 统计频率额外需要：由 train CSR 生成的每实例标量表；
- 不允许：目标 GLB 的三角形、材质、纹理、深度图或测试集可见集合；
- 不包含：真正的 HZB、三角形光栅化和动态遮挡查询。

特别限制：`baseline_aabb_hzb`（显示名 `baseline_aabb_depth_proxy`）及本页的投影面积 runner 都不能称为 HZB。前者只根据 AABB
角点的深度代理做候选打分，后者只计算屏幕投影面积；二者都没有从三角形光栅化得到最底层深度
图，也没有执行由底层深度图逐级下采样形成的层次化深度比较。因此它们只能作为 AABB depth
proxy / projected-area baseline，不能作为 HZB 实现或 HZB 消融结果。

统一入口仍然是 `load_runner`。统一评估器在创建 runner 时传入 `dataset_dir`，因此
训练集频率表由评估任务按当前数据集生成，而不是读取一个可能混入测试数据的缓存。
新增 runner 没有改变训练脚本、训练默认参数、模型权重或默认 benchmark 命令。

正式比较时仍必须固定：

1. 同一份保存的后退视锥 `candidate_ids`，关闭 GT 并集和候选上限；
2. 同一 FOV、同一 pose split、同一 GT 和同一 visible weight 语义；
3. 规则分数在 calibration 选择工作点，test 只读取冻结阈值；
4. `visible_weights` 只能称为弱重要性权重，不能称为真实像素覆盖率。

## 测试

合成测试位于：

```text
neural_instance_culling/benchmark/tests/test_visibility_baseline_runners.py
```

运行命令：

```bash
PYTHONDONTWRITEBYTECODE=1 conda run --no-capture-output -n slm_pvs \
  python -m unittest discover \
  -s neural_instance_culling/benchmark/tests \
  -p 'test_*.py' -v
```

当前测试覆盖：

- 七个 runner 的注册和构造；
- 输出长度与候选顺序保持一致；
- keep-all 不依赖 GT；
- 静态频率不读取 test 可见实例；
- train 可见实例缺失于候选时直接失败；
- 投影面积 runner 缺少 MVP 时拒绝运行；
- AABB-ray runner 缺少 MVP 时拒绝运行、只依赖相机/MVP/AABB 字段，并响应视线变化；
- bitset 不把 test-only 可见实例加入结果，且记录 train pose 数和 bitset 字节数。

## `baseline_aabb_ray` validation/calibration 诊断

### 运行口径与资源层级

本轮四个任务只运行 `baseline_aabb_ray`，分别遍历两个场景的完整 validation 和 calibration
split。命令固定为 `--device cpu`、`--poses-per-batch 4`、`--max-candidates-per-pose 0`、
`--max-eval-poses 0`，未开启 `--allow-candidate-visible-union`，因此使用数据集中保存的原始
后退视锥候选集合，不补 GT 正例、不裁剪候选。四个任务均为 `stored_candidate_set_strict`，
FOV 使用数据集记录的模型输入 66°；没有读取 test。

完整命令写入每个输出目录的 `command.txt`，命令形式如下：

```bash
conda run --no-capture-output -n slm_pvs python \
  neural_instance_culling/benchmark/evaluate_unified_pvs_metrics.py \
  --models baseline_aabb_ray \
  --dataset-dir <dataset> --runtime-meta <runtimeVisibilityMeta.json> \
  --glb-index <glbIndex.json> --glb-root <glb-root> \
  --output-dir <independent-output-dir> --split <validation|calibration> \
  --device cpu --poses-per-batch 4 \
  --max-candidates-per-pose 0 --max-eval-poses 0
```

| 场景 | 实例数 | validation pose | calibration pose | 平均候选/GT（validation） | 平均候选/GT（calibration） |
|---|---:|---:|---:|---:|---:|
| HKUST | 18,831 | 664 | 690 | 5,818.33 / 27.63 | 5,078.98 / 27.28 |
| Metropolis | 41,298 | 2,088 | 2,376 | 10,126.28 / 856.48 | 10,030.98 / 756.38 |

该 runner 位于 L0 元数据冷启动层：输入只有实例 AABB、当前相机位置/视线/FOV 和 MVP；
它不读取目标 GLB 三角形。评估器读取 GLB 索引和文件大小只为了计算资源字节指标，不能把
这些文件访问解释为 runner 使用了几何遮挡信息。

### 指标结果

以下表格取当前阈值扫描中最宽松的 `threshold=0.001` 行，用来展示该规则的最高安全上界；
它不是通过安全门选出的工作点。`pose P/R/F1/Acc/BAcc` 分别是逐 pose 后再平均的 precision、
recall、F1、accuracy 和 balanced accuracy；`weighted R` 使用数据集的弱重要性权重；
`useful cull` 为 `TN / candidate`，`bad cull` 为 `FN / candidate`；GLB 字节削减是预测
GLB 字节相对于候选 GLB 字节的平均比例。

| Split | pose P | pose R | pose F1 | pose Acc | pose BAcc | weighted R | useful cull | bad cull | 平均预测 | GLB 字节削减 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| HKUST validation | 0.091 | 0.953 | 0.135 | 0.437 | 0.678 | 0.939 | 0.377 | 0.00645 | 2,865.56 | 0.175 |
| HKUST calibration | 0.111 | 0.925 | 0.167 | 0.451 | 0.671 | 0.907 | 0.385 | 0.00897 | 2,694.43 | 0.184 |
| Metropolis validation | 0.122 | 0.950 | 0.206 | 0.295 | 0.588 | 0.985 | 0.197 | 0.00461 | 7,522.15 | 0.067 |
| Metropolis calibration | 0.154 | 0.955 | 0.248 | 0.328 | 0.593 | 0.982 | 0.199 | 0.00508 | 6,928.63 | 0.079 |

对应的跨 pose 聚合指标为：

| Split | agg precision | agg recall | agg F1 | agg accuracy | 平均 forward/total ms 每个 4-pose batch |
|---|---:|---:|---:|---:|---:|
| HKUST validation | 0.0090 | 0.932 | 0.0178 | 0.512 | 57.98 / 57.98 |
| HKUST calibration | 0.0096 | 0.945 | 0.0189 | 0.474 | 50.42 / 50.42 |
| Metropolis validation | 0.1075 | 0.944 | 0.1930 | 0.332 | 95.11 / 95.11 |
| Metropolis calibration | 0.1031 | 0.944 | 0.1859 | 0.376 | 78.28 / 78.28 |

四个 calibration 结果均没有 `pose_weighted_recall > 0.99` 的阈值行，因此 `best` 和
weighted-safe useful-cull 工作点均为 `null`。普通 F1 的最高诊断点也不安全：HKUST
calibration 为阈值 0.075、weighted recall 0.655；Metropolis calibration 为阈值 0.005、
weighted recall 0.938。降低阈值只能增加预测量，不能把该 AABB/ray 规则变成满足画面安全约束
的基线。完整 46 行 threshold rows、逐项 GLB 预算统计和日志保存在以下目录：

- `neural_instance_culling/benchmark/out/m6_baseline_aabb_ray_validation_hkust_spatial_fov66/`
- `neural_instance_culling/benchmark/out/m6_baseline_aabb_ray_calibration_hkust_spatial_fov66/`
- `neural_instance_culling/benchmark/out/m6_baseline_aabb_ray_validation_metropolis_spatial_fov66/`
- `neural_instance_culling/benchmark/out/m6_baseline_aabb_ray_calibration_metropolis_spatial_fov66/`

这些结果只属于 M6 的 validation/calibration 诊断，不是冻结 test 结果，也不为 test 选择阈值。

## 之前已完成的其他 calibration 基线

此前其他 runner 的运行命令使用完整的 calibration split、原始保存候选、`--max-candidates-per-pose 0`，未启用
GT 并集修复。四个首批 runner 的结果保存在原始 calibration 目录；包含 bitset 的复核结果
保存在带 `_bitset_v2` 后缀的独立目录；该目录同时写出信息层级、训练 pose 数、方向惩罚和
packed bitset 字节数。阈值只用于 calibration 诊断；这些结果不是冻结 test 结论。

### HKUST

数据集：`pose_csr_hkust_v3_spatial_raw_subpose_aabb_fov66_v1`，690 个 calibration view-cell，
平均候选 5,078.98、平均 GT 27.28。

| Runner | 安全工作点 | Weighted recall | Useful cull | Bad cull | 平均预测 |
|---|---:|---:|---:|---:|---:|
| keep-all | 0.001 | 1.000 | 0.000 | 0.00000 | 5,078.98 |
| static frequency | 0.100 | 0.996 | 0.573 | 0.00418 | 202.04 |
| camera distance | 0.010 | 0.990 | 0.669 | 0.00776 | 17.60 |
| projected AABB area | 无 | 未达到校准安全门 | 未定义 | 未定义 | 未定义 |
| view-cell bitset (train) | 无 | 未达到校准安全门 | 未定义 | 未定义 | 未定义 |

bitset 复核：无安全工作点。HKUST 使用 5,563 个 train pose、`13,095,302` 字节 bitset，
在二值 bitset 成员工作点（阈值 `0.001`）下的 pose-level weighted recall 为 `0.9515`，低于 `0.99`；
Metropolis 也低于 `0.99`。如果把阈值降到 `0`，零 bit 会全部保留，
退化为 keep-all，不能作为 bitset 的有效剔除工作点。

### Metropolis

数据集：`pose_csr_metropolis_spatial_dense_subpose_union_fov66_v2`，2,376 个 calibration
view-cell，平均候选 10,030.98、平均 GT 756.38。

| Runner | 安全工作点 | Weighted recall | Useful cull | Bad cull | 平均预测 |
|---|---:|---:|---:|---:|---:|
| keep-all | 0.001 | 0.998 | 0.000 | 0.00000 | 10,030.98 |
| static frequency | 无 | 未达到校准安全门 | 未定义 | 未定义 | 未定义 |
| camera distance | 0.002 | 0.997 | 0.031 | 0.00061 | 9,285.56 |
| projected AABB area | 无 | 未达到校准安全门 | 未定义 | 未定义 | 未定义 |
| view-cell bitset (train) | 无 | 未达到校准安全门 | 未定义 | 未定义 | 未定义 |

Metropolis 使用 19,296 个 train pose、`99,625,248` 字节 bitset。其 bitset 诊断工作点（阈值 `0.001`）为：pose accuracy `0.887`、balanced accuracy
`0.665`、pose precision `0.538`、pose recall `0.367`、weighted recall `0.409`、useful cull
`0.833`、bad cull `0.07921`、平均预测 `626.41`。它没有通过 weighted-recall 安全门，以上只作为失败诊断，
不应与安全工作点并列排名。

Metropolis 的结果说明候选规模和可见实例比例显著改变简单统计规则的工作区间。它不是
删除该场景的理由，但要求后续模型和基线统一报告无安全工作点的情况，并在同一候选集合
上比较 `useful cull` 与 `bad cull`，不能用裸候选削减率替代安全评价。

此前基线的正式输出位于：

- `neural_instance_culling/benchmark/out/m6_baseline_calibration_hkust_spatial_fov66/`
- `neural_instance_culling/benchmark/out/m6_baseline_calibration_metropolis_spatial_fov66/`
- `neural_instance_culling/benchmark/out/m6_baseline_calibration_hkust_spatial_fov66_bitset_v2/`
- `neural_instance_culling/benchmark/out/m6_baseline_calibration_metropolis_spatial_fov66_bitset_v2/`

## 尚未解决的问题

- 统一评估器的 `--budgets` 当前仍是 GLB 数量预算，不是严格字节预算；这是 M7 调度实验前必须单独修复的接口。
- 固定决策 runner 已在当前评测入口使用声明阈值一次评估；历史连续扫描输出只作为旧诊断保留。
- `PoseCSRSplit` 当前默认只遍历有 GT 可见实例的 pose；是否把空 GT pose 纳入 M6 主表需要在正式协议中冻结。
- `max_candidates_per_pose` 和候选并集开关仍可改变评估分母；正式 M6 不能启用它们。
- 离线投影面积规则只用于 benchmark 对照，不能据此宣称前端已经实现 GPU AABB 或 HZB。
- 本轮 `baseline_aabb_ray` 是确定性几何规则；学习型 AABB+ray MLP 已实现并完成两个场景的 validation/calibration，二者仍都属于 L0 元数据层。
- NeuralPVS 适配和真正三角形 HZB 尚未实现；当前 `baseline_aabb_hzb` 仍只能称为 AABB depth proxy，且没有可冻结的 weighted-recall 安全工作点。

## 2026-08-01 固定决策 runner 评测修正

审计发现统一评测器虽然已经记录 `fixed_keep_all` 和 `fixed_train_bitset`，但仍对这类二值/固定规则
生成完整连续阈值网格。这会制造没有实际意义的“阈值工作点”，并可能让 bitset 被误读为可校准的连续模型。

已修改：

- `evaluate_visual_utility_metrics.py`：固定决策 runner 只在其声明阈值（当前为 `0.5`）上评估；冻结清单
  提供的阈值仍优先级最高；连续 runner 保持完整 calibration 阈值网格。
- `evaluate_unified_pvs_metrics.py`：同步同一固定决策规则，保持历史入口与当前 M7 入口语义一致。
- `test_visibility_baseline_runners.py`：将 `baseline_aabb_hzb` 纳入七个 runner 的注册/顺序检查，并验证
  depth proxy 与 train bitset 的决策模式不被混淆。

验证：baseline runner 测试 4 项通过，统一评测器和 M7 评测器编译通过，M7 model-free self-test 通过。
这项修改只改变固定规则的评测表示，不改变候选集合、GT、阈值选择安全门或已有 benchmark 数值；旧的
连续扫描输出保留为历史诊断，正式重新生成时应使用新入口。

## 准入判断

第一批 L0 基线、train-only L1 bitset 以及本轮 AABB/ray L0 runner 的执行门通过：候选不被修改，
训练统计不读取 calibration/test GT，两个场景的 validation/calibration 都完成全量扫描并保留
逐工作点 JSON。`baseline_aabb_ray` 在两个场景均没有安全工作点，作为失败诊断保留，不进入安全
工作点排名。学习型 AABB+ray MLP 已补齐，但它没有改变信息层级，也不能替代真正的遮挡基线。M6 总门尚未通过，因为
NeuralPVS 和真正三角形 HZB 仍缺失；在这些
基线和固定决策接口完成前，不生成 M6 最终主表排名。

## 2026-08-01 学习型 AABB+ray 基线补充

### 变更目的

计划中的 L0 层不仅需要确定性 AABB/ray 规则，还需要一个容量受控的学习型对照，以区分
“AABB 与射线特征本身不足”与“当前主模型的固定几何、上下文和遮挡代理表征带来的收益”。该
基线只读取实例 AABB、相机射线/FOV 和当前 MVP 下的 AABB 投影特征，不读取 GLB 三角形、材质、
纹理、test 可见集合或目标模型文件。

### 实现与资源

- 新增 `aabb_ray_feature_utils.py`，训练器和 runner 共用 18 维特征布局，避免 schema 漂移。
- 新增 `train_aabb_ray_baseline.py`，使用 train split 的候选正负样本训练两层 64 单元 SiLU MLP。
- `model_runners.py` 和 `evaluate_visual_utility_metrics.py` 增加显式
  `--learned-aabb-ray-spec name|checkpoint|training_summary` 注册入口，不改变默认模型表。
- 输出目录：
  - `model/out/baseline_learned_aabb_ray_hkust_spatial_fov66_seed20260801`
  - `model/out/baseline_learned_aabb_ray_metropolis_spatial_fov66_seed20260801`
  - 两个场景的 validation/calibration 独立 benchmark 目录。
- 两场景均使用 seed `20260801`、10 epoch、400 steps/epoch、2048 行 batch、32 正样本和 256
  负样本/pose；训练摘要写明该 checkpoint 必须由共享 calibration 入口选阈值。

### 当前诊断结果

下表为 validation 上满足 `pose_weighted_recall > 0.99` 且点估计不低于 `0.9925` 的最高
pose precision 工作点；它不是 test 结果，也不构成主模型质量结论。

| 场景 | validation pose | calibration pose | validation 安全阈值 | validation weighted recall | validation useful cull | validation bad cull | validation avg pred | 最佳 validation loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| HKUST | 664 | 690 | 0.300 | 0.99182 | 0.68975 | 0.00441 | 291.21 | 0.24539 |
| Metropolis | 2,088 | 2,376 | 待完整评测摘要 | 待完整评测摘要 | 待完整评测摘要 | 待完整评测摘要 | 待完整评测摘要 | 0.48354 |

HKUST calibration 的对应工作点为阈值 `0.28`、weighted recall `0.99174`、useful cull
`0.68250`、bad cull `0.00275`、平均预测 `246.61`。该规则明显弱于需要固定场景知识的主模型，
但它建立了可复现的同信息学习型 L0 对照。

### Metropolis 完整评测核验

此前 Metropolis 评测目录为空，是因为首次启动命令没有引用 `--learned-aabb-ray-spec` 的管道分隔参数，
shell 将 `training_summary.json` 当作可执行文件，产生了 `Permission denied`。这不是数据或模型失败；修正
后使用独立目录重新完成了全部 pose：

```text
validation:  m6_learned_aabb_ray_metropolis_spatial_fov66_validation_retry3  (2,088 poses)
calibration: m6_learned_aabb_ray_metropolis_spatial_fov66_calibration_retry3 (2,376 poses)
checkpoint:  model/out/baseline_learned_aabb_ray_metropolis_spatial_fov66_seed20260801/best.pt
```

正确的注册形式必须把规格整体引用：

```bash
--models '' \
--learned-aabb-ray-spec 'm6_learned_aabb_ray_metropolis|<best.pt>|<training_summary.json>'
```

| Split | 阈值 | pose precision | pose recall | weighted recall | useful cull | bad cull | 平均预测 | 平均候选 | 平均 GT |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Metropolis validation | 0.22 | 0.12604 | 0.91424 | 0.99056 | 0.23030 | 0.00727 | 6,245.71 | 10,126.28 | 856.48 |
| Metropolis calibration | 0.20 | 0.15304 | 0.91306 | 0.99118 | 0.22468 | 0.00653 | 5,621.25 | 10,030.98 | 756.38 |

该基线只能使用实例 AABB、相机射线/FOV 和 MVP 投影特征。它没有访问目标 GLB 三角形，也不提供遮挡关系，
因此结果支持的结论是“同一 L0 元数据输入下，轻量学习规则可以达到有限的安全工作点”，不能支持
“AABB+ray 已经近似 HZB”或“它能替代固定场景特征模型”。

### 质量门判断

学习型 AABB+ray 已完成 runner/schema 和 HKUST 诊断子门；M6 总门仍未通过。真实三角形 HZB
和 NeuralPVS 适配仍未实现，`baseline_aabb_hzb` 继续只表示 AABB depth proxy。后续若真实
三角形管线无法在同一 L1/L2 信息边界下完成，将在执行记录中说明资源与公平性原因，并把 M6
拆分为“L0 冷启动基线已完成”和“L2 warm-cache 上界未完成”，绝不把 depth proxy 改名为 HZB。
