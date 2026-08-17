# 共享关系先验、逐实例校准生存场与视点矩包络 v4 实施记录

- 日期：2026-08-15
- 实验前缀：`pvs_bounded_relation_prior_instance_calibrated_moment_safety_reserve_v4`
- 当前状态：共享关系先验与逐实例校准残差的代码契约、变体/损失/残差/view-cell 严格绑定、双路径 CUDA 单步训练和 checkpoint 专属导出已通过；`scan8` 已完成 8/8 配置并冻结 `s02`，`formal80` 已按 5 变体 × 3 seed 在四张 GPU 上启动，正式评价尚未完成
- 数据访问：只使用 train、calibration 和 validation；test 未读取
- 默认版本：未修改当前 checkpoint、阈值、前端资产或部署包

## 变更目的

本轮在 2026-08-14 纠错实现上继续修复“全场只由共享关系参数生成生存场、缺少逐实例表达能力”的问题。目标仍是把重型遮挡关系建模和实例校准放在离线阶段，浏览器只查询固定的逐实例表，同时输出实例可见性、视觉效用和 GLB 下载优先级。关系数据与 PoseCSR 继续复用已冻结并通过校验的 v3 数据 schema；模型、checkpoint、运行包和实验输出升级为独立 v4，避免用旧实验名称表达新语义。

## 已实现的数据与模型契约

### 候选与查询中心

新 PoseCSR 位于：

```text
neural_instance_culling/dataset/out/
  pose_csr_hkust_v3_bounded_relation_moment_fov66_v3/
```

它分别保存：

- `candidate_camera_world`：后退 3.4641 m、FOV 66 度，只用于构造原生候选；
- `query_center_world`：view-cell 中心，只用于模型视角查询和水平圆盘矩；
- `viewcell_radius_m`：HKUST 当前恒为 2 m。

数据包含 7,999 个 pose；本轮只打开 train 2,772、calibration 168 和 validation 213 个 pose。train 包含 14,385,216 个候选引用。候选集合是原生 66 度后退相机 AABB 候选，不补入 GT，不使用前端白名单。

### 有界真实遮挡关系与生存监督

正式关系产物位于：

```text
neural_instance_culling/dataset/out/
  pvs_bounded_relation_survival_moment_envelope_safety_reserve_v3/
    bounded_relation_csr_v3/
    degree_preserving_shuffled_relation_csr_v3/
```

原生关系由 13,860 个 train 子姿态的真实三角形深度层证据构建。每个目标、方向和深度层最多保留 12 个真实来源，排序同时考虑证据置信度、像素支持和 pose 支持。

| 项目 | 正式值 |
|---|---:|
| 保留关系边 | 448,165 |
| 方向锚点 / 深度层 | 12 / 3 |
| 生存观察 | 4,163,948 |
| 遮挡事件 | 3,408,530 |
| 右删失可见观察 | 755,418 |
| 局部组 | 18,276 |
| 最大局部组实例数 | 19，约束为不超过 32 |
| 结构组 | 11,324 |
| 最大结构组实例数 | 168，占全场 0.892% |
| 最大局部空间直径 | 142.096 m，约束为 142.245 m |
| 最大结构空间直径 | 482.388 m，约束为 482.406 m |

同一实例在同一子姿态中可以同时贡献首层可见删失证据和后层遮挡事件。深度改为 train-only 分位数冻结的半径相对 `log1p` 坐标，`q01=0.0471656`、`q99=4.4478427`。

K=12 的证据质量不能描述为全部达到 90%。全量数据中 50% 的格保留全部证据，5% 分位仍为 94.56%，但最差格只保留 6.08%；该极端长尾必须在 pilot 中结合关系门、实例覆盖和最终指标继续诊断，不能沿用小样本 smoke 的 70.76% 最差值。

保持度数的置乱对照改变 390,560 / 448,165 条边，占 87.15%。它保持 source/target 度数、方向/深度分层和原生生存观察，重新计算相对几何并重建有界层级；所有观察文件与原生关系逐字节一致。

### 共享关系先验与逐实例校准残差

共享分层关系编码器先从固定几何、真实有向遮挡边、局部组和结构组生成每个实例的 `4 x 7` 生存先验。共享参数能够学习跨实例重复出现的遮挡规律，但它单独使用时会把不同实例压到同一套生成规律中，难以表达局部证据、几何细节和采样密度带来的个体差异。

v4 为每个实例增加一个同形状、零初始化的离线校准残差。最终系数定义为：

```text
最终生存系数 = 共享关系先验 + 混合系数 * 有界逐实例残差
有界逐实例残差 = 4 * tanh(原始逐实例参数 / 4)
```

残差不会从训练首步直接覆盖共享规律。前 10% optimizer step 的混合系数为 0，随后 20% step 线性增至 1。每个实例的可靠度只由 train split 中的候选出现次数和生存观察次数计算：两类计数分别进行 `log1p` 归一化，并以正计数的 95% 分位为尺度，最终取两者最大值。可靠度低的实例获得更强的二次残差约束；默认残差正则权重为 `0.02`，稀疏实例附加系数为 `3.0`。calibration、validation 和 test 均不参与可靠度计算。

checkpoint 同时保存共享先验、已应用残差和融合结果，并校验三者逐元素一致。导出器再次执行 FP16 容差内的融合校验，只把融合后的 28 个系数写入浏览器固定表。残差参数、可靠度、关系 CSR、层级分组和观察样本全部保持离线，因此前端布局仍为 `96 维几何 + 28 维融合生存系数 = 124 维 FP16`，没有新增运行时表、邻居查询或网络传播。

### 视点区域矩包络频谱

查询以九维中心 ray-space 和两个水平圆盘低秩轴表示 view-cell，不生成在线 subpose。每个频率输出正弦/余弦均值及相位相关标准差，并使用统一的 cycles 与 `2*pi` 口径。

计划登记的 8 cycles 上限已恢复。逐特征圆盘轴满足：

```text
||B[i,:]||_2 <= 1 - abs(center_feature[i])
```

因此 `2s` 的理论上界为 `4*pi*sqrt(9)*8 = 301.593`。固定圆盘传递表使用 8,192 个 FP32 值、范围 `[0,320]`，正好占 32 KiB；密集参考测试的最大绝对插值误差为 `4.77e-5`，低于 `1e-4`。旧实现把上限静默收紧到 1.25 cycles 的做法已删除。

退化圆盘或数值零方差采用机器精度门控的零输出/零梯度，避免 `sqrt(0)` 在反向传播中产生非有限梯度。

### 安全裕度工作区损失

训练目标拆为四组：

- 安全组：归一化 RVL 与正例边界尾部；
- 关系组：生存删失似然与真实关系一致性；
- 调度组：视觉效用与 GLB 排序；
- 效率组：工作阈值附近负例和无需求 GLB 字节。

效率组在前 10% optimizer step 中保持关闭，之后由停止梯度的正例低尾部安全裕度连续开启。关系、调度和效率梯度分别对安全梯度做冲突投影和范数限制。点积和范数使用 FP64 累加，FP32 写回后最多执行三次带正余量校正；最终点积仍小于零会直接终止训练。

生存观察采样器按事件/删失、12 个方向、深度桶和实例轮换。正式 `uint8/uint32` 离散字段在 gather 前规范为 `int64`，并通过一次稳定分组替代逐实例扫描全观察表。

### 关系负边监督的语义与性能纠错

真实 CUDA 遥测发现，初版关系辅助监督在 GPU 张量上逐边转换 Python 标量，并为每一条负边重新扫描完整方向/深度特征表。32,768 条正边时，该路径生成 36,943 条可重复负边，单训练步耗时 253.90 秒；同时没有使用计划登记的距离分桶，交换 source/target 后也没有重算相对几何。

现已改为与正式置乱对照一致的确定性有向双边交换：

- sampled edge index 一次传到 CPU，在相同方向、深度层和八个距离分桶内交换 target；
- 每个发生交换的原始行至多对应一条负边，保持 source/target 度数且拒绝自环和重复边；
- 返回原证据行索引，GPU 通过一次 gather 取得证据模板；
- 按交换后的 source/target AABB 重新计算相对中心与相对尺度；
- 负边仍只来自 train split，不读取 calibration、validation 或 test。

同规格 32,768 观察批的真实 RTX A6000 训练步降为 4.60 秒，较错误路径快约 55.2 倍；负边为 32,160 条。该修改同时修复了辅助监督语义和训练可执行性，没有改动正式关系 CSR、候选集合或浏览器运行资产。

## 真实 CUDA smoke

最终真实 smoke 使用两张 RTX A6000、固定 W042 几何表、正式关系和真实 GLB 字节，对完整 v4 与关闭逐实例残差的消融并行执行 1 epoch、1 optimizer step、32,768 条生存观察，并分别严格限制为 2 个 calibration 和 validation 诊断 pose。主要结果如下：

| 检查项 | 结果 |
|---|---:|
| 总 loss | 88.6099 |
| 生存监督实例覆盖 | 100% |
| 方向覆盖 | 100% |
| event/censor 类别覆盖 | 100% |
| 安全裕度门 | 0，符合 warmup 定义 |
| 关系正边 / 负边 | 32,768 / 32,160 |
| 关系梯度点积：投影前 | -0.00650 |
| 关系梯度点积：投影后 | +0.00234 |
| 调度梯度点积：投影后 | +0.01771 |
| 非有限 loss / 梯度 | 0 |
| 逐实例残差混合系数 | 完整模型 0；消融 0，符合前 10% warmup |
| checkpoint 融合最大绝对误差 | 两条路径均为 0 |
| 完整模型 / 残差消融单训练步 | 4.609 / 4.619 s |
| 完整模型 / 残差消融 CUDA 峰值已分配 | 3,357.21 / 3,344.50 MiB |

两条路径在单步 smoke 中得到相同输出是预期行为，因为完整模型仍处于残差关闭的 warmup。该结果不证明残差有效；它只证明真实数据、CUDA 反向、禁用路径、calibration 冻结、validation replay、checkpoint 融合和导出链路可执行。残差贡献必须由 12 epoch pilot 以及后续相同种子、相同 pose 的正式消融判断。

## 正式运行协议收口

### 后处理断点续跑

在正式长训仍运行期间，补齐了 v4 runner 的评价阶段续跑逻辑。此前如果验证、导出、回放、浏览器图像评价、GLB 资源评价、汇总或路线判定中的任一阶段失败，重新调用 runner 可能从头执行已完成的昂贵阶段。现在只有同时满足“上一轮该阶段明确记录为 `passed`”和“对应产物仍通过存在性或结构校验”时才复用；否则只重新执行该阶段。每次复用都会在新的 `pipeline_status` 中标注 `reused: true`，训练成员、checkpoint 和默认前端资产不被改写。

具体规则如下：

- 外部 checkpoint validator 复用已有通过记录；本地 checkpoint schema、运行表和 replay 身份校验每次重新执行；
- 运行包只有在已有导出通过记录且 `model_meta.json`、固定特征表和 checkpoint/阈值来源一致时复用；
- validation replay 只有在已有通过记录且 `validation_evaluation.json` 存在时复用，之后仍重新核对 validation pose、候选、GT、几何和运行表身份；
- 图像评价要求已有通过记录、`summary.json` 存在、正式图像评价标志为真且 GPU 门为硬件通过；资源评价要求 validation 汇总结构完整；
- 10,000 次 bootstrap 汇总和路线报告只有在 schema、validation split、成员覆盖、paired bootstrap 次数及报告文件结构均通过检查时才复用，缺失或损坏时只补做对应阶段。
- 路线完成后另外生成 `docs/evaluation/pvs_bounded_relation_survival_moment_v4_formal_validation_*.md`，包含 15 个成员的核心 validation 表、路线结论、评价产物链接和指标口径；如果旧 runner 已经写出 `evaluation_complete`，新 runner 只补发布该报告，不重复昂贵评价。

训练队列恢复也不再依赖父进程是否来得及写出总 `training_status`：只要成员目录包含完整目标 epoch、必需训练产物，并且 `model_meta.json` 中的变体、随机种子和实验成员身份与 manifest 一致，就会被复用；身份不一致或产物不完整的目录才进入新的成员 attempt。该规则覆盖父 runner 在成员完成后、总状态写入前退出的情况。

同时删除了后处理函数中一行残留的无效 f-string，并统一外部 validator 的计划记录和实际执行标签，避免未来续跑在导入 runner 或记录复用时出错。代码静态编译通过；v4 runner、验证器、评价器、汇总器、路线判定、模型和导出相关测试共 67 项通过。使用项目 `slm_pvs` 环境的两个 dry-run 分别确认 `scan8` 生成 8 个计划成员、`formal80` 生成 15 个计划成员，均保持 `testRead=false`。本次没有启动新的训练或正式评价任务。

正式运行前发现并修正了四项后处理协议问题。

### Refine 安全池排名

两个 refine seed 都有 calibration 安全工作点时，配置先进入安全池，再按 validation balanced accuracy、useful cull、precision、预测 GLB 字节、平均预测实例数、weighted recall 下界和点估计依次比较。weighted recall 在通过门后不再压过分类与资源效率。安全池为空时才使用以 weighted recall 下界优先的诊断池。validation 分类或资源字段缺失时冻结配置直接失败。

### Checkpoint 与运行表绑定

训练器只序列化一次 `best_safe.pt`，随后复制为 `best.pt`。安全成员固定选择 `best_safe.pt`，诊断成员固定选择 `best_diagnostic.pt`，不允许回退到 `last.pt`。

正式导出不读取训练目录的末轮 124 维表。它从被评价 checkpoint 对应的 W042 96 维几何表与 checkpoint 内 28 维生存系数重建专属运行表；图像、资源和 validation replay 使用这份同一成员的 bundle。运行前通过 schema、shape、变体配置、阈值来源和数值一致性检查防止成员之间误配。在 warmup 等价输出条件下，两条独立 CUDA 路径只有 60 个 FP16 生存系数发生差异，最大绝对差为 `6.1e-5`，处于 FP16 量化容差内。

### 变体、损失、残差与 view-cell 协议绑定

训练协议现在显式保存 `variant`、`lossVariant`、逐实例校准模式，以及固定水平圆盘 view-cell 的半径和候选/查询语义。runner、validator 与 exporter 分别复核这些字段，不能再由目录名代替内容身份。`without_instance_calibration_residual` 还必须同时满足：checkpoint 残差张量为零、混合系数为零、融合系数等于共享先验，且 `modelState` 中不存在逐实例残差参数。错误损失标签、非零 disabled 残差、超过 8 cycles 的频率范围和缺失 view-cell 半径均有反例测试并会直接失败。

### Validation 安全门与资源字段

validation 只回放 calibration 冻结阈值，使用与校准协议登记一致的 bootstrap 次数。正式安全资格要求 aggregate weighted recall 和其单侧 95% 下界都严格大于 `0.99`。最终两 pose smoke 的对应值为 `0.965759` 和 `0.967112`，因此会被正确标记为 validation 不安全；这不是效果结论。

同一 validation row 已产生非零资源字段：平均预测 53 个 GLB、3,364,794 字节，候选 1,610 个 GLB、282,370,264 字节。refine 排名读取这些 validation 字段，不读取校准阈值扫描行中的资源占位值。

### 图像指标回填

formal 后处理要求每个 checkpoint 单独运行硬件 Color-ID 评价，核验 NVIDIA GPU 门、validation split、checkpoint、运行表、阈值和完全相同的 view-cell/subpose key。最终汇总回填 mean/p95 miss-pixel、wrong-ID pixel 和 extra-pixel，并对图像差值执行外层 seed、内层 view-cell 的 10,000 次 paired bootstrap。图像或资源入口缺失会在训练前拒绝 formal 启动；软件 GPU、指标缺失或 key 不一致等后处理错误会保留训练结果并把 pipeline 标记为 `training_complete_evaluation_incomplete`。

### 协议验证结果

| 检查 | 结果 |
|---|---|
| v4 相关 unittest | 103 项通过 |
| `neural_instance_culling` 全部显式测试模块 | 349 项通过 |
| validator / summarizer / route self-test | 全部通过 |
| Color-ID 图像 / GLB 资源 self-test | 全部通过 |
| 真实 smoke checkpoint schema | 通过，variant/loss/残差模式/view-cell/seed/候选/几何/关系/阈值一致 |
| checkpoint 专属 runtime bundle | 通过，124 维表的 schema、shape、来源配置和生存系数一致 |
| 旧 runner dry-run | 曾通过 5 / 18 成员计划生成；已由新的 scan8 / 15 成员执行计划替代，尚待代码对齐 |

本轮没有执行正式硬件 Color-ID 图像评价，因为 pilot/refine/formal checkpoint 尚未产生；当前通过的是回填实现、硬件门拒绝逻辑和配对统计契约，不能写成已有正式图像结果。

当前默认浏览器 `InstancePVS` 仍只读取已部署旧模型 schema，尚未实现 v4 的 124 维固定表、矩包络频率和查表运行路径。v4 导出成功只证明运行资产契约成立，不能写成已经完成前端 WebGPU 接入；正式长训冻结最终 bundle 后再实现独立适配与硬件 parity，且在通过前不修改默认前端资产。

## 导出验证

导出器分别从完整模型与关闭残差的 smoke checkpoint 重建了运行包并验证运行时 allow-list。浏览器包只包含固定实例表、AABB、实例到 GLB 映射、在线查询权重、16 个频率和固定圆盘传递表，不包含逐实例残差原始参数、可靠度、关系 CSR、组 ID、生存观察、深度层或离线编码器。

| 运行资产 | 字节数 |
|---|---:|
| 18,831 x 124 FP16 固定表 | 4,670,088 |
| 在线查询权重 | 45,198 |
| 频率 | 576 |
| 圆盘传递表 | 32,768 |
| 神经资产合计 | 4,748,630，约 4.53 MiB |
| 预算 | 不超过 7 MiB，通过 |

固定表保持 `96 维几何 + 28 维生存系数`。前端仍不运行关系图、邻居查询、PointNet、subpose 展开或 AABB 八角点投影。

## 主要修改文件

- 数据：`build_triangle_depth_layer_evidence.py`、`build_viewcell_pose_csr_for_fixed_geo.py`、`build_observed_relation_csr.py`、`build_degree_preserving_shuffled_relation_csr.py`；
- 模型：`bounded_relation_survival_moment_model.py`、`bounded_hierarchical_relation_survival.py`、`viewcell_ray_space.py`、`viewcell_moment_envelope_spectral_query.py`；
- 损失与采样：`safety_reserve_operating_utility_loss.py`、`stratified_survival_sampler.py`、`survival_loss.py`；
- 训练与导出：`train_bounded_relation_survival_moment_safety.py`、`export_bounded_relation_survival_moment.py`；
- 评价：独立的 v4 runner、validator、evaluator、summarizer 和 route decision 入口及其单元测试。

## 复现命令

正式关系重建的核心命令为：

```bash
conda run -n slm_pvs python \
  neural_instance_culling/dataset/build_observed_relation_csr.py \
  --dataset-dir neural_instance_culling/dataset/out/pose_csr_hkust_v3_bounded_relation_moment_fov66_v3 \
  --runtime-meta hkust-v3/assets/runtimeVisibilityMeta.json \
  --layer-cache-dir neural_instance_culling/benchmark/out/pvs_ray_context_survival_owrb_v1_subpose5_20260811_directchrome/merged_cache_v2 \
  --output-dir neural_instance_culling/dataset/out/pvs_bounded_relation_survival_moment_envelope_safety_reserve_v3/bounded_relation_csr_v3 \
  --splits train --source-k 12
```

置乱对照构建命令为：

```bash
conda run -n slm_pvs python \
  neural_instance_culling/dataset/build_degree_preserving_shuffled_relation_csr.py \
  --relation-dir neural_instance_culling/dataset/out/pvs_bounded_relation_survival_moment_envelope_safety_reserve_v3/bounded_relation_csr_v3 \
  --runtime-meta hkust-v3/assets/runtimeVisibilityMeta.json \
  --output-dir neural_instance_culling/dataset/out/pvs_bounded_relation_survival_moment_envelope_safety_reserve_v3/degree_preserving_shuffled_relation_csr_v3 \
  --seed 20260814 --distance-buckets 8 --swap-fraction 1.0
```

训练统一由以下 runner 管理：

```bash
conda run -n slm_pvs python \
  neural_instance_culling/benchmark/run_pvs_bounded_relation_survival_moment_v4.py smoke
```

runner 的 `pilot12`、`refine24` 和 `formal80` 分别对应五成员快速验证、双 seed 八组损失/优化量级复验，以及六变体乘三 seed 的 80 epoch 长训。每个 checkpoint 只由自己的 calibration 冻结阈值；validation 只做冻结阈值 replay。

## 当前结论与后续风险

本轮可以确认实现纠错已进入真实数据和 CUDA 路径，不能确认三项创新已经提升模型效果。正式长训和评价完成后才能形成路线结论。

## v4 执行续记（2026-08-15）

### scan8 完成与冻结

八个固定参数点均使用 seed `20260801`、8 epoch、每 epoch 50 step，并完成 calibration 168 pose 与 validation 213 pose 的回放。八个成员均生成第 4、8 epoch 快照、`calibration_ready_summary.json` 和完整训练历史；没有发现非有限 loss 或梯度。

按 validation 安全池规则冻结 `s02`：

| 配置 | validation weighted recall | 单侧 LCB | balanced accuracy | useful cull | precision |
|---|---:|---:|---:|---:|---:|
| `s02` | 0.999075 | 0.998690 | 0.667823 | 0.385275 | 0.067480 |

`s02` 的 validation weighted recall 和单侧 LCB 均达到严格的 `>0.99` 安全门，但这只是单种子、8 epoch 扫描结果，不能直接替代三种子 80 epoch 正式结论；根据计划，formal80 仍然无条件执行。

### 执行前契约修正

在 formal80 启动前补齐了四类检查：

- 续跑成员必须在 `train_history.json` 中达到注册 epoch，文件残留不能伪装为完成训练；
- 只有上一轮 training status 中身份、路径和返回码均一致的成员才允许复用；失败或身份缺失的目录会创建新的 attempt；
- scan 冻结前对每个成员复核 checkpoint、变体、seed、候选/关系/几何来源、校准 168 pose 和 validation 213 pose；
- formal dry-run 也必须读取真实 `frozen_config.json`，并检查已有冻结文件与重新汇总的选择一致。

同时统一了冻结配置 schema，修复了旧 runner 产生的 schema 名称不一致问题。补丁后的 runner 与 validator 定向测试 26 项通过；前置完整 v4 相关测试 57 项通过。

### formal80 当前状态

正式输出目录为：

```text
neural_instance_culling/model/out/
  pvs_bounded_relation_prior_instance_calibrated_moment_safety_reserve_v4_formal80/
```

当前使用固定 `s02` 超参数、seed `20260801/02/03`、5 个注册变体、每个成员 80 epoch、每 epoch 100 step，四张 RTX A6000 由动态队列调度。训练日志位于：

```text
neural_instance_culling/benchmark/out/
  pvs_bounded_relation_prior_instance_calibrated_moment_safety_reserve_v4_logs/
```

本阶段不读取 test，不修改默认 checkpoint、阈值、前端资产或部署包。训练完成后由同一 runner 自动执行校准冻结阈值、完整 validation、硬件图像评价、GLB 资源评价、10,000 次 paired bootstrap 和路线报告。

后续执行顺序保持为：

1. 完成 15 个 formal80 成员，不因安全门失败取消任何成员；
2. 对每个 checkpoint 选择其 calibration 自己冻结的工作点并导出同 checkpoint 的运行表；
3. 完成 validation、图像和资源评价；
4. 汇总 seed 聚类的 10,000 次 paired bootstrap 并生成路线判定；
5. 更新正式评价报告和文档索引，最后进行默认资产未被修改的状态审计。

此前关于 `pilot12`、`refine24` 和 18 成员旧矩阵的文字属于历史实施记录，不作为本轮执行入口。

历史后续步骤如下，已由上述 formal80 流程替代：

1. 运行五成员 `pilot12`；
2. 运行双 seed `refine24`；
3. 运行六变体旧矩阵；
4. 由旧矩阵直接进入评价。

当前主要风险是极端关系格的 K=12 证据质量长尾、逐实例残差对稀疏实例过拟合、离线训练单步成本以及正式 calibration/validation 上是否存在稳定安全工作点。任何一项未通过都必须如实降级，不能修改候选、补 GT、读取 test 选阈值或提前替换前端默认资产。
