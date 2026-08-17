# 有界关系生存、视点矩包络与安全裕度效用损失纠错实施计划

- 日期：2026-08-14
- 状态：历史纠错设计依据，已由 2026-08-15 的 v4 实施与运行计划取代，不再直接执行
- 历史实验前缀：`pvs_bounded_relation_survival_moment_envelope_safety_reserve_v3`
- 前置诊断：[`pvs_integrated_spectral_survival_field_diagnosis_2026-08-14.md`](pvs_integrated_spectral_survival_field_diagnosis_2026-08-14.md)
- 当前实施记录：[`../current/pvs_bounded_relation_survival_moment_v4_implementation_2026-08-15.md`](../current/pvs_bounded_relation_survival_moment_v4_implementation_2026-08-15.md)
- 当前唯一运行入口：[`pvs_bounded_relation_survival_moment_v4_run_and_ablation_2026-08-15.md`](pvs_bounded_relation_survival_moment_v4_run_and_ablation_2026-08-15.md)

> 本文保留用于说明 v3 纠错的数学与数据设计。其“共享关系生成器直接输出全部实例生存场”的表达能力问题已由 v4 的“共享关系先验 + 有界逐实例校准残差”修复；本文中的 v3 命令、15 成员矩阵和输出目录均不再执行。

## 实施进度（2026-08-15）

阶段一代码契约与阶段二派生数据已经落实。正式关系包含 448,165 条 top-12 边和 4,163,948 条 event/censor 生存观察；保持度数的置乱对照改变 87.15% 的边，同时保持观察文件逐字节一致。真实 HKUST CUDA 单步训练、四组梯度、calibration 冻结、validation replay 和 124 维运行表导出均已通过。

真实遥测随后发现并删除了关系辅助监督中的二次复杂度旧逻辑：原实现逐条同步 GPU 标量并为每条负边扫描整表，还遗漏距离分桶与交换后相对几何重算。当前实现复用正式对照的分层双边交换，一次 CPU 构造、一次 GPU gather，并重算 source-target 相对中心和尺度。同规格 32,768 观察批的训练步由 253.90 秒降为 4.60 秒，约加速 55.2 倍；负边数量由异常的 36,943 条收敛为 32,160 条，正式关系数据和浏览器资产均未改变。

实现验证还发现并修复了正式二进制方向编号、`uint32` gather、采样器逐实例全表扫描、圆盘零方差反向、FP32 梯度投影残差以及 8 cycles 查表范围等仅靠小型测试无法暴露的问题。全量 K=12 的最差证据保留质量为 6.08%，并非原 smoke 的 70.76%，后续必须作为 pilot 风险继续报告。

当前进度不构成模型效果结论。阶段三 `pilot12`、阶段四 `refine24` 和阶段五 15 成员 `formal80` 仍须按本计划执行；在正式 validation 和前端硬件门通过前，默认模型、阈值和前端资产保持不变。

## Material Passport

- Origin Skill：academic-research-suite / experiment-agent
- Origin Mode：plan
- Verification Status：CODE_AUDITED_CORRECTIVE_PLAN_NOT_YET_IMPLEMENTED
- 数据访问边界：允许 train、calibration、validation；最终模型和阈值冻结前禁止读取 test
- 候选边界：保持原生 66 度后退相机 AABB 候选，不补入 GT
- 前端边界：不在线运行关系图、邻居查询、subpose、PointNet、HZB 或全量 AABB 八角点投影

## 一、目标与执行约束

本计划不是在当前 v2 上继续堆叠参数，而是修复已经确认的数学、图结构、监督和损失实现错误，再重新验证三项论文创新：

1. 分层遮挡关系生存网络；
2. 视点区域积分频谱查询；
3. 在 RVL 高召回能力之上的安全约束资源效用损失。

三项创新的论文叙事保持不变：重型关系提取、图聚合和生存场生成全部离线完成；浏览器只读取固定实例表，并以当前视点执行一次轻量查询，同时输出实例可见性、实例视觉效用和 GLB 下载优先级。

实施必须遵守以下约束：

- `pvs_hierarchical_survival_integrated_spectral_threshold_aligned_utility_v2_formal80_20260814_rerun` 已自然完成 15/15，产物保持冻结，不作为 v3 代码或结果目录复用；
- 不在运行中的目录内改写 checkpoint、日志、manifest 或结果；
- 当前 v2 完成后先执行独立 validation replay 和 paired bootstrap，作为失败实现的正式诊断证据；
- 新代码和新数据使用 v3 schema 与独立输出目录，不复用 v2 名称表达新语义；
- v3 通过安全门和前端 smoke 前，不修改当前默认 checkpoint、阈值、前端资产或部署包；
- test split 只允许在最终变体、超参数、阈值规则和导出 schema 全部冻结后读取一次。

## 二、修正后的整体模型

### 2.1 离线阶段

```text
真实三角形深度层证据
  -> 同一实例的可见像素删失证据 + 遮挡像素事件证据
  -> 目标/方向/深度层内的真实遮挡来源排序
  -> 有界关系层级与方向深度注意力
  -> 每实例 4 x 7 遮挡生存系数
  -> 96 维几何 + 28 维生存系数固定表
```

### 2.2 浏览器阶段

```text
当前 view-cell 锚点与相机朝向
  -> 后退 3.4641 m、66 度相机生成候选集合
  -> 当前 view-cell 中心生成视角查询均值
  -> 水平圆盘低秩不确定性矩
  -> 16 个联合频率的均值与相位相关方差包络
  -> 固定生存场查询 + 共享轻量主干
  -> 实例可见性 / 实例视觉效用 / GLB 下载优先级
  -> 当前真实 60 度视锥过滤与实例级显示
```

候选相机和模型查询中心必须成为两个明确字段：候选相机负责保守覆盖，view-cell 中心负责描述 GT subpose 分布。二者不能继续复用同一个 `camera_world`。

### 2.3 运行资产预算

固定实例表仍为：

```text
96 维几何 + 28 维遮挡生存系数 = 124 个 FP16
```

HKUST 的固定表大小为：

```text
18,831 x 124 x 2 = 4,670,088 bytes = 4.454 MiB
```

关系 CSR、层级、深度层、观察样本和离线图编码器不导出。查询网络和三个任务头的权重预算不超过 2.5 MiB，全部神经资产硬门仍为 7 MiB。

## 三、创新一：有界分层遮挡关系生存网络

### 3.1 研究定义

该模块输入：

- 每个实例 96 维固定几何；
- train-only 的真实“前方来源实例遮挡后方目标实例”有向边；
- 每条边的方向、深度层、深度间隔、像素支持、pose 支持和证据置信度；
- 由真实关系和空间约束共同构建的有界局部/结构层级。

该模块输出：

- 每个实例 `4 x 7` 个单调遮挡生存系数；
- 仅用于训练诊断的关系增量、关系门和边注意力统计。

需要它的原因是：实例可见性是目标形状、潜在遮挡来源和观察方向的联合函数。关系网络把真实遮挡来源离线压缩为逐实例方向生存场，浏览器无需再次查邻居。

### 3.2 深度层证据修复

修改：

- `neural_instance_culling/dataset/build_triangle_depth_layer_evidence.py`
- 对应 dataset tests 与 relation meta schema

当前 bug 是同一实例只要在任意后层出现，就会覆盖其第一层可见证据并整体写成 `event=1`。改为保留两类独立观察：

```text
第一层可见像素 -> right-censored observation，weight=log1p(visible_pixels)
后层遮挡像素   -> event observation，weight=log1p(occluded_pixels)
```

同一实例、同一方向、同一 subpose 可以同时产生一条删失观察和一条事件观察。不得再用单一布尔值覆盖两种证据。

产物新增统计：

- event/censor 原始数量与加权比例；
- 同时部分可见和部分遮挡的实例/pose 数；
- 每方向、每深度层、每实例覆盖；
- 观察深度分位数；
- 训练 split 来源和候选口径。

首层 ID 与原 Color-ID 可见结果、深度严格递增、候选子集和 train-only 门继续保留。正式证据仍禁止使用 AABB 重叠作为遮挡事件。

### 3.3 生存深度坐标修复

删除当前高度挤压的：

\[
\rho=d/(d+r_i).
\]

改为 train-only 分位数归一化的半径相对对数深度：

\[
z_i(d)=
\operatorname{clip}
\left(
\frac{\log(1+d/(r_i+\epsilon))-q_{0.01}}
{q_{0.99}-q_{0.01}},
0,1
\right).
\]

其中 `q01/q99` 只由 train observation 冻结并写入 dataset/checkpoint/export meta。该变换随距离严格单调，保留实例尺度信息，并把大量远距离细小构件从 `[0.94,1]` 的拥挤区间展开。

前端已经计算距离和实例半径，额外只需一次 `log1p`；现有九维 ray 查询本身也包含对数距离，所以 WebGPU 可以共享中间结果。

### 3.4 真实来源选择，不使用最近邻

修改：

- `neural_instance_culling/dataset/build_observed_relation_csr.py`
- `neural_instance_culling/model/common/train_observed_relation_csr.py`

每个目标实例、每个球面方向、每个深度层只保留真实遮挡证据最强的最多 8 个来源。来源排序分数固定为 train-only 证据的单调组合：

```text
edge_score = evidence_confidence
           * log1p(pixel_support)
           * log1p(1 + pose_support_count)
```

三个分项先在当前目标/方向/深度格内归一化，避免大表面或高频采样区域只因绝对像素数占据全部来源。top-k 只从真实深度层边中选择，不引入 KNN、AABB 相交或动态 teacher。

CSR 同时保存被截断总置信质量和保留质量，用于判断 top-k 是否丢失过多证据。若任一方向/深度格的保留质量低于 90%，pilot 可以把 K 从 8 提升到 12；K 的改变必须生成独立命名的关系产物，不能运行中临时修改。

### 3.5 有界层级替换 connected components

删除无容量约束的 connected-components 分组。新层级使用“强关系优先、有容量和空间直径上限的确定性合并”：

1. 将有向边按对称强度、实例 ID 稳定排序；
2. 依次尝试合并两个组；
3. 局部组只有在合并后实例数不超过 32、世界空间直径不超过登记局部尺度时才允许合并；
4. 未合并实例保留为单例，不通过弱桥接边继续传染；
5. 局部组再以相同方式构建结构组，每个结构组最多包含 64 个局部组，并设置结构空间直径上限；
6. 所有尺度、上限、组大小直方图和最大组 ID 写入 meta。

局部和结构空间尺度先由 train 实例中心距离分布的固定分位数生成，再写入 manifest；validation/test 不参与分组。层级契约硬门为：

- 局部组最大实例数 `<=32`；
- 结构组最大局部组数 `<=64`；
- 不存在覆盖超过 10% 全场实例的单一结构组；
- 所有实例恰好属于一个局部组，所有局部组恰好属于一个结构组；
- 无越界 ID、空洞 ID 或由非 train 边产生的合并。

### 3.6 置信度注意力替换普通平均

修改：

- `neural_instance_culling/model/common/hierarchical_relation_survival.py`

当前 `segment_mean` 改为目标/方向/深度格内的 masked attention：

```text
message = MLP(source_geo, target_geo, relative_geometry, direction, depth)
attention_logit = learned_score(message) + log(confidence + eps)
relation_token = sum(softmax(attention_logit) * message)
```

显式 `log(confidence)` 先验保证大量极弱边不会仅因数量稀释少量强边。三个深度层继续保持顺序信息，但改为有序门控而不是把层内来源先做等权平均。

实例、局部组和结构组均使用有界 attention pooling。解码输出采用：

```text
geometry_base_coefficients + relation_gate * relation_delta_coefficients
```

并记录 `relation_gate`、`relation_delta_norm / base_norm`、移除真实边后的系数变化。该残差结构保留几何稳定基线，同时使关系增量可解释、可测量。

### 3.7 关系直接监督

新增仅在离线训练使用的关系一致性辅助损失：

- 真边与同方向/深度/距离分层内的负边二分类；
- 来源在目标前方的深度顺序回归；
- 真实边置信度排序。

负边从 train-only 实例中生成，不使用 dynamic-pool teacher，不读取 validation/test。该辅助项的作用是让关系编码器必须学习真实来源-目标结构，避免最终可见性 MLP 完全绕过关系消息。

### 3.8 正确的 shuffled control

当前简单 ID 平移删除。正式负对照使用确定性的有向双边交换，在同方向、同深度层、相近距离桶内执行：

```text
(source_a -> target_a, source_b -> target_b)
  =>
(source_a -> target_b, source_b -> target_a)
```

交换必须保持 source/target 度数、方向/深度边数分布和证据强度分布，拒绝自环与重复边。相对几何特征必须按置乱后的 source/target 重新计算，有界层级也必须从置乱图重新构建，不能复用真实图的 group IDs。

## 四、创新二：视点区域低秩矩包络频谱查询

### 4.1 研究定义

该模块输入：

- 当前 view-cell 中心到实例中心的九维 ray-space 查询；
- view-cell 水平圆盘在 ray-space 的两个低秩不确定性轴；
- 每实例固定生存系数和归一化深度。

该模块输出：

- 4 维生存方向基；
- 8 维 view-cell 边界频谱摘要；
- 8 维生存语义。

需要它的原因是：view-cell PVS 是区域内潜在可见并集，单点 Fourier 查询不能表示边界位置变化；前端又不能运行多个 subpose。解析矩包络以一次查询近似区域内频率响应的均值和不确定性。

### 4.2 分离候选相机与查询中心

修改：

- `neural_instance_culling/dataset/build_viewcell_pose_csr_for_fixed_geo.py`
- `neural_instance_culling/model/pose_csr_dataset.py`
- `neural_instance_culling/model/hierarchical_relation_survival_integrated_model.py`
- 后续 exporter 和前端 Worker/WebGPU 输入 schema

新数据 schema 明确保存：

```text
candidate_camera_world = viewcell_center - forward * back_offset
query_center_world     = viewcell_center
```

候选 CSR 和 MVP 继续只由 `candidate_camera_world + FOV66` 生成。模型 ray、区域均值和低秩矩只由 `query_center_world + FOV66` 生成。前端触发一次预测时把当前锚点记为 view-cell 中心；在锚点范围内不低频重跑预测，超出范围后建立新锚点。

### 4.3 用水平圆盘的低秩映射替换启发式对角方差

修改：

- `neural_instance_culling/model/common/viewcell_integrated_spectral_query.py`

HKUST 当前登记 view-cell 是世界水平面内的均匀圆盘。先用九维 ray-space 特征对相机位置的一阶 Jacobian，把单位圆盘映射为两个九维低秩轴：

\[
B=[J_xR,\ J_zR]\in\mathbb R^{9\times2}.
\]

若只看二阶协方差，则：

\[
\Sigma_x\approx\frac14BB^T.
\]

但是正式实现不能仅凭该协方差把圆盘改写成高斯分布。对任一联合频率 `f`，圆盘在线性化 ray-space 中的径向参数为：

\[
s=2\pi\lVert B^Tf\rVert.
\]

每个频率只需要两个低秩投影，不构造完整 9x9 协方差、不展开 subpose，也不传输 81 个协方差值。其他场景若登记矩形或非水平 view-cell，必须由 dataset meta 提供对应形状及其频谱传递函数，不能继续硬编码 HKUST 的圆盘半径。

### 4.4 修复 Fourier 单位

联合频率统一登记为 cycles，相位固定为：

\[
\phi=2\pi f^T\mu.
\]

均匀圆盘的频谱传递函数使用：

\[
\chi(s)=\frac{2J_1(s)}{s},\qquad \chi(0)=1.
\]

因此一阶频率响应为 `chi(s)`，二阶矩需要的双频响应为 `chi(2s)`。这在“ray-space 对相机位移的一阶线性化”条件下对应连续均匀圆盘，不再把圆盘未经说明地替换成高斯；线性化误差仍必须单独测量。

当前环境 PyTorch 2.5.1 的 `torch.special.bessel_j1` 不提供 autograd，不能直接放入可学习频率路径。它只用于离线生成参考值。训练端和 WGSL 统一读取同一张固定 FP32 径向传递表，并执行相同的分段线性插值；训练端对格内插值坐标保留梯度，使频率仍可学习。表大小硬门为 32 KiB，网格范围和密度由误差测试冻结，最大绝对函数误差小于 `1e-4`，有限差分梯度误差必须单独登记。超出范围时只能使用登记并验证过的渐近近似，不能静默截断。schema、PyTorch、导出器和 WGSL 必须记录同一 cycles 单位、`2*pi` 参数、查表范围、插值公式和误差界。旧高斯公式及缺失 `(2*pi)^2` 的公式直接删除，不保留 compatibility flag。

### 4.5 从均值改为相位相关矩包络

当前每个频率只输出均值正弦、均值余弦和一个与相位无关的 unresolved energy。新实现解析计算：

```text
E[sin(phi)]
E[cos(phi)]
Std[sin(phi)]
Std[cos(phi)]
```

二阶矩通过 `2f` 的圆盘响应得到：

\[
E[\sin^2\phi]=\frac{1-E[\cos2\phi]}2,
\qquad
E[\cos^2\phi]=\frac{1+E[\cos2\phi]}2.
\]

均值描述区域内稳定响应，标准差描述边界位置可能出现但会被平均隐藏的高频变化。查询网络分别编码稳定矩和边界矩，生成：

- 4 维生存方向基，用于查询 `4 x 7` 生存系数；
- 8 维边界频谱摘要，直接进入可见性共享主干。

这样不把 16 个频率全部压缩进同一个 4 维瓶颈，也不增加逐实例固定表大小。该表示仍不是精确的可见并集，但比单纯频谱均值提供了学习保守上包络所需的二阶信息。

### 4.6 频率与计算预算

首版保持 16 个可学习联合频率和最大范数 8，不同时扫描频率数、关系宽度和损失，避免变量混杂。浏览器每候选执行：

- 一个中心 ray-space 查询；
- 两个低秩不确定性轴；
- 16 次相位、两个低秩投影，以及一阶/双频圆盘传递查表；
- 一个 4 维方向基和一个 8 维边界摘要；
- 一次共享 MLP。

禁止前端生成真实 subpose、循环多次网络前向或访问关系 CSR。正式晋级要求 WebGPU 10k 候选 p95 不高于当前 Fourier117 硬件路径的 1.10 倍；真实移动设备可用后继续使用 p95 `<50 ms` 门。

## 五、创新三：安全裕度驱动的工作区资源效用损失

### 5.1 研究定义

该损失输入：

- 实例可见性 logits 和 GT；
- pose 内归一化视觉权重；
- dense-subpose 可见命中率；
- 实例到 GLB 的映射、真实字节和下载效用；
- train-only 采样的工作阈值。

该损失输出：

- 保留原 RVL 作用的安全梯度；
- 只在正例尾部已有安全裕度时启用的实例负样本和 GLB 资源梯度；
- 独立的生存关系、视觉效用和下载排序辅助梯度。

需要它的原因是：weighted recall 只保护重要 GT，不惩罚 false positive；普通 BCE 又容易受巨大负样本规模和类别不平衡影响。训练必须在高召回安全前提下，直接压缩实际工作阈值附近的无效实例和无需求 GLB。

### 5.2 新模块与旧模块清理

新增正式模块：

```text
neural_instance_culling/model/common/safety_reserve_operating_utility_loss.py
```

在 v3 接管主线后删除 `viewcell_quality_rvl_loss.py` 的旧正式含义，不增加别名或双重默认。当前运行 v2 完成前不删除其源文件。

### 5.3 安全核心

保留 `rvl_strong_v2` 的主要高召回结构，并使用已经登记的 pose 内视觉权重归一化。安全核心为：

\[
L_{safe}=L_{RVL}+\lambda_{boundary}L_{boundary-tail}.
\]

边界正例风险结合视觉权重、漏检概率和 subpose 稀有度：

\[
r_i=q_i(1-p_i)
\left[1+\lambda_{rare}(1-\sqrt{h_i})\right].
\]

每个 pose 对最高风险的正例质量尾部做 CVaR。该项保护只在少量合法位置可见但仍属于 PVS 并集的实例。

### 5.4 连续工作阈值分布

不读取 calibration 阈值参与反向传播。每个 train step 使用确定性 RNG 从以下对数均匀工作区采样两个阈值：

```text
log(tau) ~ Uniform(log(1e-4), log(0.2))
```

并周期性插入固定锚点：

```text
{0.001, 0.005, 0.02, 0.10}
```

该范围覆盖当前真实冻结阈值，也允许分数分离改善后进入中间工作区。阈值序列只依赖 train seed 和 global step，写入 manifest，不读取 calibration、validation 或 test。

### 5.5 阈值局部实例负样本

对 GT 不可见实例计算阈值附近的 soft keep，并只聚合每 pose 最高风险的一小部分负样本：

\[
L_{negative-band}
=
\operatorname{CVaR}_{\beta}
\left[
\sigma\left(\frac{l_i-\operatorname{logit}(\tau)}T\right)
\right].
\]

这直接提高 specificity、accuracy、balanced accuracy 和 precision，避免所有容易负样本把梯度平均到接近零。`beta` 首版固定为 2%，只在 pilot 证明过稀或过宽时产生新配置扫描。

### 5.6 用有界 soft-max 替换 noisy-OR

前端实际规则是某 GLB 内任一实例超过阈值即进入请求集合，更接近实例 logits 的最大值，不是独立事件 noisy-OR。

训练端对每个 GLB 取最高 8 个实例 logits，再使用温度 softmax 加权平均形成有界 smooth maximum：

```text
group_logit = sum(softmax(top_logits / T_group) * top_logits)
soft_request = sigmoid((group_logit - logit(tau)) / T_request)
```

固定 top-8 使梯度和组大小解耦，避免含 5,610 个实例的 GLB 因数量自动饱和。只对当前 pose 没有 GT 可见实例的 GLB 计算真实字节成本：

\[
L_{glb-resource}
=
\frac{\sum_{g\notin G_p}c_g\,soft\_request_g}
{\sum_{g\notin G_p}c_g+\epsilon}.
\]

训练成本使用固定 `log1p(bytes)` 归一化，正式报告继续使用原始字节。

### 5.7 安全裕度门

资源项不能从第一步常开。对当前采样阈值计算正例低尾部 logit 与阈值 logit 的加权裕度：

\[
m=Q^{weighted}_{0.01}(l_{positive})-\operatorname{logit}(\tau).
\]

在前 10% optimizer steps 内资源门固定为零；之后使用停止梯度的连续门：

\[
g_{reserve}=\sigma((m-m_0)/T_g).
\]

效率目标为：

\[
L_{efficiency}=g_{reserve}
(\lambda_{negative}L_{negative-band}
+\lambda_{glb}L_{glb-resource}).
\]

安全裕度门不是 calibration 安全证明，只是防止模型尚未学会正例时就用漏检换剔除。正式安全资格仍只由 calibration 冻结阈值和 validation weighted recall/LCB 决定。

### 5.8 梯度分组与投影

当前把生存、效用、下载和正则全部混入 safety gradient 的做法删除。新训练明确拆分：

```text
G_safe      <- RVL + boundary-positive tail
G_relation  <- survival censoring + relation consistency
G_schedule  <- visual utility + GLB ranking
G_efficiency<- negative band + no-demand GLB bytes
```

`G_relation + G_schedule` 和 `G_efficiency` 分别相对 `G_safe` 做冲突投影；只有点积小于零的分量被移除。两个辅助梯度分别限制在 `G_safe` 范数的预登记比例内，不能再由一个巨大的复合梯度决定资源方向。

每 step 记录：

- 四组梯度投影前后范数；
- 与安全梯度的点积；
- 投影和限幅比例；
- 安全裕度门值；
- 当前阈值及阈值附近正负样本数量。

pilot 的有效性检查为：资源梯度不能在超过 80% step 中低于安全梯度的 1%，也不能在超过 80% step 中持续触发上限。前者说明损失仍未产生作用，后者说明效率项过强。

### 5.9 总目标

\[
L=
L_{safe}
+\lambda_{survival}L_{survival}
+\lambda_{relation}L_{relation-consistency}
+\lambda_{utility}L_{visual-utility}
+\lambda_{download}L_{glb-priority}
+L_{efficiency}
+\lambda_{reg}L_{reg}.
\]

训练器不再把一个 `quality_total` 同时作为日志总量和另一套手工重组目标。每个分项只在一个地方组合，测试必须验证日志总和与实际反向目标一致。

## 六、生存观察训练调度

修改：

- `neural_instance_culling/model/common/survival_loss.py`
- `neural_instance_culling/model/train_hierarchical_relation_survival_integrated.py`

删除固定 `linspace(4096)`。新增 `StratifiedSurvivalObservationSampler`：

1. 只读取 train observation；
2. 按 event/censor、12 个方向、深度分位桶和实例 ID 建立索引；
3. 每个 epoch 对各桶做由 `seed + epoch` 决定的稳定 shuffle；
4. 每 step 默认读取 8,192 条并沿全局流顺序轮换；
5. event/censor 使用分层采样并携带抽样概率校正权重；
6. 每个有观察的实例至少每个 epoch 获得一次直接监督；
7. 日志记录本 epoch 唯一观察数、唯一实例数和方向覆盖。

若 8,192 条显存或训练时间不合格，只能通过已登记 pilot 把 batch 改为 4,096；即使减小 batch，也必须保持轮换和实例覆盖，不能恢复固定样本。

## 七、模型和训练入口改造

### 7.1 模型文件

主要修改：

- `neural_instance_culling/model/common/hierarchical_relation_survival.py`
- `neural_instance_culling/model/common/viewcell_integrated_spectral_query.py`
- `neural_instance_culling/model/hierarchical_relation_survival_integrated_model.py`
- `neural_instance_culling/model/train_hierarchical_relation_survival_integrated.py`

v3 模型输入明确为：

```text
固定几何 96
生存方向基 4
生存语义 8
view-cell 边界频谱摘要 8
中心 ray-space 9
低秩不确定性摘要 4
归一化深度 1
```

固定表仍为 124 维；新增 8 维边界摘要和低秩摘要均在线解析生成，不增加逐实例资产。

### 7.2 训练入口

当前 v2 formal 完成后，新建稳定入口：

```text
neural_instance_culling/model/train_bounded_relation_survival_moment_safety.py
```

旧训练器不复用 v3 名称。v3 稳定并接管后，删除错误 v2 默认注册和旧逻辑，不建立 compatibility alias。

训练入口必须：

- 校验 query center 与 candidate camera 均存在；
- 校验关系 artifact 为 corrected event/censor 和 bounded hierarchy schema；
- 校验 train/calibration/validation 候选口径和逐 pose 内容；
- 校验观察深度分位数和 event/censor 覆盖；
- 每个 checkpoint 使用自身 calibration 冻结阈值；
- 保存 `best_safe.pt` 和明确标记的 `best_diagnostic.pt`；
- 永不使用 validation/test 选择阈值或 checkpoint；
- 输出各损失和梯度分组的完整日志。

## 八、评价、runner 与后处理修复

### 8.1 当前 pipeline 缺口

修改：

- `neural_instance_culling/benchmark/evaluate_pvs_hierarchical_relation_survival_integrated.py`
- `neural_instance_culling/benchmark/summarize_pvs_hierarchical_relation_survival_integrated.py`
- `neural_instance_culling/benchmark/validate_pvs_hierarchical_relation_survival_integrated.py`
- 新 v3 runner

新 runner 的 formal 模式必须在训练成员全部成功后自动执行：

```text
checkpoint/schema validation
  -> calibration-frozen validation replay
  -> pose/candidate/GT semantic validation
  -> 10,000 paired bootstrap
  -> image and resource evaluation manifest
  -> route decision and report
```

不得再以 `formal_member_summary.json` 存在作为整个正式实验完成条件。任何后处理失败必须保留训练结果并把 pipeline 状态标为 `training_complete_evaluation_incomplete`，不能重复训练。

### 8.2 指标

每个工作点同时报告：

- aggregate 与 pose-macro recall、precision、F1、Jaccard；
- weighted recall 和单侧 95% LCB；
- accuracy、balanced accuracy、specificity；
- useful cull、bad cull、TP/FP/FN/TN；
- 平均预测数、预测/候选、预测/GT；
- GLB 数量、字节削减、相同视觉效用所需字节；
- 下载头 NDCG、预算效用和 required-GLB recall；
- Color-ID miss/wrong/extra pixel；
- CUDA、WebGPU、主线程和内存。

画面安全主门仍为 validation aggregate weighted recall `>0.99` 且其单侧 LCB `>0.99`。普通 pose recall 只作诊断。满足安全门后优先比较 balanced accuracy、useful cull、precision、平均预测数和 GLB 字节。

### 8.3 统计

正式 validation 使用三个相同 seed 和相同 pose 的 10,000 次 paired bootstrap：外层重采样 seed，内层在 seed 内重采样 pose。每项创新相对 full 的移除消融报告差值、95% 区间、方向和是否跨零。

## 九、导出和前端改造

### 9.1 导出器

修改或新增：

- `neural_instance_culling/model/export_hierarchical_relation_survival_integrated.py`
- v3 exporter tests

导出 meta 必须包含：

- `candidateCameraSemantics`；
- `queryCenterSemantics`；
- `viewcellShape=horizontal_disk` 和半径；
- 低秩圆盘映射、Fourier cycles 单位、`2*pi` 径向参数、传递查表范围和误差界；
- 生存深度 `q01/q99`；
- 124 维固定表布局；
- calibration 冻结阈值和安全状态；
- dataset、候选和 relation artifact 的 schema、shape 与来源。

导出包禁止包含 relation CSR、group IDs、observation、深度层或离线编码器输入。

### 9.2 WebGPU

新模型达到 validation 晋级门后再修改：

- `slm2viewer` 的神经剔除 Worker/WebGPU backend；
- parity 生成脚本和硬件 benchmark。

前端逻辑保持：

1. 以当前锚点构造后退 66 度候选相机；
2. 用当前锚点本身作为 query center；
3. GPU gather 124 维固定表；
4. GPU 计算两个低秩轴和 16 个频谱矩；
5. GPU 输出实例可见性、效用和下载分数；
6. 当前 60 度真实相机再次做实例级视锥过滤；
7. GLB 聚合只用于下载，不能替代实例级显示。

不新增 CPU/GPU 往返的全量实例中间结果。预测触发继续使用“超出 view-cell 范围再预测”，不恢复偷偷低频更新。

### 9.3 硬件验收

WebGPU parity 必须使用 AGENTS.md 登记的 Vulkan Chrome 参数，调用 `requestAdapter({powerPreference:'high-performance'})` 并保存 adapter info、WebGL renderer 和同窗口 `nvidia-smi/pmon`。SwiftShader 只能记录数值 parity，不能进入硬件性能结论。

## 十、代码测试矩阵

### 10.1 数据和关系

1. 同一实例同时可见和遮挡时必须输出独立 event/censor 观察；
2. event/censor 加权总量与原像素聚合守恒；
3. observation sampler 连续两个 step 不得返回同一固定索引集合；
4. 一个 epoch 覆盖所有有观察实例；
5. 新深度坐标严格单调且分位数不再挤压到 1；
6. local/structural group 上限、空间直径和无巨型分量门通过；
7. attention 权重在每个非空格内和为 1，空格输出零；
8. degree-preserving shuffle 保持度数/方向/深度分布，重算相对几何和层级；
9. train-only 关系构建拒绝 calibration、validation 和 test。

### 10.2 积分频谱

1. 零半径时解析矩严格退化为 point Fourier；
2. 相位和圆盘径向参数均包含正确的 `2*pi` cycles 换算，schema 与代码一致；
3. query center 等于 view-cell center，不等于 candidate back camera；
4. 水平圆盘没有世界 Y 位移；
5. 圆盘传递函数及其一、二阶矩与 100,000 个 Monte Carlo 水平圆盘样本在预登记误差内一致，并分别报告查表误差和 ray-space 一阶线性化误差；
6. 径向传递表的插值梯度与中心有限差分一致，禁止调用无 autograd 的 `torch.special.bessel_j1` 参与反向路径；
7. 频率置换只产生对应通道置换；
8. PyTorch/导出/WGSL FP32 与 FP16 parity 通过；
9. 10k 候选硬件 WebGPU 输出有限且 canvas 非空。

### 10.3 损失和梯度

1. 工作阈值采样覆盖 `[1e-4,0.2]` 且只依赖 train seed/step；
2. GLB smooth-max 对复制大量相同低 logit 实例基本不变，不再随组大小饱和；
3. 最高实例 logit 上升时 soft request 单调上升；
4. 安全 warmup 内 efficiency gradient 为零；
5. 安全裕度增加时资源门单调开启；
6. 关系、调度和效率梯度投影后与安全梯度点积不为负；
7. 日志分项之和等于实际 objective；
8. 所有损失、梯度和 dual/EMA 状态有限；
9. calibration/test 张量传入训练损失必须失败。

### 10.4 runner 与评价

1. formal manifest 的变体数、seed 数和成员数正确；
2. 每个成员使用相同的候选、GT 和 pose 顺序；
3. threshold 只能来自自身 calibration；
4. validation/test 修改不能改变 checkpoint 或冻结阈值；
5. 训练完成而评价失败时不得重训；
6. paired bootstrap 为 10,000 次 seed/pose 配对采样；
7. download logits 改变时下载指标变化、实例分类指标不变；
8. runtime bundle 不包含离线图资源。

## 十一、实施阶段

### 阶段零：完成并封存当前 v2

1. 已保持原进程自然运行至结束，没有发送停止或改参信号；
2. 已核对 15/15 member、三个 seed 和 checkpoint 完整性；
3. 执行独立 validation replay、候选口径校验和 10,000 paired bootstrap；
4. 写 v2 正式诊断报告，明确数学和监督 bug；
5. 不导出、不部署 v2；
6. 从当前 README 主行动入口移除 v2，保留一份失败实验报告和必要复现产物。

### 阶段一：代码契约修复

按以下顺序实现，前一步测试未通过不得进入下一步：

1. event/censor 证据与随机分层 observation sampler；
2. 新深度坐标与 bounded hierarchy；
3. 置信度 attention 和正确 shuffled control；
4. query center、低秩圆盘映射、`2*pi`、圆盘传递函数和矩包络；
5. 安全裕度工作区损失和梯度分组；
6. evaluator/runner 自动后处理；
7. exporter schema 和软件数值 parity。

### 阶段二：重建派生数据

首轮不重新采样 GT，也不改变候选集合。优先从现有 train-only 深度层缓存和 subpose sidecar 重建：

```text
corrected_triangle_event_censor_v3/
bounded_relation_csr_v3/
viewcell_query_geometry_v3/
```

新产物必须保留原始数据来源、生成参数和 schema。只有当重建后真实边覆盖仍无法满足方向/实例覆盖门，才登记额外 train-only 深度层渲染；正式补采必须使用 Chrome NVIDIA Vulkan 硬件路径并生成 GPU evidence，不能使用软件后端进入正式数据。

### 阶段三：四 GPU 快速功能验证

先使用 seed `20260801`、12 epoch、每 epoch 100 step，同时运行四个累积成员：

| 成员 | 关系 | 视点查询 | 损失 | 目的 |
|---|---|---|---|---|
| `control` | 容量匹配几何系数 | 同频率 point query | 归一化 RVL | 修复后的控制基线 |
| `relation` | 有界真实关系生存 | 同频率 point query | 归一化 RVL | 快速检查真实关系是否产生增量 |
| `relation_moment` | 有界真实关系生存 | 低秩矩包络 | 归一化 RVL | 检查区域表达是否保持安全尾部 |
| `full` | 有界真实关系生存 | 低秩矩包络 | 安全裕度效用损失 | 检查三项组合 |

快速阶段只用于发现实现错误和明显方向性，不做 paired bootstrap，不写论文正式结论。以下项目是诊断门：

- 所有成员数值稳定、候选摘要一致；
- relation gate 与 relation delta 非零，真实边移除会改变系数；
- Full calibration 能找到 weighted recall `>0.99` 的候选阈值；
- validation weighted recall 不低于 0.985；
- 相比上一个累积成员，balanced accuracy、precision、useful cull 或平均预测数至少一项有改善趋势，且 weighted recall 没有下降超过 0.01；
- 资源梯度不是长期无效或长期撞上限。

若发现非有限数值、schema/候选/GT 错误、关系消息完全断路或解析矩不满足数值契约，必须先修复对应代码并重跑该成员。若代码契约全部通过但模型指标未达到上述诊断目标，记录失败原因并继续选择扫描中的诊断最优配置；诊断指标不能取消后续完整长训。

### 阶段四：小规模超参数复验

架构参数先固定，只扫描损失和优化量级。使用 8 个预先生成的组合、seed `20260801/20260802`、24 epoch，四 GPU 队列执行。首轮范围：

| 参数 | 候选 |
|---|---|
| learning rate | `1e-4, 2e-4, 3e-4` |
| survival weight | `0.10, 0.25, 0.40` |
| relation consistency weight | `0.05, 0.10` |
| boundary tail weight | `0.15, 0.30, 0.50` |
| negative-band weight | `0.01, 0.03, 0.06` |
| GLB resource weight | `0.005, 0.015, 0.03` |
| auxiliary gradient cap | `0.10, 0.25` |
| efficiency gradient cap | `0.10, 0.25` |

不使用任意加权总分。先筛选两个 seed 均有 calibration 安全工作点的配置，再按 validation weighted recall/LCB、balanced accuracy、useful cull、precision、平均预测数、GLB 字节和 seed 方差做 Pareto 选择。若没有安全配置，在确认代码契约无误后按相同 Pareto 口径冻结诊断最优配置，并仍然进入正式长训；安全门只决定结果能否晋级论文主模型，不决定是否获得完整训练曲线和消融结果。

### 阶段五：正式长训与核心消融

正式配置冻结后，从头训练以下 5 个变体，每个 `3 seeds x 80 epochs`，共 15 个成员：

| 变体 | 作用 |
|---|---|
| `full` | 三项创新完整组合 |
| `without_bounded_relation` | 容量匹配几何系数，评价关系生存网络贡献 |
| `degree_preserving_shuffled_relation` | 评价真实来源-目标身份是否被使用 |
| `without_viewcell_moment_envelope` | 同容量 point query，评价区域矩包络贡献 |
| `without_safety_reserve_utility` | 只用归一化 RVL，评价新损失总贡献 |

四张 GPU 使用持久单卡 worker 队列，同一 GPU 同时只运行一个成员。训练不中途因指标差早停；只有非有限数值、候选/GT/schema 错误或基础设施不可恢复时暂停受影响成员。

### 阶段六：正式评价和前端晋级

15 个成员完成后自动执行：

1. 完整 validation replay；
2. 10,000 paired bootstrap；
3. 硬件 Color-ID 图像评价；
4. 下载头与启发式基线比较；
5. CUDA、资产大小和软件 parity；
6. 只有 Full 通过论文晋级门后，才实现并测量硬件 WebGPU；
7. 所有内容冻结后，最终模型读取一次 test。

## 十二、论文晋级条件

Full 三个 seed 必须全部满足：

- validation aggregate weighted recall `>0.99`；
- 同指标单侧 95% LCB `>0.99`；
- 候选、GT 和 pose 顺序完全一致；
- 图像平均与 p95 miss-pixel 没有稳定恶化；
- 相比修复后控制基线，precision 提升至少 0.02、balanced accuracy/useful cull 提升至少 0.01、GLB 字节下降至少 5% 或运行成本明确下降中至少一项的 95% 区间不跨零；
- 全部神经资产 `<=7 MiB`；
- 硬件 WebGPU 10k 候选 p95 不高于当前 Fourier117 路径 1.10 倍；
- 浏览器不运行在线关系图、邻居查询或 subpose。

单项创新只有在对应移除消融中产生稳定收益，才写成论文正贡献。若模块只有非零 gate 或改变输出，却没有改善安全、分类、图像、下载或运行成本，则降级为辅助表示或失败消融。

## 十三、输出目录

```text
neural_instance_culling/dataset/out/
  pvs_bounded_relation_survival_moment_envelope_safety_reserve_v3/

neural_instance_culling/model/out/
  pvs_bounded_relation_survival_moment_envelope_safety_reserve_v3_pilot12/
  pvs_bounded_relation_survival_moment_envelope_safety_reserve_v3_refine24/
  pvs_bounded_relation_survival_moment_envelope_safety_reserve_v3_formal80/

neural_instance_culling/benchmark/out/
  pvs_bounded_relation_survival_moment_envelope_safety_reserve_v3_formal_validation/
```

每个目录必须包含不可覆盖 manifest、输入路径与 schema、代码 commit、GPU、日志、checkpoint、阈值来源、候选口径和 `testRead`。smoke 产物完成后删除，不在根目录或默认 runner 中保留临时名称。

## 十四、完成定义

本计划只有在以下条件全部满足后才算完成：

- 当前 v2 formal80 已封存并完成正式诊断；
- P0 代码错误全部有回归测试；
- corrected relation/data artifacts 可复现；
- 快速验证与双 seed 复验完成；
- 15 个三 seed、80 epoch 正式成员全部完成；
- validation、paired bootstrap、图像、下载、CUDA、资产和 WebGPU 评价完成；
- 论文贡献与失败消融有清晰边界；
- 默认前端只在最终模型通过所有门后更新；
- 文档索引、当前模型说明、导出 schema 和部署说明同步更新。
