# PVS V5 泛化模型统一架构与实验规范

日期：2026-09-18  
状态：设计冻结前规范，不是性能报告  
方法工作名：GCOF-PVS V5

## 1. 设计结论

V5 不继承 V4 的逐项修补路线。模型只完成一条链路：共享网络把新场景的局部几何和纯几何
遮挡邻域编译为紧凑方向场，浏览器查询该场并输出一个实例分数。训练以“视觉安全约束下减少
冗余保留”为单一任务目标。

V5 采用两份根目录设计稿中的下列内容：

- 局部表面几何编码，不复用含场景归一化位置的旧 96D 特征；
- 纯几何方向关系图，不读取可见标签或 train-observed relation；
- 单层关系编译器、固定低阶方向基和解析 survival field；
- 对圆盘和定向盒都适用的 9 点区域查询；
- 召回约束下的 GT 归一化冗余目标。

V5 删除下列 V4 机制：

- scene ID、instance embedding 和可学习 per-instance residual；
- 多层 local/structural 图传播；
- learned direction basis、spectral moment 和 boundary summary；
- utility/download 独立任务头；
- recall guard、CVaR、tail mining、margin curriculum 和关系辅助分类；
- 依赖目标场景 train 分位数的深度归一化。

模型输出一个连续分数。实例 PVS 使用冻结阈值，GLB 调度对同一资源的候选实例取最大分数。
V5 不为下载排序再训练一套不相关模型。

## 2. “训练一次即可用于新场景”的严格含义

最终部署模型只训练一次。新场景允许执行确定性的几何预处理，包括局部表面采样、AABB
关系构图以及共享模型前向编译；不允许使用该场景可见标签更新网络权重或实例表。

把 HKUST、IFCBench、Sponza、Viking Village 和 Big City 全部加入训练，只能证明同一共享模型
覆盖五个已知场景，不能证明未见场景泛化。论文使用三套互不混淆的协议：

| 协议 | 模型训练数据 | 用途 |
|---|---|---|
| Shared in-domain | 五个真实场景 train + 合成训练集 | 架构选择、消融和已知场景效果 |
| 五折 LOSO | 合成训练集 + 四个真实 source scene | 评价第五个完整未见场景 |
| Universal final | 合成训练集 + 五个真实场景 train | 最终发布的一份通用权重 |

论文还应增加一个从未参与结构选择和 LOSO 训练的新公开场景作为 blind holdout。优先选择
Robot Lab 或 Industrial Set v3.0。Universal final 冻结后才转换该场景、生成评价 GT 并读取
结果。没有 blind holdout 时，论文只声明“五场景 LOSO 跨场景迁移”，不声明开放世界泛化。

## 3. 训练数据

### 3.1 五个现有真实场景

训练使用以下注册版本，每个场景只保留一个正式单位协议：

| 场景 | 单位来源 | 训练角色 |
|---|---|---|
| HKUST | 原生 BIM 构件单位 | 大型室外 BIM/校园 |
| IFCBench/Metropolis | 当前正式实例单位 | 大型合成 BIM 城市 |
| Sponza 64 KiB | Connected-SAH streaming units | 室内图形学场景 |
| Viking Village 64 KiB | Connected-SAH streaming units | 室外村落 |
| Big City 64 KiB | Connected-SAH streaming units | 高密度城市 |

Viking 128 KiB 只用于分割粒度对照，不作为第六个独立训练场景，避免同一几何重复进入训练并
抬高样本量。所有真实场景继续使用各自冻结的 train/calibration/validation/test 身份。训练器
不能按 pose 随机重划整个数据集。

每个 source scene 提供四类资产：

1. 真实变换后的局部表面点和法向；
2. AABB 与纯几何 proxy relation；
3. train pose 的 candidate、visible label 和原始 visible weight；
4. 只由 source train 几何生成的外部首次命中 probe。

LOSO held-out scene 只允许读取前两类资产来生成运行时表。其 calibration/test 标签只由对应
评价模式读取，probe 不得进入 source-only 训练。

### 3.2 合成训练集

五个真实场景不足以覆盖新场景分布。V5 增加 120 个程序化场景，按生成器 seed 划分，不能按
pose 划分：

```text
96 synthetic train scenes
12 synthetic validation scenes
12 synthetic diagnostic scenes
```

生成器固定五类结构，每类 24 个场景：

- 房间、门洞和长走廊；
- 多楼校园、庭院和连廊；
- 城市街谷、密集街区和高低建筑混合；
- 工业管线、设备、梁柱和多层平台；
- 大量重复构件与非重复网格混合的杂乱场景。

每个场景包含 256 至 4096 个 renderable units。生成器随机化尺度、密度、层数、通道宽度、
遮挡深度、单位大小分布和重复率，但不提供 scene ID 或语义类别给模型。几何来源使用程序化
primitive 和许可清楚、与 blind holdout 无关的 mesh bank。LOSO fold 不能从 held-out scene
抽取 mesh 或统计量扩充合成集。

合成场景使用与真实场景相同的 candidate、Color-ID、区域 GT 和 probe 生成器。view-cell 同时
覆盖圆盘和定向盒，尺寸范围在生成器配置中固定。单位粒度按 32/64/128 KiB 三档生成，用于学习
对 streaming granularity 的鲁棒性；正式真实场景仍按上一节的唯一版本评价。

### 3.3 场景平衡与增强

每个 optimizer step 只处理一个 scene。真实场景按固定 round-robin 轮转，两个真实 scene step
后插入一个 synthetic scene step；synthetic scene 在 96 个训练场景中均匀选择。这样大城市
不会因 candidate observation 更多而淹没小场景。

每个真实 source scene 接收 36,000 次更新。synthetic pool 的更新数固定为真实更新总数的一半：

```text
四 source LOSO fold: 4 x 36,000 real + 72,000 synthetic = 216,000 updates
Universal final:      5 x 36,000 real + 90,000 synthetic = 270,000 updates
```

每步从当前 scene 均匀采样 4 个 candidate 非空 pose，保留 GT=0 的纯负 pose，并使用全部候选；
同时采样 8192 个该 scene 的 field probe observation。实现固定先均匀抽取 512 个 probe unit，
再从每个 unit 的 576 条 ray 与 13 个距离档位中均匀抽取 16 个 observation。由于每个有效 unit
拥有相同数量的 ray，该分组抽样与全表逐 observation 均匀抽样具有相同边缘分布，同时将每步
因 field 监督引入的唯一几何 unit 上界固定为 512；它不减少监督数量，也不做 event balancing
或困难样本挖掘。主训练不使用 ambiguity hard sampling。

训练对整个 scene、相机和几何同步应用固定序列的全局 yaw 旋转，标签不变，用于减少世界朝向
记忆。等比例缩放和平移只做一致性测试，不作为扩大样本数量的手段。

## 4. V5 模型架构

### 4.1 共享局部几何编码器

每个 unit 按真实实例变换和实际世界三角形面积固定采样 256 个表面点。输入为以 AABB 中心和
半对角线归一化的局部 xyz、单位法向以及三个尺寸比例。网络为：

```text
point: 6 -> 32 -> 64 -> 64, SiLU
pool: max(64) || mean(64)
unit: 131 -> 64 -> 32, tanh
output: z_i in R^32
```

编码器不使用 BatchNorm、scene bounds、世界中心、实例编号或场景编号。服务器为新场景离线
生成 `z_i`，浏览器不运行点云编码器。

### 4.2 纯几何方向关系图

使用正二十面体 12 个顶点作为固定方向 anchor。对每个 target-anchor，根据 AABB 正交投影
重叠和前后深度区间选取最多 K=8 个 potential occluder。排序只使用投影覆盖和 target-relative
depth gap。

每条边为 8D scale-free feature：相对方向 3D、相对距离、半径比、target/source 投影覆盖比和
相对 depth gap。构图 schema 必须写入 `usesVisibilityLabels=false`；V5 Full 拒绝读取旧
train-observed relation。

### 4.3 单层遮挡场编译器

```text
[z_target(32), z_source(32), edge(8)]
    -> 72 -> 64 -> 32 edge message
    -> masked attention within target-anchor
    -> h_ik
```

聚合器同时保留 `log1p(neighbor_count)` 和覆盖和，避免 softmax 抹掉邻域数量。七参数 anchor
响应为：

```text
base:  [z_i, anchor] -> 35 -> 32 -> 7
delta: [h_ik, count, overlap_sum, anchor] -> 37 -> 32 -> 7
q_ik = base + has_neighbor * delta
```

12x7 响应通过固定一阶方向基 `[1,x,y,z]` 的伪逆投影为 `C_i in R^(4x7)`。投影矩阵是 buffer，
不是 Parameter。

### 4.4 解析外部命中场

`C_i` 表示面积均匀表面探针沿查询方向在给定目标中心深度内未命中其他 unit 的概率分布。距离采用：

$$t=\log(1+d/r_i).$$

七个参数产生无命中质量和两个截断 logistic survival 分量。实现必须满足 `S(0)=1`、固定方向
随距离不增，并在 PyTorch、WebGPU 和 WASM 中使用相同参数变换。该场是最终可见性的中间
统计，不宣称等于有限物体的真实可见概率。

field probe 只在 source train scene 生成。每个 unit 使用 16 个面积均匀表面起点、12 个 anchor
加 24 个固定 Fibonacci 方向。射线仍从表面出发以避免自身相交，但命中值存为
`max(0, (hit-target_center) dot direction)`，与运行时从目标中心到 support 的方向深度完全同坐标。
新场景部署不生成 probe。

### 4.5 区域查询

圆盘使用中心和 8 个等角圆周点；定向盒使用中心和 8 个角点。支持点来自数据 manifest，不能
把 IFCBench 的盒状区域改称圆盘。九次查询共享同一个 `C_i`，压缩为：

```text
S_center, S_max, S_mean, S_min
```

最终 query geometry 严格按以下顺序构造 16D，训练、PyTorch 导出、WebGPU 和 WASM 不得各自
重新定义：

| 维度 | 内容 |
|---|---|
| 0–2 | unit 中心指向区域中心的世界方向单位向量 |
| 3–5 | 区域中心指向 unit 的方向在相机 right/up/forward 基中的分量 |
| 6 | `log(1 + d/r_i)` |
| 7 | `r_i/(d+r_i)` |
| 8–10 | 三个区域半轴长度除以 `d+r_i+max(half_axes)`；圆盘为 `(R,0,R)` |
| 11–12 | `tan(FOV_x/2), tan(FOV_y/2)` |
| 13 | 区域类型：圆盘 0、定向盒 1 |
| 14 | `near/(d+r_i)` |
| 15 | `log(1+far/(d+r_i))` |

`near/far` 来自真实采样或运行相机配置；历史 CSR 未逐 pose 保存 clip plane 时，场景注册表必须
显式登记该批 GT 的 Color-ID 相机默认值，不能在训练器中静默填另一个常数。最终头为：

```text
z_i(32) + field_stats(4) + query_geometry(16)
    -> 52 -> 32 -> 1 visibility logit
```

浏览器运行时没有图传播。每个 unit 下发 32D FP16 `z_i`、4x7 FP16 `C_i`、AABB 和 resource ID，
理论主体约 148 B/unit；正式数字从导出文件实测。

## 5. 统一损失

### 5.1 优化问题

V5 只包含一个任务优化问题和一个表示监督。对 source scene `s`，用 source train 统计：

- `G_s`：可见 unit occurrence 总数；
- `W_s`：原始 visible weight 总和。

定义 `h_keep(l)=softplus(l)/ln(2)`、`h_miss(l)=softplus(-l)/ln(2)`：

$$J_{extra,s}=\frac{1}{G_s}\sum_{y=0}h_{keep}(\ell),$$

$$R_{count,s}=\frac{1}{G_s}\sum_{y=1}h_{miss}(\ell),$$

$$R_{visual,s}=\frac{1}{W_s}\sum_{y=1}w\,h_{miss}(\ell).$$

训练目标为：

$$\min_\theta\frac{1}{|S|}\sum_s[J_{extra,s}+0.25L_{field,s}]$$

约束：

$$R_{count,s}\le0.02,\qquad R_{visual,s}\le0.01.$$

视觉约束对应当前论文的主要安全目标；较松的普通实例约束只防止模型通过集中漏掉大量小单元
换取漂亮的 weighted recall。它不替代 weighted recall，也不升级为新的校准硬门。

五个真实 source scene 各自维护两个非负拉格朗日乘子；96 个 synthetic train scene 不再各自
维护稀疏更新的乘子，而是按五个固定结构 family 共享五组乘子。模型参数下降、乘子按所在约束组
的预算违例投影上升。这样每个 synthetic family 可累计约 18k 次 dual 更新，而不是每个合成场景
只有约 938 次。乘子只读取 source train 风险，不进入网络、不导出，也不由 held-out scene 估计。

### 5.2 唯一的表示监督

`L_field` 是外部首次命中 observation 的 event/right-censor NLL。系数固定为 0.25。它定义
中间场的物理含义，不承担最终 PVS safety。只有附录敏感性允许比较 0.125/0.5，不根据单场景
validation CNOR 选择权重。

### 5.3 明确删除的损失项

V5 Full 没有下列项：

```text
pose-balanced BCE
recall guard
pose CVaR
hard positive/negative tail
pairwise margin
relation classification/depth ranking
per-instance calibration regularization
loss warmup/ramp curriculum
```

任务层的全部代码应能写成：

```python
loss = (risk.extra
        + dual_count * risk.miss_count
        + dual_visual * risk.miss_visual
        + 0.25 * field_nll)
```

### 5.4 校准

当前论文主工作点保持不变：在 calibration 上选择满足 weighted recall `>0.99` 且单侧 95%
bootstrap LCB `>0.99` 的最高分数变化点。没有 LCB 合格点但平均 weighted recall `>0.99` 的
成员保留为 mean-target 结果，不伪装为严格安全成员。

普通实例 recall、FN/GT 和 Bad Cull 必须报告，但不作为新的 99% 硬门。这样训练中的 count
constraint 负责避免结构性小实例崩溃，校准仍只承担论文既定的视觉安全职责。

LOSO 同时报告：

- `source_global`：阈值只来自各 source calibration，目标场景零标签；
- `target_calibrated`：权重冻结，只用目标 calibration 选一个标量阈值。

## 6. 训练实现

优化器使用 AdamW，初始学习率 `2e-4`、weight decay `1e-5`，按 optimizer step cosine 降到 0，
全局梯度范数裁剪为 5。训练固定走到最后一次更新，不按 validation 挑 epoch。

### 6.1 正式长训前参数扫描

正文训练前只扫描优化动力学，不扫描模型容量、关系 K、方向基阶数、支持点数或 loss 组成，避免
把架构搜索伪装成消融。扫描固定使用 FULL、一个开发 seed、五个真实场景 train 和全部 96 个
synthetic train scene；calibration 只选阈值，validation 只选参数，test 始终不可读。

第一阶段执行 6 个 `12,000 update` 的从头 pilot：

```text
model learning rate: 1e-4, 2e-4, 4e-4
dual learning rate:  1e-3, 3e-3
weight decay:        1e-5 fixed
field coefficient:  0.25 fixed
```

所有成员使用相同 scene/pose/synthetic RNG 序列、余弦调度、零初始化非负乘子和相同 bootstrap
设置。快速扫描始终保留相对最优的两个配置，不能因没有成员达到严格安全门而取消后续确认。
两个候选各自从头执行 `36,000 update` 单 seed 确认；最终参数按以下词典序冻结：

1. validation 上达到 strict LCB 工作点的场景数；
2. validation 上达到 mean-target 工作点的场景数；
3. 五场景等权的 weighted recall LCB；
4. 在安全层级相同的前提下，五场景等权 CNOR、Useful Cull；
5. 五场景等权 predicted/GT，越低越优。

pilot 还必须输出训练动力学图，不作为提前取消既定矩阵的门控：横轴分别使用 real scene-local
update 和 synthetic family-local update，纵轴绘制 `J_extra`、`R_count`、`R_visual`、
`lambda_count`、`lambda_visual`。重点检查前 2k–5k update 内是否出现 `J_extra` 快速下降、
两个 miss risk 急升而 dual 长期接近零。若所有学习率组合都出现该模式，再登记并比较 dual 初值
或短暂约束预热；在观察到轨迹前不向正式目标加入 warmup、bias 修正或额外 loss。

正式 preflight 逐场景报告零 logit 的解析初值：`J_extra=N_neg/N_pos`、`R_count=1`、
`R_visual=1`，并同时记录 train candidate/visible occurrence。该结果用于解释不同 prevalence
场景的初始梯度尺度，但不替代 pilot 的实际参数轨迹。

若参数扫描结果不好，可以在上述两个学习率轴的相邻数量级内追加一次最多 4 个成员的局部扫描，
但不得改变损失项、数据权限或评价顺序。参数冻结后，正文 15 个消融和 45 个 LOSO 模型全部从头
训练，不从 pilot checkpoint 微调。后续允许的“微调”仅指对同一冻结架构和数据协议调整已登记的
优化器参数并重新从头训练；不得按单个目标场景标签更新 universal/LOSO 权重。

几何编码、关系编译、field NLL 和最终 PVS loss 端到端更新。大场景使用精确梯度重计算：先对
当前 step 涉及的唯一 unit 生成无图 `z_cache`，下游分块累计 `z_leaf.grad`，再按几何 chunk
重算编码器并回传。减小 chunk 只能改变显存，不得改变 pose batch、候选、损失分母或更新数。

checkpoint 保存共享模型、优化器、scheduler、五个真实场景与五个 synthetic family 的乘子、
场景到 dual group 的固定映射、数据流 RNG 和每个 source 的更新次数。训练与恢复都拒绝未知
schema、旧的逐 synthetic-scene dual 状态、旧 V4 loss 字段和 target label 权限错误。

## 7. 精简消融

正文只保留 Full 加四个因果对照。消融不按实现模块逐项删除，而是分别检验论文的四条核心
主张：场景关系、结构化表示、外部命中监督和约束目标。所有变体在 Universal shared in-domain 协议上训练
三种子，使用相同五场景/合成数据序列、更新数、量化和校准协议。

| ID | 受控表示或目标 | 论文问题 |
|---|---|---|
| FULL | 完整 V5 | 最终方法 |
| GEOMETRY_FIELD | 共享 MLP 只从 `z_i` 生成相同 4x7 场；无邻域输入，其他均同 Full | 新场景中目标自身几何是否足以替代场景遮挡关系 |
| GENERIC_RELATION_28 | 使用相同 proxy graph 和关系编译器，但输出同容量 28D 通用 latent；无解析 field/probe NLL | 收益来自关系信息和容量，还是具有方向/距离语义的结构化场 |
| FULL_NO_FIELD_NLL | 架构、4x7 场、解析查询和约束任务目标均与 Full 相同，但不读取 external-hit probe，不加入场 NLL | Full 的收益来自结构化场本身，还是仅来自额外射线监督 |
| PBCE_OBJECTIVE | 架构、field probe 和数据完全同 Full，只把约束任务目标换成 pose-balanced BCE | 直接优化安全约束下冗余是否优于普通分类训练 |

`GENERIC_RELATION_28` 的运行时 latent 与 Full 的 4x7 场同为 28 个 FP16 数。它把 12 个 anchor
token 的完整 `12x7=84D` 有序证据一次性线性压成 28D，并将 `z_i + latent + query16` 送入查询头。
它不再先逐 anchor 做 `7->28` 后求均值，因此不会把方向结构和有效信息秩人为压到 7 以下。
该对照的共享网络参数量允许略大于 Full，使结论对 Full 更保守；双方运行时逐 unit 表仍同为 28 个数。

`FULL_NO_FIELD_NLL` 与 Full 使用同一个模型类和运行时资产结构，唯一差别是训练时不实例化 probe
读取器且场损失严格为零。因而 `FULL_NO_FIELD_NLL` 对 Full 只测量 external-hit 监督价值；
`GENERIC_RELATION_28` 对 `FULL_NO_FIELD_NLL` 才用于判断结构化方向场归纳偏置本身的价值。

Full 结果在所有表中复用。旧 V4、Keep-All、AABB-query MLP 和 HZB 属于 baseline，不混进
消融矩阵。Free per-instance field、residual、SH 阶数、K 和支持点数量都不进入正文核心消融。
若 reviewer 或实现诊断需要，SH0/SH2 只作为附录单种子资产敏感性，不参与方法选择。

训练规模：正文消融 5 配置 x 3 seeds，共 15 个共享模型。

## 8. 泛化实验

五折 LOSO 只训练三种模型：

```text
GEOMETRY_FIELD
GENERIC_RELATION_28
FULL
```

每种模型执行 5 folds x 3 seeds，共 45 个训练。每个 fold 的 held-out scene 完全不参与参数、
对偶乘子、source normalizer、synthetic mesh bank 或阈值训练。每个模型同时输出 source_global
和 target_calibrated 两个结果，不重复训练。

Universal final 直接复用正文消融中的 FULL 三种子。架构和 seed 选择规则冻结后，导出一份
主发布权重，并在 blind holdout 上只执行几何编译和推理。

V5 核心训练总数为：

```text
15 main ablation models
45 LOSO models
= 60 models
```

这 60 个模型覆盖四个核心方法主张与跨场景泛化；不执行两份草案中 156 个模型的完整矩阵。

## 9. 论文结果表

### 9.1 Shared in-domain

五场景分别报告三种子 mean ± sample std：

- weighted recall 与 LCB；
- ordinary recall、FN/GT 和 Bad Cull；
- CNOR、Useful Cull、FP/GT 和 predicted/GT；
- pose PR-AUC、正样本比例和 AP lift；
- GLB bytes reduction 与图像 miss/PER。

### 9.2 LOSO

每个 held-out scene 报 source_global 和 target_calibrated。跨场景均值按 scene 等权，不能混池
所有 observation。source_global 未达到安全目标时保留原阈值和失败结果，不能用目标标签补救。

### 9.3 系统

报告实际 runtime asset bytes、bytes/unit、server compile time、WebGPU/WASM p50/p95、候选规模
拟合以及 streaming Bytes@95/99。HZB 使用相同区域协议，资产驻留成本单独列出。

## 10. 执行顺序

1. 冻结当前 V4 和正在完成的 V4 pilot，不再增加 V4 损失配置。
2. 构建 V5 local-surface、proxy graph 和 external-hit probe schema，并完成权限测试。
3. 实现共享模型、约束风险、对偶更新、梯度重计算和 CPU 数值测试。
4. 用五个真实场景的小子集完成一个 shared seed smoke，验证 loss、显存和跨场景 batch。
5. 按 6.1 节完成 FULL 参数扫描和双候选确认，冻结优化器参数。
6. 跑 FULL、GEOMETRY_FIELD、GENERIC_RELATION_28、FULL_NO_FIELD_NLL、PBCE_OBJECTIVE 的单 seed 开发训练；只读
   calibration/validation。
7. 若 FULL 没有相对三个对照改善平均安全—紧致前沿，先检查 proxy graph recall 和 field NLL，
   不增加新的 loss 项。
8. 架构冻结后完成 15 个正文消融模型。
9. 完成 45 个 LOSO 模型，再训练/复用 Universal final。
10. 最后读取 frozen test、blind holdout，并执行图像、HZB、runtime 和 streaming 实验。

## 11. 采用条件与论文主张

V5 成为论文主线至少需要满足：

- Shared Full 在五场景平均安全—紧致前沿上不弱于 scene-specific V4；
- FULL 稳定优于 GEOMETRY_FIELD，证明纯几何关系提供可迁移场景上下文；
- FULL 稳定优于同容量 GENERIC_RELATION_28，证明结构化方向场不只是 28D 通用 latent；
- FULL 稳定优于 FULL_NO_FIELD_NLL，证明 external-hit 监督确实改善了结构化场，而不是只增加训练成本；
- FULL 在同一架构下优于 PBCE_OBJECTIVE，证明约束目标改善安全—紧致折中；
- LOSO FULL 在多数 held-out scene 上优于 GEOMETRY_FIELD 和 GENERIC_RELATION_28；
- target-calibrated 不更新权重即可恢复合格工作点；
- blind holdout 不依赖场景微调即可产生非零有效剔除。

论文应使用“cross-scene transfer over heterogeneous scenes”或“geometry-compiled shared model”。
除非 blind holdout 和更多外部场景支持，不使用“universal visibility model”。

方法主张收敛为：服务器用共享、场景无关的网络把纯几何潜在遮挡邻域编译成紧凑方向场；浏览器
在渲染几何到达前查询该场；训练在视觉安全与基本实例覆盖约束下直接减少冗余 PVS。局部几何
编码、attention、低阶方向基和拉格朗日更新本身都不是独立创新点。

## 12. 代码边界

新实现使用独立 V5 namespace：

```text
neural_instance_culling/model/v5/
neural_instance_culling/dataset/v5/
neural_instance_culling/benchmark/v5/
```

V5 入口拒绝 V4 的 scene-normalized 96D、train-observed relation、instance residual 和旧 loss
schedule。开发阶段 V4 保留为冻结 baseline；V5 通过采用条件后，前端默认资产一次性切换到 V5，
不增加双默认路径或兼容开关。

设计来源：根目录《面向泛化的紧凑遮挡场PVS_新架构与实验设计.md》和
《GCOF-PVS_V6.1_召回约束版完整规范.md》。本规范对两份草案作出训练数据、约束强度、消融
规模和泛化评价方面的最终取舍。

## 13. 真实 external-hit probe 生成器（2026-09-18）

本次实现把 7.3 的场监督从旧 JSONL 记录落实为真实三角形射线资产。生成器读取已有
`[unit,256,6]` 表面点二进制，每个 unit 固定取前 16 个点；方向固定为 12 个正二十面体方向
加 24 个 Fibonacci 方向。GLB 经现有 Three.js loader 解码后按 renderable object/instance
逐批匹配到 unit，结果立即追加到紧凑的 `Float32Array` 世界位置块和 `Uint32Array` unit 归属块；
不再把全场三角形保留为 JS `{a,b,c,...}` 对象数组。全部 GLB 完成后只做一次 typed-array
扁平化并建立合并的 `BufferGeometry`/`three-mesh-bvh`，射线索引只保留位置、BVH 和平行的
unit ID 数组。每个 GLB 处理完成后会释放 geometry、material、texture 和 GLTFParser 缓存，
可选地在 `--expose-gc` 下按间隔触发 V8 回收。

BVH 仍查询最近外部真实三角形；射线起点沿方向偏移 `1e-5*r`，但存储深度统一改为
`max(0, (hit-target_center) dot direction)`。当前 unit 的全部三角形都会被跳过，目标中心坐标超过 `1024*r` 的结果用
`hitDistances=NaN` 表示右删失。正式 Color-ID 协议的 `DoubleSide` 规则用于 BVH 射线查询，
源材质的透明度字段不会被误当成 Color-ID 的可见性规则。

修改文件：

- `neural_instance_culling/dataset/v5/generate_external_hit_probes.mjs`
- `neural_instance_culling/dataset/v5/test_generate_external_hit_probes.mjs`
- `neural_instance_culling/dataset/v5/schemas.py`（只接受中心深度列式 v2 manifest）
- `neural_instance_culling/package.json`、`package-lock.json`（加入 `three-mesh-bvh`）

输出目录固定暴露 `unitIds`、`directions`、`hitDistances`、`maxDistances`、`startIds`、
`directionIds` 六个小端无头二进制列，行序为 `[unit][directionId][startId]`，每个 unit 576 行。
manifest 使用 `parallel_external_hit_target_depth_current_status-v2`、source-train 权限字段、13 个距离比和
列 dtype/shape；训练器均匀抽取 ray 后即时展开距离事件，不生成每条 ray 的 13 份 JSON 记录。
生成器支持无哈希进度恢复、连续分片和按 unit ID 合并。

验证命令：

```bash
node neural_instance_culling/dataset/v5/test_generate_external_hit_probes.mjs
node --check neural_instance_culling/dataset/v5/generate_external_hit_probes.mjs
python -m py_compile neural_instance_culling/dataset/v5/schemas.py
```

测试覆盖两个真实 GLB 盒子、反向面 DoubleSide 命中、当前 unit 全部跳过、1024 倍尺度右删失、
不同表面起点命中同一点时中心深度一致、六列 shape/dtype、逆序分片合并和完整输出恢复；另有 10,000 三角形多分块规模 smoke，验证
三角形对象保留数为 0、位置/归属均为 typed arrays，且 BVH 索引不含 `triangles` 对象数组。
画面安全指标、剔除效率指标和前端延迟尚未在本生成器任务中实现，需由后续训练与 benchmark
读取列式资产后报告。

### 13.1 大场景内存修复与重跑约定（2026-09-18）

Viking Village 的旧实现是在全场解码后同时保留每个三角形的多个 JS 数组、对象和材质引用，
随后再复制成 BVH 位置数组，导致三角形对象数随全场规模线性膨胀并触及 V8 4 GiB 堆上限。
当前实现的长期存活结构只有：合并 `BufferGeometry.position`（每三角形 9 个 `float32`）、
`triangleUnitIds`（每三角形 1 个 `uint32`）以及 `three-mesh-bvh` 的加速结构。逐 GLB 的匹配
数组只在回调期间存在，GLB 场景和 parser 缓存随后立即释放。`--gc-every` 只是回收辅助，不是
正确性的前提，也不是通过增大 `--max-old-space-size` 绕过问题。

重跑时继续使用原有连续 shard、progress sidecar、resume 和 unit-ID merge 语义。建议先用
`--expose-gc`、较小的 shard 和明确日志运行：

首次替换旧的完整输出时在下面命令中加入 `--no-resume`；若任务中断，后续重跑时去掉
`--no-resume` 即按 progress sidecar 继续。

```bash
V5_ROOT=neural_instance_culling/dataset/out/pvs_v5_geometry_compilation_v1
VIKING_ROOT="$V5_ROOT/viking_village_64k"

for shard in $(seq 0 7); do
  node --expose-gc neural_instance_culling/dataset/v5/generate_external_hit_probes.mjs \
    --scene-id viking_village_64k \
    --assets-dir neural_instance_culling/dataset/out/standard_graphics_connected_sah_64k_v1/viking_village_64k/assets \
    --runtime-meta neural_instance_culling/dataset/out/standard_graphics_connected_sah_64k_v1/viking_village_64k/assets/runtimeVisibilityMeta.json \
    --glb-index neural_instance_culling/dataset/out/standard_graphics_connected_sah_64k_v1/viking_village_64k/assets/glbIndex.json \
    --surface-manifest "$VIKING_ROOT/surface/surface_manifest.json" \
    --output-dir "$VIKING_ROOT/probe_shards/shard_${shard}" \
    --shard-index "$shard" --shard-count 8 \
    --progress-every 25 --gc-every 25 \
    > "$VIKING_ROOT/logs/probes_shard_${shard}_stdout.log" \
    2> "$VIKING_ROOT/logs/probes_shard_${shard}_stderr.log"
done
```

Viking 分片完成后合并：

```bash
node --input-type=module -e '
import { mergeExternalHitProbeShards } from "./neural_instance_culling/dataset/v5/generate_external_hit_probes.mjs";
mergeExternalHitProbeShards({
  shardDirs: Array.from({ length: 8 }, (_, i) => `neural_instance_culling/dataset/out/pvs_v5_geometry_compilation_v1/viking_village_64k/probe_shards/shard_${i}`),
  outputDir: "neural_instance_culling/dataset/out/pvs_v5_geometry_compilation_v1/viking_village_64k/probes",
});
'
```

Big City 使用相同入口；其规模较小可使用 4 个 shard：

```bash
CITY_ROOT="$V5_ROOT/big_city_64k"
for shard in $(seq 0 3); do
  node --expose-gc neural_instance_culling/dataset/v5/generate_external_hit_probes.mjs \
    --scene-id big_city_64k \
    --assets-dir neural_instance_culling/dataset/out/standard_graphics_connected_sah_64k_v1/bigcity_64k/assets \
    --runtime-meta neural_instance_culling/dataset/out/standard_graphics_connected_sah_64k_v1/bigcity_64k/assets/runtimeVisibilityMeta.json \
    --glb-index neural_instance_culling/dataset/out/standard_graphics_connected_sah_64k_v1/bigcity_64k/assets/glbIndex.json \
    --surface-manifest "$CITY_ROOT/surface/surface_manifest.json" \
    --output-dir "$CITY_ROOT/probe_shards/shard_${shard}" \
    --shard-index "$shard" --shard-count 4 \
    --progress-every 25 --gc-every 25 \
    > "$CITY_ROOT/logs/probes_shard_${shard}_stdout.log" \
    2> "$CITY_ROOT/logs/probes_shard_${shard}_stderr.log"
done
```

Big City 分片完成后合并：

```bash
node --input-type=module -e '
import { mergeExternalHitProbeShards } from "./neural_instance_culling/dataset/v5/generate_external_hit_probes.mjs";
mergeExternalHitProbeShards({
  shardDirs: Array.from({ length: 4 }, (_, i) => `neural_instance_culling/dataset/out/pvs_v5_geometry_compilation_v1/big_city_64k/probe_shards/shard_${i}`),
  outputDir: "neural_instance_culling/dataset/out/pvs_v5_geometry_compilation_v1/big_city_64k/probes",
});
'
```

分片完成后仍通过现有 `mergeExternalHitProbeShards` 按 unit ID 合并，不改变列式 manifest。
本次代码验证命令为：

```bash
node --check neural_instance_culling/dataset/v5/generate_external_hit_probes.mjs
node neural_instance_culling/dataset/v5/test_generate_external_hit_probes.mjs
git diff --check -- neural_instance_culling/dataset/v5/generate_external_hit_probes.mjs \
  neural_instance_culling/dataset/v5/test_generate_external_hit_probes.mjs
```

## 14. 正式训练前语义审查修正（2026-09-18）

本轮静态审查在启动参数扫描前冻结了以下契约，旧错误资产不保留兼容入口：

1. 12 个 anchor 只从 `dataset/v5/directions.json` 读取。relation manifest 显式保存有序
   `anchorDirections`，数据加载器和模型编译器都逐项核对，Node probe 生成器读取同一资源。
2. external-hit probe 升级为 target-centered v2。旧 surface-origin v1 manifest 会直接失败；
   五个真实场景和 96 个 synthetic train scene 的 probe 必须重建，PoseCSR/Color-ID GT 不重采样。
3. `GENERIC_RELATION_28` 读取完整 84D anchor evidence 后压成 28D，不再先求方向均值。
4. shared 核心消融新增 `FULL_NO_FIELD_NLL`，正式矩阵为 5 变体 x 3 seeds；LOSO 仍为原三变体。
5. 场景风险的逆采样因子改用 candidate 非空的 `eligiblePoseCount`；preflight 同时报告总 train pose、
   eligible pose 和 candidate-empty pose 数。
6. train 标签若包含退化 unit 立即失败；calibration/validation/test 仍只在对应 score sidecar 被授权
   读取时检查，禁止为预检提前读取 test 标签。

合成场景使用 `--rebuild-compiled` 只重建 deterministic surface/relation/probe 和编译 manifest，
保留现有 PoseCSR、Color-ID GT、runtime metadata 与 split。正式恢复生成前先运行：

```bash
conda run --no-capture-output -n slm_pvs \
  python -m neural_instance_culling.dataset.v5.generate_synthetic_datasets \
  --all --rebuild-compiled --device cuda \
  > synthetic_rebuild_stdout.log 2> synthetic_rebuild_stderr.log
```

当前 Python compiled artifact 仍是 FP32，V5 WebGPU/WASM query kernel 尚未实现。这两项属于正式
runtime 实验前的系统任务：完成 FP16 导出、PyTorch FP32 对 FP16 安全重放和浏览器 parity 后，
才能报告 148 B/unit 或 WebGPU/WASM 硬件延迟；此前只报告 Python FP32 实际字节和耗时。
compiled manifest 固定写 `pythonInferenceReady=true`、`browserRuntimeReady=false`，不再使用含义
不明确的 `runtimeReady`。
