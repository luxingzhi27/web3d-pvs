# GCOF-PVS V6.1：面向泛化与 PVS 紧致度的完整重构规范

**版本：2026-09-18 / `gcof-v61-recall-constrained-v1`**  
**适用项目：`luxingzhi27/web3d-pvs`**  
**定位：架构、训练目标、消融配置、数据接口与复现规范；不是已验证的性能报告。**

本文以用户提供的《面向泛化的紧凑遮挡场 PVS：新模型架构、损失函数与消融实验完整设计》（2026-09-16）为基础，保留“几何关系 → 方向遮挡场 → 少量区域解析查询”的组织方式，将 FP/GTP 目标与“不能靠漏检换剔除”的要求一并纳入正式约束优化。V6.1 取代 V6 固定权重的覆盖—冗余主目标；V6 目标仍完整保留为 L_FIXED_GT 对照。本文的实验矩阵、训练预算与汇总规则事先固定，不包含根据阶段结果决定是否继续的试验路线。[S0]

## 1. 方法总览与本版的明确修订

### 1.1 一句话方法

**服务器使用共享网络，把不依赖可见性标签的几何邻域编译成每个资源单位的紧凑方向遮挡场；浏览器只查询该场和一个小分类头，在渲染几何到达前预测区域潜在可见集合。训练在普通实例召回与视觉加权召回的预算约束下，最小化相对于真实 PVS 规模的冗余风险。**

整体链路：

```text
服务器 / 离线：
实例真实变换后的网格 → 局部表面点与法向 → 共享几何编码器 → 32D z_i
实例 AABB → 12 方向纯几何潜在关系 → 单层关系编译器 → 12×7 响应
12×7 响应 → 固定方向基投影 → 4×7 遮挡场 C_i

下发：z_i、C_i、AABB、资源映射、共享小网络权重

浏览器 / 在线：
查询区域 → 9 个支持位置 → 同一 C_i 的解析查询 → 4 个无遮挡统计量
32D z_i + 4D 场统计 + 16D 相机/区域几何 → 52→32→1 → 可见分数
冻结阈值 → 实例保留集合 → GLB 映射与既有下载调度
```

### 1.2 相对于上传方案，哪些保持、哪些修改

| 项目 | 上传方案 | 本版正式定义与原因 |
|---|---|---|
| 主链路 | 几何关系编译为方向场 | 保持；不切换成另一个 Feature-HZB 项目 |
| 几何输入 | 既有固定 96D → 32D | 改为局部表面点/法向 → 共享 32D；旧 96D 输入实际含场景归一化位置，不能默认无位置记忆 [S1] |
| 方向锚点 | 旧 12 个方向 | 改为正二十面体的 12 个顶点；固定基投影和二阶消融都要求矩阵满秩 |
| 关系编译 | 一层聚合、base+delta | 保持；明确空邻域掩码、方向条件与邻域数量信息 |
| 方向场 | 一阶基、4×7 | 保持为主配置；0/1/2 阶分别独立报告，禁止看结果后悄悄换主配置 |
| 深度变量 | d/(d+r) | 改为 log(1+d/r)，避免远距离全部挤近 1；不使用目标场景分位数 |
| 生存语义 | 容易被理解成物体必然可见 | 明确定义为“平行探针首次外部命中”的无遮挡概率；不是有限物体可见性的精确公式 |
| 区域查询 | 圆盘 9 点 | 保持 9 次解析查询，但读取真实区域类型；不能把 IFCBench 的盒状采样协议改称圆盘 [S2] |
| 分类头 | 39→32→1 | 改为 52→32→1，补足观察朝向、视场角、裁剪距离；否则候选内的部分出视野无法区分 |
| 可见性损失 | 逐 pose、逐类别平衡 BCE | 改为召回预算约束下的 GT 归一化冗余风险；旧 PBCE 与固定覆盖—冗余目标都保留为对照 |
| 冗余目标 | 没有显式 FP/GTP 项 | GT 归一化负例对数风险为被最小化目标；普通及视觉加权漏检风险分别为约束 |
| 安全语义 | 主要保护 weighted recall | 同时保护普通实例与原始视觉权重；主表使用双召回工作点，旧 weighted-only 结果单独比较 |
| 场监督权重 | 默认 1:1 | 主配置固定 0.25；同为 NLL 不意味着梯度量级相等。0.125/0.5 为预定敏感性 |
| 结果选择 | 筛选后保留约十组 | 改为固定完整矩阵与固定训练终点；所有种子保留 |

这些修改是本版设计，而不是上传文档已经包含的结论。尤其不能将“96D 改成 32D”“增加 FP/GTP 项”直接声称为已经证明的创新或提升。

## 2. 任务、适用范围与数据契约

### 2.1 任务符号

场景 s 由独立可渲染单位组成。对查询区域 p，记候选集合为 C_p，真实区域可见集合为 V_p，不可见候选为 O_p=C_p\V_p，候选数为 n_p，真实可见数为 g_p。模型输出 logit ℓ_pi 与分数 q_pi=σ(ℓ_pi)。可见为正类。

真实标签仍由注册的 Color-ID/深度渲染协议产生：单位在注册的任一观察样本中满足可见判据则为正。输入关系图不能读取这些标签。**用于监督与评价的可见集合、用于编译器输入的几何关系、用于场监督的外部命中探针是三类不同资产。**

### 2.2 本次不混改的内容

HKUST 保留原有 5926/659/730/684 个 train/calibration/validation/test pose 的身份、单位划分与候选协议；其他场景从各自注册 manifest 读取实际划分。不得把旧 128 KiB 划分、Connected-SAH 新划分和各版本的候选统计混合。实际引用的仓库快照为 `b21aff24915fdf8f0525b759b622fab6675e47ee`。[S2,S3]

训练必须遍历候选非空的 pose，包括 g_p=0 的纯负 pose。候选为空的 pose没有网络样本，但保留在评价统计中。不得用“候选 ∪ GT”掩盖候选筛选遗漏。GT 不在 C_p 内时，应记录候选遗漏 FN；不能把这部分算成网络可修复错误，也不能事后补进候选。

HKUST 的既有 test 已经被查看并参与了研究讨论，因此应标明为既有测试集的回顾性复现。改变结构后重新运行同一测试集不是“此前从未接触的盲测”。新增独立空间 holdout 时另建版本，并对所有比较方法使用同一组新 pose。

### 2.3 区域真值不能被支持点重新定义

9 个支持点是网络的查询近似，不是 GT 的定义。原数据若用 4、32 或其他数量的观察样本构造并集，必须记录并保留。IFCBench 当前记录是四个 `camera_aligned_box` 样本的并集，不能只换一个字段就称为整个连续圆盘的可见并集。[S2]

正式表逐场景列出：区域形状、尺寸/半轴、GT 样本数、相机朝向约束、候选视场角、真实渲染视场角。主任务继承同一 pose 的固定相机朝向，不隐含“区域内任意转头”的全向可见性。

适用内容以静态、可获得三角形表面的 Mesh/BIM/GLB 为主。Alpha mask 必须按现有材质规则采样或明确单独处理。透明体、3DGS 和无表面点云不在本版“无需改动即可支持”的主张中。

## 3. 泛化契约：哪些数据允许进入模型

Full 不包含实例编号嵌入、场景编号、自由实例系数或目标场景微调参数。实例索引仅用于张量寻址，不能输入网络。服务器可以读取新场景的全部几何以构图和编码，这叫**新场景几何预处理**，不叫“新场景零预处理”。

共享参数包括几何编码器、关系消息网络、聚合器、场响应头、最终可见性头；它们在一个跨场景 fold 中只训练一份。目标场景几何和 AABB 可参与自身资产生成，但目标场景的可见标签、探针命中监督、阈值统计不能参与 source-only 模型训练。

Full 输入不允许使用 scene bounds 归一化的世界坐标、目标场景深度分位数、可见频次或 train-observed 关系置信度。训练期的 GT/视觉权重平均数以及两个对偶乘子属于源训练集优化元数据，不传入网络、不导出、不从目标场景估计。每个 source scene 有独立优化约束，但共享同一组模型参数；这不是目标场景实例记忆。

## 4. 几何描述：从局部表面直接得到 32D

### 4.1 几何采样与真实变换

对每个 unit 使用其真实实例矩阵变换顶点；法向使用逆转置并归一化。不能根据 AABB 中心和统一尺度“恢复”旋转后的实例几何。对非均匀缩放后表面按实际世界三角形面积采样，固定采样 256 点。

记 AABB 中心 c_i、半对角线 r_i>0；局部坐标为 (x-c_i)/r_i。每点输入为局部 xyz 与单位法向，共 6D。附加三个尺寸比例 size_i/max(size_i)，不附加中心和场景大小。退化零面积几何单独记录，不向编码器填造假的点云。

采样由固定 geometry hash 和单独的采样种子确定；不随训练种子改变。该 hash 不作为特征。这样种子比较反映优化差异，而不是偷偷换了输入点云。

### 4.2 编码网络

```text
逐点：6 → 32 → 64 → 64，层间 SiLU
聚合：max(64) 与 mean(64) 拼接为 128D
拼接：128D + 3D 尺寸比例 = 131D
输出：131 → 64 → 32，最后 tanh
```

不用 BatchNorm、Dropout、场景归一化层或实例表。这样可以跨场景共享，且几何编码器能使用后文的精确梯度重计算。服务器离线导出 z_i，客户端不执行点云编码。

旧 96D 资产仍用于冻结 V4 基线，不能把它简单降维后称为本版的“局部纯几何编码”。旧 exporter 中明确拼入了 `world_norm`、`center_norm` 和 `size_norm`。[S1]

## 5. 纯几何关系图

### 5.1 锚点与坐标

令 φ=(1+√5)/2。12 个单位锚点为下列向量的归一化结果，顺序由代码固定：

```text
(0, ±1, ±φ), (±1, ±φ, 0), (±φ, 0, ±1)
```

锚点 a_k 表示从目标朝潜在观察者的方向。选择与 a_k 最不平行的坐标轴 e，设 u=normalize(e×a_k)，v=a_k×u。与旧的 12 个水平/斜向锚点不共用编号；探针方向、投影矩阵和导出 schema 全部更新。

### 5.2 候选边判定

把 unit 的 AABB 投影到 u、v、a_k，得到二维矩形与深度区间。j→i 进入候选关系，需满足：j≠i；两个投影矩形有正面积重叠；source 的前向深度上界超过 target 的后向下界。最后一条采用区间判定，不仅使用中心正负号，以免排除大包围盒跨越目标深度的情况。

设 A_ij 为交叠面积，A_i、A_j 为投影面积，gap=max(0,d_j^-−d_i^+)。排序分数固定为

$$h_{ji}^{k}=\frac{A_{ij}/A_i}{1+gap/r_i}.$$

按 h 从大到小、gap 从小到大、几何内容 hash 递增取 K=8。源 ID 只作完全重合情况下的最终确定性平局键。此分数是低开销候选选择规则，不是精确遮挡概率。

### 5.3 每条边的 8 个特征

| 位置 | 数值 |
|---|---|
| 0–2 | normalize(c_j−c_i)，重合中心时置零 |
| 3 | log(1+‖c_j−c_i‖/r_i) |
| 4 | log(r_j/r_i) |
| 5 | A_ij/A_i |
| 6 | A_ij/A_j |
| 7 | log(1+gap/r_i) |

所有距离以同一 target 半径为参考；分母的数值 epsilon 与目标尺度绑定。保持 FP32 特征，不能在缓存中随机裁掉距离远但覆盖大的 source。

### 5.4 图的存储与实现边界

离线使用 `[N,12,K]` source index 和有效掩码，或者等价 CSR；边特征为 `[N,12,K,8]`。无邻居位置不得错误读取 source=0 的真实几何；即使填了零号占位索引，消息与聚合也必须被 mask。

生产构图使用二维空间索引检索投影重叠；CPU 二次枚举只用于小场景的正确性参照。**最坏情况仍可能接近 N²**，不能因为用了 R-tree 就声称所有场景构图复杂度 O(N log N)。记录候选边数、最终边数、各锚点度分布、耗时和峰值内存。

稀疏锚点、正交投影与 top-K 会遗漏某些透视遮挡和遮挡物联合覆盖。关系图是信息压缩近似，不是保守遮挡证明。后续网络无法保证恢复已经丢失的细节。

## 6. 单层场编译器

### 6.1 消息与聚合

$$m_{ji}=\phi([z_i,z_j,e_{ji}]),\qquad \phi:72\to64\to32.$$

每个目标—锚点单元内部计算 `Linear(32,1)` 后做 masked softmax，得到 h_ik=Σ_j α_ji m_ji。保留两个附加量：log(1+有效邻居数)、log(1+所选关系的目标覆盖比之和)。它们防止归一化 attention 完全抹去“有一个遮挡物还是多个遮挡物”的信息。

空单元直接得到 h=0、数量=0、覆盖和=0，不能对全负无穷 softmax 产生 NaN。

### 6.2 七参数响应

```text
base 输入：[z_i(32), anchor(3)]，35 → 32 → 7
关系增量：[h_ik(32), count(1), overlap_sum(1), anchor(3)]，37 → 32 → 7
q_ik = base + has_neighbor × delta
```

`has_neighbor` 是确定性布尔掩码，不是另一个可学习 gate。给 base 锚点方向是为了允许对象自身形状产生方向依赖；只有一个各向同性 base 会限制无邻域时的表达。无 per-instance residual、无局部/结构层级传播、无边分类/置信度排序辅助头。

### 6.3 固定低阶方向投影

主配置使用 b(d)=[1,d_x,d_y,d_z]，B_k=b(a_k)，预计算 P=B⁺。C_i=P Q_i，得到 `[4,7]`。这里使用的是未归一化实球谐等价多项式基，不混用其他库的标准化 SH 系数。

0 阶为 `[1]`；2 阶附加 `[xy,yz,zx,x²−y²,3z²−1]`，共 9 维。12 个锚点构成的矩阵分别检查秩 1/4/9。P 用 FP64 计算、FP32 保存，是 buffer 而不是 Parameter。

一阶主配置是预先固定的紧凑容量选择，不是“物理上一定充分”。二阶结果即使更好也如实报告，不能把二阶试验结果替换到一阶主模型名下。

## 7. 遮挡场的精确定义与解析表达

### 7.1 为什么不把 S 直接称为单位可见概率

有限对象可能只有一小部分露出；不同表面点被不同遮挡物遮住；沿一个方向改变观察距离也可能产生透视显露。因此“真实 unit 是否可见”一般不等价于一个关于距离单调下降的 survival 函数。

本版 S_i(a,d) 定义为：**从该单位的一个面积均匀表面探针起点，沿方向 a 行进距离 d 之前，没有遇到其他单位的遮挡表面的概率。** 对固定起点和方向，首次外部命中距离明确；对起点求平均后仍保持关于 d 的单调性。它是最终 PVS 的可解释中间统计，不是最终标签本身。

这也解释了为什么保留小可见性头：它要把无遮挡探针统计、目标形状、区域和相机条件映射为有限单位的区域可见标签。

### 7.2 距离与两分量模型

令 t=log(1+d/r_i)≥0。对查询方向 a，θ=b(a)^T C_i∈R⁷。参数转换：

$$p_0=\sigma(\theta_0),\quad \pi=\operatorname{softmax}(\theta_{1:3}),\quad
\mu=\operatorname{softplus}(\theta_{3:5}),\quad
s=0.02+\operatorname{softplus}(\theta_{5:7}).$$

定义每个分量的、在 t=0 截断归一化后的生存函数：

$$Q_m(t)=\exp\left[\operatorname{softplus}(-\mu_m/s_m)-\operatorname{softplus}((t-\mu_m)/s_m)\right].$$

总场为

$$S_i(a,d)=p_0+(1-p_0)\sum_{m=1}^{2}\pi_m Q_m(t).$$

因此 S(0)=1，S∈(0,1]，固定方向下随 d 不增；p_0 是模型中的无外部命中质量。两个混合分量无须排序，交换分量不会改变函数。它不承诺在训练探针有限范围之外准确估计无穷远概率。

旧 `d/(d+r)` 加限制在 [0.05,0.95] 的中心参数会把很远的遮挡结构挤在狭窄区间；本版不限制 μ 的上界，也不读取场景深度分位数。

### 7.3 场监督的生成

只为训练 source scene 生成监督：每个 unit 面积均匀选取 16 个探针起点；方向固定为 12 个正二十面体方向加 24 个固定 Fibonacci 球方向，共 36 个。每条射线只寻找第一个**其他 unit** 的表面，跳过当前整个 unit；材质和双面规则与注册渲染协议保持一致。

以 r_i 为尺度查询距离比 `[.25,.5,1,2,4,8,16,32,64,128,256,512,1024]`，只需一次最近命中查询即可产生 13 个截至距离标签。射线偏移为 10⁻⁵ r_i，并明确命中距离从原探针点计量。超出最大追踪距离未命中只作为该距离内未观察到命中，不能给出“无限远绝对无遮挡”真值。

e=1 表示 h≤d，e=0 表示 h>d。实际存储最近命中距离 h、最大有效追踪距离和 probe ID，不重复存储 13 份射线几何。训练从 unit、方向、起点、距离格点均匀抽取 8192 条有效观测。无 event 平衡采样、无隐藏困难采样；若实现非均匀采样，必须另立配置并带正确重要性权重。

**不能直接沿用旧 survival 观察文件的 event 和归一化深度。** 旧坐标、起点和事件语义不同，本版必须生成 `parallel_external_hit_current_status-v1` 资产。这是本版新增离线成本。

## 8. 区域查询与最终分类

### 8.1 支持点

圆盘：中心加 8 个等角度圆周点。圆盘轴来自区域 manifest，默认固定世界水平基，不由当前相机 yaw 随意改变同一缓存区域的采样位置。

定向盒：中心加 8 个角点，使用 manifest 中的三个有向半轴。五点对照在圆盘取中心与四个轴向端点，在盒上取中心与四个固定四面体角点；代码规定点顺序。

每点使用目标中心指向支持点的方向和中心距离查询同一场。这是把表面平行探针统计用于有限目标观察的近似，偏差由最终标签监督，而不是声称每个探针就是实际相机光线。

### 8.2 四个统计量与十六个相机特征

统计量固定为 `[S_center,S_max,S_mean,S_min]`。一支持点时四项相同。`S_max` 是采样支持位置中的最大无遮挡统计，不是整个连续区域可见概率的数学上界。

最终查询几何按以下顺序构造 16D：

| 位置 | 内容 |
|---|---|
| 0–2 | 目标→区域中心的世界方向单位向量 |
| 3–5 | 区域中心→目标方向在相机 right/up/forward 基中的分量 |
| 6 | log(1+d/r_i) |
| 7 | r_i/(d+r_i) |
| 8–10 | 区域三个半轴长度除以 d+r_i+max(半轴长度)；圆盘为 (R,0,R) |
| 11–12 | tan(FOV_x/2)、tan(FOV_y/2) |
| 13 | 区域类型：圆盘 0、盒 1 |
| 14 | near/(d+r_i) |
| 15 | log(1+far/(d+r_i)) |

相机投影矩阵、near/far 必须来自真实查询配置，不能填不同场景的常数。r=0 的退化单位不进入常规网络；相机与中心重合时用确定性零方向及保留标记记录该边界处理。各方法使用相同的几何无效输入处理。

### 8.3 小分类头与资源调度

输入为 32+4+16=52D，网络 `Linear(52,32) → SiLU → Linear(32,1)`。它没有 utility、download 或另一个校准 residual 头。

分类头理论上仍可能忽略场统计，因此必须报告无场对照和场直接输出对照，不能仅凭结构图宣称“网络决策一定来自物理场”。损失输出是用于分类/排序的分数，不因用了 sigmoid 就自动是已校准的真实可见概率。

实例分数聚合到共享 GLB 时，固定使用该资源候选实例分数的 max；需要资源即可下载一次。不要把多个相关实例的概率用独立事件公式相乘。既有 scheduler 的缓存、并发和带宽策略保持相同，使系统对比只改变可见性输入。

## 9. 损失：召回约束下的 GT 归一化冗余风险

### 9.1 为什么不能只加一个 FP/GTP 惩罚

只最小化误保留，会偏好“全部判不可见”；只提高召回，会偏好“全部保留”。固定的正负损失加权虽然可训练，但并不明确规定这两种错误谁必须优先受控。因此 V6.1 把问题写成：**普通可见实例和重要可见内容的遗漏不得超过预算，在此约束下尽量缩小额外保留的集合。**

上一版“训练完全不需要高召回目标，只靠校准负责安全”的说法过于绝对。训练必须提供对高召回有利的分数分离；校准只能选工作点，不能创造本来不存在的排序能力。两者应职责清晰，但不能认为高召回训练必然多余。

这里的“约束”指优化问题中的代理风险约束，不意味着任何优化器都保证求得可行解，更不意味着任意未知场景零漏检。普通非凸拉格朗日优化不存在这种无条件保证；本实现不是文献中随机混合模型等理论结果的直接实现。[S8]

### 9.2 同时保护两种召回，不能用一个替代另一个

令 G_s 为 source scene 训练集全部 GT 单位出现次数，W_s 为这些 GT 的原始 visible_weights 总和。普通实例漏检率是 FN/G_s；视觉加权漏检率是遗漏权重/W_s。

HKUST 既有冻结结果中，两者确实不同：weighted recall=0.996065，而普通 aggregate recall=0.961017。因此“weighted recall>99%”不能表述为“99% 实例不漏检”。[S9]

本版同时约束：

$$FNR_{count}\le\epsilon_c,\qquad FNR_{visual}\le\epsilon_w.$$

训练使用其可微上界；主配置固定两个代理预算均为 0.005。它们是预注册的训练预算，不是已达到的性能；正式校准另要求普通与加权召回均大于 0.99。目标不是用极低的总 loss 数值冒充已满足召回。

### 9.3 可微风险的精确定义

为使阈值语义明确，训练代理的参考阈值固定为 τ₀=0.5，定义：

$$h_{\mathrm{miss}}(\ell)=\frac{\operatorname{softplus}(-\ell)}{\ln 2},\qquad
h_{\mathrm{keep}}(\ell)=\frac{\operatorname{softplus}(\ell)}{\ln 2}.$$

主冗余目标是：

$$\boxed{J_{FP,s}=\frac1{G_s}\sum_{p,i\in O_p}h_{\mathrm{keep}}(\ell_{pi})}.$$

普通漏检代理风险为：

$$\boxed{R_{c,s}=\frac1{G_s}\sum_{p,i\in V_p}h_{\mathrm{miss}}(\ell_{pi})}.$$

视觉加权漏检代理风险为：

$$\boxed{R_{w,s}=\frac1{W_s}\sum_{p,i\in V_p}w_{pi}\,h_{\mathrm{miss}}(\ell_{pi})}.$$

这里 w 是原始 visible_weights，不经过 floor、power、按最大值压缩或正负类别平衡。普通约束单独保护小实例；视觉约束保护视觉贡献。主方法不再需要上一版有界正例重要度 a；它仅供 L_PBCE 与 L_FIXED_GT 控制实验使用。

候选筛选造成的遗漏也不能隐藏。若完整 GT 中有单位不在候选内，应分别在 R_c 与 R_w 的分子加上候选遗漏计数及遗漏权重常数。网络无法降低这部分常数。GT 非空而候选为空的 pose 是候选契约错误，不得在训练采样前无声丢弃。

### 9.4 为什么这三项确实对应“少漏检、少误保留”

在 q≥0.5 时，h_keep(ℓ)≥1；在 q<0.5 时，h_miss(ℓ)>1。故对同一完整数据集：

$$\boxed{FP/G_s\le J_{FP,s},\qquad FN/G_s\le R_{c,s},\qquad
wFN/W_s\le R_{w,s}.}$$

这是由每个样本相加得到的确定性不等式，包含上述候选遗漏常数。只有完整数据集代理风险实际小于预算时，才可推出该数据集在 0.5 阈值下的对应错误率小于预算。小批次估计、未见样本或另一个阈值不能直接套用这一结论。

对任意最终阈值 0<τ<1，完整数据上的关系改为：

$$FPGT(\tau)\le\frac{\ln2}{-\ln(1-\tau)}J_{FP},\qquad
FNR(\tau)\le m_C+\frac{\ln2}{-\ln\tau}R_c^{\mathrm{net}}.$$

其中 m_C 是候选筛选的固定漏检率，R_c^{net} 只含候选内正例的对数项；无候选遗漏时 m_C=0、R_c^{net}=R_c。固定遗漏不能随阈值一起缩小。视觉加权 FNR 用固定遗漏权重比例与对应候选内风险作同样分解。阈值改变会改变界的松紧；正式结果必须用冻结阈值重新计算硬 TP/FP/FN/TN，而不是报告这些替代界。

导数也符合纠错直觉：对高置信错误负例，∂h_keep/∂ℓ=q/ln2 接近常数；对高置信漏检正例，∂h_miss/∂ℓ=−(1−q)/ln2 接近负常数。相比单用 sigmoid 概率质量，在 q≈1 的错误 FP 上，对数项不会因为 q(1−q) 接近零而失去纠正梯度。[S6]

这些是逐 logit 的梯度性质，不是“共享网络每次参数更新一定同时改善所有样本”的保证。表示能力不足、相互冲突标签、错误 GT 仍会阻碍训练。

### 9.5 正式优化问题与总目标

对每个 source scene 都约束，而不是仅在所有场景混池后平均：

$$\boxed{\min_\theta\;\frac1{|\mathcal S|}\sum_s
[J_{FP,s}(\theta)+0.25L_{surv,s}(\theta)]
\quad\text{s.t.}\quad
R_{c,s}\le0.005,\;R_{w,s}\le0.005\;(\forall s).}$$

实现使用两个非负对偶乘子 η_c,s、η_w,s：

$$\mathcal L_s=J_{FP,s}+\eta_{c,s}(R_{c,s}-\epsilon_c)
+\eta_{w,s}(R_{w,s}-\epsilon_w)+0.25L_{surv,s}.$$

对模型参数做下降；对两个乘子做投影上升。两个乘子及其更新开关只存在训练进程/恢复 checkpoint 中，不在浏览器网络或场景运行资产中。每步处理一个 source scene，仅更新该 scene 的两个乘子。

模型反向传播时可以省去与 θ 无关的 −ηε 常数，故核心代码仍是：

```python
loss = (risks.excess
        + dual.eta_count * risks.miss_count
        + dual.eta_visual * risks.miss_visual
        + 0.25 * survival_nll)
```

这不是 PBCE + recall guard + CVaR + tail 的另一轮叠加：没有独立 BCE 分类项、困难挖掘、排序间隔或温度课程。任务层只有一个冗余目标和两种明确的安全预算；场监督属于表示层。

### 9.6 对偶更新的固定实现

令 κ_s=max(1,N_s/G_s)，其中 N_s 为 source train 全部不可见候选出现次数。初始化 η_c=η_w=κ_s/2；单约束消融则启用的那一个初始化为 κ_s。这样总初始正例压力对齐该 source 的负正样本数量比，而不是从极小权重开始任由负例占优。

每次成功完成一个 optimizer step 后，用该步旧前向输出的风险估计更新：

$$\eta_{c,s}\leftarrow\max\{0,\eta_{c,s}+0.01\kappa_s(\widehat R_{c,s}-\epsilon_c)\},$$
$$\eta_{w,s}\leftarrow\max\{0,\eta_{w,s}+0.01\kappa_s(\widehat R_{w,s}-\epsilon_w)\}.$$

超过预算，保护相应正例的压力增加；低于预算，压力逐渐减少，让冗余目标继续压低不可见分数。乘子不交给 AdamW，不做 weight decay，不反向穿过更新规则，也不使用 validation/calibration/test 风险更新。不存在额外 EMA、CVaR、warm-up 或分段 loss schedule。

步长 0.01 是预注册优化器超参数，不是理论上最优的数。0.005/0.02 的完整对照与预算 0.0025/0.01 的对照均在固定矩阵中。普通非凸训练可能振荡或未满足预算，必须记录；不能宣称这个规则一定降低 seed 方差。[S8]

### 9.7 空 GT、分块与无偏小批次估计

源训练集预计算 M_s（候选非空 pose 数）、ḡ_s=G_s/M_s、w̄_s=W_s/M_s。完整候选 pose 均匀采样 B=4，包括 GT=0 的 pose：

$$\widehat J_{FP,s}=\frac{\sum_{batch,O}h_{\mathrm{keep}}}{B\bar g_s},\quad
\widehat R_{c,s}=\frac{\sum_{batch,V}h_{\mathrm{miss}}}{B\bar g_s},\quad
\widehat R_{w,s}=\frac{\sum_{batch,V}w h_{\mathrm{miss}}}{B\bar w_s}.$$

加上该 batch 候选遗漏常数。分母是源训练集常数，不是当前 batch 的 GT 数，不是 `g_p+epsilon`。纯负 batch 的两种正例风险可以为零，但负例风险仍正常计算。原始视觉权重整体乘一个正常数时，加权风险不变。

目标单位分块时，每块用同一个完整 batch 分母；乘子在全部块反向传播期间冻结。先汇总所有块的三项风险，再只做一次模型更新和一次对偶更新。候选遗漏常数只加一次。梯度累积跨多个 pose microbatch 时，应使用合并后的 B，并只在真正 optimizer step 时更新对偶。

全部 source train 的 G_s、W_s 必须为正；源目标统计不得从 held-out 场景估计。若候选筛选的固定遗漏已超过预算，属于本约束问题的输入不可行性，应明确报告候选契约错误，不能通过减小分母、删标签或无限调大乘子掩盖。

### 9.8 完整训练步示例

```python
# normalizer 与 dual 来自当前 SOURCE scene，均只由 source train 初始化。
# batch.raw_visible_weights 不经过旧 positive importance 压缩。
risks = recall_risks(logits, batch.target, batch.raw_visible_weights,
                     normalizer, batch_pose_count=4)
loss = dual.model_objective(risks) + 0.25 * survival_nll
optimizer.zero_grad(set_to_none=True)
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
optimizer.step()
dual.update(risks, split="train")  # 使用旧前向风险；每个有效更新仅一次
scheduler.step()
```

必须记录 J_FP、R_c、R_w、两个预算差、两个乘子、乘子加权项、场 NLL、模型梯度范数。每 900 个当前 source 更新，把固定 source train 诊断子集上的风险与硬 FN/FP 单独报告；诊断结果不参与 checkpoint 选择或对偶更新。不能用小批次曲线短暂低于预算代替完整训练/测试结果。

### 9.9 保留直接软 FP/GTP 和固定权重方案作对照

`L_SOFT` 仅把 J_FP 换成 Σ负例q/G，保留同样两个预算和对偶更新，检验概率质量与对数惩罚的实际差异。此时在 τ₀=0.5 只有 FPGT≤2J_soft，不能沿用 J_log 的常数。

`L_FIXED_MULT` 保留 Full 的完全相同风险公式、原始权重、归一化、初始化和场监督，仅冻结两个乘子不更新。这一组才是隔离“自适应对偶更新”的单因素控制。

`L_FIXED_GT` 完整保留 V6 的 `L_cov + L_tight + 0.25L_surv`，以及有界视觉权重 a 与无 ln2 的公式；无对偶控制。这是旧固定权重目标的历史方案对照，并非只改变对偶更新一个因素。`L_PBCE` 保留逐 pose 逐类别平衡 BCE。

仅改变 BCE 的分母，在单场景、等样本权重情形可能只是整体常数缩放，不会创造新的最优排序。本版真正检验的是安全约束、训练分布和实际高召回处的冗余是否改善，而不是将换分母宣传为新理论。

### 9.10 “一定有用”的验收语义

可以在数学和单元测试中确认：两种退化解分别受罚、错误 logit 的梯度方向正确、完整样本上的替代上界成立、零 GT 不爆炸、分块不改目标。

不能在尚未跑真实训练时确认：它一定比 NeuralPVS 损失更好、一定达到所有场景 99%、一定更稳定、一定更少下载。最终有效性只指：**在同一普通/加权召回工作点下，FP/GTP 或真实资源成本降低，并同时报告置信区间、失败种子与画面误差。**

若要求任意未知查询严格零漏检，学习损失本身不提供此能力。需额外的几何保守证据或对不确定候选保留；保留全部候选是极端参照，但不是有效剔除成果。

## 10. 与 NeuralPVS 的明确对照

NeuralPVS 官方实现使用带 FP/FN 平方项的 weighted Dice，另有 `NoGuessLoss`；后者按几何占据量归一化可见与不可见区域的预测质量。[S4,S5]

候选单位适配对照按每个 pose 的一个二分类通道计算：

$$TP=\sum qy,\quad FP_2=\sum q^2(1-y)^2,\quad FN_2=\sum(1-q)^2y^2,$$
$$D=\frac{2TP}{2TP+0.001FP_2+0.999FN_2},$$
$$L_{NG}=1+\frac{\sum_{y=0}q-\sum_{y=1}q}{n_p},\quad
L_{task}=0.99(1-D)+0.01L_{NG}.$$

分母 epsilon=10⁻⁶，不私自增加使空正例 Dice 改变梯度的平滑分子。纯负 pose 的 Dice 项可能没有负例梯度，NoGuess 仍起作用；如实保留并报告。最终仍加同一个 0.25 L_surv。

NeuralPVS 的逻辑是重罚漏检的 Dice 与抑制过量可见预测的 RVL 协同；RVL 单独不是数学零漏检保证。本版沿用“保护覆盖、抑制冗余”的任务逻辑，但以明确的普通/加权召回预算和 GT 归一化目标表达，而不是直接搬用其固定 Dice/RVL 权重。[S4,S5,S7]

必须称为“NeuralPVS 损失的候选单位适配对照”，不能称为完整 NeuralPVS 方法复现。差异在于评价单位、归一化对象与风险解释，不在于包装一个新的缩写。原 weighted Dice 是**加重 FN 而非减轻 FN**。

## 11. 完整训练协议

### 11.1 固定配置

| 项目 | 正式值 |
|---|---|
| 训练种子 | 20260801、20260802、20260803 |
| 每个 source scene 更新次数 | 36,000 |
| 每步 pose | 4，单一 source scene，全部候选，不截断 |
| 每步场观测 | 8,192 |
| 几何/网络计算 | FP32；不采用混合精度作为隐含变量 |
| 优化器 | AdamW，lr=2×10⁻⁴，weight_decay=10⁻⁵ |
| 学习率 | 按 optimizer step 的 cosine，从初值下降到 0 |
| 梯度裁剪 | 全部 trainable parameters 的全局 L2 norm 上限 5 |
| 主目标 | GT 对数冗余目标 + 两个召回预算；γ=0.25 固定 |
| 训练预算与乘子 | ε_count=ε_visual=0.005；每个 source 两个投影上升乘子，详见第 9 节 |
| 最终 checkpoint | 固定总更新次数的最后一个，不挑最佳 seed/epoch |
| 随机增强 | 主矩阵不做随机场景旋转；平移/缩放作为确定性一致性测试 |
| 区域/几何 GT | 注册版本、hash 和样本规则固定 |

单场景共 36,000 步；四 source 的 LOSO 共 144,000 步，每个 source 恰好 36,000 步。source 选择采用固定种子打乱的四场景轮转表，防止随机抽样使场景暴露次数不同。每个 source 内从候选非空 pose 均匀有放回取样，保留零 GT pose。

对应种子的所有变体使用相同 pose 序列、相同 probe 抽样流。几何采样、图构造、初始化、pose、probe、bootstrap 分别使用独立随机数生成器；删除某个模块不应改变下一批训练数据。

最终可见性头 bias 可统一初始化为 source 训练分布的 logit 正例率；仅用 source train，所有可学习分类头对照相同。若使用默认零 bias，也必须作为整个矩阵统一配置。本版配置生成器要求生产实现采用零 bias，避免额外的数据依赖初始化。字段写入 manifest。

### 11.2 不使用 detach 破坏端到端训练

每步有三条梯度路径：可见性头→场→关系编译器；可见性头→z_i；场监督→场→关系编译器→目标及 source 的 z。不能把上一轮导出的 FP16 表当成训练期永不更新的特征，又宣称几何与关系端到端学习。

### 11.3 大场景的精确梯度重计算

单步先求出候选目标、探针监督目标和它们所需 source 的唯一单位集合 U。对 U 的几何编码采用如下链式法则缓存，不保留整个点云编码计算图：

```python
with torch.no_grad():
    z_cache = encode_required_units_in_chunks(points, ratios)
z_leaf = z_cache.detach().requires_grad_(True)

# downstream 按 target 分块；每块沿用相同的全 batch 损失分母。
# 累积 compiler/head 参数梯度，以及 z_leaf.grad。
backward_all_visibility_and_probe_chunks(z_leaf)

# 此时还没有 optimizer.step()。
for ids in unique_geometry_chunks:
    z = geometry_encoder(points[ids], ratios[ids])
    z.backward(z_leaf.grad[ids])

clip_grad_norm_(all_trainable_parameters, 5.0)
optimizer.step()
```

在无 Dropout/BatchNorm、无中途更新的条件下，这是原目标的链式法则重计算，不是冻结特征的近似。每个 source 节点的梯度必须从所有 target 累积。target chunk=128，几何重算 chunk=256，OOM 时只减小这两种内存块尺寸，不改 pose batch、候选数、损失分母或训练步数。

参考包提供该机制的 CPU 梯度一致性测试。生产训练器须扩展到完整候选/探针分块；不能直接把示例中一次性 downstream 图用于城市规模场景。

### 11.4 从同一表示生成监督、查询和导出

训练使用共享解析场函数；导出与 WebGPU/WASM 必须实现完全相同的基顺序、log 距离、混合参数变换、区域支持点顺序。量化后做 calibration：z_i、C_i 转 FP16，按 FP32 解码查询；小网络权重保持 FP32。禁止先用未量化网络选择阈值，再默认同阈值部署到另一套量化资产。

## 12. 校准、泛化和评价分离

### 12.1 本场景训练

只用该场景 train 更新模型参数和该 scene 的两个对偶乘子。训练到固定终点并导出后，在 calibration 上扫描实际分数变化点。正式主表 `dual_recall` 使用普通实例 recall、原始视觉 weighted recall 及各自单侧 95% pose-bootstrap 下界，四个值都严格大于 0.99 的最高阈值。

同时单独报告 `weighted_only_legacy` 兼容工作点：沿用旧项目只要求 weighted recall 和对应下界大于 0.99。它用于解释与 V4 旧任务的差异，不能把该工作点较低的 FP/GTP 放进双召回主表。所有新旧方法在同一工作点定义下比较；原 V4 的旧数值阈值并不自动满足新要求。

validation/test 只重放各自冻结阈值，不重新选阈值。两项单独的 95% 下界不声称构成联合 95% 保证；扫描、样本相关性与分布漂移也不被普通 bootstrap 自动解决。本版仍称为经验统计工作点，不称为任意连续观察位置的严格 PVS 证明。

### 12.2 五折跨场景训练

场景固定为 HKUST、IFCBench/Metropolis、Sponza、Viking、Big City。每折只使用四个 source 的 train 更新同一组参数；held-out 场景只通过几何编码器和纯几何图生成资产。

每折报告两种工作点，复用同一个模型和目标分数：

| 模式 | 阈值来源 | 允许使用目标标签吗 |
|---|---|---|
| `source_global` | 每个 source calibration 单独满足经验安全条件的最大共同阈值 | 不允许 |
| `target_calibrated` | 目标 calibration 选择一个标量阈值 | 只允许这一步，禁止梯度更新 |

source_global 在目标场景任一种召回或对应下界不满足 0.99 时照实报告，不移动阈值、不删该场景、不把 target_calibrated 数字补到其位置。target_calibrated 不能写成严格的无标签 zero-shot。

在所有源场景扫描共同阈值时，使用统一候选阈值集及固定 bootstrap 索引。保留 keep-all 作为明确部署回退点，但“模型未达到合格工作点”与“keep-all 回退安全”要分开统计，不能把回退算成一个有效剔除成功的 seed。

### 12.3 泛化不等于没有场景资产

新场景仍有各自 z/C 表和关系图，但这些由共享网络前向生成，不是场景内优化的自由参数。LOSO 证明的也是这五类场景之间的迁移，不是任意城市、任意动态场景或任意表示的普适保证。

平移/等比例缩放测试应保持输出近似不变；旋转测试只报告鲁棒性。有限锚点、top-K 与普通 PointNet 并不构成严格旋转等变网络。

## 13. 消融矩阵：问题驱动、配置唯一、完整重训

所有主消融在同一 HKUST 数据契约、三个种子和固定 36,000 更新下从头训练。Full 只训练一次，在不同表里复用同一个结果。每个变体使用本身量化资产的 calibration 阈值，不能共享 Full 的数值阈值。

每个变体都报告**实际参与 forward 的参数量**。保留了未使用层却把它们计入“容量匹配”，不构成公平控制。

### 13.1 统一 Full 配置

以下 JSON 与配置包 FULL.json 相同；训练/数据/评价字段见展开文件。主任务通过两个源训练约束保护覆盖，不额外混入旧 recall guard。

```json
{
  "model": {
    "geometry_input": "local_surface_xyz_normal",
    "points_per_unit": 256,
    "geometry_dim": 32,
    "field_source": "geometry_proxy",
    "anchors": "icosahedron12",
    "topk": 8,
    "edge_dim": 8,
    "message_dim": 32,
    "edge_hidden": 64,
    "pool": "attention",
    "field_order": 1,
    "field_parameter_dim": 7,
    "support_count": 9,
    "decoder": "mlp",
    "decoder_hidden": 32,
    "query_geometry_dim": 16,
    "instance_residual": false
  },
  "loss": {
    "task": "recall_constrained_gt_log",
    "tight_weight": 1.0,
    "survival_weight": 0.25,
    "positive_importance": "raw_visual_and_uniform_count",
    "normalization": "source_scene_gt_and_raw_weight",
    "dual": {
      "epsilon_count": 0.005,
      "epsilon_visual": 0.005,
      "dual_lr": 0.01,
      "count_enabled": true,
      "visual_enabled": true,
      "reference_threshold": 0.5,
      "initialization": "max_negative_gt_ratio_divided_by_enabled_constraints",
      "step_scale": "max_negative_gt_ratio_or_one",
      "update_split": "source_train_only",
      "update_enabled": true
    },
    "relation_aux_weight": 0.0,
    "explicit_l2_weight": 0.0,
    "residual_l2_weight": 0.0
  }
}
```

### 13.2 A 组：必须同时检验“保覆盖”与“压冗余”

除表中修改外，架构、场监督 γ=0.25、训练数据序列、训练预算、量化和校准规则完全相同。每个控制都在双召回主工作点和旧 weighted-only 兼容点分别报告。

| ID | 精确改动 | 回答的问题 |
|---|---|---|
| FULL | 第 9 节约束目标，两个预算均 0.005 | 完整方法 |
| L_FIXED_MULT | 风险与初始化完全同 Full；update_enabled=false，乘子全程固定 | 自适应约束压力是否必要 |
| L_FIXED_GT | V6 固定 L_cov+L_tight，无对偶；保留原有有界正例权重 | 召回约束是否优于固定权衡 |
| L_PBCE | 逐 pose/类别平衡 BCE；正例用旧有界权重；无对偶 | 简单分类目标是否已足够 |
| L_SOFT | 负例项仅换成 Σq/G；两个约束不变 | 直接概率质量是否存在饱和问题及实际差异 |
| L_NO_TIGHT | excess_weight=0；保留两个约束、场 NLL | 没有冗余压力是否退化成过量保留 |
| L_TIGHT_ONLY | 只优化 J_FP+场 NLL；无正例任务约束和无对偶 | 没有覆盖保护是否退化成漏检/校准后全保留 |
| L_WEIGHTED_ONLY | 只启用视觉加权约束；count 乘子恒为 0 | 高 weighted recall 是否掩盖小实例漏检 |
| L_COUNT_ONLY | 只启用普通约束；visual 乘子恒为 0 | 普通召回是否足以保护重要内容 |
| L_NPVS | 第 10 节官方损失的候选单位适配；无新对偶 | 与 Dice+RVL 的同架构对照 |
| L_LEGACY | 旧 PBCE+guard+tail 任务目标；无新对偶 | 复杂旧训练目标的同架构对照 |
| L_NO_SURV | γ=0，其他与 FULL 相同 | 场的显式命中监督是否必要 |

L_TIGHT_ONLY 不可以因为 calibration 只能全保留就省略；L_NO_TIGHT 不可以因为分类 accuracy 看似较高就宣称有效。这两组直接检验损失是否排除错误的退化策略。

L_LEGACY 完整固定以下旧 HKUST 参数，不能遗漏课程定义：

```yaml
recall_guard_weight: 0.3
recall_target: 0.99
recall_temperature: 0.05
pose_cvar_fraction: 0.25
pose_cvar_weight: 0.25
separation_weight: 0.2
positive_mass_fraction: 0.005
positive_count_cap: 64
negative_fraction: 0.01
negative_count_cap: 256
margin: 0.5
logit_temperature: 0.25
positive_importance_floor: 0.5
positive_importance_power: 0.5
guard_zero_fraction: 0.1
guard_middle_end_fraction: 0.3
guard_middle_scale: 0.25
tail_zero_fraction: 0.1
tail_ramp_end_fraction: 0.15
```

只有 L_LEGACY 使用这些课程；关系辅助、实例残差和额外 L2 仍按该组共同配置关闭。直接包装冻结 V4 的旧函数，不能凭 loss 名字重写成另一个实现。

### 13.3 B 组：信息来自自身几何、场景关系，还是实例记忆？

| ID | 字段/结构的精确定义 | 可学习实例表 | 运行时字段 |
|---|---|---|---|
| R_NO_FIELD | field_source=none；不构图、不计算 C；四个 S 统计置零；γ=0；保留 52D 头布局 | 无 | z 与公共 metadata |
| R_FREE_FIELD | C 为 Parameter[N,4,7]，N(0,0.02) 初始化；无 compiler；相同场查询与监督 | 有 | z+C |
| R_GEOMETRY | C=MLP(z_i)，输出 28；无任何 source/neighbour 输入 | 无 | z+C |
| FULL | 纯几何图、单层关系编译、固定基投影 | 无 | z+C |
| R_SHUFFLED | 使用固定扰乱的 source 配对，重新计算 8D 几何特征；其余网络不变 | 无 | z+C |
| R_MEAN | 所有有效 source 使用 1/K_valid 权重；移除 attention score 层；数量/覆盖和仍保留 | 无 | z+C |
| R_RESIDUAL | C=C_compiler+2tanh(R_i/2)，R_i∈R^(4×7)，零初始化；全程启用，额外 0.01 mean(residual²) | 有 | 导出融合后 z+C |

R_RESIDUAL 是本版简化的实例记忆对照，不是完整复刻旧 residual 的 reliability/warmup 机制。它故意不引入额外 curriculum。无实例表的 Full 不使用该项正则。

R_GEOMETRY 的参数匹配只针对 Full 的**场编译部分**：使用 `32→h1→h2→28`，穷举 h1,h2∈[16,512]，选择参数误差最小且宽度差最小的组合；若能在 1% 内匹配则记录匹配结果，否则如实记录误差，不增加未使用参数。所有变体的共享几何编码器与最后一层头保持相同。

R_SHUFFLED 在每个 anchor 内采用固定种子、保持目标度数的 source 对换；去除自环和重复，排序键固定。重新计算相对方向、距离、尺度与重叠；不把原真实边特征贴在错误 source 上。此对照破坏了部分几何关系，不能称为只有拓扑变化而其余输入分布完全不变。正式记录改变的边比例。

自由场和 residual 不具有新单位零样本推理能力，LOSO 列写 N/A。它们的零样本实验不能通过在目标场景补训练来“完成”。

### 13.4 C 组：低阶方向场是否足够？

| ID | 方向基 | 场维度 | 主配置几何+场 FP16 浮点数 |
|---|---|---:|---:|
| D_SH0 | [1] | 1×7 | 39 |
| FULL | [1,x,y,z] | 4×7 | 60 |
| D_SH2 | 完整固定二阶多项式基 | 9×7 | 95 |

三者都保留同样 12 个几何锚点、同样 K=8 和相同七参数响应头，仅固定投影矩阵和运行时基数不同。不能把减少锚点、减少 graph 输入一并作为 SH0 的变化。

SH2 的收益必须与额外传输、解析查询成本共同报告；不预写“SH1 必然最佳”的预期结论。

### 13.5 D 组：区域信息来自多点支持还是分类头？

| ID | 场支持点 | 最终头 | 训练标签 |
|---|---:|---|---|
| Q_CENTER | 1；统计为四个相同 S | 与 Full 相同，仍输入真实区域尺寸 | 原区域标签 |
| Q_FIVE | 5；圆盘十字/盒四面体固定点 | 与 Full 相同 | 原区域标签 |
| FULL | 9 | 与 Full 相同 | 原区域标签 |
| Q_FIELD_MAX | 9 | 删除 MLP，q=S_max；以数值安全的 logit(q) 接主风险 | 原区域标签 |

Q_CENTER 不是“真正单视点标签训练的模型”。它只是在相同区域任务下删除额外支持点，区域尺寸仍在输入，才能隔离支持查询的作用。

Q_FIELD_MAX 只需下发 C、AABB 与资源映射；z 仍参与服务器编译，但客户端不用 z。这组的导出必须真的删除不使用的 z/MLP，不能继续拿 Full 的资产大小代替。

另设 `POINT_UNION9` 方法参照：单独训练一个以真实单视点 GT 为监督、region extent=0 的同类模型，在 9 个区域支持点分别运行分类头，用 max/布尔并集聚合后再校准整体阈值。不能用 Q_CENTER 的区域标签模型冒充单点模型；9 次预测误差相关，也不使用独立性乘法。该参照在 HKUST 固定三个种子，属于单独 baseline job，不计入上述 23 个核心配置。

### 13.6 E 组：固定附录敏感性

| ID | 唯一变化 |
|---|---|
| S_G16、S_G64 | geometry_dim=16/64；相关输入维度自动推导 |
| S_K4、S_K16 | 每锚点 top-K=4/16；同一几何边排序 |
| S_DUAL005、S_DUAL002 | dual_lr=0.005/0.02；预算不变 |
| S_BUDGET0025、S_BUDGET001 | 两个训练代理预算同时设为 0.0025/0.01；正式评价仍是同一个 99% 工作点 |
| S_SURV0125、S_SURV05 | γ=0.125/0.5 |

这些是预定义配置，不以某个种子先跑出的结果决定保留谁。主配置仍为 g=32、K=8、两个预算 0.005、dual_lr=0.01、γ=0.25。

## 14. 固定训练矩阵与公平比较

机器配置生成器输出 23 个核心配置、10 个敏感性配置，共 33 个；HKUST 每个三个种子，共 99 个训练。

LOSO 固定 Full、L_PBCE、R_GEOMETRY，五折各三个种子，共 45 个模型；每个模型复用分数报告 source_global/target_calibrated，不重复训练。其他四个场景的 in-scene Full 各三个种子，共 12 个。

**模型与消融矩阵合计 156 个训练，10,476,000 个 optimizer updates。** 额外 AABB-query 与 POINT_UNION9 基线共 33 个，全部机器学习任务合计 189 个训练。完整工作量较大，但这是显式、固定的实验范围，不虚称为十来个训练，也不根据指标筛掉未完成配置。真实总耗时必须由实际硬件/step 测量估算。

对应种子的变体共享相同 pose/probe 抽样序列，共有模块使用名称独立的随机初始化流。checkpoint 同时保存模型、AdamW、scheduler、各 source 的两个乘子及 update_enabled、RNG、更新计数及数据 hash；恢复必须重建所有状态。崩溃恢复不换 seed，不把技术失败改称性能失败或剔除出统计。

## 15. 基线选择及其代码定义

### 15.1 本项目中可以公平落地的比较

| 基线 | 输入与算法 | 比较目的 |
|---|---|---|
| Keep-All | 保留完整候选，资源去重 | 零误剔除但无紧致度的参照 |
| AABB-query MLP | 3D 尺寸比例 + 16D 相机几何；19→64→64→1；主覆盖/冗余风险；无场监督 | 没有学习几何/关系资产时的共享网络下限 |
| 冻结 V4 | 使用原 96D+28D、原字段和原阈值；只在完全相同数据 hash 下引用旧结果 | 旧系统历史比较；不是新架构的单因素消融 |
| 无损几何外壳 HZB | 真实可访问几何、同区域查询与同候选 | 几何驻留的质量/成本参照 |
| 同启动字节外壳 HZB | 与当前 Full 的实际启动传输预算匹配，重新计算预算 | 验证轻量学习资产相对几何代理的收益 |
| POINT_UNION9 | 单点监督、9 次完整头调用、最终并集 | 区域解析查询相对于重复完整推理的价值 |

AABB-query MLP 的 19D 是本版新共享基线，不冒充仓库既有 18D AABB+Ray 实现。它在五个场景各三种子和五折 LOSO 各三种子训练，总共 30 个；POINT_UNION9 为 HKUST 三个训练。加上它们，机器学习训练总预算为 189 个模型。其他 HZB/Keep-All 不训练神经网络。

HZB 必须明确 Point 与 Region：单视点 HZB 不能直接与区域并集比较。区域 HZB 使用注册 GT 观察样本或明确的相同区域采样集合做并集，报告全部查询成本、深度绘制成本、资产与峰值内存。不同近似程度不能隐藏成同一个“HZB”数字。

### 15.2 NeuralPVS 的方法与损失分开

L_NPVS 只是损失对照。完整 NeuralPVS 属于另一种输入表示与预测单位；只有把其结果映射到完全相同 unit、GT、视场和区域协议后才能填写同一结果表。仅引用论文原生 froxel 数字时单独列为文献参照，不能和当前实例 FP/GTP 混画成直接优劣排名。[S7]

类似地，标签关系图不是理论性能上界。若作为附录参照，名称应为“额外标签信息参照”，而非 oracle upper bound。

## 16. 评价指标：FP/GTP 与 CNOR 不再互相近似

### 16.1 集合与安全

对每个 scene/seed/阈值先累加 TP、FP、FN、TN，主紧致度：

$$FPGT=\frac{FP}{TP+FN},\quad FNR=\frac{FN}{TP+FN},\quad
Inflation=\frac{TP+FP}{TP+FN}=1-FNR+FPGT.$$

weighted recall=Σ正例保留权重/Σ全部正例权重，使用原始 visible_weights；V6.1 主约束同样使用原始权重，只有旧损失对照保留有界 a。不能把它写成普通实例 recall，也不能把 1−weighted recall 写成未加权 FNR。

CNOR 继续严格使用：

$$CNOR=\frac{\sum_{p:n_p>0}TN_p/n_p}{\sum_{p:n_p>0}(TN_p+FP_p)/n_p}.$$

Useful Cull 的 aggregate 版本为 ΣTN/Σn。CNOR 与 aggregate specificity 的分母不同，不能将 CNOR 代入 `(1−s)(1−π)/π` 求正式 FP/GTP。

FP/GTP 的 pose-macro 版本只对 g_p>0 求平均并标明数量；g_p=0 时该比值无定义，不填 0。全部零 GT pose 的 FP 仍进入 aggregate 分子。主表同时列出空 GT pose 数和其平均/总 FP。

每个 seed 先单独求场景指标，再对三个 seed 给 mean±sample std；不能用平均 precision/平均 recall 的非线性组合代替平均 FP/GTP。跨场景平均是每个场景 FP/GTP 的等权平均，不把所有城市候选混在一起。

### 16.2 阈值无关与稳定性

报告普通 AP 的 pose-macro 与 aggregate 两种口径、各自正例比例；weighted ROC/AP 若另外实现须独立命名。增加低误剔除区域内的 WR—FP/GTP 曲线，在 validation 上绘制，不在 test 上选择曲线中最好点。

三种子稳定性包括 FP/GTP、CNOR、Useful Cull 的均值和样本标准差；双召回可校准比例；普通/加权约束违约率和两个对偶乘子的轨迹；每 pose 的分数秩相关；冻结阈值保留集合的 Jaccard；边界附近正负样本分布。不要仅凭全体负例主导的 Pearson>某常数就判定稳定。

概率阈值的标准差只作诊断。一个常数 bias 或温度变化就能移动阈值而不改变排序，因此“阈值更接近 0.5”“阈值方差更小”不能单独作为性能改进。

额外记录 source probe 的 NLL、Brier 分数、距离单调性违例、同方向不同深度曲线，以及 Full/R_FREE_FIELD/R_GEOMETRY 在低训练支持单位上的差异。支持度用 train 的候选次数定义；零支持单位单列，其余按训练统计四分位数分桶。

### 16.3 图像、资源与运行成本

图像报告同一冻结工作点的 miss-pixel、wrong-ID、PER、p95 和最大尾部错误。实例 FP/GTP 不等于像素错误，也不等于浪费下载字节。

资源指标先通过显式实例→GLB 映射进行 OR/max 去重，再计算下载字节、Bytes@95/99、达到参考画面覆盖的时间。真实可见资源本来必须下载时，其中不可见实例不应再次计为额外下载的完整 GLB。

浏览器计时包含候选生成、场查询、MLP、阈值筛选、GPU/CPU 回读与资源聚合。分别记录 GPU kernel 时间与端到端 wall time；不以 GPU 时间代替真实下载调度等待。

主资产预算为 32 FP16 + 28 FP16 + 6 FP32 AABB + 1 uint32 映射，理论 148 B/unit。实际文件还包括头部、padding、共享权重和 manifest，全部从导出文件实测。GPU 存储可能有对齐/展开开销，与传输大小分开。9 次解析函数不代表一定比旧频域查询更快，速度必须测试。

## 17. 代码与文件边界

V6 的几何/图/解析场接口保持不变；新增 `recall_constrained.py`、源训练 normalizer 元数据和对偶 checkpoint 字段。新配置 schema 必须拒绝把 `gt_log_fixed_v6` 默认为正式约束目标。所有新 API 都是执行 AI 需按本文实现的生产合同，不表示仓库已经有完整训练入口。

### 17.1 新命名空间，不覆盖 V4

```text
neural_instance_culling/model/v6/
  geometry_encoder.py
  proxy_graph.py
  field_compiler.py
  directional_basis.py
  survival_field.py
  region_query.py
  visibility_head.py
  risk_loss.py
  recall_constrained.py
  gradient_cache.py
  model.py
  train.py
  export.py
neural_instance_culling/benchmark/v6/
  controls.py
  build_matrix.py
  run_matrix.py
  evaluate.py
  summarize.py
neural_instance_culling/dataset/v6/
  build_local_surface_points.py
  build_geometry_proxy_graph.py
  build_external_hit_probes.py
  adapter.py
```

这些是需要实现的新路径，不是声称仓库已经存在。参考包的 `reference/gcof_core.py` 提供可运行数学核心，不能替代生产场景解析、稀疏构图、追踪监督生成、长训恢复和 WebGPU 实现。

### 17.2 必须分开的资产 schema

| 资产 | 必需字段与限制 |
|---|---|
| local surface points | unit order hash、真实 transform hash、sampling seed、[N,256,6]、[N,3] |
| geometry proxy graph | geometry-only、usesVisibilityLabels=false、anchors hash、K、mask、edge schema |
| external hit probes | source scene、ray schema、单位/方向/起点/命中距离、最大追踪距离 |
| pose dataset | split IDs、candidate CSR、GT、raw weights、真实区域半轴与相机参数 |
| source risk normalizer | source TRAIN 的 M、G、N、W、ḡ、w̄、候选遗漏常数、dataset hash；不导出 |
| checkpoint | resolved config、参数、优化器、scheduler、各 source 对偶乘子及其更新状态、全部 RNG、已处理 update/source 计数 |
| runtime | z/C 的 dtype/shape、基常量、head 权重、instance order、GLB map、量化方式 |
| evaluation sidecar | pose offsets、unit IDs、量化部署版本的 score、label、raw weight、frozen threshold |

源/目标权限必须由 fold manifest 指定，不由路径中是否出现字符串 `test` 猜测。读取 target geometry 与读取 target labels 是不同权限。部署编译流程不加载 target probe 文件。

### 17.3 模型接口

```python
z = geometry_encoder(local_points, extent_ratios)     # [U,32]
C = compiler(z_target, z_source, edge8, mask)          # [T,4,7]
logS = field(C, support_directions, support_distances, radii)
stats = region_stats(logS)                            # [Q,4]
query16 = build_query_geometry(...)
logits = visibility_head(z_for_queries, stats, query16)
```

T 是唯一目标单位数，Q 是 pose×candidate 出现次数；同一单位在不同 pose 出现，C 只编译一次但查询不同。不能错误把 unique-unit 批次当成 pose 批次计算 loss。

### 17.4 训练命令的合同

```bash
# 生成配置：参考包现在即可执行，不启动训练。
python build_configs.py

# 下列是按本文实现后生产入口应支持的命令形式。
python -m neural_instance_culling.benchmark.v6.run_matrix \
  --jobs /path/to/configs/jobs.json \
  --scene-registry /path/to/scenes.json \
  --output-root /path/to/outputs/gcof_v61 \
  --gpu-ids 0 1 2 3
```

生产 runner 根据 jobs.json 独立执行每个 run，输出 resolved config 与 stdout/stderr。不得遇到未知字段就忽略，未知 schema/维度必须抛错；尤其不能把 V4 的 `spectral_mode`、旧 depth quantile 或某个 loss schedule 默认带入。

### 17.5 数值与梯度测试

随附 35 项 CPU 单测覆盖：方向投影秩、输入尺度/点排列一致性、空邻域、邻居排列、场边界与单调性、场梯度、1/5/9 点、正例质量归一化、纯负 batch、极端错误 logits、分块 loss 等价、FP/GTP 替代上界、NeuralPVS 适配纯负梯度、几何缓存梯度一致性及查询维度。

生产新增测试还应包括：真实实例旋转/非均匀缩放后的表面采样；纯几何构图与小场景暴力参照一致；无 target label 读取；候选/GT 单位顺序；断点恢复与连续训练的更新次数一致；FP16 导出后 PyTorch/WebGPU/WASM 对同一输入的数值偏差与阈值翻转计数。新增约束单测还覆盖硬 FP/FN 上界、漏检小实例、候选遗漏常数、原始权重尺度不变、对偶上升/下降方向、禁止 calibration 更新和乘子 checkpoint 恢复。CPU 单测通过不表示真实场景训练、跨场景安全性或生产接口已验证。

## 18. 论文贡献与创新边界

### 18.1 应当作为论文中心的内容

**贡献一：部署约束下的几何到遮挡场编译。** 针对“渲染几何尚未到达浏览器”的约束，将服务器可用的场景几何邻域编译为固定大小的单位描述；客户端无需图传播或完整几何遮挡计算。重点是同一编码机制能在新场景生成资产，而不是每个场景重新拟合一张自由表。

**贡献二：具有明确中间语义的紧凑查询表示。** 将方向条件下的外部命中统计压缩为低阶解析场，用少量区域支持位置共享查询，再通过轻量头处理有限对象与相机条件。这里的技术贡献要由关系、自由场、方向阶数、支持数和直接场对照共同证明。

**贡献三：召回受约束的紧致 PVS 学习。** 以 GT 归一化冗余为目标，同时保护普通实例及视觉加权覆盖；区分训练代理、约束求解、后训练校准和硬 FP/GTP。拉格朗日优化、对数上界本身并非新理论；贡献只能建立在任务适配与同工作点实验收益上。

**贡献四：跨场景与真实浏览器证据。** source-only 编译、目标阈值校准与本场景训练明确分开；同时报告单位级紧致度、画面误差、真实资源收益、资产和端侧时间。scheduler 本身不被包装成新调度算法。

### 18.2 建议的论文表述

> We compile geometry-derived neighbourhoods into compact directional occlusion fields using shared, scene-independent networks. A small set of analytic region queries supports visibility prediction before render geometry arrives. Training minimizes GT-normalized excess visibility risk subject to instance-count and visually weighted coverage constraints, while operating thresholds are calibrated separately.

这是一项待验证的方法主张，不预写“优于现有方法”“稳定降低方差”“达到零样本安全保证”。SH、PointNet、attention、survival 混合和对数损失分别都不是本工作首创；新意应体现在它们围绕具体信息与部署限制形成的、能被严格对照支持的设计。

### 18.3 不能宣称的结论

不宣称一阶方向场能精确表示任意遮挡；不宣称 9 点覆盖连续区域的所有显露；不宣称 source calibration 使任意目标场景都满足 99%；不宣称模型无目标训练就没有目标预处理；不宣称 FP/GTP 小等于真实下载必然同比减少；不把额外标签关系称为数学上界。

架构的核心价值是否成立，最终取决于实测的风险—紧致度—字节—时延关系和跨场景对照，而不是让每一个新增组件都出现一个小数点提升。

## 19. 随附文件的使用范围

`configs/` 包含完整展开配置、156 项模型/消融/迁移 job 清单与统计；`baseline_jobs.json` 另列 33 项学习基线。`reference/gcof_core.py`、`recall_constrained.py` 与两份测试文件可运行，用于生产实现对照。它们没有伪装成已经连接现有数据的长训程序。

交给执行 AI 时，应同时提供本文、配置包和现有仓库。执行 AI 的工作是按本规范实现明确的数据适配、构图、probe 生成、训练/导出和 evaluator，不是重新选择方法、不根据早期结果改主结构，也不是把文中的新 API 名字当成已经存在的调用。

## 20. 依据与引用

[S0] 用户上传《面向泛化的紧凑遮挡场 PVS：新模型架构、损失函数与消融实验完整设计》，版本 2026-09-16。本文保留其主链路和问题驱动消融框架；第 1.2 节逐项列出本次修订。

[S1] 项目几何 exporter，核对快照 b21aff24915fdf8f0525b759b622fab6675e47ee：`neural_instance_culling/model/prepare_fixed_geometry_features.py`。其 16D 点输入包含局部点、场景归一化世界位置、中心和尺度。https://github.com/luxingzhi27/web3d-pvs/blob/b21aff24915fdf8f0525b759b622fab6675e47ee/neural_instance_culling/model/prepare_fixed_geometry_features.py

[S2] 项目当前主线记录：`docs/current/pvs_mainline_2026-08-23.md`，核对同一快照，含 HKUST split 与 IFCBench 四样本盒状并集语义。https://github.com/luxingzhi27/web3d-pvs/blob/b21aff24915fdf8f0525b759b622fab6675e47ee/docs/current/pvs_mainline_2026-08-23.md

[S3] 项目训练、损失与消融入口：`model/pvs_model.py`、`model/common/visibility_loss.py`、`benchmark/run_pvs.py`，相对根目录 `neural_instance_culling/`，同一快照。冻结旧入口只用于 V4 历史或任务损失对照。

[S4] NeuralPVS 官方 weighted Dice 实现 `losses/dice.py`，读取的 blob 为 a3f1849d409d70166a84cd6189136eea352245a4；FP/FN 为平方软项。https://github.com/windingwind/neuralpvs/blob/946088616cad18de81cde12fecd6ab204e52eac9/losses/dice.py

[S5] NeuralPVS 官方 `losses/no_guess.py`，读取的 blob 为 f7e5a09db4db4123362c94acff510b2b18c56210；按几何量归一化。https://github.com/windingwind/neuralpvs/blob/946088616cad18de81cde12fecd6ab204e52eac9/losses/no_guess.py

[S6] PyTorch 官方 BCEWithLogitsLoss 数值稳定性说明。https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html

[S7] Wang et al. NeuralPVS: Learned Estimation of Potentially Visible Sets, arXiv:2509.24677, 2025。https://arxiv.org/abs/2509.24677

本文的架构参数、召回受约束的 GT 对数风险、探针监督约定和固定实验矩阵是本次提出的设计；引用上述资料不表示这些设计已经由原文验证。

[S8] Cotter, Jiang, Sridharan. Two-Player Games for Efficient Non-Convex Constrained Optimization. ALT/PMLR 98, 2019. 文献明确区分非凸拉格朗日优化、代理约束及可证明的解表示；本文的简单投影乘子更新不直接继承其理论保证。https://proceedings.mlr.press/v98/cotter19a.html

[S9] 项目 HKUST 冻结结果，快照 b21aff24915fdf8f0525b759b622fab6675e47ee，`docs/evaluation/pvs_paper_baselines_contracts_2026-09-09.md`。weighted recall 0.996065 与普通 recall 0.961017 是不同口径。https://github.com/luxingzhi27/web3d-pvs/blob/b21aff24915fdf8f0525b759b622fab6675e47ee/docs/evaluation/pvs_paper_baselines_contracts_2026-09-09.md
