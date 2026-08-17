# 分层遮挡关系生存网络、视点区域积分频谱查询与质量风险损失实施记录

日期：2026-08-13
状态：v1 架构、实现、pilot 和 32 epoch 学习曲线的历史记录；当前行动只以[论文全创新组合扫参、长训与消融执行计划](pvs_full_innovation_hyperparameter_scan_longtrain_ablation_2026-08-13.md)为准。默认模型、阈值和前端资产保持不变。

## 研究目的

当前正式实验已经证明两个基础事实。第一，实例可见性确实依赖场景中的遮挡关系，但把关系压缩成普通上下文向量后，它与遮挡生存场表达的信息高度重叠。第二，117 维固定 Fourier 视角编码明显优于九维直接视角输入，说明可见性边界包含小型多层感知机难以直接学习的高频变化。RVL 主干也已表现出稳定的高召回作用，但尚未直接约束 view-cell 内稀有可见情况和无需求 GLB 字节。

本路线把论文方案收敛为三个相互配合、可以独立验证的候选创新点：

1. 借用 Graph U-Net 的多尺度编码、逐实例解码和跳跃连接思想，构造由真实三角形遮挡边驱动的分层关系网络，并让它直接生成逐实例遮挡生存场；
2. 把固定、逐坐标展开的 Fourier 输入改成视点区域感知、方向可学习且由遮挡关系场约束的紧凑频谱查询，用较少频率表达尖锐边界，同时抑制 view-cell 内的混叠和不稳定振荡。
3. 保留当前有效的 RVL 吸引/排斥主干，在其上加入视点区域质量尾部风险和 GLB 资源排斥，并通过安全余量门控与梯度投影防止资源优化伤害重要可见实例。

分层关系构建与编码全部放在离线训练和资产导出阶段；损失创新只在训练时执行。浏览器仍只接收后退相机候选实例，读取固定实例表，并执行一次轻量视角查询；不运行图传播、邻居搜索、点云编码、子位姿展开或 HZB 构造。

三个创新点分别回答三个问题：离线阶段怎样把“谁在什么方向和深度遮挡谁”烘焙成紧凑实例资产；前端怎样用一次 view-cell 查询稳定读取这份遮挡分布；训练怎样在保护重要画面的前提下减少误报实例和无需求 GLB 下载。只有快速验证和正式长训均支持的模块，才写入论文贡献和默认模型。

## v1 实验身份与保护边界

本记录覆盖的 v1 实验统一使用前缀 `pvs_hierarchical_relation_survival_integrated_spectral_quality_rvl_v1`。该身份只用于复现已经完成的实现、pilot 和学习曲线，不再承载后续扫参或正式长训。

统一输出根固定为：

- 训练期关系数据：`neural_instance_culling/dataset/out/pvs_hierarchical_relation_survival_integrated_spectral_quality_rvl_v1/`；
- checkpoint 与导出中间结果：`neural_instance_culling/model/out/pvs_hierarchical_relation_survival_integrated_spectral_quality_rvl_v1/`；
- calibration、validation、图像评价与统计：`neural_instance_culling/benchmark/out/pvs_hierarchical_relation_survival_integrated_spectral_quality_rvl_v1/`；
- pilot 报告：`docs/evaluation/pvs_hierarchical_relation_survival_integrated_spectral_quality_rvl_v1_pilot.md`；
- v1 正式长训报告未创建，因为旧 Formal80 未达到启动门；不能把计划文件名当作已有结果。

这些路径记录 v1 产物的来源。现有完整或 partial 成员均不得覆盖，也不得换名后冒充 v2 结果。

该路线实施期间保持以下边界：

- 当前 HKUST 前端模型 `pvs_directional_occlusion_proxy_encoder_rvl_strong_v2_full40_best`、阈值 `0.02` 和部署资产保持不变；
- 历史强参考使用已冻结的 `rvl_strong_v2`、自由逐实例生存场和 Fourier117 查询定义，不能悄然修改后仍沿用同名；
- 新实验使用相同的空间隔离 train/calibration/validation、相同后退相机候选 CSR、相同实例 GT、相同 `visible_weights`、相同实例到 GLB 映射和候选口径；
- `test` 在模型结构、checkpoint、阈值、资产和论文对比均冻结前不得读取；
- 不补入 GT 可见实例，不改变候选集合，不使用前端白名单或运行时多 subpose 查询修复结果；
- 所有训练从头开始。pilot 结果只能筛选实现和参数，不能作为正式论文统计结论。

当前实现仅支持单进程单卡训练，不把四张 GPU 误写成分布式训练。四张卡用于同时运行四个独立成员；每个成员必须保存独立 stdout、stderr、训练元数据、checkpoint 和校准摘要。

## 已有证据与问题定位

### 普通上下文与生存场的重叠

以 2026-08-11 修正 pooled-context 定义后的正式矩阵为准，在 RVL 控制损失下：

| 比较 | pose precision 差值 | pose useful cull 差值 | pose weighted recall 差值 | 解释 |
|---|---:|---:|---:|---|
| 无生存场时加入方向上下文 | -0.0118 `[-0.0248, 0.0013]` | -0.0128 `[-0.0258, 0.0014]` | +0.0020 `[0.0000, 0.0041]` | weighted recall 略升，但分类与有效剔除没有稳定改善，预测 GLB 字节反而稳定增加 |
| 已有生存场时再加入方向上下文 | +0.0001 `[-0.0067, 0.0097]` | -0.0011 `[-0.0055, 0.0023]` | +0.0003 `[-0.0034, 0.0030]` | 各项接近零且主要区间跨零 |

因此，修正版证据不支持普通上下文具有稳定的独立分类或剔除贡献。继续扩大上下文宽度也没有解决问题：32 维增至 64 维后，precision、balanced accuracy、useful cull 和 weighted recall 均未稳定改善，固定表还由 5.88 MB 增至 7.08 MB。旧 2026-08-09 汇总使用了已被修正的上下文定义，其“无生存场时上下文明显提高 useful cull”等数字不再作为依据。

后续不再保留“普通上下文头 + 独立生存场”这两个并行槽位。关系编码器应把实例间关系直接转化成具有明确遮挡语义的生存场参数。完整证据见[修正正式矩阵](../evaluation/pvs_direction_depth_relation_survival_constrained_v1_surface_fallback_owrb_formal40_pooled_context_20260811_formal_validation.md)和[补充机制矩阵](../evaluation/pvs_ray_context_survival_owrb_supplement_v1_formal40_20260812_formal_validation.md)。

### 生存场已经表现出的有效信号

修正版正式矩阵中，RVL 条件下加入生存场具有明确分类信号：上下文关闭时，pose precision、balanced accuracy、F1 和 useful cull 分别提高 `8.31`、`5.54`、`8.14` 和 `8.23` 个百分点，平均预测数减少约 `89`；上下文开启时，相应改善为 `9.50`、`5.30`、`8.34` 和 `9.40` 个百分点，平均预测数减少约 `157`。两组 weighted recall 差值区间均跨零；后一组 pose bad cull 增加 `0.92` 个百分点，区间 `[0.27, 1.53]`，说明效率收益仍需安全约束解释。跨 RVL 与旧安全损失混合计算的全因子结论受强交互影响，不能替代这两组成对比较。

这组结果支持继续研究“方向和距离可查询的遮挡生存表示”，但不证明现有生存场已经通过安全系统路线。当前效果较好的生存场主要由逐实例自由参数学习，真实遮挡关系编码器输出的普通上下文可以被最终可见性网络绕过。它尚未证明“目标实例与遮挡来源之间的关系”真正生成了有效生存场，因此新实验必须同时比较真实来源、置乱来源和等容量自由参数。

### 当前 Fourier 编码为什么有效

当前九个视角标量由三维单位视线方向和六个距离、投影位置及角尺度标量组成。117 维版本采用固定二倍频带：

- 三个单位视线方向分量各展开十个频带，共 63 维；
- 其余六个标量各展开四个频带，共 54 维；
- 合计 117 维。

相比九维直接输入，117 维 Fourier 编码在正式补充矩阵中取得了以下差值：

| 指标 | 差值 | 95% 配对区间 |
|---|---:|---:|
| pose precision | +0.2164 | [0.1961, 0.2372] |
| pose balanced accuracy | +0.0555 | [0.0468, 0.0646] |
| pose useful cull | +0.0552 | [0.0420, 0.0681] |
| pose bad cull | -0.0179 | [-0.0249, -0.0123] |
| miss-pixel rate | -0.0035 | [-0.0070, -0.0009] |

代价是可见性头输入从 177 维增至 285 维，CUDA p95 增加约 1 ms，预测的绝对 GLB 字节增加约 15.58 MB。Fourier 编码解决了小型 ReLU 网络的频谱偏置，但当前实现仍有四个问题：

1. 频率固定在坐标轴方向，不能主动对齐实例遮挡边界在联合“方向、距离、屏幕位置”空间中的朝向；
2. 所有候选都计算同样的十层高频，高频数量与当前 view-cell 尺度和实例距离无关；
3. 全局正弦基逼近突变边界会产生振荡，继续增加频带容易带来过拟合、阈值敏感和小范围相机扰动不稳定；
4. Fourier 特征直接进入最终可见性头，网络可以依靠高维视角特征拟合结果，而不使用离线遮挡关系。

## 候选创新一：真实遮挡边驱动的分层关系生存网络

### 与旧 Graph U-Net 路线的关系

Graph U-Net 值得保留的核心思想有三项：编码端逐层扩大感受野、解码端恢复到每个原始节点、同尺度跳跃连接保留细粒度信息。这些机制适合“先理解场景级关系，再给每个实例恢复一个输出”的任务。

原始 Graph U-Net 的通用节点打分池化不直接适合本项目。可见性中的邻接关系具有方向和深度顺序，同一个近邻既可能完全无关，也可能在特定方向成为主要遮挡来源。新网络采用以下约束：

- 图边来自真实三角形深度层和表面深度比较，不使用 KNN 作为遮挡边；
- 每条边保留“前方实例指向被遮挡实例”的方向，不把有向关系退化成无向邻接；
- 池化时保留球面方向和深度层，不对全部方向做全局 mean/max；
- 解码结果直接生成生存场系数，不再生成与生存场语义重叠的普通上下文向量；
- 整个图网络只在离线运行，前端不导出图、边或图网络权重。

### 图的定义

以实例为节点构造训练期有向遮挡图

\[
\mathcal G=(\mathcal V,\mathcal E).
\]

节点 \(i\) 的初始特征包含：

- PointNet++ 风格离线编码得到的 96 维实例几何；
- 归一化实例中心、尺度和包围半径；
- 训练采样中该实例的有效遮挡证据覆盖率。

若训练代表子位姿的深度层中，实例 \(j\) 位于实例 \(i\) 前方，则建立有向边 \(j\rightarrow i\)。边特征包含：

- 真实观察方向及其球面方向区域；
- 前后深度间隔和归一化距离层；
- 遮挡像素支持、pose 支持和证据置信度；
- 两实例中心差、尺度比和几何特征组合；
- 证据来自深度剥离还是 GLB 表面点补全。

正式图不能直接使用当前 `sourceK=8` 的 top-k 关系表。现有审计已经记录其丢弃了 2,258,881 个超出 top-k 的唯一来源实例。后续需要从 train-only 深度缓存构建“在已登记采样、最多六层深度剥离和表面补全覆盖内，不按来源 top-k 截断”的关系 CSR 摘要，并记录 render pose、候选口径、深度层范围和表面补全来源。该摘要仍不是无限深度、全视角的完整场景遮挡图。当前丢失 source identity 的 target-level 压缩关系场也不能作为图网络输入。

### 方向保持的分层编码

计划使用三个离线尺度：实例层、局部遮挡群层和大型结构遮挡群层。层级通过真实遮挡边的支持强度进行重边合并，并用最大空间直径约束避免把远距离、偶然共现的节点合成一个群。粗化后的边只由原始真实边聚合产生，不创建没有观测依据的新遮挡边。

各层节点数量随场景和遮挡图规模变化，不把任意规模场景压缩成固定数量的全局 token。关系 CSR 和分层图只供离线分块训练使用，因此可以保留比前端资产更丰富的来源信息。

每一层为节点维护“入射遮挡消息”和“出射遮挡消息”。入射消息描述哪些结构可能挡住当前节点，是生存场的主要来源；出射消息描述当前节点在场景中的遮挡角色，作为结构辅助信息。消息先在同一方向、同一深度层内聚合，再跨相邻深度层和相邻球面方向传播，避免相反视角的关系被混合。

编码端逐层将细实例关系汇聚到遮挡群和大型结构，使一个实例能够获得墙体、楼层或大设备等长程遮挡结构的信息。解码端按保存的群成员映射逐层上采样，并与对应实例层特征做跳跃连接，最终恢复每个实例的独立表示。跳跃连接保证细小构件不会在粗层聚合中消失。

### 直接输出遮挡生存场

解码后的每实例特征不输出普通上下文。系数头直接生成

\[
A_i\in\mathbb R^{R\times 7},
\]

其中 \(R\) 是共享方向查询基的秩。本轮固定 \(R=4\)，不再把 \(R=6\) 留作训练后的可选分支。七个参数继续描述“该方向无阻挡概率”和两个首遮挡深度分量。查询方向 \(\mathbf d\) 和视点区域尺度生成共享基 \(\psi(\mathbf d)\)，再计算

\[
\theta_i(\mathbf d)=\psi(\mathbf d)^\top A_i.
\]

经过正值、归一化和深度排序变换后得到随查询距离单调非增的生存概率

\[
S_i(\mathbf d,\rho)=P(\text{在方向 }\mathbf d\text{、归一化距离 }\rho\text{ 内尚未遇到有效遮挡}).
\]

这一步把“谁可能挡住谁”的图关系转化成前端可直接查询的物理语义量。最终可见性头只接收实例几何、轻量视角标量和生存语义，不再读取普通上下文，也不允许把生存场原始系数平铺后绕过查询过程。

### 监督信号

网络同时接受两类已有真实监督：

1. 深度剥离事件与右删失监督。观察到前方遮挡层时为事件样本，第一层可见且未观察到遮挡时为右删失样本；按真实像素支持和 pose 支持加权，优化生存似然。
2. view-cell 潜在可见集合监督。任一合法子位姿可见即视为该 view-cell 的正例，使用保留高召回作用的可见性损失训练最终输出，并共享视觉效用和 GLB 调度头。

每个解码尺度增加轻量深监督，将对应粗节点的事件/删失统计用于训练该尺度的生存预测。深监督只用于改善离线梯度传播；导出时只保留最终逐实例系数。

### 为什么它与已失败的关系残差路线不同

此前关系残差直接对最终可见性分数做双向修正，实验中常用增加漏检换取预测数下降。这里的关系网络只负责生成可解释的遮挡生存分布，由事件/删失监督约束；最终安全决策仍由统一可见性头和 calibration 阈值完成。

此前 target-level 关系场在拟合后丢失来源身份，无法证明具体遮挡来源被使用。新网络在离线传播全过程中保留有向来源边，只有最终导出时才压缩成逐实例系数。

此前普通上下文与自由参数生存场并行，最终网络可以忽略前者。新网络移除自由逐实例生存参数和普通上下文，生存场只能由共享关系编码器根据几何与真实遮挡边生成。

### 前端资产和计算预算

若使用 \(R=4\)，固定表为 `96 维几何 + 28 维生存系数 = 124 维`；HKUST 18,831 个实例的 FP16 原始表约为 4,670,088 bytes。若使用 \(R=6\)，固定表为 138 维，约为 5,197,356 bytes。两者都不超过当前 156 维实验表的 5,875,272 bytes，也远低于已部署 352 维方向代理表。

浏览器不加载关系图和离线编码器。每个候选只进行一次共享方向基计算、一个小型系数内积、生存函数计算和紧凑可见性头前向，运行复杂度仍与候选数线性相关，单候选成本为常数。

### 必须先补齐的关系数据结构

正式关系网络不能直接读取当前 `sourceK=8` 表，也不能读取已经丢失来源身份的 target-level 关系场。实现前先新增 train-only 的 `pvs-viewcell-train-observed-relation-csr-v1`：

- 行索引按 `target instance → spherical direction → depth shell` 排列；
- CSR 项保留 `source instance`、render pose 支持、像素支持、深度间隔一阶/二阶统计、相对中心和尺度统计、证据置信度与来源类型；
- 在同一 render pose 内先对来源—目标关系去重，再跨 pose 聚合；在已登记的六层深度剥离与表面补全覆盖内不做来源 top-k 截断；
- 元数据必须保存 train split、实际 render pose 顺序、canonical/render candidate 语义、FOV、分辨率、深度层数、surface fallback 摘要及输入路径、schema 和 shape；
- “不做 top-k”只描述已观测深度层与补全范围，不能命名成完整全场景遮挡图；
- 图分层映射和原始 CSR 只用于离线训练，不导出到浏览器。

构建后先执行覆盖和语义检查：`source/target` ID 有效、深度事件严格递增、关系只来自 train、候选口径一致、右删失与遮挡事件不混用。任何一项失败都停止训练，不通过兼容分支回退到旧 top-k 表。

## 候选创新二：视点区域积分的关系条件化频谱查询

### 设计目标

单纯增加 Fourier 频带可以拟合更尖锐的边界，但不能解决 view-cell 内部的不连续性。模型标签表示整个视点区域的潜在可见并集：区域内只要存在一个合法子位姿能看到实例，就必须保留该实例。把区域内 Fourier 响应简单求均值会淹没少量但重要的可见子位姿；只查询中心点又会对高频边界产生混叠和抖动。

新的“视点区域积分频谱查询”同时表达中心相位、区域平均后的可信幅度和区域内尚未解析的高频能量。它借鉴 Fourier Features 的高频表达、mip-NeRF 的积分位置编码、Ref-NeRF 的尺度感知方向编码和可学习 Fourier 特征，但监督语义改为 view-cell 潜在可见集合，并将频谱结果用于查询遮挡生存场。

### 从九维查询到区域分布

对实例 \(i\)，仍以九维无哈希视角量作为基础：

\[
\mathbf q_i=[\text{三维单位视线方向},\text{归一化距离},\text{前向夹角},u,v,\text{水平角尺度},\text{垂直角尺度}].
\]

视点区域用均值 \(\boldsymbol\mu_i\) 和对角不确定度 \(\boldsymbol\Sigma_i\) 近似。浏览器不采样多个子位姿。不确定度由已知 view-cell 平移半径、yaw/pitch 扰动范围、实例距离和 66° 模型视场角解析估算：

- 方向不确定度约为“视点区域半径 / 实例距离 + 旋转扰动”；
- 距离不确定度由视点区域半径与实例距离归一化得到；
- 屏幕位置不确定度由方向不确定度和视场角尺度得到。

这些量只需要少量加、乘、倒数和平方运算，不需要运行子位姿循环或 AABB 八角点投影。

首轮只对当前 HKUST 契约实现解析不确定度：水平圆盘半径 `2 m`、固定高度和固定朝向、模型输入 FOV `66°`、后退距离 `3.4641 m`。当前前端一旦位置离开该圆盘或朝向/FOV 发生变化就重新建立预测锚点，因此训练端的区域分布必须与这一运行门完全一致。以后若允许 yaw/pitch 范围，必须先扩展数据契约和候选包络，再扩展协方差公式，不能先在编码器中虚构未采样的方向范围。

### 可学习的联合频率

当前编码分别对每个坐标使用固定 \(2^k\) 频率。新编码学习 \(K\) 个联合频率向量

\[
\mathbf b_k\in\mathbb R^9,
\]

使频率能够对齐“方向、距离、屏幕位置和角尺度”共同形成的可见性边界。频率向量在对数均匀的低、中、高三个频率组中初始化，并约束其范数和组内多样性，防止全部频率坍缩到同一个高频或记忆单一场景方向。

对于每个频率，先计算中心相位和区域衰减：

\[
t_{ik}=2\pi\mathbf b_k^\top\boldsymbol\mu_i,
\qquad
a_{ik}=\exp\left(-\frac12\mathbf b_k^\top\boldsymbol\Sigma_i\mathbf b_k\right).
\]

再构造三元频谱特征：

\[
\mathbf z_{ik}=
\left[
a_{ik}\sin t_{ik},
a_{ik}\cos t_{ik},
\sqrt{\max(0,1-a_{ik}^2)}
\right].
\]

前两项是在视点区域内积分后的稳定相位响应；第三项是单位复相位的解析标准差，表示该频率在当前区域内有多少变化无法由单点相位可靠确定。高频或近距离大视点区域会产生更大的未解析相位能量，明确告诉模型“该实例可能跨越可见性边界”。因此编码没有把区域内少量可见情况直接平均成不可见。单位视线始终使用三维笛卡尔向量而非 yaw 角，球面方位跨越 \(-\pi/\pi\) 时不会产生人为接缝。

本轮固定 \(K=16\)，连同原始九维量共 57 维。它只需计算 16 组正弦/余弦；当前 Fourier117 实际计算 54 组逐坐标频率相位。新编码的三角函数组数约减少 70%，同时允许频率方向参与训练。若该配置未通过快速验证，本轮频谱创新直接判定未通过，不在结果产生后追加 \(K=24\) 或更多频率。

### 低频主体与局部边界分支

连续频谱无法在有限维度下精确表示数学突变。实际目标是让分类阈值形成清晰边界，同时避免全局 Fourier 在边界两侧振铃。共享频率分成两组：

- 低频主体描述大范围、平滑的方向和距离变化，所有候选都使用；
- 高频边界组只在生存场显示较大局部斜率、较高遮挡不确定度或较大的未解析频谱能量时参与查询。

高频边界组采用连续软门控相乘，不在 WebGPU 中为不同候选执行动态分支。边界组不直接给最终 logit 增加自由残差。它与低频主体共同生成一个最多六维的共享查询基，再读取分层关系网络导出的生存场系数：

\[
\boldsymbol\psi_i=\operatorname{QueryNet}(\mathbf q_i,\mathbf z_i),
\qquad
\boldsymbol\theta_i=\boldsymbol\psi_i^\top A_i.
\]

最终可见性头只看到 96 维几何、九维直接视角量、八维生存语义，以及“平均未解析能量、最大未解析能量”两个区域不确定度摘要，共 115 维；57 维频谱不会原样拼入最终头。这个结构迫使高频视角表达服务于“查询离线遮挡关系”，降低模型绕过生存场、直接记忆相机到实例标签的可能性。

### 高频与不连续性处理的合理边界

该设计可以改善三类问题：

1. 可学习联合频率比轴对齐频率更容易表示倾斜的遮挡边界；
2. 区域衰减抑制超出 view-cell 分辨能力的高频振荡，降低相机轻微扰动引起的分数跳变；
3. 未解析能量通道把边界风险显式交给高召回训练目标，避免积分平均掩盖细小但重要的可见区域。

它不能保证精确恢复任意不连续集合，也不能仅凭结构宣称优于 Fourier117。若学习频率在跨场景测试中出现方向记忆，应限制为共享球面方向频率并加强旋转增强。若积分编码仍然过度平滑，再考虑 WIRE 启发的少量局部 Gabor/小波边界基；该扩展不进入首轮方案，避免一次混入过多变量。

### 实现与数值门

首轮固定 `K=16`，频率参数在所有实例之间共享，不允许逐实例频率或相机哈希。实现必须同时提供 PyTorch FP32、导出 FP16 和浏览器查询三条路径，并覆盖以下单元测试：

- 零区域尺度时退化为普通可学习 Fourier 响应；
- 区域尺度增大时积分幅度非增、未解析能量非减；
- 三维方向跨越球面方位接缝时输出连续；
- 频率范数约束和低/中/高频组不发生整体坍缩；
- 同一导出样本的 PyTorch FP32、FP16 和 WebGPU 数值误差在登记范围内；
- WebGPU 若返回 SwiftShader，只能完成软件数值 parity，不能记录硬件延迟。

训练端记录每组频率范数、软门均值、积分衰减分布和边界实例翻转率。不得通过追加频率数把紧凑查询重新变成高维输入堆叠。

## 候选创新三：视点区域质量风险与资源排斥 RVL

### 设计目标和输入

当前 `rvl_strong_v2` 对不平衡候选集合有效，正式实验也表明旧“安全约束效用损失”虽然提高 weighted recall，却通过大量过预测降低 precision、balanced accuracy、useful cull 和资源效率。因此新损失保留 RVL 的 BCE、Tversky、集合数量、难负排序及 FN/FP 吸引—排斥主干，不用一个全新的约束优化器替换它。

对 view-cell (v) 中候选实例 (i)，新增训练输入为：

- 可见标签 (y_{vi}) 和模型 logit (z_{vi})；
- 当前 Three.js Color-ID 数据中的 `visible_weights`。对 HKUST 当前数据，它是各合法 subpose 中最大屏幕覆盖率的 parts-per-million，可作为视觉质量证据；若换用 rvcServer `component_weights`，必须降级称为弱重要性证据；
- `visible_hit_counts / successful_subpose_count`，表示该实例在多少合法 subpose 中出现，仅用于保护 view-cell 并集中的稀有正例；
- 实例到 GLB 的映射和真实 GLB 字节成本。

先用现有 source view-cell 数据生成只读 sidecar，将 `visible_hit_counts` 对齐到正式空间 CSR。sidecar 必须验证可见 ID、权重、pose 顺序、split 和候选口径，不修改主 CSR、候选集合或 GT。

### 质量尾部风险

令温度化软分数为 (s_{vi}=\sigma(z_{vi}/T))，归一化屏幕质量为 (w_{vi}=\log(1+\text{visible\_weight}_{vi}))，子位姿命中率为 (h_{vi}\in[0,1])。危险正例质量定义为

\[
q_{vi}=w_{vi}\,[1+\beta(1-\sqrt{h_{vi}})].
\]

它同时保护屏幕贡献大的实例和只在少量合法子位姿出现的 view-cell 边界实例。每个 pose 的软质量漏检风险为

\[
R_v^q=1-\frac{\sum_i y_{vi}q_{vi}s_{vi}}
{\sum_i y_{vi}q_{vi}+\epsilon}.
\]

按累计 (q) 质量选择最低分正例尾部 (H_v^\alpha)，质量风险项为

\[
L_{\mathrm{quality}}=
\frac1{|V|}\sum_v\left[
\operatorname{ReLU}(R_v^q-\epsilon_q)^2+
\frac{\sum_{i\in H_v^\alpha}q_{vi}\,\operatorname{softplus}(m_+-z_{vi})}
{\sum_{i\in H_v^\alpha}q_{vi}+\epsilon}
\right].
\]

本轮固定 `T=0.10`、`alpha=0.01`、`epsilon_q=0.005`、`beta=1.0`。这些值是预注册的快速验证和正式训练配置，不在看到 validation 结果后重新扫描，也不是统计安全证明。若现有批次只含一个 pose，尾部在 pose 内按质量累计计算；跨 pose 最差风险至少需要两个 pose 的 batch，不能在单 pose 批次中伪造 CVaR。

### GLB 资源排斥

一个 GLB 只要有一个预测实例就可能被请求。使用可微 noisy-OR 得到软请求概率

\[
P_{vg}=1-\prod_{i:g(i)=g}(1-s_{vi}).
\]

令 (Y_{vg}=\max_{i:g(i)=g}y_{vi})。资源项只处罚 (Y_{vg}=0) 的无需求 GLB：

\[
L_{\mathrm{resource}}=
\frac1{|V|}\sum_v
\frac{\sum_{g:Y_{vg}=0}\bar c_g P_{vg}}
{\sum_{g\in C_v}\bar c_g+\epsilon},
\]

其中 \(\bar c_g\) 是由真实文件字节归一化得到的 GLB 成本。含有任一 GT 可见实例的 GLB 不受此排斥项处罚，避免粗粒度下载信号污染实例级遮挡判别。该项减少“误报实例带来的额外请求字节”，不替代已有视觉效用和下载优先级头。

### 安全门控和梯度投影

总损失定义为

\[
L=L_{\mathrm{RVL\ strong\ v2}}+\lambda_qL_{\mathrm{quality}}+
g\,\lambda_rL_{\mathrm{resource}}+L_{\mathrm{survival}}+L_{\mathrm{utility}}+L_{\mathrm{download}}.
\]

资源门 (g) 使用训练期质量召回的 detached EMA，并在 warm-up 后连续开启；它只决定是否优化资源，不能当作 calibration 安全证书。安全梯度定义为 (g_s=\nabla(L_{\mathrm{RVL}}+\lambda_qL_{\mathrm{quality}}))，资源梯度为 (g_r=\nabla(\lambda_rL_{\mathrm{resource}}))。当两者内积小于零时，投影掉与安全梯度冲突的分量，再限制 ‖(g_r)‖ 不超过 `0.25 × ‖g_s‖`。本轮固定 `lambda_q=0.5`，pilot 只比较 `lambda_r ∈ {0.05, 0.10}`；若两者均通过门，依次以 validation 额外无需求 GLB 字节更少、balanced accuracy 更高、precision 更高作为 tie-break。RVL 内部权重不重新扫描。

训练记录必须包含质量风险、稀有正例尾部、资源项、门控比例、投影触发率、投影前后梯度范数、软质量召回和预测 GLB 字节。正式安全仍由每个 checkpoint 自己的 calibration 阈值和 weighted recall 单侧置信下界决定。

完整联合训练固定沿用当前强参考的辅助项和监督：生存删失似然权重 `0.25`，实例视觉效用回归/排序权重 `0.18`，GLB 必需性与效用/字节排序权重 `0.20`，参数正则权重 `1e-5`。实例视觉效用目标由 pose 内 `log1p(visible_weights)` 归一化得到；GLB 排序目标按同一 GLB 中 GT 可见实例的效用和真实字节成本构造，训练时比较必需 GLB 与无需求 GLB。正式评价在候选 GLB 字节预算的 `10%/25%/50%` 三个固定档位报告 utility recall 与 NDCG，并报告达到完整模型同等视觉效用所需的绝对字节。留一变体不得修改这些辅助损失和监督。

### 与已有损失试验的区别

- 旧安全约束损失用对偶变量整体推高召回，正式结果出现明显过预测；新损失保留已验证的 RVL 主干，只在危险质量尾部增加保护。
- 旧校准/weighted-tail pilot 主要围绕固定 `0.5` 锚点和普通 `visible_weights` 尾部；新路线不把 `0.5` 设为安全硬门，显式加入合法 subpose 稀有度和无需求 GLB 字节。
- 旧选择性纠错直接改变最终 logit，容易以漏检换取剔除；新资源梯度必须经过安全余量门控和冲突投影。
- calibration 和 test 规则不进入损失反向传播。训练期 EMA、温度和 margin 都不能描述成统计置信保证。

## 联合模型的数据流

```text
离线 GLB 表面点与实例变换
        │
        ├─ PointNet++ 风格几何编码 ───────────────→ 96 维实例几何
        │
train-only 三角形深度层与表面深度补全
        │
        └─ 观测覆盖内、不做来源 top-k 截断的有向遮挡关系图
                  │
                  └─ 分层关系编码与逐实例解码 ───→ R×7 生存场系数

浏览器后退相机候选 + 当前 view-cell 范围
        │
        └─ 九维视角均值与区域不确定度
                  │
                  └─ 视点区域积分频谱查询 ───────→ R 维共享查询基
                                                       │
96 维几何 + 生存场语义 + 九维直接视角 ───────────────┼─→ 可见性
                                                       ├─→ 视觉效用
                                                       └─→ GLB 下载优先级
```

这条路径形成清晰分工：分层关系网络离线回答“场景中哪些结构会在不同方向和深度挡住该实例”；频谱查询在线回答“当前 view-cell 应当如何读取这份遮挡分布”；统一输出头负责在 weighted recall 安全约束下决定显示和下载。

## v1 验证协议摘要

v1 依次实现并快速检查了关系来源、频谱查询和质量/资源损失三个矩阵。关系矩阵包含自由生存场、单尺度真实关系、分层真实关系和来源置乱；频谱矩阵包含 Fourier117、可学习中心点查询和区域积分查询；损失矩阵包含 RVL、质量尾部以及两档资源项。完整成员定义、命令和冻结 manifest 保留在对应代码与结果目录中，不再复制为后续待办。

该协议最初使用单 seed、8 epoch pilot，随后对 R0/R2 执行双 seed、32 epoch 学习曲线。结果已经证明“继续增加 epoch”不能稳定解决分层关系模型的过预测，因此旧 `module_recheck16` 和旧 `formal80` 均未启动。后续代码修复、超参数扫描、全组合长训和消融只按新的 v2 执行计划进行。

## 2026-08-13 公共频谱查询模块实现记录

本次先实现候选创新二的独立公共算子，不接入当前默认模型、训练 runner、旧 exporter 或前端资产。实现文件为 `neural_instance_culling/model/common/viewcell_integrated_spectral_query.py`，公共 schema 为 `view-cell-integrated-spectral-query-v1`。

- 输入：一次候选批次的九维中心视角量，以及由当前 view-cell 半径和实例距离解析得到的九维对角方差；默认遵循半径 `2 m`、固定朝向、模型 FOV `66°` 的现行契约。
- 频谱：固定 `K=16` 个共享可学习九维联合频率；每个频率输出 Gaussian 积分后的 sin、cos 和未解析相位能量，完整频谱维度为 `9 + 16 * 3 = 57`。
- 查询：低频主体、中频过渡和高频边界分支由连续 sigmoid 门控合并，关系生存场语义可作为八维条件输入；可选的 `R x 7` 生存系数只做一次单批次字段查询。
- 运行边界：模块没有 subpose 展开、邻居搜索、图传播或 AABB 八角点投影；零方差严格退化为中心相位查询。
- 测试：`neural_instance_culling/model/common/tests/test_viewcell_integrated_spectral_query.py` 覆盖有限性、零不确定度、积分单调性、解析方差、关系生存条件、频率分组和导出 schema。

该记录只证明当时的公共算子和单元测试，不证明模型训练收益、weighted recall 安全门、浏览器 WebGPU parity 或移动端性能；后续验证以 v2 唯一执行计划为准。

## 2026-08-13 训练专用关系 CSR 与最小分层编码器实现记录

本次先完成候选创新一的独立训练代码线，不接入当前默认模型、训练 runner、旧 `ray-context-relation-evidence-v2` 接口、旧 exporter 或前端资产。实现文件为 `neural_instance_culling/model/common/train_observed_relation_csr.py`、`neural_instance_culling/model/common/hierarchical_relation_survival.py`，测试为 `neural_instance_culling/model/tests/test_train_observed_relation_csr_hierarchy.py`。

- **CSR/schema 输入与输出：** 输入显式的候选 ID、pose 顺序和 CSR offsets，以及 train-only 三角形深度层/表面补全的有向关系行；输出按 `target instance → direction bin → depth shell` 排列的独立二进制 CSR。每条边保留 source instance、pose/像素支持、前后深度间隔及其统计、相对中心/尺度、证据置信度和来源类型；同一行不允许重复 source，且当前 schema 明确不做 source top-k 截断。
- **校验边界：** 元数据必须声明 `pvs-viewcell-train-observed-relation-csr-v1`、`trainOnly=true`、仅含 `train` split、候选 canonical/render 语义、输入路径与 shape、深度层数、FOV 和字段语义。校验器拒绝 split 泄漏、候选口径不一致、source/target 越界、自环、非正深度间隔、非法方向/深度桶以及把 event 与 right-censored observation 混用；候选校验只读取 native candidate CSR，不读取或并入 GT visible IDs。
- **分层编码器输入与输出：** 输入固定实例几何特征、经过 schema 校验的 CSR，以及 `instance → local group → structural group` 的连续映射；先按 target/direction/depth segment 聚合有向 source→target 消息，再做实例层、局部遮挡群层、结构群层编码和逐实例解码，固定输出 `[N, 4, 7]` 生存场系数。方向和深度桶在实例更新前保持显式，避免把相反方向或不同深度顺序混成无向池化。
- **依赖资源与运行命令：** 依赖当前已登记的 train 深度缓存、surface fallback 关系、candidate CSR 和实例几何表；本次只运行 `conda run -n slm_pvs python -m unittest neural_instance_culling.model.tests.test_train_observed_relation_csr_hierarchy -v`、旧证据回归测试和 `conda run -n slm_pvs python -m py_compile ...`，没有启动训练、导出、浏览器采样或 benchmark。
- **测试结果：** 新增关系/分层测试 `8/8` 通过；既有 `test_triangle_depth_layer_evidence.py` 与 `test_ray_context_relation_evidence.py` 回归测试 `10/10` 通过。schema 测试中的“候选摘要/候选顺序”是候选一致性指标，“source/target 深度顺序”是关系语义校验；编码器输出形状、有限性、分层 segment 计数、方向反转后的 target 归属、反向传播和生存概率随距离单调不增是网络结构测试，不是画面安全指标。

该阶段之后已经完成关系 CSR builder 接线、三层图映射、关系生存删失损失、统一训练入口和 R0--R3 pilot。仍未完成的正式验证包括容量匹配关系消融、正确的下载头评价、修正后的导出 schema、Color-ID 图像评价和硬件 WebGPU 性能；因此不能据此修改默认 checkpoint、阈值或前端资产。

## 2026-08-13 Pilot 执行结果与阶段判定

本轮已完成三个独立 pilot：关系来源矩阵（R0--R3）、频谱查询矩阵（S0--S2）和损失矩阵（L0--L3）。每个矩阵使用 seed `20260801`、8 epoch、相同的 213 个 validation pose 和相同的原生候选集合；三个汇总均保留 10,000 次配对 bootstrap 字段。本轮没有读取 test split，完整数据与指标见独立的 [pilot 诊断报告](../evaluation/pvs_hierarchical_relation_survival_integrated_spectral_quality_rvl_v1_pilot.md)。

训练入口在 pilot 阶段用 32 个 calibration pose 快速冻结阈值，因此 calibration 中若干成员的 weighted recall 及其下界超过 `0.99` 只表示快速筛选通过，不能推广到完整 validation。冻结各自阈值后重放全部 213 个 validation pose，最高 weighted recall 为单尺度关系 `R1_single_scale_relation` 的 `0.883763`；积分频谱 `S2_integrated_spectral` 为 `0.878301`。关系、频谱和损失矩阵所有成员均低于 `0.99`，所以没有合格的正式安全工作点。本轮不能直接进入 `formal80`，也不能据此替换默认模型、阈值或前端资产。

快速结果仍有诊断价值：分层关系版本 `R2` 相比自由生存场 `R0` 提升 aggregate precision `+0.0366`，但 weighted recall 下降 `-0.0216`；区域积分频谱 `S2` 相比可学习频谱点查询 `S1` 的 weighted recall 仅提升 `+0.00088`，precision 下降 `-0.00168`；质量尾部风险 `L1` 相比 RVL 控制的 weighted recall 提升 `+0.00184`，precision 差异未形成稳定证据；资源排斥版本 `L2/L3` 虽提高 weighted recall 约 `+0.052`，却显著增加平均预测数和预测 GLB 字节。useful cull、precision 或平均预测数的单项变化不能抵消完整 validation 安全门失败。

阶段判定为：实现、单元测试、schema self-test、pilot 和 32 epoch 学习曲线均已完成。分层关系模型可以在完整 calibration 下通过 validation 安全门，但效率未稳定改善，32 epoch 晋级门失败；完整结果见[学习曲线评价](../evaluation/pvs_hierarchical_relation_survival_learning_curve_e32_2026-08-13.md)，后续只按[v2 唯一执行计划](pvs_full_innovation_hyperparameter_scan_longtrain_ablation_2026-08-13.md)先修复损失和评价契约，再扫描全组合。不得通过降低 weighted recall 要求、共享阈值、补入 GT 或直接启动长训来掩盖该问题。默认模型和前端资产继续保持不变。

## 可形成的论文贡献

若上述验证通过，论文可将模型贡献概括为：

1. **分层遮挡关系烘焙。** 使用真实三角形有向遮挡图进行多尺度编码与逐实例恢复，把实例形状、相对位置和遮挡来源直接压缩成连续、距离单调的逐实例生存场；重型关系传播全部离线执行。
2. **视点区域边界频谱查询。** 使用可学习联合频率和区域积分衰减，在单次查询中同时表达中心方向、可解析高频和 view-cell 边界不确定性，并用该编码查询离线生存场，而非把高维 Fourier 特征直接交给通用分类头。
3. **视点区域质量风险与资源排斥 RVL。** 在保留 RVL 高召回主干的基础上，用屏幕质量和合法子位姿稀有度保护危险正例，并对无需求 GLB 的额外请求字节施加安全门控的排斥，使训练目标直接对应画面风险与下载成本。

移动端友好的统一调度是三个创新共同满足的系统属性：浏览器只加载紧凑实例几何与生存系数，用同一轻量查询同时产生实例可见性、视觉效用和 GLB 下载优先级。它不替代第三项损失创新。

当前只有“Fourier117 明显优于 direct9”“生存场具有较强分类/剔除信号”和“RVL 主干具有高召回作用”获得了正式实验支持。分层真实关系生成、视点区域积分频谱以及质量风险与资源排斥扩展均是待验证机制；在快速复验、80 epoch 三种子长训和留一消融完成前，三者都不能写成已证实贡献，也不能替换默认模型。

## 相关研究与借鉴边界

- [Graph U-Nets](https://proceedings.mlr.press/v97/gao19a.html)：借鉴图编码器/解码器、池化/反池化和逐节点恢复；本方案改用真实有向遮挡边并输出实例生存场。
- [Hierarchical Graph Representation Learning with Differentiable Pooling](https://arxiv.org/abs/1806.08804)：支持学习或构造多尺度图表征；本方案优先使用可审计的离线关系粗化，避免前端动态图池化。
- [Superpoint Transformer](https://openaccess.thecvf.com/content/ICCV2023/html/Robert_Efficient_3D_Semantic_Segmentation_with_Superpoint_Transformer_ICCV_2023_paper.html)：说明大型三维场景中的层次关系表征可以保持紧凑；其任务和在线结构不直接复用。
- [Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains](https://arxiv.org/abs/2006.10739)：解释 Fourier 特征缓解小型 MLP 频谱偏置的原因。
- [Mip-NeRF](https://openaccess.thecvf.com/content/ICCV2021/html/Barron_Mip-NeRF_A_Multiscale_Representation_for_Anti-Aliasing_Neural_Radiance_Fields_ICCV_2021_paper.html)：借鉴对区域内 Fourier 响应做解析积分和按尺度衰减高频；本项目保留未解析能量，以符合 view-cell 可见并集的保守语义。
- [Ref-NeRF](https://openaccess.thecvf.com/content/CVPR2022/html/Verbin_Ref-NeRF_Structured_View-Dependent_Appearance_for_Neural_Radiance_Fields_CVPR_2022_paper.html)：借鉴球面方向基随不确定尺度衰减高阶响应的思想；本项目预测遮挡生存而非反射外观。
- [Learnable Fourier Features for Multi-Dimensional Spatial Positional Encoding](https://arxiv.org/abs/2106.02795)：支持用可学习联合频率替代完全固定、逐坐标的位置编码。
- [WIRE: Wavelet Implicit Neural Representations](https://openaccess.thecvf.com/content/CVPR2023/html/Saragadam_WIRE_Wavelet_Implicit_Neural_Representations_CVPR_2023_paper.html)：为局部 Gabor/小波边界基提供后续参考；该分支暂不进入首轮实现。
