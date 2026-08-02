# NeuralStreamWeb3D 2026 年 9 月投稿研究与实施规划

初版日期：2026-07-31  
本次审计更新：2026-08-01  
状态：投稿前研究设计，尚未形成可投稿结论  
主目标：Eurographics 2027，2026-09-25 摘要登记，2026-10-01 全文截止  
严格九月全文备选：滚动期刊，依据最终成果侧重点选择 The Visual Computer、IEEE Transactions on Multimedia 或 Multimedia Systems

## 1. 执行结论

NeuralStreamWeb3D 当前已经形成完整原型：对离线预处理的静态实例化三维场景，使用较重的点云和关系编码器生成固定实例特征；浏览器只读取固定特征表，以当前相机到实例的视线方向及少量射线空间标量执行 WebGPU 查询；模型共享实例表征，同时输出实例可见性、视觉效用和可下载 GLB 资源优先级。前端使用 66° 后退相机产生模型候选和预取结果，再以真实 60° 相机按实例收紧最终显示集合。

当前证据可以支持一个有限结论：在 HKUST 的场景内随机划分上，该方法能够在较高重要构件召回下缩小候选实例和候选 GLB 字节量。当前证据不能支持跨场景泛化、图像近似无损、移动端实时、方向遮挡代理具有独立因果贡献、联合下载头优于可见性后处理等主张。

投稿前必须先解决六个阻断项：

1. 当前阈值 `0.64` 是在完整测试集上扫描得到的，现有结果属于测试集校准的探索性结果。
2. 每轮验证使用变化的随机验证子集，不同 checkpoint 没有在完全相同的验证样本上比较。
3. 显式方向遮挡抑制输出长期接近零，遮挡代理的实际贡献尚未建立。
4. 当前 view-cell 随机划分没有空间隔离，不能排除相邻视点泄漏和场景记忆。
5. 正式实例级同位姿图像评价和真实移动设备 WebGPU 评价尚未完成。
6. Metropolis 的候选正样本补入、点云缓存和 `instance_to_glb` 索引语义存在冲突，暂不能作为论文主表证据。

因此，现有 HKUST `0.64` 结果只能放在研究日志或论文的 preliminary evidence 中，不能作为最终 test 主结果。正式协议必须是：完整且固定的验证/校准集合选择 checkpoint 和阈值，冻结后测试集只运行一次。

## 2. 当前系统与探索性证据

### 2.1 当前方法

当前模型对每个实例离线生成 352 维半精度浮点特征：96 维几何特征、64 维场景上下文特征和 `8 个方向 × 3 个深度层 × 8 维` 的 192 维方向遮挡代理。运行时对候选实例构造单位视线方向、距离、屏幕相对位置和角尺度，使用轻量查询头输出：

- 实例可见性分数，用于实例级显示过滤；
- 实例视觉效用分数，用于估计当前画面价值；
- GLB 下载分数，用于请求、解码和缓存排序。

方向遮挡代理的监督来自训练视点中的真实可见集合、实例包围盒屏幕重叠和投影深度关系。其准确名称应为“投影几何弱监督的方向遮挡代理”，不能称为真实深度缓冲、神经层次深度缓冲、精确遮挡场或无监督遮挡发现。

前端当前仍由模型分数与规则调度共同完成下载：模型提供分数，规则层负责立即请求/预取队列、并发、缓存、取消和过期处理。因此论文不能声称下载完全由模型端到端控制。

### 2.2 端侧冷启动资源边界

本项目所说的“没有提前拿到场景中模型数据”，必须定义为：浏览器作出第一轮可见性与下载决策时，目标 GLB 的三角形、材质和纹理尚未下载、解码或上传 GPU。端侧仍会先获得一份远小于完整模型资产的启动数据，包括：

- 实例编号、AABB 和实例到 GLB 的映射；
- 场景空间索引或特征页目录；
- 当前区域需要的固定实例特征页；
- 与场景共享或按场景导出的轻量查询头权重。

因此准确论文术语是“模型几何资产到达前的冷启动推理”或“无需运行时原始几何的端侧查询”，不能写成“端侧完全没有任何场景数据”。如果完整固定特征表必须先整体下载，它本身也是冷启动字节成本，必须从节省的 GLB 字节中扣除。

系统应区分三个运行阶段：

| 阶段 | 端侧可用数据 | 允许的方法 | 主要问题 |
|---|---|---|---|
| Cold-0 | 查询头、实例 AABB/映射、当前特征页，无目标 GLB 几何 | 本方法、AABB/统计/bitset 基线 | 第一轮剔除和下载顺序 |
| Cold-1 | 已下载少量优先 GLB，部分几何驻留 | 本方法与部分几何方法混合 | 渐进完善画面和预取 |
| Warm | 所需几何大部分驻留 | 标准 HZB、CHC++ 等 | 稳态渲染剔除 |

论文主问题发生在 Cold-0/Cold-1。Warm HZB 是质量和稳态性能上界，不是“尚未下载几何时”的同条件竞争者。

### 2.3 可见性到下载效用的级联输出

代码中的三个输出具有明确有向关系。设当前视线查询后的实例特征为 `q_i`，则当前实现可概括为：

```text
p_i = sigmoid(visibility_head(q_i))
u_i = sigmoid(utility_head(q_i, p_i))
d_i = sigmoid(download_head(q_i, p_i, u_i))
D_g = max(d_i), i 属于 GLB g
```

其中 `p_i` 是实例可见性概率，`u_i` 是根据可见性概率和实例查询特征预测的视觉效用，`d_i` 是同时读取可见性和视觉效用的实例下载分数，`D_g` 是前端当前使用的 GLB 级最大聚合优先级。视觉效用监督目前由 pose 内可见权重 `log(1+w_i)` 归一化得到；下载监督把同一 GLB 的可见需求、可见权重和资源成本用于排序。

该数据流可由模型中的级联计算（`compute_logits_with_aux`、`compute_utility_logits`、`compute_download_logits`）和前端 `InstancePVS` 的 GLB max 聚合直接复核，论文实现图应与这条真实代码路径一致。

这条级联是本项目“算-传统一”的核心，不应只描述成普通共享干路多任务学习。与此同时，当前 MLP 并不保证视觉效用随可见性单调，也不保证 GLB 最大聚合是预算最优。投稿前应比较当前自由级联与更可解释的受约束分解：

```text
u_i = p_i * salience_i, salience_i >= 0
U_g = 1 - product(1 - u_i), i 属于 GLB g
priority_g = scheduler(U_g, bytes_g, decode_ms_g, cache_g)
```

该分解让不可见概率直接抑制即时视觉效用，同时用饱和聚合避免同一 GLB 中大量低效用实例简单累加。是否采用它由 validation/calibration 上的调度和图像实验决定，不能只凭叙事替换当前实现。

### 2.4 HKUST 探索性结果

当前 HKUST 数据含 7,999 个 view-cell，划分为 6,585/730/684 个训练、验证和测试 view-cell；测试集中 684 个 view-cell 唯一且候选不截断。现有测试集校准工作点为：

| 指标 | 数值 |
|---|---:|
| 阈值 | 0.64 |
| Pose precision | 0.8156 |
| Pose recall | 0.8680 |
| Pose F1 | 0.8160 |
| Weighted recall | 0.990071 |
| Instance accuracy | 0.9533 |
| Balanced accuracy | 0.9046 |
| Useful cull | 0.87897 |
| Bad cull | 0.00959 |
| 平均候选/GT/预测实例 | 4640.57 / 103.25 / 186.63 |
| GLB 字节削减 | 0.85837 |

这些数字说明模型更倾向保护屏幕贡献高的实例，但普通 pose recall 仍只有 0.8680，aggregate recall 为 0.7560。`weighted recall=0.990071` 不能被解释为所有构件或所有像素接近无损，其高于 0.99 的裕量也只有约 `0.000071`。

### 2.5 结果可信度分级

| 结论 | 当前可信度 | 原因 |
|---|---|---|
| HKUST 场景内模型可缩小候选集合 | 中等 | 有完整 684 view-cell 评测，但阈值使用 test 校准 |
| 重要构件得到较高保护 | 中等 | weighted recall 较高，但缺图像长尾和置信下界 |
| 方向遮挡代理有效 | 低 | 显式抑制均值约 `1e-9` 到 `1e-7`，缺因果干预 |
| 下载头优于后处理 | 低 | 缺同预算启发式和独立排序器基线 |
| 跨区域/跨场景泛化 | 未建立 | 当前随机 split 且每场景独立训练 |
| 移动端轻量实时 | 未建立 | Metropolis 10,793 候选曾记录约 3.9 秒浏览器 smoke |
| 图像近似无损 | 未建立 | 正式 instance-ID/RGB 同位姿评价未运行 |

## 3. 研究问题

### 3.1 主研究问题

> 对离线预处理的大规模静态实例化三维场景，在目标 GLB 几何、材质和纹理尚未到达端侧时，固定场景特征页与轻量视线条件查询，能否相对于同信息条件的传统和学习型基线，在客户端计算、启动内存和下载字节预算下取得更优的可见图像效用 Pareto 前沿？

这里的“固定场景特征表”输入场景实例几何、上下文和方向遮挡证据，输出浏览器可分页读取的实例表征；“视线条件查询”输入当前相机与候选实例的射线空间关系，输出实例显示价值和 GLB 下载价值；“图像效用 Pareto 前沿”同时考察画面损失、有效剔除、下载字节和运行延迟，不由单个 F1 决定。

### 3.2 子问题与可检验假设

| 编号 | 子问题 | 可检验假设 |
|---|---|---|
| RQ1 | 方向和上下文编码是否真实改善安全剔除？ | 在同一验证校准规则下，完整代理相对几何+视线基线提高 useful cull 或降低同图像效用所需字节，且三种子置信区间不跨零 |
| RQ2 | 可见性到视觉效用再到下载分数的级联调度，是否优于可见性后处理和独立排序器？ | 在相同字节或时间预算下，级联头的视觉效用曲线面积更高，达到 95%/99% 效用所需字节或时间更少 |
| RQ3 | 方法能否泛化到未见空间区域和未见场景？ | 空间块留出显著优于位置/频率记忆基线；跨场景零样本和少样本适配结果可量化且不通过邻近视点泄漏获得 |
| RQ4 | 浏览器实现的实际扩展边界是什么？ | 候选生成、WebGPU 查询、回读、调度和实例矩阵更新在不同候选规模下具有可解释曲线，分页和紧凑回读可降低启动与 p95 延迟 |

### 3.3 FINER 可行性评估

FINER 分别表示可行性、趣味性、新颖性、伦理性和相关性。

| 维度 | 判断 | 约束 |
|---|---|---|
| 可行性 | 条件可行 | 已有模型、数据和 WebGPU 原型；八周内必须收缩到两项核心贡献，四块 GPU 并行训练 |
| 趣味性 | 高 | 学习型可见性与 Web3D 资源流送之间存在明确系统问题 |
| 新颖性 | 条件成立 | “固定特征+视线 MLP”本身不新；新颖性必须来自方向代理的因果价值和预算感知统一调度，或来自完整浏览器系统 Pareto 结果 |
| 伦理性 | 低风险 | 无人类受试者；需要核对公开数据许可、模型资产许可和设备测量记录 |
| 相关性 | 高 | 对大规模 Web3D、远程 BIM/数字孪生和移动端渐进加载直接相关 |

## 4. 文献定位与研究缺口

### 4.1 检索协议

本轮检索截至 2026-07-31，使用 ACM Digital Library、IEEE Xplore、Wiley、Crossref、arXiv、W3C 和 OGC 官方记录交叉核验。核心检索组合包括：

- `potentially visible set neural from-region visibility`；
- `point set neural visibility ray direction`；
- `hierarchical z buffer occlusion culling`；
- `view-dependent progressive mesh web streaming`；
- `WebGPU browser neural inference latency`；
- `learning to rank cost budget utility byte`。

纳入条件是工作直接涉及潜在可见集、遮挡、视点区域、渐进三维流送、浏览器推理或预算排序；排除只解决单帧离线图像分类、无三维场景关系或无法核验正式来源的材料。

### 4.2 关键文献矩阵

| 方向 | 工作 | 与本项目的关系 |
|---|---|---|
| 图形学遮挡 | Hierarchical Z-Buffer Visibility, SIGGRAPH 1993, [DOI](https://doi.org/10.1145/166117.166147) | 完整几何 warm-cache 遮挡基线；不能用 AABB 深度代理冒充 |
| 图形学遮挡 | CHC++, CGF 2008, [DOI](https://doi.org/10.1111/j.1467-8659.2008.01119.x) | 时空相干遮挡查询基线和延迟参考 |
| 可见性集合 | Hardly-Visible Sets, CGF 2000, [DOI](https://doi.org/10.1111/1467-8659.00442) | 已把低可见度、像素贡献和资源简化联系起来，限制“首次统一”主张 |
| 点集可见性 | Direct Visibility of Point Sets, TOG 2007, [DOI](https://doi.org/10.1145/1276377.1276407) | 点集直接可见性的经典图形学参照 |
| 视点区域 | Camera Offset Space, TOG 2019, [DOI](https://doi.org/10.1145/3355089.3356530) | 分析相机区域变化下的可见性稳定性 |
| 视点区域 | Guided Visibility Sampling++, PACM CGIT 2021, [DOI](https://doi.org/10.1145/3451266) | 视点区域采样和 PVS 构建基线 |
| 视点区域 | Trim Regions, TOG 2023, [DOI](https://doi.org/10.1145/3592434) | 强 from-region 可见性基线，需计完整几何和构建成本 |
| 神经 PVS | NeuralPVS, SIGGRAPH Asia 2025, [DOI](https://doi.org/10.1145/3757377.3763822) | 已提出神经 from-region PVS，不能声称首个神经 PVS |
| 神经可见性 | Neural Visibility of Point Sets, SIGGRAPH Asia 2025, [DOI](https://doi.org/10.1145/3757377.3763869) | 已使用视角无关点特征和视线条件查询，限制单纯架构新颖性 |
| 动态可见性 | Disocclusion Buffer, SIGGRAPH Asia 2025, [DOI](https://doi.org/10.1145/3757377.3763981) | 可作为增强的动态可见性对照，需完整资源成本说明 |
| 渐进几何 | Progressive Meshes, SIGGRAPH 1996, [DOI](https://doi.org/10.1145/237170.237216) | 视相关几何渐进传输的基础 |
| 渐进流送 | Receiver-driven View-dependent Progressive Mesh Streaming, [DOI](https://doi.org/10.1145/1496046.1496049) | 从客户端视点与网络预算驱动渐进请求的直接参照 |
| Web 三维 | Nexus, Graphical Models 2016, [DOI](https://doi.org/10.1016/j.gmod.2016.09.002) | 大规模 Web 几何的多分辨率和流送系统参照 |
| 标准流送 | OGC 3D Tiles, [标准](https://www.ogc.org/standard/3dtiles/) | 屏幕空间误差层次调度基线 |
| Web 推理 | WebInf, IEEE ICPADS 2023, [DOI](https://doi.org/10.1109/ICPADS60453.2023.00333) | 浏览器端推理执行和端云资源约束参考 |
| Web 推理 | WeInfer, ACM 2025, [DOI](https://doi.org/10.1145/3696410.3714553) | Web 端推理性能与执行系统参考 |
| 排序学习 | RankNet, ICML 2005, [DOI](https://doi.org/10.1145/1102351.1102363) | 成对 GLB 排序基线 |
| 排序学习 | ListNet, ICML 2007, [DOI](https://doi.org/10.1145/1273496.1273513) | 列表级下载排序基线 |
| 预算排序 | Cost-Sensitive Learning to Rank, AAAI 2019, [DOI](https://doi.org/10.1609/aaai.v33i01.33014570) | 将视觉效用与字节/时间成本统一到预算排序 |
| 浏览器计算 | WebGPU, W3C, [规范](https://www.w3.org/TR/webgpu/) | 浏览器实现与可复现兼容性依据 |

### 4.3 文献脉络综合

传统遮挡剔除以 HZB、硬件查询和时空相干为代表，优势是几何关系精确、warm-cache 性能成熟，前提是遮挡体和被遮挡体的几何已经可用。离线 PVS 和 view-cell 方法把昂贵计算前移，用位图或区域可见集换取运行时快速查询，但存储随场景、cell 数和方向分辨率增长，对 cell 外视点适应有限。

NeuralPVS 证明了神经网络可以学习 from-region PVS，Neural Visibility of Point Sets 证明了视角无关点特征结合视线条件查询可以预测可见性。这两类工作削弱了“固定特征+视线查询”本身的新颖性，同时说明用学习表示压缩可见性知识具有研究基础。它们没有直接回答本项目中的两个工程粒度：实例是否显示，以及尚未到达的 GLB 是否值得优先下载。

Progressive Meshes、Nexus、3D Tiles 和接收端驱动流送主要依据几何误差、视点和层次组织资源，擅长在可用几何层次中选择 LOD 或 tile。它们通常不学习大规模实例间遮挡，也不把潜在可见集合的安全约束作为下载排序的第一目标。Hardly-Visible Sets 已经指出像素贡献和资源简化应联合考虑，因此本项目不能声称首次连接可见性与资源价值，只能把贡献落到冷启动浏览器约束下的统一可执行模型。

RankNet、ListNet 和成本敏感排序提供了从实例/GLB 列表中学习预算顺序的方法；WebInf、WeInfer 和 WebGPU 相关工作提供了浏览器推理与算子实现背景。它们分别缺少三维遮挡语义或场景流送目标。本项目的研究机会位于这些方向交叉处：用紧凑场景知识在原始几何到达前预测可见性，再将可见性显式传入视觉效用和成本感知下载级联，并在自定义浏览器运行路径上验证净收益。

### 4.4 研究缺口

基于当前已核验文献，不能把“神经 PVS”“视角无关点特征+视线查询”或“可见性影响资源选择”单独作为新贡献。更可辩护的缺口是：现有神经可见性工作与大规模 Web3D 的实例级显示、GLB 粒度资源成本、冷缓存加载和浏览器端算子约束之间仍缺少一套经过端到端验证的统一方案。

本项目应验证的组合缺口为：

1. 如何把场景特定的几何、上下文和方向遮挡知识压缩为浏览器可分页的固定实例表，使端侧在目标 GLB 几何到达前作出第一轮决策，而不执行在线点云编码；
2. 如何通过 `可见性 -> 视觉效用 -> 下载分数` 的显式级联保持实例级显示与 GLB 级下载两个粒度，并优化预算内可见图像效用；
3. 如何把启动元数据和神经特征页自身的下载成本也计入收益，在 WebGPU、CPU 候选生成、GPU 回读、网络和解码共同约束下报告端到端 Pareto，而不是只报告离线二分类指标。

### 4.5 主张边界

可以在证据通过后声称：

> 对离线预处理的静态实例化场景，系统将场景特定几何、上下文和投影几何弱监督的方向遮挡知识压缩为可分页固定实例表，使浏览器在目标 GLB 几何资产到达前，通过轻量视线条件查询和 `可见性 -> 视觉效用 -> 下载分数` 级联，为实例级安全可见性过滤与 GLB 级预算感知流送提供统一决策依据。

不得声称：

- 首个神经 PVS；
- 首个视角无关特征与射线查询模型；
- 零场景先验或运行时不依赖任何离线场景知识；
- 总运行时间、总资产或总内存与实例数量无关；
- 在没有真实设备测量前达到移动端实时；
- weighted recall 单独证明画面无损；
- AABB 深度代理等价于标准几何 HZB；
- 当前规则和模型共同调度等价于完全端到端神经下载。

运行时单实例输入维度与屏幕分辨率无关，但总特征大小和总计算仍随活动实例或候选实例数量增长。

当前自定义 WGSL 查询不依赖 ONNX Runtime、TensorFlow.js 等通用模型框架，这是包体和算子融合方面的工程特性。只有在与通用 Web 推理框架比较后证明其显著降低启动字节、初始化和 p95 查询延迟，才能作为系统贡献；不能把“没有使用现有模型框架”本身写成算法创新。

## 5. 贡献路线与失败分支

### 5.1 路线 A：方向代理成立

只有当代理干预和重训练消融证明方向关系具有独立收益时，论文贡献写为：

1. 固定场景知识压缩：把实例几何、上下文和方向遮挡证据离线编码为可部署实例表；
2. 方向遮挡代理：以投影几何弱监督学习相机无关的方向/深度代理，并由当前视线轻量选择；
3. 级联双粒度统一调度：可见性概率进入视觉效用头，可见性与效用共同产生下载分数，再聚合为 GLB 预算价值；
4. 几何资产到达前的浏览器系统：自定义 WebGPU 查询、实例级渲染过滤、特征分页和预算调度的端到端实现。

### 5.2 路线 B：方向代理不成立

若代理清零、置换或删除后没有稳定退化，必须删除“方向遮挡代理有效”的核心主张，简化或移除显式抑制分支。论文转为系统路线：固定特征分页、轻量射线查询、实例/GLB 双粒度调度和 WebGPU 端到端 Pareto。该路线不再以 Eurographics 的模型新颖性为主要卖点，应优先考虑 The Visual Computer、IEEE Transactions on Multimedia 或 Multimedia Systems。

路线选择必须由 2026-08-16 前完成的因果消融决定，不能在写作阶段用措辞掩盖失败结果。

## 6. 正式数据与划分协议

### 6.1 数据审计先行

正式训练前必须生成逐场景数据审计报告并满足：

- `visible_ids` 严格属于前端可执行的后退视锥 `candidate_ids`，不能依赖运行时无法复现的 GT 强制补入；
- 实例 AABB、实例到 GLB 映射、点云缓存行语义和固定特征行语义一致；
- HKUST 3,273 行点云缓存明确记录为 GLB 原型缓存，并通过 `instance_to_glb` 解析；
- Metropolis 41,298 行点云缓存必须明确是实例缓存还是 GLB 原型缓存。当前代码按 `instance_to_glb` 索引，若它是实例缓存则现有训练读取语义错误；
- Metropolis 的 `candidateMissVisible=228,974` 必须消除或证明候选生成在前端可以同样补入；否则该场景不能进入主表；
- 60°真实相机、66°采样/模型相机、宽高比和后退距离只从统一配置读取。

### 6.2 四路空间隔离划分

随机 view-cell split 只保留为探索性对照。正式数据按空间块和方向分层生成四路划分：

1. `train`：训练离线编码器和查询头；
2. `validation`：完整固定集合选择 checkpoint，不随 epoch 更换子集；
3. `calibration`：选择阈值和安全工作点；
4. `test`：冻结模型和阈值后只执行一次。

空间块大小应大于 view-cell 位置扰动直径，并尽量沿建筑/街区边界划分。相邻块不能跨集合共享；每个集合按街道缝隙、建筑近旁、广场、外围、天空、远景和方向扇区分层。生成后保存不可变 manifest、随机种子和 SHA-256。

方向泛化另设一个不参与主阈值选择的方向留出集，检查同一空间位置下未见 yaw/pitch 扇区。若只有两个独立大场景，只能报告双向跨场景迁移试验；若要声称通用跨场景编码器，至少需要三个具有不同几何和实例分布的大场景。

### 6.3 checkpoint 与阈值冻结

正式协议如下：

```text
固定完整 validation：选择 checkpoint
独立 calibration：要求 weighted recall > 0.99，再最大化 pose precision
冻结 checkpoint、阈值、量化和全部后处理参数
test：只运行一次并报告全部指标，不再扫描、不再回退
```

为降低仅高出 0.99 极小裕量的风险，calibration 使用预注册安全裕量：点估计至少 `0.9925`，且按 view-cell 聚类 bootstrap 的单侧 95% 下置信界大于 0.99。论文主安全结论要求 test 的 weighted recall 点估计严格大于 0.99；若 test 下置信界也大于 0.99，可进一步声称统计意义上的稳定满足。若 test 未达到阈值，只能如实报告失败，不能回到 test 重新调阈值。

## 7. 基线与公平比较

基线先按端侧可用信息分层，再在层内比较质量和成本：

| 信息层级 | 可用信息 | 同层方法 | 论文用途 |
|---|---|---|---|
| L0 元数据冷启动 | AABB、实例到 GLB 映射、资源字节，不含三角形 | 视锥全保留、静态频率、AABB depth-proxy、位置/AABB+ray | 证明仅靠轻量元数据能做到什么 |
| L1 压缩场景知识 | L0 + bitset/froxel/固定神经特征页和查询权重 | 离线 PVS bitset、NeuralPVS、本方法及消融 | 论文主公平比较，计入表示自身字节和启动时间 |
| L2 完整几何 warm-cache | 已下载并解码三角形，可生成深度 | 标准 HZB、CHC++、Trim Regions 等 | 图形学质量/稳态上界，不参与 Cold-0 胜负 |

L1 方法必须同时报告压缩表示的总字节、首个区域页字节和下载/解码时间。若本方法先下载 14 MB 特征再节省 10 MB GLB，它在冷启动字节上没有净收益。

### 7.1 可见性基线

| 级别 | 基线 | 资源假设 | 角色 |
|---|---|---|---|
| 必做 | 66°后退视锥全保留 | 仅实例 AABB | 零遮挡能力、最高安全性的下界 |
| 必做 | 离线 view-cell PVS bitset | 同一采样 GT 和离散 cell 索引 | 比较存储、查询和未见位置泛化 |
| 必做 | 静态实例可见频率 | 训练集统计 | 证伪模型只记住可见频率 |
| 必做 | 位置/视线 MLP | 实例位置、AABB、射线特征 | 证伪场景位置记忆 |
| 必做 | AABB+视线 MLP | AABB 与射线特征 | 无点云几何基线 |
| 必做 | 几何+视线 MLP | 固定几何特征 | 上下文和代理的直接母基线 |
| 必做 | AABB depth-proxy | 仅 AABB 投影矩形和近深度 | 显示名 `baseline_aabb_depth_proxy`；`baseline_aabb_hzb` 仅为历史兼容 key |
| 必做 | 标准几何 HZB | L2：已下载三角形和深度预通过 | warm-cache 图形学上界，计入深度构建与映射成本 |
| 必做 | Dynamic-pool | 动态邻居传播 | 质量/延迟上界，证明当前代理的替代价值 |
| 必做 | NeuralPVS 适配 | 其所需 froxel/几何表示 | 同 FOV、同 view-cell 和实例映射的学习型基线 |
| 增强 | Neural Visibility of Point Sets 风格适配 | 场景表面点及视线 | 算法上界，完整报告点资产和运行成本 |
| 增强 | Trim Regions / Disocclusion Buffer | 完整几何 | 强 from-region 对照，区分冷/热缓存 |

所有方法使用相同 66°模型视锥、60°真实显示视锥、view-cell、测试 pose 和 GT。同层方法使用相同信息预算；跨层结果只能表示质量/资源上界。需要完整几何的方法分别报告冷启动不可用条件和 warm-cache 条件，并把几何下载、上传 GPU、深度/froxel 构建及实例映射纳入总成本。

### 7.2 下载调度基线

| 基线 | 输入 | 目的 |
|---|---|---|
| FIFO/原始 preload 顺序 | GLB 列表 | 工程下界 |
| 距离优先 | 相机到 AABB/实例距离 | 简单空间启发式 |
| 投影面积优先 | AABB 屏幕面积 | 图像贡献启发式 |
| 小资源优先 | GLB 字节 | 快速完成数量的启发式 |
| 可见性 max/sum 聚合 | 实例可见概率 | 判断专用下载头是否只复现可见性后处理 |
| 可见性门控效用 `p × salience` | 实例可见概率和学习到的显著性 | 检验受约束级联是否优于自由 MLP |
| Noisy-OR GLB 聚合 | 同一 GLB 的实例效用 | 对比当前 max 聚合并处理多实例累积价值 |
| 预测效用/字节 | 实例效用和 GLB 成本 | 强贪心基线 |
| 独立 RankNet/ListNet/成本敏感排序器 | 相同输入和容量 | 判断共享训练是否带来收益 |
| 当前联合下载头 | 共享查询表征 | 待验证方法 |
| GT 效用/字节 oracle | 真实像素效用和成本 | 可达到上界，不参与公平运行时胜负 |
| 3D Tiles 屏幕空间误差 | 层次和几何误差 | 增强的标准流送对照 |

比较时固定相机轨迹、网络 trace、并发数、缓存初态和字节/时间预算。不能让一个方法多下载后再比较更高效用。

### 7.3 系统基线

在模型权重和输入完全相同的条件下比较：当前自定义 WGSL、ONNX Runtime Web 或 TensorFlow.js 的 WebGPU 后端（若算子可表达）、CPU/WASM fallback。该组只比较包体、启动、延迟和内存，不比较预测精度差异；精度一致性另做 FP32/FP16/WGSL 对照。

## 8. 必做实验设计

### 8.1 方向代理因果实验

先对同一 checkpoint 做推理期干预，以最低成本判断模型是否依赖代理：

- 代理全部清零；
- 代理替换为同分布随机值；
- 八个方向取平均；
- 方向循环移位；
- 跨实例随机置换代理；
- 上下文清零和跨实例置换。

再进行三随机种子重训练消融：

```text
AABB + ray
geometry + ray
geometry + context + ray
geometry + context + proxy + ray，不含显式 inhibition
完整模型
```

所有变体独立在 calibration 按同一安全规则选择阈值。不能共享当前 `0.64`，也不能用 test 为每个变体找最优点。记录基础可见性 logit、代理门控熵、抑制量分布和最终 logit 变化，避免只从最终指标猜测分支是否工作。

代理贡献的 Go 条件为：相对 `geometry + context + ray`，完整模型在安全约束下至少提高 useful cull 2 个百分点，且三种子 paired bootstrap 95% 置信区间下界大于零；同时 `bad cull` 增量及其区间上界均不超过预注册的 `+0.002`。若 M5 已有同预算图像/字节证据，也可使用达到相同图像效用时减少 10% 下载字节且区间不跨零的替代门。未达到时执行路线 B。

### 8.2 损失与任务头实验

当前数量损失要求概率总量接近 GT，而预算损失允许约 `2.5×GT`；RVL 误报项又按可见实例数归一化，可能随候选规模改变损失比例。先统一记录各损失的未加权值、加权值和梯度范数，再进行：

- 无 RVL、均匀 RVL、证据 RVL、成本感知 RVL；
- 无数量约束、仅数量约束、仅预算约束、统一预算约束；
- 可见性单任务；
- 可见性和下载独立模型；
- 共享干路双任务；
- 共享干路但下载梯度停止回传到可见性表征；
- 当前完整多任务模型。

损失权重只在 validation/calibration 搜索。多组权重比较使用统一预算，不能因某一组搜索次数更多而获得优势。

### 8.3 实例级图像安全实验

重写现有图像管线，使其使用真实 GLB、真实 60°相机、相同 pose 和实例 ID 缓冲，不再只比较 GLB ID。管线开发和阈值前检查只使用 validation/calibration；所有方法冻结后，才在 one-shot test 中为每个 view-cell 至少生成：

1. 全 GT 实例参考图；
2. 各方法冻结阈值后的预测图；
3. 漏实例像素掩码；
4. 多显示实例像素掩码；
5. 错误实例 ID 掩码；
6. RGB 差异图。

主指标为 miss-pixel rate、wrong-ID pixel rate、extra-pixel rate 和图像 PER；辅助报告 SSIM，并在可用时报告 FLIP。按 view-cell 报告均值、中位数、95 分位和最差 5% 区域，单独统计细小、细长、近景和短暂出现构件。

建议安全门槛为 mean miss-pixel rate `<0.5%`、95 分位 `<1%`。若普通 recall 低于 0.95，只有图像门槛和人工长尾检查同时通过时，才能解释为漏失主要来自低视觉贡献实例。

### 8.4 统一下载调度实验

训练监督从单纯“GLB 是否含可见实例”扩展为预算内视觉效用，至少包含：可见权重、缺失图像损失、GLB 字节、实测解码时间、缓存状态和预取时间价值。实例可见性和 GLB 调度共享基础表征，但保持不同输出粒度。

首先验证级联结构本身，使用相同查询特征和近似参数量比较：

- `visibility-only`：GLB 分数为实例可见性 max/sum；
- `independent-utility`：效用头不读取可见性概率；
- `current-cascade`：效用读取可见性，下载头读取可见性和效用；
- `visibility-gated`：`utility = visibility × salience`；
- `independent-ranker`：独立 GLB 排序模型；
- `stop-gradient-cascade`：保留级联输入，但下载损失不改变可见性表征。

GLB 聚合比较 max、sum、top-k sum 和 Noisy-OR。训练、阈值和聚合规则都只在 validation/calibration 决定。这样能够回答收益来自显式可见性-效用级联、共享表征、GLB 聚合，还是仅来自更大模型。

固定导航轨迹下使用冷缓存、温缓存、4G 和 Wi-Fi trace，报告：

- `utility@bytes` 和 `utility@time`；
- 效用-字节曲线面积；
- 达到 90%/95%/98%/99% 可见效用所需字节和时间；
- 导航过程 missing-utility integral；
- 无效下载字节；
- 首个有用画面和首个足够完整画面的时间；
- GLB 解码和上传造成的 p95 帧时间。

联合头的 Go 条件为：相对最佳启发式或等容量独立排序器，在相同效用下减少至少 15% 字节或首屏时间，三种子/多轨迹置信区间不跨零。未通过时删除“联合调度优于后处理”的主贡献，把下载头降为工程组件。

### 8.5 泛化实验

泛化分三层报告，不能混为一个数字：

1. 场景内空间块留出：检验相邻 view-cell 泄漏；
2. 场景内方向留出：检验未见观察方向；
3. 跨场景：单场景训练后零样本迁移、联合训练、1%/5%/10% 少样本适配。

同时比较静态可见频率、实例 ID embedding、位置+视线 MLP 和 AABB+视线 MLP，证伪模型只记住场景位置。若独立场景少于三个或 leave-one-scene-out 明显失败，论文只声明“按场景离线预处理和训练”，并报告适配成本，不使用“通用编码器”表述。

### 8.6 运行时与规模实验

先修复真实视锥与神经预测的更新耦合：神经预测保持跨出 view-cell/后退范围才重算，真实 60°实例视锥应在相机更新时独立刷新，不能沿用旧显示集合。实例绑定不匹配不得静默退化为整 GLB 显示；该路径必须报错或显式计为兼容降级。

为避免端侧先遍历全场景实例，建议把运行输入改为“轻量空间目录 + 当前页实例特征”：离线仅用实例 AABB 建立紧凑 BVH/网格页，运行时由 66°后退视锥遍历相交节点，只请求这些节点对应的特征页，再对叶节点中的实例执行神经查询。远距离节点可以先做组级保守保留或预算裁剪，但最终显示仍保持实例粒度。该结构不需要目标 GLB 三角形，并使首轮工作量取决于活动空间页和候选数量，而不是全场景实例总数。

必须比较四条路径：全实例 CPU AABB 扫描、CPU 空间层次+特征页、GPU 全实例 AABB、GPU 空间层次+compact。对每条路径分别报告目录字节、页命中率、候选漏失、查询数量和端到端延迟；任何层次裁剪造成的漏实例都计入整体 FN，不能在神经模型指标之外隐藏。

运行时分项计时：

- CPU/Worker 空间候选生成；
- WebGPU buffer 更新和 query dispatch；
- GPU 查询；
- GPU 结果压缩与 readback；
- CPU GLB 聚合和 top-k；
- 实例矩阵/可见数量更新；
- GLB 下载、解码和 GPU 上传；
- 总调度延迟和帧时间。

候选规模使用 256、512、1k、2k、4k、8k、10k 和 16k 分桶，报告 cold/warm p50、p95、p99、峰值内存和能耗。设备至少包含桌面独显、桌面/笔记本集显和两个 Android 性能档位。

系统优化按收益顺序实施：

1. BVH/空间层次减少参与推理的实例；
2. GPU compact，仅回读预测实例和必要 GLB top-k；
3. 按空间页懒加载固定实例特征，避免单体特征文件随全场景增长；
4. FP16 对照 INT8 和低秩代理压缩；
5. GLB 原型特征加实例位置/尺寸/姿态残差，利用实例化复用。

“移动端轻量”的最低门槛建议为 10k 候选下桌面 WebGPU p95 `<20 ms`、移动端 p95 `<50 ms`、主线程附加工作 `<2 ms` 且无明显帧停顿。当前约 3.9 秒 smoke 属于明确 No-Go；若八周内无法达到门槛，论文必须改为“浏览器可部署原型”并完整报告限制。

### 8.7 导出一致性实验

对固定 pose 和候选集合逐层比较：

- PyTorch FP32；
- 导出 FP16；
- WGSL/WebGPU；
- 可选 INT8。

报告最大/平均 logit 误差、阈值翻转数量、实例集合 Jaccard、GLB 排序 Kendall/Spearman 相关性和 top-k 重合率。任何导出后 weighted recall 降到 0.99 以下都视为导出失败，不能通过前端另改阈值修补。

## 9. 指标与统计协议

### 9.1 画面安全

- pose-level 和 aggregate precision、recall、F1、Jaccard；
- weighted recall 及其权重来源；
- `FN/GT` 与 `bad cull = FN/candidate`；
- image PER、miss-pixel、wrong-ID、extra-pixel；
- 平均、95 分位和最差 5% view-cell。

### 9.2 有效剔除与资源

- `useful cull = TN/candidate`；
- 平均 candidate、GT、prediction 及其比例；
- GLB 数量和字节削减；
- 运行时特征总大小、每实例大小、空间页大小和初始页大小。
- Cold-0 bootstrap bytes：作出第一轮预测前必须下载的元数据、特征页和权重；
- time-to-first-visibility-decision：页面启动到第一轮可见性/下载计划产生的时间；
- net saved bytes：避免下载的 GLB 字节减去神经权重、特征页和额外元数据字节；
- break-even poses/time：特征启动成本被累计节省抵消所需的相机查询次数或浏览时间；
- 无目标 GLB 几何访问证明：第一轮 dispatch 前的网络请求和 GPU buffer 清单。

### 9.3 下载和体验

- 固定预算视觉效用；
- 达到目标效用的字节/时间；
- 首个有用画面、足够完整画面；
- missing-utility integral 和浪费字节；
- 解码、GPU 上传、帧时间和交互卡顿。
- 在 1/2/5/10/20 MB 冷启动总预算下的视觉效用，其中预算包含神经启动资产；
- 相对 keep-all、bitset、NeuralPVS 和 3D Tiles 的净字节收益，而非只统计 GLB 子集内部削减。

### 9.4 统计方法

- 实验单位为 view-cell；图像实验以 view-cell 聚类，不能把候选实例或像素当独立样本；
- 正式模型至少 3 个随机种子，建议关键主表 5 个种子；
- 报告 seed 间 `mean ± std`；
- 同一 test view-cell 上的方法差值执行分场景、分区域 paired stratified bootstrap，10,000 次，报告 95% CI 和绝对效应量；
- weighted recall 报告单侧 95% 下置信界；其他主指标使用双侧区间；
- 下载实验按相同轨迹和网络 trace 配对；运行时报告 p50/p95/p99；
- 多个损失、消融和 baseline 的显著性比较使用 Holm 校正；
- 预先指定主指标，避免从大量指标中只挑有利结果。

## 10. 实验矩阵与优先级

### 10.1 Must-have

| ID | 实验/实现 | 通过标准 | 截止 |
|---|---|---|---|
| M0 | 修复 validation/calibration/test 协议 | test 无阈值扫描，完整固定 validation | 08-03 |
| M1 | HKUST/Metropolis 数据语义审计 | 点云、候选、AABB、映射全部一致 | 08-05 |
| M2 | 空间块四路 split | manifest 固定，无相邻块跨集合 | 08-07 |
| M3 | 代理推理期干预 | 得出是否依赖代理的明确诊断 | 08-09 |
| M4 | 核心消融三种子 | 代理/上下文/RVL 的独立效应与 CI | 08-16 |
| M5 | 实例级 60°图像评价管线 | validation/calibration 完整运行，final test 保持封存 | 08-18 |
| M6 | 可见性主 baseline | keep-all、bitset、频率、AABB/ray、HZB、NeuralPVS | 08-23 |
| M7 | 级联效用与下载调度 baseline | visibility 聚合、自由/门控级联、GLB 聚合、独立 ranker、oracle | 08-25 |
| M8 | 冷/热缓存导航实验 | utility-byte/time 主曲线完整 | 08-28 |
| M9 | 浏览器正确性、空间分页与分项计时 | 无旧视锥驻留和整 GLB 静默降级；Cold-0 不访问目标 GLB 几何 | 08-20 |
| M10 | 真实设备 benchmark | 桌面两类+Android 两档，p50/p95/p99 | 09-03 |
| M11 | 空间与跨场景泛化 | 空间块、方向、零/少样本完整 | 09-06 |
| M12 | FP32/FP16/WGSL 一致性 | 冻结阈值后安全约束不退化 | 09-06 |
| M13 | 正式三种子 one-shot test | 集合、图像、调度和运行指标一次完成，主表/附录表冻结 | 09-13 |

### 10.2 Enhanced

- Trim Regions、Disocclusion Buffer、Neural Visibility of Point Sets 风格适配；
- 3D Tiles 屏幕空间误差流送对照；
- INT8、低秩代理和原型+残差压缩；
- 第三个独立大型场景和完整 leave-one-scene-out；
- 5 个随机种子；
- FLIP、能耗和更长真实用户导航 trace。

### 10.3 Optional

- ONNX Runtime Web/TensorFlow.js 后端；
- 不同方向单元、深度层和来源数量的大范围结构搜索；
- 高级 GPU 全流程 GLB top-k；
- 动态场景或可变几何扩展。

Optional 任务不能挤占 M0-M13。论文主结果没有冻结前，不进行大规模新架构探索。

## 11. 代码实施顺序

### P0：先修实验有效性

1. 修改 `train_directional_occlusion_proxy_encoder.py`，把 checkpoint 选择固定为完整 validation，把阈值选择移到独立 calibration；移除训练结束后 test 阈值扫描。
2. 扩展 `common/threshold_selection.py`，输入预注册安全裕量和校准来源，并在输出 meta 中写入 split、阈值、规则和 manifest hash。
3. 修改 CSR 构建器，支持空间块四路 split 和不可变 manifest。
4. 增加资源语义校验，明确点云是 GLB 原型对齐还是实例对齐；删除模型中互相矛盾的注释和隐式猜测。
5. 修复 Metropolis 候选漏正样本与点云索引后再训练。

### P1：建立核心科学证据

1. 新增统一 intervention/ablation runner，不创建含糊临时目录；
2. 重写实例级图像评价；
3. 增加频率、位置、AABB/ray、几何/ray、真实 HZB 和 NeuralPVS runner；
4. 新增自由级联/可见性门控级联、共享/独立下载排序、GLB 聚合基线和固定网络 trace runner；
5. 所有结果输出逐 view-cell 明细，便于 paired bootstrap。

### P2：建立系统证据

1. 解耦低频神经预测和高频真实视锥实例过滤；
2. 增加候选、dispatch、GPU、readback、聚合、矩阵更新和资源解码计时；
3. 实现空间层次和特征分页；
4. 实现 GPU compact，减少全候选回读；
5. 建立真实 Android Chrome 自动化或可重复手工 benchmark 协议。

### P3：冻结与写作

1. 冻结数据 manifest、代码 commit、conda 环境、浏览器和 GPU 信息；
2. 三种子完整重训；
3. calibration 选择并冻结阈值；
4. one-shot test；
5. 生成表格、曲线、失败案例、视频和 artifact manifest；
6. 任何 test 后参数变化都建立新实验并废弃旧 test，不覆盖结果。

## 12. 复现入口和待新增命令

当前入口为：

```bash
# 训练
conda run -n slm_pvs python \
  neural_instance_culling/model/train_directional_occlusion_proxy_encoder.py \
  --device cuda --epochs 40 --amp \
  --target-weighted-recall 0.99 \
  > train_stdout.log 2> train_stderr.log

# 统一集合与资源指标
conda run -n slm_pvs python \
  neural_instance_culling/benchmark/evaluate_unified_pvs_metrics.py \
  ...

# 当前图像评价入口，必须先升级为 instance-ID 口径
conda run -n slm_pvs python \
  neural_instance_culling/benchmark/evaluate_viewcell_image_per.py \
  ...

# 前端静态检查
cd slm2viewer
npm test
```

`npm test` 目前只验证资产、数量和阈值等静态契约，不等价于浏览器推理正确性或性能测试。

投稿协议实施后应新增稳定入口，建议命名为：

```text
neural_instance_culling/tools/build_spatial_split_manifest.py
neural_instance_culling/benchmark/calibrate_visibility_threshold.py
neural_instance_culling/benchmark/evaluate_frozen_test.py
neural_instance_culling/benchmark/evaluate_proxy_interventions.py
neural_instance_culling/benchmark/evaluate_instance_image_metrics.py
neural_instance_culling/benchmark/evaluate_streaming_qoe.py
slm2viewer/scripts/benchmark_webgpu_runtime.mjs
```

这些名称代表不同职责，不能复用旧实验名改变含义。

## 13. 八月至投稿排期

| 周期 | 研究与工程交付 | 决策 |
|---|---|---|
| 07-31 至 08-03 | M0：固定验证/校准/test 协议；冻结当前探索性结果 | 未修复 test 污染则停止后续正式评测 |
| 08-03 至 08-09 | M1-M3：两场景数据审计、空间 split、代理推理干预 | 代理完全不敏感则立即转路线 B |
| 08-10 至 08-16 | M4：三种子核心消融；启动 PVS baseline | 08-16 冻结贡献路线 A/B |
| 08-17 至 08-23 | M5、M6、M9：图像评价、传统/学习基线、前端正确性和计时 | 图像或前端错误未解决则 No-Go |
| 08-24 至 08-30 | M7、M8：下载排序和网络 QoE；并行完成系统优化 | 联合头无增益则删除调度创新主张 |
| 08-31 至 09-06 | M10-M12：移动设备、泛化、导出一致性 | 冻结正式架构、数据和实验协议 |
| 09-07 至 09-13 | M13：三种子完整训练、calibration、one-shot test | 09-13 冻结全部主结果 |
| 09-14 至 09-20 | 论文初稿、主图表、失败案例、补充材料和视频 | 09-20 完成内部完整审稿 |
| 09-21 至 09-25 | 反方审稿、统计复核、Eurographics 摘要登记 | 09-25 提交摘要和完整标题/作者信息 |
| 09-26 至 10-01 | 只修写作、图表、artifact 和合规，不再调模型 | 10-01 提交 Eurographics 全文 |

四块 GPU 的并行方式：三块分别运行三个正式 seed，第四块运行一个 baseline 或导出一致性/图像任务。禁止四块卡并行做无预注册的大范围权重搜索，导致主实验来不及完成。

## 14. 投稿渠道决策

### 14.1 主目标：Eurographics 2027

官方 [Call for Papers](https://eg2027.isti.cnr.it/call-for-papers) 列出的摘要登记时间为 2026-09-25 23:59 UTC，全文截止为 2026-10-01 23:59 UTC，录用论文发表于 Computer Graphics Forum。其计算机图形学、渲染、深度学习图形学和图形系统范围与本项目最匹配。

Go 条件：方向代理路线 A 成立，或系统路线 B 在真实设备、图像安全和统一流送上形成足够强的系统贡献；主要基线、空间泛化和统计全部完成。

### 14.2 条件目标：IEEE VR 2027

官方 [Papers CFP](https://ieeevr.org/2027/contribute/papers) 要求 2026-08-24 AoE 前登记摘要，2026-08-31 AoE 前提交全文，正文 4-9 页。只有在 08-16 前主要实验已经完成，且论文能够明确服务沉浸式 VR/AR/MR 或三维用户界面时才考虑。纯浏览器大场景优化存在 scope 风险，不作为默认路线。

### 14.3 九月滚动期刊备选

若 Eurographics Go 条件未满足但系统证据完整，可在九月底提交滚动期刊：

| 渠道 | 适用叙事 | 条件 |
|---|---|---|
| [The Visual Computer](https://link.springer.com/journal/371/aims-and-scope) | 可见性、图形算法和 WebGPU 图形系统 | 图像、传统基线和算法消融较强 |
| [IEEE Transactions on Multimedia](https://signalprocessingsociety.org/publications-resources/ieee-transactions-multimedia) | 视觉效用、网络预算、资源流送和端侧推理 | 调度/QoE 和多媒体系统贡献较强，初稿页数按官方规则控制 |
| [Multimedia Systems](https://link.springer.com/journal/530/aims-and-scope) | 工程系统、传输、缓存和移动端部署 | 算法新颖性较弱但端到端系统证据扎实 |

VRST/SUI 的 2 页 poster/demo 不作为同一成果的并行正式投稿，避免重复投稿和新颖性冲突。CHI 缺少明确人因与用户研究，不是合适备选。

## 15. 论文定位、标题和摘要论证链

### 15.1 一句话定位

> NeuralStreamWeb3D 将静态实例化场景压缩为可分页的实例知识表，使浏览器在目标几何资产到达前，以自定义 WebGPU 视线查询级联预测实例可见性、视觉效用和 GLB 下载优先级，并在画面安全约束下联合减少端侧计算与传输成本。

这句话同时限定了静态场景、离线预处理、端侧已有紧凑知识页、原始几何尚未到达、实例和 GLB 两个粒度。任何摘要版本都不应删掉这些边界。

### 15.2 候选标题

算法与系统证据同时成立时：

```text
NeuralStreamWeb3D: Lightweight Ray-Conditioned Visibility and
Utility-Aware Streaming Before Geometry Arrival
```

方向代理因果证据较强时：

```text
Directional Scene-Knowledge Proxies for Geometry-Free Cold-Start
Visibility and Streaming in Instanced Web3D Scenes
```

系统路线 B 时：

```text
Scene-Knowledge Paging for Lightweight Visibility and
Budget-Aware Asset Streaming in Web3D
```

第二个标题中的 `Geometry-Free Cold-Start` 必须在摘要第一段立即解释为“不需要目标 GLB 原始几何，但需要紧凑场景元数据和特征页”，否则容易被审稿人理解为零场景先验。

### 15.3 创新优先级

| 优先级 | 创新点 | 成立所需证据 |
|---|---|---|
| I1 | 几何资产到达前的压缩场景知识查询 | 同 L1 信息预算 baseline、bootstrap bytes、净字节和 first-decision 时间 |
| I2 | `可见性 -> 视觉效用 -> 下载分数` 级联双粒度决策 | 自由/门控级联、visibility 聚合、独立 ranker 和 GLB 聚合消融 |
| I3 | 方向遮挡代理 | 清零、平均、方向移位、跨实例置换和三种子重训练因果证据 |
| I4 | 空间目录+特征分页+GPU compact | 不同场景/候选规模的启动字节、p95 延迟和无额外 FN |
| I5 | 无通用框架的融合 WGSL 查询 | 与 ORT Web/TF.js/WASM 同权重比较后的包体、初始化和延迟收益 |

I1 和 I2 是最贴合项目目标的核心；I3 失败时可以降级；I4 是轻量化主张的系统支撑；I5 只有基准显著时进入贡献列表。

### 15.4 摘要论证链

摘要按以下五句话展开：

1. 大规模 Web3D 在模型资产尚未到达时缺少可用于遮挡剔除和下载排序的几何，完整 HZB 等方法无法直接工作；
2. 现有神经 PVS 或点可见性方法没有同时解决实例显示、GLB 成本和浏览器端冷启动预算；
3. 本文把场景知识压缩为分页实例表，并以轻量视线查询级联产生可见性、视觉效用和下载分数；
4. 系统在自定义 WebGPU 中完成候选查询和紧凑输出，真实 60°视锥保持实例级显示，GLB 级分数负责传输；
5. 在空间隔离、多场景、图像、字节和真实设备实验中，相对最强同信息 baseline 报告安全约束下的 Pareto 改善。

第五句中的数值只能在 M13 one-shot test 后填写。

### 15.5 主图与主表

- 图 1：Cold-0/Cold-1/Warm 时间线，标出何时只有元数据/特征页、何时 GLB 几何到达；
- 图 2：离线场景知识压缩与 `p_i -> u_i -> d_i -> D_g` 级联；
- 图 3：安全约束下的 image utility、net bytes 和 p95 latency 三维 Pareto；
- 图 4：空间页与 GPU compact 的候选规模曲线；
- 表 1：L0/L1/L2 baseline 的资源可用性和公平性；
- 表 2：主结果，包含 weighted/ordinary recall、miss-pixel、useful/bad cull、bootstrap/net bytes 和移动 p95；
- 表 3：方向代理和效用级联因果消融；
- 表 4：跨空间块、方向和场景泛化。

## 16. Go/No-Go 门槛

| 门槛 | Go | No-Go 后处理 |
|---|---|---|
| 协议 | 阈值只在 calibration 选择，test one-shot | 结果不得进入论文主表 |
| 数据 | 两场景候选、点云和映射语义可复现 | 有问题场景移出主表并重采样/重训 |
| Cold-0 | 第一轮决策前不请求目标 GLB 几何；中位导航 trace 的 net saved bytes 为正，且 first-decision 早于 keep-all 首批关键 GLB 解码完成 | 删除“几何到达前有效”主张或先完成特征分页 |
| 安全 | 每场景 test weighted recall 点估计 `>0.99`，完整报告普通 recall 和 FN | 不重新扫 test；报告失败并训练新版本 |
| 图像 | mean miss-pixel `<0.5%`，p95 `<1%`，无系统性 pop-in | test 失败后不得调阈值；当前实验 No-Go，下一实验重新冻结协议 |
| 代理 | useful cull +2 pp 或同效用字节 -10%，CI 不跨零 | 转路线 B，删除代理主张 |
| 调度 | 相对最佳启发式/独立 ranker 同效用字节或首屏时间 -15% | 下载头降为工程组件 |
| 系统 | 10k 候选 desktop p95 `<20 ms`、mobile `<50 ms`、主线程 `<2 ms` | 删除移动实时主张；评估期刊系统路线 |
| 自定义 WGSL | 相对最佳通用 Web 推理框架在包体、启动或 p95 至少一项有显著收益且其余不明显退化 | 仅作为实现细节，不列论文贡献 |
| 泛化 | 至少空间块泛化成立；通用 claim 需至少 3 场景 | 限定为每场景训练并报告适配成本 |
| 统计 | 至少 3 seeds、paired CI、主指标预注册 | 不提交高要求会议 |

测试污染、代理无因果增益、实例级图像安全失败、真实移动性能失败四者中任一项未被论文诚实降级，都构成 Eurographics No-Go。

## 17. 预期论文结构

1. 引言：Web3D 中实例显示与 GLB 下载的双粒度矛盾；
2. 相关工作：PVS/遮挡、神经可见性、三维流送和预算排序；
3. 问题定义：Cold-0/Cold-1/Warm 资源模型、66°查询、60°显示、实例/GLB 输出和净字节预算；
4. 方法：固定场景知识压缩、方向代理或简化上下文、射线查询和可见性-效用-下载级联；
5. 浏览器系统：空间目录、特征分页、自定义 WGSL、WebGPU compact、缓存和调度；
6. 实验协议：空间划分、validation/calibration/test、基线、公平资源假设和统计；
7. 结果：安全/剔除、图像、调度、泛化、运行时和消融；
8. 局限：场景特定离线预处理、静态几何、线性候选依赖和设备兼容性；
9. 结论。

主图建议为“图像效用-净字节-延迟 Pareto”而不是单一 F1 柱状图。主表同时列 weighted recall、普通 recall、useful/bad cull、miss-pixel、bootstrap/net bytes、p95 延迟和运行资产大小。

## 18. 复现与文档交付

投稿前必须冻结：

- git commit 和 dirty worktree 说明；
- conda 环境文件、CUDA/PyTorch/Node/Chrome/WebGPU backend 版本；
- 每个数据集、证据、checkpoint、前端资产和 benchmark 的 SHA-256/字节大小；
- 采样和 split manifest；
- 每个训练的完整命令、stdout/stderr、seed 和 GPU；
- calibration 阈值记录和 test one-shot 标记；
- 逐 view-cell 指标和统计脚本；
- 设备、网络 trace、缓存初态和导航轨迹；
- 失败实验及删除/保留决定。

由于生成数据、权重和 benchmark 当前多数被 `.gitignore` 排除，必须额外建立 artifact manifest 和稳定外部存储位置。任何正式实验完成后均按 `AGENTS.md` 更新 `docs/evaluation/` 或 `docs/experiments/`，不能只保留终端输出。

## 19. 第一周立即执行清单

1. 给现有 HKUST `0.64` 结果加 `exploratory_test_calibrated` 标记，停止称其为正式 test；
2. 实现固定 validation、独立 calibration 和 one-shot test；
3. 生成 HKUST/Metropolis 空间四路 split manifest；
4. 修复 Metropolis 点云与候选语义；
5. 运行代理清零、平均、方向移位和跨实例置换；
6. 建立实例级 60° image-ID 渲染最小 smoke；
7. 修复真实视锥高频刷新与整 GLB 降级测试；
8. 增加 Cold-0 网络请求审计，证明第一轮决策前没有读取目标 GLB 几何；
9. 在 08-09 前输出第一份 Go/No-Go 报告。

在以上事项完成前，不继续搜索新的损失权重或扩大模型结构。

## 20. 原始需求覆盖审计

| 原始需求 | 文档证据 | 完成判断 |
|---|---|---|
| 按 Academic Research Suite 组织研究计划 | 第 3 节 RQ/FINER、第 4 节检索与综合、第 6-10 节方法和实验、第 16 节门槛 | 已覆盖 |
| 基于当前实验现状 | 第 2 节模型、代码级联、HKUST 指标和可信度；第 1 节六项阻断问题 | 已覆盖，现有 0.64 明确降级为探索性 |
| 相关文献综述 | 第 4 节检索协议、17 项核心来源、脉络综合和研究缺口 | 已覆盖 |
| 九月投稿渠道 | 第 13-14 节排期、Eurographics/IEEE VR/滚动期刊条件路线 | 已覆盖并使用官方日期 |
| 涉及的研究方向和可发表贡献 | 第 3、5、15 节研究问题、路线 A/B、创新优先级和摘要论证链 | 已覆盖 |
| 需要添加的实验 | 第 8、10、13 节实验设计、M0-M13 和逐周交付 | 已覆盖 |
| 合理 baseline | 第 7 节 L0/L1/L2 信息预算分层、可见性/调度/系统基线 | 已覆盖 |
| 需要测定的指标 | 第 9 节画面、安全、剔除、Cold-0、调度、系统和统计指标 | 已覆盖 |
| 端侧轻量化 | 第 8.6 节空间目录、特征分页、GPU compact、量化；第 16 节系统门槛 | 已覆盖，等待后续实验验证 |
| 原始模型资产未到达时工作 | 第 2.2 节 Cold-0 定义、第 7 节公平信息层级、第 9 节启动/净字节指标 | 已准确限定并覆盖 |
| 可见性驱动视觉效用和下载头 | 第 2.3 节真实代码级联、第 8.4 节自由/门控级联消融 | 已覆盖 |
| 不依赖通用模型框架 | 第 4.5、7.3、15.3 节将自定义 WGSL 作为待基准验证的系统特性 | 已覆盖且未过度主张 |
| 实现优化和更多创新点 | 第 8.4、8.6、11、15.3 节级联效用、分页、原型复用、量化和自定义 WGSL | 已覆盖并排定优先级 |

本文件完成的是可投稿研究计划，不代表 M0-M13 已经执行。后续每项实验必须按其门槛产生独立代码、日志、指标报告和 one-shot test 证据；计划本身不能替代实验结果。

## 21. 实际执行状态（2026-08-01）

本节只记录当前仓库中已经由代码和审计文件证明的状态，不把计划目标当作实验结果。

| 阶段 | 当前状态 | 证据/剩余工作 |
|---|---|---|
| M0 | 子门通过，总门未通过 | 固定 validation、独立 calibration、frozen test 和严格候选语义已实现；已修正 `evaluate_frozen_test.py` 与当前统一评估器的调用契约并加入运行期 fixture；HKUST/Metropolis 正式训练仍在运行，完成后还需核对 bootstrap、冻结阈值、逐 pose 明细和 one-shot test |
| M1 | 正式资源门通过 | `m1_hkust_spatial_resource_audit_2026-08-01.json`、`m1_metropolis_spatial_resource_audit_2026-08-01.json`；旧随机数据审计失败记录仍保留，不能混入正式结果 |
| M2 | 空间隔离门通过，类别风险已登记 | `m2_spatial_split_execution_2026-08-01.md`；Metropolis sky 物理位置稀疏，M11 需按类别报告限制，K4 稳定性仍待验证 |
| M3-M4 | M3 未执行正式版本；M4 消融入口已补齐 | 旧模型的推理期干预仍只能作为 exploratory evidence；已加入 checkpoint 可追溯的 `AABB+ray`、`geometry+ray`、`geometry+context+ray` 输入消融及辅助损失屏蔽，正式训练仍须等待 GPU 资源和空间正式 checkpoint 审计后按同一 calibration 规则完成 |
| M5 | split/实例语义与 schema smoke 通过，图像质量门未通过 | `docs/evaluation/m5_instance_id_buffer_schema_2026-08-01.md`、`docs/experiments/m5_formal_split_alignment_2026-08-01.md`；正式图像入口现在默认使用 Pose CSR 空间标签并完整遍历 split，真实冻结模型的 validation/calibration/test 图像评价仍待执行 |
| M6 | L0 子门通过，总门未通过 | 七个 model-free runner、确定性 AABB+ray 和学习型 AABB+ray MLP 已在 HKUST/Metropolis 完整 validation/calibration；真实三角形 HZB 与 NeuralPVS 仍未实现，不能以 depth proxy 代替 |
| M7-M8 | 协议和 model-free baseline 子门通过，总门未通过 | Metropolis 修正后 calibration 已完成，弱效用 byte 回放可复现；独立 ranker、真实解码时间、真实网络/设备轨迹和联合头质量比较仍缺失 |
| M9-M13 | 未执行或证据不足 | 仍需浏览器正确性/规模、真实设备、泛化、导出一致性和最终 one-shot test 证据；现有 exploratory 输出不能直接进入主表 |

### 21.1 当前执行核验（2026-08-01 08:10）

- M0-M5 的 Python 回归测试全部通过（当前 benchmark unittest 共 31 项）；`evaluate_proxy_interventions.py --self-test` 通过。该结果只证明协议、候选严格性、冻结测试入口和 Color-ID schema 的代码契约，不替代正式模型、图像或设备质量门。
- HKUST 正式空间训练 `pvs_directional_occlusion_proxy_encoder_rvl_strong_v2_full40_hkust_spatial_fov66_seed20260801_protocolfix` 正在运行，已完成第 16/40 个 epoch；截至核验时非有限 loss/gradient 跳过计数均为零。训练结束前不读取中间 checkpoint 进入正式 test。
- Metropolis 正式空间训练 `pvs_directional_occlusion_proxy_encoder_rvl_strong_v2_full40_metropolis_spatial_fov66_seed20260801_protocolfix_bs1_gpu2` 正在 GPU 2 上运行第 1/40 个 epoch，使用显式 seed `20260801`、FP32、pose-set batch size 1；同样未发现非有限 loss/gradient。GPU 1 上另有一个未显式指定 seed 的历史诊断运行，保留作诊断，禁止纳入主表。
- Metropolis 的 M7 baseline validation 已生成完整摘要；calibration 仍在运行，尚未冻结阈值。HKUST baseline validation/calibration 结果仍只作为校准诊断，不能替代正式模型的 frozen test。
- M6 的 `baseline_aabb_hzb` 继续严格标记为 `baseline_aabb_depth_proxy`：它只光栅化 AABB 矩形，不是三角形 HZB。AABB+ray MLP、真实几何 HZB 和 NeuralPVS 适配仍是未完成项，不能用现有 proxy 结果填补。

本次执行还修复了两个会污染所有后续指标的问题：正式训练/benchmark 不再自动将 GT 可见实例补进候选，候选漏正样本会直接失败；冻结测试入口不再使用过期的评估器参数契约。前者的详细记录见 `docs/experiments/m0_fixed_validation_calibration_protocol_2026-08-01.md` 和 `docs/evaluation/m1_resource_semantics_audit_2026-08-01.md`，后者及其回归测试见 `docs/experiments/m0_frozen_test_audit_2026-08-01.md`。

### 21.2 评测语义回归覆盖（2026-08-01）

本轮补充了 `docs/experiments/m0_evaluation_semantics_regression_2026-08-01.md` 和
`test_evaluation_protocol_semantics.py`。新增测试覆盖非法/非有限 `visible_weights`、零弱效用姿态的
`not_applicable` 语义，以及 M3 对 pre-test calibration 摘要和 one-shot test 摘要的选择规则；同时修复
pose/GLB 聚合时丢失 `utilityStatus` 的报告缺陷。当前 benchmark 测试总数为 30 项，全部通过。

该结果只通过了 M0 的协议回归子门，不改变正式准入要求。HKUST/Metropolis 训练、最终 bootstrap 校准、
固定特征导出和 one-shot test 仍未完成，因此 M0 总门及投稿主表仍保持未通过状态。

### 21.3 M4 RVL-off 控制变量审计（2026-08-01）

审计发现 loss profile 会覆盖显式 `--rvl-mode off`，使 RVL-off 消融不能真正关闭 RVL。已在训练器中
引入 `resolve_loss_profile`：profile 只提供默认值，显式命令行模式优先；新增回归测试覆盖 `off` 和默认
`evidence` 两条路径。该修复已记录在 `docs/experiments/m4_registered_input_ablation_protocol_2026-08-01.md`。

当前 M4 仍没有正式三种子指标；已有训练进程启动于该修复前，不能追溯视为 RVL-off 或其他正式消融。

### 21.4 M6 runner 清单核对（2026-08-01）

实际注册表包含 7 个 runner：keep-all、train 频率、相机距离、投影 AABB 面积、确定性
`AABB+ray`、AABB depth proxy 和 train view-cell bitset。M6 文档已同步该数量；其中 depth proxy
仍只代表 CPU AABB 深度近似，不是三角形 HZB。学习型 AABB+ray、真实几何 HZB 和 NeuralPVS
同口径适配仍未完成，不能以现有 7 个 runner 代替计划中的完整 M6 门。

### 21.5 M6 固定决策规则校正（2026-08-01）

`keep-all` 和 train view-cell bitset 已在 runner 中声明固定决策模式，但旧评测器仍对其扫描连续阈值。
现已统一为使用固定阈值一次评估；连续几何/学习 runner 继续执行 calibration 阈值网格。该修正避免将
二值查表基线包装成可调连续模型，相关测试和记录见 `m6_visibility_baseline_runners_2026-08-01.md`。

### 21.6 M7 baseline 修正后重跑（2026-08-01）

固定决策口径修正后，已在独立输出目录重新启动 HKUST 和 Metropolis 的完整 calibration baseline
组合。任务使用严格保存候选和真实 GLB 字节，不读取 test；在生成完整 `summary.json` 前不报告数值，
旧摘要只保留为修正前诊断。启动与输出记录见 `m7_unified_download_scheduling_2026-08-01.md`。

### 21.7 M7 非适用效用聚合修复（2026-08-01）

完整 Metropolis calibration 暴露了预算汇总器对 `None`/`not_applicable` 的处理缺陷：同一字段在不同
pose 中同时出现数值和合法非适用值时，旧实现把 `None` 传给 `float`，导致任务在 594 个 pose 遍历完后
失败。修复后的实现只平均实际数值，并保留状态字段；它不填零、不修改 GT、不修改候选，也不改变任何
模型决策。新增回归测试后 benchmark 测试共 30 项全部通过。HKUST 修复后 calibration 已完成；
Metropolis 使用独立 retry1 目录重跑，待摘要生成后才进入 M7 数值审计。

### 21.8 M6/M7 执行核验（2026-08-01）

- 学习型 AABB+ray MLP 已在 HKUST 和 Metropolis 完成完整 validation/calibration；Metropolis 的 calibration 工作点为阈值 `0.20`、weighted recall `0.99118`、useful cull `0.22468`、bad cull `0.00653`。该结果是 L0 元数据冷启动基线，不读取目标 GLB 三角形。
- Metropolis M7 修正后 model-free calibration 已完成 `2,376` 个 pose。七个 runner 中只有 keep-all 和 camera-distance 达到 weighted-recall 安全条件；其余 runner 明确无安全工作点。该摘要支持 baseline 参照，不支持联合下载头主张。
- M6 总门仍未通过：`baseline_aabb_hzb` 仍是 AABB depth proxy，真实三角形 HZB 和 NeuralPVS 适配没有被伪造；M7/M8 总门仍未通过：独立排序器、设备成本、真实网络轨迹和联合级联比较尚未完成。

### 21.9 正式训练收尾链路（2026-08-01）

为保证 M0 不因训练任务结束后的人工衔接遗漏而停在半成品状态，新增并启动
`neural_instance_culling/benchmark/run_formal_training_followup.sh`。它只监听当前登记的两条
`rvl_strong_v2` 空间训练输出；每个输出出现 `calibration_ready_summary.json` 后，先生成不可变 frozen
manifest，再在独立目录中对完整 test split 执行一次 `evaluate_frozen_test.py evaluate`。脚本不扫描 test
阈值、不补 GT 候选、不裁剪候选，也不覆盖既有输出。当前 tmux 会话为 `formal_m0_followup`，该记录在
正式摘要生成前不改变 M0 的“总门未通过”判断。

### 21.10 三角形 HZB 基线生成链路（2026-08-01）

新增 `benchmark/build_triangle_hzb_cache_browser.mjs`，将真实本地 GLB 三角形交给浏览器
Three.js 光栅化，输出线性视深 level-0，并在浏览器中逐级进行最小池化；
`benchmark/triangle_hzb.py` 和 `model_runners.py` 已完成缓存读取、AABB 投影查询和严格候选
接口。一次 1 GLB/1 pose 的 smoke 产生了 7 层、`64x36`、level-0 深度范围
`0.029750–1.000000` 的缓存，Python runner 对 7,405 个候选返回有限结果。

该 smoke 使用的是非完整 GLB 子集，缓存被明确标记 `formalReady=false`，不能进入 M6 正式表格。
完整 HKUST/Metropolis validation/calibration、三角形 HZB 与标准深度渲染的同位姿校验、真实 GPU
后端和性能计时仍未完成；因此 M6 总门仍保持未通过。详细记录见
`docs/experiments/m6_triangle_hzb_baseline_2026-08-01.md`。

### 21.11 M3 正式收尾自动化（2026-08-01）

新增 `neural_instance_culling/benchmark/run_formal_m3_interventions.sh` 和对应执行记录。该脚本等待正式训练产生无 test 结果的
`calibration_ready_summary.json`，并等待 M0 one-shot test 完成后，在完整 validation split 上运行全部代理/上下文推理期干预。
每个变体使用同一候选 CSR、同一 calibration 阈值和同一 pose 计划，保存逐 pose 指标、logit/门控诊断和 10,000 次配对 bootstrap。
它不覆盖已有输出，也不打开 test split。正式指标产生前，M3 仍标记为“未执行正式版本”。

### 21.12 M4 抑制头控制与矩阵收尾（2026-08-01）

审计发现原 M4 变体虽然能屏蔽固定输入，却没有办法单独关闭显式抑制头。模型新增默认开启的
`usesExplicitInhibition` 配置字段和 `--disable-explicit-inhibition` 控制；关闭时保留相同 state-dict
结构，但将抑制输出固定为零，加载器会从 checkpoint 恢复该语义。新增
`run_formal_m4_ablation_matrix.sh`，登记五种输入/抑制变体和三个随机种子，等待现有训练作业结束后按
固定 GPU 槽运行。该脚本已在 `formal_m4_matrix` tmux 会话中启动并等待现有训练进程；当前 M4 正式指标仍未生成。

另有 `run_formal_m4_validation_evaluations.sh` 收尾会话入口，负责在矩阵完成后锁定同一 validation
候选并生成逐 pose 配对 bootstrap；它的输出不会打开 test split。

### 21.13 M6 三角形 HZB 与 M7 独立 RankNet 收尾（2026-08-01）

M6 已完成两场景完整 validation/calibration 的三角形 HZB 缓存和查询：HKUST 覆盖 3,273 个 GLB、1,354
个 pose，缓存约 266 MB；Metropolis 覆盖 3,669 个 GLB、4,464 个 pose，缓存约 878 MB。两套缓存的
level-0 都是 256x144，查询 FOV 为 66°，前端真实渲染 FOV 为 60°，Chrome 构建后端记录为
headless SwiftShader/WebGL。两个场景都没有满足 `weighted recall > 0.99` 的阈值，因此 M6 只通过
“完整几何 warm-cache 基线可复现”子门，总门仍未通过；详细数值见
`m6_triangle_hzb_baseline_2026-08-01.md`。

M7 已完成独立 RankNet 下载排序器的 HKUST/Metropolis 训练和完整 validation/calibration 评测，分别
遍历 664/690 与 2,088/2,376 个 pose，严格使用保存候选集合和本地 GLB 字节。HKUST 的最佳 validation
RankNet loss 为 `0.0435587`，Metropolis 为 `0.2489221`。在独立排序器的弱效用教师下，HKUST 20 MiB
预算的 utility recall 为 `0.8794`（validation），Metropolis 为 `0.8311`；这些结果只证明字节预算
曲线可复现，不能代替可见性安全指标，也不能证明联合级联优于现有后处理。M7 总门仍等待主线模型、
冷/温轨迹、真实解码时间和 paired 调度结果。

本轮 benchmark 回归已达到 33 项单元测试全部通过，visual utility/trajectory self-test、Python 编译、
Node 语法检查和差异检查均通过。正式主线训练、M0 frozen test、M3/M4 消融仍由持久会话继续执行，
本节不把中间 checkpoint 写入投稿主表。

### 21.14 执行状态更新（2026-08-01）

本次更新只写入已经实际产生的产物，不把后台任务的预期结果提前写入主表：

- M0 HKUST：正式空间模型已完成 40 epoch、calibration 安全校准和一次完整 frozen test。阈值为
  `0.02`，test 遍历 `722` 个唯一 pose，`testEvaluationCount=1`；weighted recall 为 `0.997047`，
  pose recall 为 `0.946084`，useful cull 为 `0.759446`，bad cull 为 `0.004613`。冻结入口首次运行曾因
  评测器导入契约错误在读取 test 前失败，失败 claim 被单独保留，修复后使用同一 immutable manifest
  在新目录完成真正测试。
- M3 HKUST：在完整 validation 的 `664` 个 pose 上完成九种推理干预和 `10,000` 次 paired bootstrap。
  代理清零相对 baseline 的 useful-cull 差值为 `-0.001293`，95% CI 为 `[-0.002912, 0.000235]`；
  weighted-recall 差值为 `-0.000202`，95% CI 为 `[-0.000988, 0.001003]`。代理分支会改变输出，
  但当前未满足预注册的独立 useful-cull 增益门槛，不能单独宣称方向代理贡献成立。
- Metropolis 正式主线仍在 GPU 2 上运行，当前约为 `31/40` epoch；其 calibration、冻结 test 和 M3
  结果尚未产生。M4 三种子矩阵和其 validation 配对评估继续等待主线训练完成。
- M5 的 canonical view-cell center 语义入口修复已推送；真实 GLB v3 smoke 通过，但仍是小规模 smoke，
  完整 validation/calibration 图像评价及 one-shot test 图像门尚未通过。
- 回归验证更新为 benchmark Python `35 tests, OK`、前端当前完整性 smoke 通过、M9 前端回归 `8` 项通过。
  本轮提交均采用追加 commit 并推送到 `origin/main`，没有删除或改写既有历史。

### 21.15 HKUST 正式模型导出（2026-08-01）

HKUST 已完成 `pvs_directional_occlusion_proxy_encoder_rvl_strong_v2_full40_hkust_spatial_fov66_seed20260801_protocolfix_retry2`
的前端导出。导出器原先只接受旧式 `eval_summary.json`，而正式训练使用
`calibration_ready_summary.json`；本次补充了正式校准摘要解析，并强制验证
`testEvaluationCount=0`、冻结阈值一致性、weighted recall 点估计和 bootstrap 下限，再写出
`pvs_directional_occlusion_proxy_encoder_rvl_strong_v2_full40_best`。导出结果阈值为 `0.02`，
calibration weighted recall 为 `0.9930808`，单侧 bootstrap 95% 下界为 `0.9909417`，前端运行资产为
`13,844,856` bytes。默认前端、静态完整性 smoke、M9 审计和部署打包脚本已同步切换；旧 `w042`
仅保留为历史资产。

本次变更没有改变模型权重或 test 结果，也没有把 M5 图像门、Metropolis M0/M3 或 M4 消融标记为完成。
M9 空间 AABB 审计在强模型元数据下仍集合一致，但索引 p50/p95 比全扫描更慢，空间分页和性能优化仍是
未完成工作。详细记录见 `docs/experiments/m0_hkust_formal_export_2026-08-01.md`。

### 21.16 M6 NeuralPVS 与 M10 设备审计（2026-08-01）

NeuralPVS 审计确认当前仓库没有可公平运行的原始或适配实现。论文方法需要真实深度片段、视锥体体素表示、体素到实例的映射以及相应的三维网络；现有 Color-ID 实例并集、AABB 投影 froxel 诊断和 AABB depth proxy 都不能替代这些输入。因此 M6 只保留 L0 元数据基线和 L2 三角形 HZB warm-cache 子门，NeuralPVS 子门未实现，不能生成或宣称伪造的比较结果。详见 `docs/experiments/m6_neuralpvs_baseline_audit_2026-08-01.md`。

M10 审计完成桌面 headless smoke：WebGPU 路径可运行，但实际适配器是 SwiftShader；没有硬件独显/集显的正式分布，也没有两个 Android 性能档位、网络轨迹或完整分项计时。因此当前只证明浏览器启动和 Cold-0 请求顺序，不能宣称移动端实时。详见 `docs/frontend/m10_device_benchmark_2026-08-01.md`。

### 21.17 M5 HKUST validation 图像门（2026-08-01）

冻结模型在完整 `213` 个 validation view-cell 上完成真实 GLB、实例级 Color-ID 评价。聚合
`miss-pixel rate=0.4947%`，但按 view-cell 的均值为 `0.5409%`、p95 为 `3.1973%`、最大值为
`13.1422%`；self-consistency PER 为 0，说明渲染绑定链路自身没有出现错误。由于计划门槛要求
mean `<0.5%` 且 p95 `<1%`，M5 图像门保持 No-Go。漏像素集中在少数大型高贡献构件，不能归因于
低视觉效用长尾，也不能通过 test 阈值扫描修复。完整记录见
`docs/evaluation/m5_hkust_formal_validation_image_2026-08-01.md`。

### 21.18 M3/M7 正式结果补录（2026-08-01）

- M3 两场景正式推理干预均已完成。HKUST 的代理清零相对 baseline 的 useful-cull 差值为
  `-0.001293`，95% CI `[-0.002912, 0.000235]`；Metropolis 的代理清零会把 weighted recall
  从 `0.992301` 降至 `0.223644`，跨实例代理置换降至 `0.696713`。代理确实参与模型决策，
  但清零时的额外剔除伴随严重漏检，不能称为有效效率增益。M3 机制诊断完成，独立贡献主张等待
  M4 三种子消融。
- M7 HKUST 正式 validation/calibration 已完成。主线 calibration 工作点为阈值 `0.05`、weighted
  recall `0.990532`、useful cull `0.881718`、bad cull `0.005222`。在 20 MiB 弱效用预算下，
  validation 的 visibility-only、current-cascade、visibility-gated、independent-utility 效用召回
  分别为 `0.9440`、`0.9254`、`0.9421`、`0.8794`；这些不是像素效用或真实下载时间结论。
- M5 HKUST calibration 图像评价已完成，聚合 miss-pixel rate 为 `0.2660%`，按 view-cell 均值为
  `0.2935%`、p95 为 `1.2934%`，因此均值门通过但 p95 图像安全门仍 No-Go。M7 Metropolis validation/calibration 已完成，主线
  calibration 工作点为阈值 `0.48`、weighted recall `0.990281`、useful cull `0.580003`、bad cull
  `0.008333`，20 MiB validation 的 visibility-only/current-cascade 效用召回为 `0.9912/0.9921`。
  M4 三种子训练已启动。M0、M3、M5、M6、M7、M10 的正式门控状态仍以各自报告为准，尚未将任何
  No-Go 门改为通过。

### 21.19 M5 HKUST calibration 图像门补录（2026-08-01）

HKUST calibration 的 `168` 个 view-cell 已在同一 Chrome 页面中完成真实 GLB、实例级 Color-ID reference
和 prediction 渲染。聚合 miss-pixel rate 为 `0.2660%`，按 view-cell 的均值为 `0.2935%`、p95 为
`1.2934%`，self-consistency PER 为 `0`。均值 `<0.5%` 的子门通过，但 p95 `<1%` 的长尾门失败；漏像素
集中于少数高视觉贡献构件。完整结果见 `docs/evaluation/m5_hkust_formal_calibration_image_2026-08-01.md`。

该结果支持“实例级图像评价管线可以运行且绑定自洽”，不支持当前模型通过图像安全门。test 图像评价仍封存，
不能依据 calibration 或 test 重新扫描阈值掩盖长尾错误。

### 21.20 M10 移动设备测试方案固化（2026-08-01）

当前服务器没有实体 Android 设备、ADB 和可核验的硬件移动 WebGPU 适配器，因此 M10 不生成移动端性能
数值，也不把桌面 SwiftShader smoke 外推为移动端结果。移动设备测试方案已写入
`docs/frontend/m10_device_benchmark_2026-08-01.md`，后续按固定 commit、模型、FOV、viewport 和三条
可回放轨迹执行高性能/中端两档真实 Android 设备测试，分别覆盖冷缓存、温缓存、受控 Wi-Fi/4G trace
以及 `256` 至 `16k` 的候选规模桶。

正式数据必须记录硬件 WebGPU adapter、空间候选、特征页命中、buffer 更新、dispatch、readback、实例
聚合、GLB 下载/解码/上传、首个有用画面、主线程时间、帧时间、内存和温度，并按设备/场景/缓存/轨迹
报告 p50/p95/p99。10k 候选的建议门槛为移动端查询 p95 `<50 ms`、主线程附加工作 `<2 ms` 且无明显
帧停顿；没有真实样本的桶必须标记缺失。该方案已固化，但 M10 质量门仍为 No-Go，不能用计划替代实测。

### 21.21 M9 跨桶 AABB 索引正确性修复（2026-08-02）

强制空间索引审计在 `far=100/500/1000` 下发现，完整 AABB 的 Three.js 保守视锥测试结果可能在空间桶拆分后
消失，导致索引漏候选。当前 `InstancePVS` 将所有跨桶实例放入溢出列表，只有完全处于单桶内的实例使用桶索引。
修复后四种 far、每种 128 个 HKUST pose 均为 `0/128` 集合差异；结果见
`docs/evaluation/m9_spatial_aabb_index_audit_2026-08-02.json` 和
`neural_instance_culling/benchmark/out/m9_spatial_aabb_index_far*_20260802.json`。

该修复只通过空间索引正确性子门，far=2000 时索引 p50 `12.20 ms` 高于全扫描 `0.69 ms`，因此不形成空间索引
加速主张，默认查询仍保留超过桶数量上限时的全量扫描回退。移动设备仍无真实证据，沿用 M10 的测试方案和 No-Go
状态。

### 21.22 M12 后退相机一致性修复（2026-08-02）

M12 的首轮 WGSL parity 参考未应用当前前端的后退相机位移，因而比较了不同查询位置。现已让参考生成器读取导出的
`predictionCameraMode` 和 `pvsBackOffsetM`，并保存实际 `predictionPosition`。协议对齐后的 16 个固定 case、7,845 个
候选值中，FP16/WGSL 阈值翻转率为 `0`，可见性 logit 平均绝对误差/p99 为 `0.000507/0.001489`，下载排序
Spearman/top-10% Jaccard 为 `0.999996/1.000000`。原始结果见
`neural_instance_culling/benchmark/out/m12_webgpu_parity_hkust_strong_v2_back_camera_20260802/`。

该结果通过 M12 数值一致性子门，但浏览器适配器为 SwiftShader，不能转化为硬件 GPU 或移动端性能结论。M10 仍保持
真实设备证据缺失；移动端测试协议和相对 FP16 的预注册正确性门限见
`docs/frontend/m10_device_benchmark_2026-08-01.md`。

### 21.23 M8 当前主线轨迹回放与 M11 少样本重试（2026-08-02）

M8 首次当前主线回放把主线下载头和独立 RankNet 放在同一次 `current-cascade` 评估中，因 RankNet 没有
下载头被评估器拒绝；六个失败日志保留。随后新增 `run_m8_formal_trajectory_replay.sh`，将当前级联与
独立排序器拆为两个兼容的 score-mode 批次，在相同轨迹、成本、带宽和预算下重跑。三条固定轨迹均值为：

| 场景 | 当前级联最终弱效用召回 | RankNet 最终弱效用召回 | 当前级联 20 MiB | RankNet 20 MiB |
|---|---:|---:|---:|---:|
| HKUST | `0.8749 ± 0.0153` | `0.8542 ± 0.0175` | `0.1982 ± 0.0102` | `0.1948 ± 0.0199` |
| Metropolis | `0.9059 ± 0.0170` | `0.6700 ± 0.0227` | `0.3004 ± 0.0398` | `0.1509 ± 0.0443` |

这些数值是三条确定性离线轨迹的均值和样本标准差，效用定义为 `log1p(visible_weights)`，不是像素覆盖率；
没有真实网络、设备解码/上传和多种子 paired bootstrap，因此 M8 联合调度质量门仍未通过，只作为当前主线
离线调度诊断。M11 的 Metropolis 少样本 `retry2` 在 1% 适配中触发显存不足，保留失败目录；`retry3`
已改用单 pose 训练和冻结 test 批次、较小特征导出批次、默认 AMP 与显存分段，当前正在 GPU 3 运行，
尚未产出适配指标。

### 21.24 NeuralPVS 官方实现审计更正（2026-08-02）

上一条 M6 记录把“当前仓库没有公平适配链路”误写成“公开资料不足以复现”。该表述已更正：NeuralPVS
官方训练/推理仓库可以获得，审计副本 commit 为 `946088616cad18de81cde12fecd6ab204e52eac9`，其中包含
OACNN/VNet、三维交错模块、数据集读取器、训练器、推理器和 Dice/focal/repulsive/no-guess 类损失实现；
官方数据格式为 `gv/*.bin.gz` 与 `pvv/*.bin.gz` 的 bit-packed 三维体素网格。

更正不改变 M6 质量门：当前项目仍没有把三个场景的真实深度/实例 ID 采样转换为官方 froxel 语义，也没有
完成 froxel 到实例的保守映射、严格 candidate CSR runner、同 FOV/view-cell/calibration/test 评测和冷启动
资源核算；`slm_pvs` 当前也没有官方所需的 `spconv`/`cupy` 依赖。因此现有 AABB depth proxy、三角形 HZB
warm-cache 和实例级 MLP 结果仍不得冒充 NeuralPVS，M6 保持 `No-Go / adaptation not implemented`。
详细审计见 `docs/experiments/m6_neuralpvs_baseline_audit_2026-08-01.md`。

### 21.25 M4-v2 完整因子消融与安全工作点计划（2026-08-03）

上一版 M4 只用 `full - geometry_context_ray` 的 useful-cull 和 bad-cull 门判定方向代理路线，不能区分
“方向代理是否被模型使用”“方向代理是否改善分类”“方向代理是否改善安全约束下的剔除效率”以及“方向代理是否
改善系统资源成本”。本节冻结 M4-v2 的执行协议；在本节和对应独立协议提交前，不启动 M4-v2 新评测或新训练。

#### 计划边界和不可变产物

- 不修改、不覆盖 `neural_instance_culling/benchmark/out/m4_formal_matrix_validation_summary.json`、
  `neural_instance_culling/benchmark/out/m4_formal_route_decision.json` 和
  `docs/evaluation/m4_formal_matrix_validation_2026-08-02.md`。
- M4-v2 所有结果写入 `neural_instance_culling/benchmark/out/m4_formal_matrix_validation_v2/`；路线判定写入
  `m4_formal_route_decision_v2.json` 和同名 Markdown；报告使用独立的
  `docs/evaluation/m4_formal_matrix_validation_v2_*.md` 名称。
- 现有 B/C/D 变体的逐 pose validation interventions 优先复用，只重新汇总，不重复训练；缺失的 A 变体
  `geometry_context_ray_no_inhibition` 没有可证明等价的现成 checkpoint，必须用独立输出目录训练或明确把
  推理期近似降级为诊断而不能纳入正式因果主表。
- 现有 `aabb_ray` 和 `geometry_ray` 继续保留为逐级输入基线，不参与 2×2 因子效应的代数计算。

#### 2×2 因子设计

因子一是方向遮挡代理是否输入查询头；因子二是显式遮挡抑制头是否启用。四个正式成员固定为：

| 变体 | 方向代理 | 显式抑制 | 代号 |
|---|---|---|---|
| `geometry_context_ray_no_inhibition` | 关闭 | 关闭 | A |
| `geometry_context_proxy_ray_no_inhibition` | 开启 | 关闭 | B |
| `geometry_context_ray` | 关闭 | 开启 | C |
| `full` | 开启 | 开启 | D |

三种 seed 仍为 `20260801`、`20260802`、`20260803`。正式因子差异逐 pose 配对计算：方向代理无抑制主效应
为 `B-A`，显式抑制无代理主效应为 `C-A`，有抑制时方向代理效应为 `D-C`，交互效应为
`D-B-C+A`。不能把 `D-C` 单独解释为方向代理的全部贡献。

#### 指标和数据语义

对每个 pose 使用严格存储的后退相机候选 `C`、真实可见集合 `G` 和预测集合 `P`：
`TP=P∩G`、`FP=P-G`、`FN=G-P`、`TN=C-(P∪G)`。所有成员必须使用同一 validation pose 顺序、同一候选
集合、同一 GT 和同一候选哈希；禁止 GT union、候选截断和前端白名单修复。

每个工作点同时输出 pose-level 宏平均和所有 pose 合并的 aggregate 结果，包括 pose/aggregate recall、
weighted recall、visual utility recall、bad cull、precision、F1、Jaccard、accuracy、balanced accuracy、
specificity、useful cull、平均 TP/FP/FN/TN、平均预测数、预测/候选比、预测/GT 比、GLB 数量和字节削减。
`visible_weights` 仍只能解释为可见重要性代理，不能称为真实像素覆盖率。若同位姿图像结果或统一 GLB
成本结果不存在，字段写为 `not_available`，不能从 weighted recall 或 useful cull 推断 miss-pixel、
wrong-ID、extra-pixel 或下载收益。

运行成本字段至少记录 forward latency、输入维度、固定特征表字节；只有真实浏览器数据存在时才记录 WebGPU、
主线程和内存，不用服务器 GPU 或 SwiftShader 填补移动端数字。

#### 阈值工作点和统计方法

每个 checkpoint 只能使用自己的 calibration 集冻结阈值，validation/test 不选阈值。安全工作点必须同时满足
pose recall `>=0.95`、weighted recall `>0.99`，并在校准协议支持时满足 weighted recall 置信下界 `>0.99`；
若无阈值满足，报告 `no_qualified_safety_workpoint`，不降低门槛参与排名。安全工作点在约束内最大化 useful
cull；best F1、最高 precision 和固定阈值只作为诊断工作点。

额外记录但不替代原始指标：

```text
safety_factor = min(1, pose_recall / 0.95) * min(1, weighted_recall / 0.99)
safety_adjusted_useful_cull = useful_cull * safety_factor
```

三 seed 使用同一 pose 的 paired bootstrap，先按 seed 聚类，再在 seed 内对 pose 重采样，至少 10,000 次。对
`B-A`、`C-A`、`D-C`、`D-B` 和 `D-B-C+A` 的所有核心指标输出差值、95% 区间、方向和是否跨零。

#### 路线判定

为避免“明显下降”在结果产生后临时解释，M4-v2 将 recall/weighted recall 差值 95% 区间下界低于 `-0.01` 登记为安全层失败，bad-cull 差值区间上界固定不超过 `+0.002`；两项均在 pose 宏平均和 aggregate 口径检查。

路线判定分三层。第一层要求方向代理不造成 pose/weighted recall 的明显安全下降，bad-cull 增量满足预注册上限。
第二层要求在相同安全工作点下，precision、balanced accuracy、F1、useful cull 或预测/GLB 成本至少一项的
配对区间稳定改善；第三层要求相同视觉效用下 GLB 字节/首屏时间下降，或 miss-pixel/p95 miss-pixel 稳定改善。
只有“被模型使用”但所有效果指标区间跨零的方向代理，才降级为辅助表征。不能用任意加权分数、降低阈值或
单独 useful-cull 增益替代上述判定。

#### 执行和验收顺序

1. 提交本节和 `m4_formal_matrix_validation_v2_protocol_2026-08-03.md`，冻结协议后再执行。
2. 审计 15 份旧 `interventions.json` 的 schema、阈值来源、pose 数、候选 hash 和 TP/FP/FN/TN，生成 v2
   独立 manifest；不读取 test。
3. 为 A 变体补齐三种 seed 的独立训练/validation/calibration；B/C/D 和逐级基线只复用已有有效产物。
4. 用独立 v2 汇总脚本生成 pose 宏平均、aggregate、资源字段、四个主效应和交互 bootstrap；字段缺失必须显式
   标注，不默认为零。
5. 用独立 v2 路线脚本生成安全工作点和三层路线判定，执行 schema/self-test/unittest。
6. 生成 M4-v2 正式报告，更新 `docs/README.md`，不改现有 Route B 结论或默认前端模型。

本计划的可复现协议正文见 `docs/experiments/m4_formal_matrix_validation_v2_protocol_2026-08-03.md`。
