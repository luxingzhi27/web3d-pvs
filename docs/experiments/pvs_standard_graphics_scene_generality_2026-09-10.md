# 标准图形学场景的 PVS 泛化实验方案

日期：2026-09-10；2026-09-11 更新执行状态

状态：方案冻结并执行中。Sponza 已完成正式采样、Pose CSR 和 train-only 三角形关系，Full 三种子与 AABB 扫描正在运行；Big City 正在执行正式硬件 Color-ID 采样。

## 1. 结论与实验定位

标准图形学场景不需要具备 BIM 构件语义，也不需要存在“一个 GLB 原型被多个实例复用”。当前框架实际要求的是一组带稠密编号、世界 AABB、固定几何特征和可见性标签的独立可渲染单位。对没有原型复用的场景，直接使用：

```text
一个 renderable unit
  -> 一个 componentGlobalId
  -> 一个独立 GLB/resource ID
  -> 一行固定运行特征
```

即 `instanceCount == globalGlbCount`，且 `instance_to_glb[unit_id] == unit_id`。这不会改变 V4 模型、候选生成、Color-ID 采样、关系特征或统一指标的数学含义。

这组实验的主要任务是证明：

1. 方法能够在没有 IFC/BIM 语义、没有原型复用的普通图形场景上进行单位级 from-region PVS；
2. 在相同 renderable units、候选、GT 和 test split 上，Full V4 相比 Keep-All 和 AABB + Ray MLP 具有更好的安全-剔除折中；
3. 与 WebGPU Batched Geometry-shell Hi-Z 相比，神经方法用更小的启动资产在正式几何尚未到达前给出可用 PVS；
4. 推理成本随候选单位数的增长规律在 BIM 和非 BIM 场景中一致。

它不用于证明 BIM 的原型复用收益。HKUST/IFCBench 的 GLB 数量、字节削减和真实 progressive streaming 仍是资源复用与网络收益的主实验；标准场景的正文结果优先报告单位级剔除、图像安全、启动资产和运行时间。

## 2. 论文中的问题定义

论文统一使用 **renderable unit** 表示可独立判断可见性的静态渲染单位：

- BIM 中的 component instance 是 renderable unit；
- 普通场景中的源 scene-graph object 或自动生成的 mesh cluster 也是 renderable unit；
- streaming unit 只在资源下载实验中使用。标准场景采用一单位一资源时，两者数值相同，但概念仍需区分。

因此论文方法不是“只能预测 IFC 构件”，而是：

> 对独立可渲染单位的场景专属可见性信号进行离线编码，并在客户端以轻量视角查询恢复 view-cell PVS。

当前模型仍是逐场景预处理和逐场景训练。多个标准场景可以证明表示不依赖 BIM 语义，但不能称为对未见场景的 zero-shot generalization。论文应把方法定位为 scene-specific visibility codec，类似场景专属 BVH、PVS、lightmap 或 LOD，而不是通用场景分类器。

## 3. 与 NeuralPVS 的关系

[NeuralPVS](https://arxiv.org/abs/2509.24677) 在 Sponza、Robot Lab、Viking Village、Big City 和 Industrial Set v3.0 上评价，正式结果使用 60 秒、60 Hz 的相机轨迹，即每场景 3600 帧。其 GT 每个 viewcell 使用 1000 个采样位置。官方[训练仓库](https://github.com/windingwind/neuralpvs)和[渲染仓库](https://github.com/DerThomy/NeuralPVS)已经公开。

采用其中的场景有两个价值：建立与近期 from-region PVS 工作的场景联系，并检验本文表示对非 BIM 几何分布的适用性。但两种方法的输出和训练目标不同：

| 项目 | NeuralPVS | 本项目 |
|---|---|---|
| 运行输入 | 当前场景的 froxelized geometry grid | 场景专属固定单位特征和当前视角 |
| 输出 | froxel PVS | renderable-unit PVS 和单位分数 |
| 训练定位 | synthetic scenes 训练后应用到未见场景 | 每场景离线训练和压缩 |
| 主要系统问题 | 几何可用后的快速 from-region PVS | 正式几何下载前决定保留/下载哪些单位 |

因此不能把 NeuralPVS 论文中的 FNR、FPR、时间或内存数字直接放进同一数值表。只有在以下条件全部满足后，才能增加正式 NeuralPVS 数值行：

1. 使用同一场景版本、相机、view-cell、候选范围和 GT 采样；
2. 将官方 froxel 输出通过预先登记的保守规则映射为同一 renderable-unit 集合；
3. 使用同一指标实现重新计算，而不是摘录论文结果；
4. 明确标注 NeuralPVS 是 zero-shot predictor，而本文是 scene-specific codec。

第一阶段不实现这个跨输出空间映射。标准场景主表比较 Keep-All、AABB + Ray MLP、WebGPU Batched Geometry-shell Hi-Z 和 Full V4；NeuralPVS 作为相关工作和场景选择依据。官方实现能够稳定运行后，再单独登记 NeuralPVS-aligned protocol，不能临时修改当前 test 口径。

本项目使用的 Sponza 固定为 Khronos glTF 分发，不声称与 NeuralPVS Unity 工程中的 `Sponza_Modular.FBX` 是逐三角形相同版本；因此 Sponza 只提供共同标准场景语境，不用于直接复现或比较 NeuralPVS 论文数字。Big City 则来自 NeuralPVS 官方资产，但在候选、GT 和输出单位未对齐前同样不放入直接数值行。

## 4. 场景选择与许可门

### 4.1 首选矩阵

| 优先级 | 场景 | 作用 | 当前许可/来源状态 |
|---|---|---|---|
| P0 | Sponza | 室内、多层遮挡，先打通完整转换和训练链 | 固定为 Khronos `glTF-Sample-Assets/Models/Sponza` 当前检出版本；模型文件声明 CRYENGINE Limited License，实验只保留本地派生资产，不随论文结果包再分发 |
| P1 | Viking Village | 大型室外村落，与 NeuralPVS 评价场景对齐 | Unity Asset Store EULA；允许本地论文实验不等于允许再分发转换资产 |
| P1 | Big City | 大型室外城市，与 NeuralPVS 评价场景对齐 | 已从 NeuralPVS 官方资产取得 `Assets/BigCity/scene.gltf`，来源审计通过，并已完成本项目固定单位转换；派生 GLB 仅作为本地实验产物，不据此声明公开再分发许可 |

NeuralPVS 论文报告的 primitive 规模约为 Sponza `0.3M`、Viking Village `7.2M`、Big City `15.7M`。这些是该论文场景版本的说明；本项目 Big City 的来源审计另记录了 `29` 个源 primitive、`15,711,990` 个源三角形和 `12,855,583` 个源顶点，不能用论文概数替代本项目统计。

如果 Viking Village 因许可或转换失败不能使用，按固定顺序替换为 San Miguel、Bistro 或 Power Plant。替代场景必须先记录来源版本、论文使用许可、是否允许发布派生 GLB 和相机轨迹来源；不能为了得到更好的模型结果临时换场景。

### 4.2 最小和目标配置

- 最小可发表补充：Sponza 加一个大型室外标准场景；
- 目标配置：Sponza、Viking Village、Big City 三个场景；
- 不在本轮增加 Robot Lab 和 Industrial Set v3.0：前者的官方第三方清单标记 license unknown，后者受 Unity Asset Store EULA 约束且规模较小。

每个场景在转换前输出 `scene_source_audit.json`，只记录来源、版本、坐标单位、格式、材质类别、三角形/节点统计和许可边界。许可证不清楚时停止该场景，不产生可被误用的正式结果。

### 4.3 Big City 已完成产物

Big City 的来源记录为 `neural_instance_culling/dataset/out/standard_graphics_sources/bigcity_source_audit.json`，源文件为 `neural_instance_culling/dataset/out/standard_graphics_sources/neuralpvs_official/repo/Assets/BigCity/scene.gltf`。审计结果为 `29` 个静态 `OPAQUE` primitive、`15,711,990` 个三角形、`12,855,583` 个顶点，无动画、skin、BLEND 或排除 primitive。

固定 `128 KiB` 目标的转换产物位于 `neural_instance_culling/dataset/out/standard_graphics_scenes/bigcity_128k/assets/`，并包含 `scene_source_audit.json`、`scene_audit.json`、`conversionManifest.json`、`runtimeVisibilityMeta.json` 和 `glbIndex.json`。转换 manifest 明确 `oneUnitPerResource=true`，生成 `2734` 个 renderable units、`2734` 个 instance 和 `2734` 个 GLB/resource；实际单位 GLB 字节为 p50 `93,464 B`、p95 `116,968.2 B`、最大 `131,088 B`，总计 `260,906,928 B`。这些是来源与资产转换事实，不是 PVS、图像或运行时性能结果。

### 4.4 Sponza 已完成产物

Sponza 固定为 Khronos `glTF-Sample-Assets` 的 `Models/Sponza/glTF/Sponza.gltf` 分发。来源审计记录 `103` 个静态 primitive、`262,267` 个三角形、`192,496` 个顶点，其中 `89` 个 OPAQUE、`14` 个 MASK，没有 BLEND、动画或 skin。该分发的模型许可文件声明 CRYENGINE Limited License，因此只用于本地论文实验，不将源文件、纹理或派生 GLB 纳入可公开结果包。

固定 `128 KiB` 转换产物位于 `neural_instance_culling/dataset/out/standard_graphics_scenes/sponza_128k/assets/`，生成 `132` 个一单位一资源的 renderable units，总 GLB 字节为 `7,188,496 B`，单元字节 p50 `48,476 B`、p95 `121,103.8 B`、最大 `125,244 B`。其中 `110` 个输出单位为 OPAQUE、`22` 个为保留 alpha 纹理的 MASK。AABB 体积碰撞判定会把非闭合建筑 shell 内部全部排除，因此 Sponza 正式计划改用真实三角形最近表面距离；非闭合 shell 不执行内部实体判定，也不把输出宣称为 navmesh。

### 4.5 固定几何表与标准场景 HZB 外壳

两个标准场景均已按每单位 `1024` 个归一化表面点生成离线点缓存。Sponza 为 `132/132` 成功、Big City 为 `2734/2734` 成功，均无 decode failure、fallback 或 empty geometry。随后使用同一个冻结的 HKUST 几何编码器导出 96D float16 固定表，得到 Sponza `[132,96]` 和 Big City `[2734,96]`；两表数值均有限。该编码器不在标准场景上更新，标准场景的 train/calibration/validation/test 只用于后续 PVS 模型，因此不存在目标场景 test 对几何编码器的监督泄漏。论文必须将其描述为共享冻结几何特征提取器，不能描述成标准场景专属预训练。

标准场景 lossless shell 也已导出。Sponza 外壳仅含 `110` 个确定 OPAQUE 单位、`227,327` 个三角形，22 个 MASK 单位不写遮挡深度；压缩几何为 `1,732,508 B`，完整运行资产为 `1,844,914 B`。Big City 的 `2734` 个单位均为 OPAQUE，外壳含 `15,711,990` 个三角形；压缩几何为 `103,401,989 B`，含 AABB、映射和 metadata 的完整运行资产以 offline report 为准。它们目前只是 HZB 启动资产，不是 HZB 精度或耗时结果；equal-asset shell 必须等对应神经运行资产冻结后再导出。

## 5. 固定 renderable-unit 转换协议

### 5.1 原则

单位化必须只读取源 scene graph、几何、材质和压缩字节，不能读取可见标签、模型分数或 HZB 结果。所有场景使用同一规则，避免按场景手工切分来改善结果。

默认目标为约 `128 KiB` 压缩几何/单位。该数值是 Web streaming 粒度，不是三角形数量门限。正式主结果固定使用 `128 KiB`，不按场景调参；`64/128/256 KiB` 只作为一个代表场景上的粒度敏感性实验。

### 5.2 转换步骤

1. 读取 glTF/GLB；Unity 场景先由固定版本 Unity 导出为 glTF，并保留节点路径、primitive、世界变换和 alpha mode。
2. 将静态节点世界变换固化到几何；动画、骨骼和动态对象不进入本轮静态 PVS 实验，并单独统计排除数量。
3. 每个源 renderable node/primitive 先作为一个自然单位。多个节点引用同一 mesh 也不做原型复用，分别展开为独立单位，以保持所有标准场景的一对一资源协议。
4. 对超过目标的 primitive，先按三角形连通分量分组，再按三角形质心的 Morton 顺序进行确定性空间切分，直到每个输出单位接近目标大小。
5. 对同一源节点、同一透明类别且空间相邻的小分量，按 Morton 顺序贪心合并到目标大小；不跨源节点合并，不混合 `OPAQUE`、`MASK` 和 `BLEND`。
6. 每个单位保留完整三角形，不做减面；顶点仅在单位内部重索引。输出后使用固定 Meshopt 参数压缩，并报告实际 p50/p95/最大单位字节。
7. 按源节点路径、primitive 序号和 cluster 序号排序后分配连续 `unit_id`，不得依赖文件系统遍历顺序。
8. 每个单位输出一个 `task-0/glb/LOD0/sub_<unit_id>.glb`，`globalGlbId=componentGlobalId=unit_id`，并生成 `sceneWeb.json`、`glbIndex.json`、`runtimeVisibilityMeta.json` 和 `conversionManifest.json`。

目标大小是软目标。为了命中精确字节而反复依据可见性拆分没有科学意义；正式统计必须报告实际字节分布、单位数、三角形分布和 AABB 尺度分布。

### 5.3 材质与可见性边界

- `OPAQUE` 静态单位进入全部模型和 HZB 实验；
- `MASK` 单位在 reference Color-ID 中保留 alpha test，作为 occludee 评价；geometry-only HZB 不下载 alpha texture，因此不把它当作遮挡深度写入者，离线关系证据也不把其完整三角形当成实体遮挡面；
- `BLEND` 单位不进入静态遮挡学习与剔除候选，运行时始终保留，并在场景统计中报告数量和字节；
- 双面几何在 reference render 中遵循原材质；geometry-shell HZB 继续关闭背面剔除以保持保守；
- 缺材质或 alpha 语义不明确的 primitive 按非遮挡物处理，不能为了提高 HZB 剔除率假设为不透明。

Color-ID 采样输出单位 ID 和屏幕覆盖 parts-per-million。同一 view-cell 的单位 GT 对成功 subpose 取并集，同一单位的 `visible_weight` 取 subpose 最大屏幕覆盖，和现有数据集协议一致。

## 6. 数据集与训练协议

每个标准场景独立完成：

```text
source scene
  -> deterministic renderable-unit conversion
  -> runtime meta + one-to-one unit/resource map
  -> legal camera-space audit
  -> representative view-cell plan
  -> hardware Color-ID subpose sampling
  -> train/calibration/validation/test split
  -> triangle relation evidence + fixed 96D geometry
  -> Full V4 and AABB + Ray MLP training
  -> calibration-frozen validation selection
  -> one frozen test read
```

截至 2026-09-11，Big City 与 Sponza 已完成来源审计、deterministic conversion 和四路空间相机计划。固定 view-cell 半尺寸为 `0.5/0.5/0.25 m`，外接半径 `0.75 m`，另加 `0.05 m` 表面余量，因此中心到障碍的安全半径为 `0.8 m`。Big City 以扩张单位 AABB 保守排除碰撞，得到 `674` 个中心和 `8088` 个 pose，train/calibration/validation/test 为 `5916/480/984/708`；Sponza 以 `262,267` 个真实三角形的最近表面距离排除碰撞，得到 `202` 个中心和 `2424` 个 pose，四路为 `1728/132/276/288`。两场景均尚无 Color-ID subpose 采样、Pose CSR、关系证据、固定特征、Full/AABB 训练、calibration/validation/test、Hi-Z 结果或性能数字；因此本节后续步骤仍是待执行协议。

### 6.1 相机与 view-cell

- 主协议继续使用真实显示相机 `60 degrees` 和后退候选/模型相机 `66 degrees`；aspect 从实际 viewport 读取。
- 代表相机沿可导航区域自动生成，Sponza 只在建筑可行走空间，室外场景沿道路/开放地面；不得在墙内、地下或几何内部均匀撒点。
- view-cell 的物理半尺寸根据场景单位尺度预先登记，不能直接套用 HKUST 的 `2 m` 或 IFCBench 的 `2.5 m`。
- 每个 view-cell 的方向固定，subpose 只改变合法位置；GT 是所有成功 subpose 的可见单位并集。
- 主训练采样数沿用当前可承受的场景协议；另在固定 100 个 validation cells 上补到 128 个嵌套采样点，报告 GT 收敛。不能把 NeuralPVS 的 1000 点数字写成本文已经执行的采样量。

本轮标准场景已预登记 view-cell 半尺寸为 `0.5/0.5/0.25 m`，主数据每 cell 使用 `16` 个确定性伪随机 subpose。由此 Big City 和 Sponza 的待渲染 Color-ID 相机数分别为 `129,408` 和 `38,784`。NeuralPVS 文献中的 `1000 positions/view-cell` 只在 summary 中标为 related-work reference，不是本文实际采样数。

### 6.2 Split 与模型

- 先按空间区域分块，再在区域层分配 `train/calibration/validation/test`，避免同一走廊或相邻相机泄漏到多个 split；
- 比例固定为约 `72/8/10/10`，分别对应 train/calibration/validation/test；实际唯一 view-cell 数进入场景统计；
- Full V4 使用现有 96D 几何、28D 结构化生存场和相同综合损失，不增加标准场景专用头；
- AABB + Ray MLP 使用与 Full 相同的训练步数、三 seed、安全校准规则和候选集合；
- 每个 checkpoint 只在 calibration 冻结阈值，validation 选模型，test 只读取一次；
- 先用 Sponza 完成单 seed 管线 smoke，确认语义后仍需执行登记的三 seed 正式训练，smoke 不能进入论文表格。

## 7. 正式比较与指标

所有方法使用同一单位、相同 `66 degrees` candidate CSR、相同区域 GT、相同 split 和统一 evaluator：

| 方法 | 标准场景中的作用 |
|---|---|
| Keep-All | 安全上界和零剔除参照 |
| AABB + Ray MLP | 检验普通包围盒/视角分类是否足够 |
| WebGPU Batched Geometry-shell Hi-Z | 几何已作为单独启动外壳到达后的强在线遮挡基线 |
| Full V4 | 本文场景专属紧凑 PVS 表示 |

本文 Hi-Z/HZB 只使用唯一算法 `opaque depth -> max pyramid -> conservative projected AABB test`，外壳 metadata 使用修正后的 `geometry-shell-hzb-v2` schema。它同时报告 Point60 和 Region66；Region66 使用与学习方法相同的登记 subpose 并集，所有 subpose 的总运行时间全部计入；Point60 只证明当前相机遮挡测试能力，不能和一次查询覆盖区域的模型时间混写。

### 7.1 主指标

| 类别 | 必须报告 |
|---|---|
| 排序 | pose-macro PR-AUC、对应 pose prevalence、AP lift；aggregate AP 放附录 |
| 画面安全 | Visible Recall、False Occlusion Rate、weighted recall、weighted recall 单侧 95% LCB |
| 遮挡能力 | Occlusion Recall（即 specificity）、Useful Cull Ratio、Bad Cull Ratio |
| 分类诊断 | precision、balanced accuracy、平均候选和平均保留单位数 |
| 图像 | PER、miss-pixel、wrong-ID、extra-pixel，均给 mean/p95/aggregate |
| 系统成本 | 神经/外壳传输字节、解码后内存、冷启动、纯查询 p50/p95、候选规模拟合 |

`False Occlusion Rate = 1 - Visible Recall`，分母是真正可见单位；`Bad Cull Ratio = FN / candidate`，分母是全部候选。二者必须同时报告且不能互换。完整定义继续以[统一指标协议](../evaluation/unified_pvs_metrics_evaluation.md)为准。

### 7.2 标准场景不进入哪些主结论

- 不把一单位一 GLB 的字节削减和 HKUST/IFCBench 的原型复用 streaming 表混合求平均；
- 不把 source node 数量、unit 数量或原论文 primitive 数量互相替代；
- 不用标准场景单 seed 结果补足 HKUST 消融；核心消融仍由 HKUST 三 seed 负责；
- 不宣称跨场景/zero-shot；每个场景分别训练的事实进入表注和 limitations；
- 不摘录 NeuralPVS 原论文数字作为同协议 baseline。

如需补充标准场景 streaming，只在固定 `128 KiB` 单位协议内报告 `units/bytes-to-visible-weight coverage`，并与 BIM streaming 分表。由于这里没有原型复用，其作用是验证连续分数的排序能力，不是证明资源共享收益。

## 8. 论文表图设计

### 8.1 正文或附录表

`Table G1 - Standard-scene statistics`：

```text
Scene | Source triangles | Units | Unit bytes p50/p95 | Scene extent |
Eligible opaque/mask bytes | Excluded blend bytes | Split | Avg candidate | GT/candidate
```

`Table G2 - Non-BIM culling generality`：

```text
Scene | Method | Pose AP | Prevalence | Visible Recall | Occlusion Recall |
False Occlusion Rate | Weighted Recall | Useful Cull | Bad Cull | Avg retained
```

`Table G3 - Asset/runtime trade-off`：

```text
Scene | Method | Startup MB | Decoded MB | Point/Region queries |
Query p50/p95 | Useful Cull at the frozen safe point | Image miss p95
```

### 8.2 图

1. 启动资产大小使用对数横轴，纵轴为满足安全门后的 Useful Cull；放 Keep-All、equal-asset shell、lossless shell 和 Full V4。
2. 候选单位数-延迟图同时放 HKUST、IFCBench 和标准场景，检查同一前端 kernel 的规模趋势。
3. 每个标准场景固定展示 median、p95 和 worst miss 三个 pose 的 Reference / Prediction / Difference，不挑选只有低误差的画面。
4. 单位粒度敏感性图使用 `64/128/256 KiB`，横轴为实际 p50 unit bytes，纵轴分别为候选数、Useful Cull、模型资产和延迟。

标准场景可先放附录；如果三场景结果稳定且显著强化 generality，再把 G2 和资产-剔除 Pareto 提升到正文。

## 9. 最小实现范围

当前通用转换实现位于独立目录 `neural_instance_culling/tools/graphics_scene_importer/`，不把通用场景转换继续塞进 `glb_instancer`：

| 模块 | 单一职责 |
|---|---|
| `read_gltf.mjs` | 读取节点、primitive、变换和材质透明类别 |
| `partition_units.mjs` | 按固定目标大小完成连通分量、Morton 切分和小簇合并 |
| `write_slm_scene.mjs` | 写一单位一 GLB、sceneWeb 和 conversion manifest |
| `audit_scene.mjs` | 输出单位、三角形、字节、AABB 和材质统计 |
| `test/fixture_scene.mjs` | 构造多节点、超大 primitive、MASK/BLEND 的小型回归场景 |

现有 `build_scene_runtime_meta.mjs`、`generate_glb_points_v3.mjs`、Color-ID sampler、Pose CSR、关系证据、V4 trainer、统一 evaluator 和 HZB runner继续复用。只在这些现有入口确实无法表达一对一单位协议时做小范围修改；不增加旧 schema 兼容层、双重默认路径或标准场景专用模型。

## 10. 分阶段执行方案

### Phase G0：来源和场景体检

1. 固定 Sponza 的具体分发版本并记录署名/许可；Big City 来源审计已完成，若纳入目标矩阵则继续核对本地实验的派生资产边界，并审计 Viking Village 来源。
2. 只读统计源节点、primitive、三角形、材质、动画、场景范围和坐标单位。
3. 确定合法相机区域生成方式，产出场景预览和 20 个固定审计相机。

完成条件：至少 Sponza 和一个大型室外场景通过来源、坐标和可渲染性审计。

### Phase G1：转换器与 Sponza 端到端 smoke

1. 实现上述五个小模块和 fixture 测试；通用导入器实现与 Big City 资产转换产物已经存在；
2. 以 `128 KiB` 转换 Sponza；
3. 构建 runtime meta、GLB 点、固定 96D 几何和小规模 Color-ID 数据；
4. 验证 `visible_ids subset candidate_ids`、ID 颜色解码、AABB、alpha policy 和一对一映射；
5. 单 seed 短训只验证链路，不保留为结果。

完成条件：相同相机的原场景与单位化 reference render 无缺失，正式 evaluator 能完整运行且没有 ID/材质语义错误。

### Phase G2：正式数据和三 seed

1. 冻结 `128 KiB` 转换输出；
2. 生成空间隔离 split 和完整 view-cell 采样；
3. 生成关系证据与固定特征；
4. Full V4、AABB + Ray MLP 各完成三 seed 正式训练；
5. calibration 冻结阈值，validation 选成员，随后一次读取 test；
6. 运行 Keep-All、Hi-Z、图像评价和纯推理时间。

Sponza 完成后按相同转换参数执行 Viking Village；Big City 已有 `128 KiB` 转换输出，后续直接从相机/采样和数据集阶段继续。许可不通过时按预登记替代顺序切换场景，不改变已有结果。

### Phase G3：敏感性与论文产物

1. 只在预登记的一个代表场景生成 `64/128/256 KiB` 三套独立单位数据；
2. 每套使用相同相机区域和同一 split 区域归属，但因单位 ID 改变而分别采样、训练和评价；
3. 生成 G1-G3 表、资产-剔除 Pareto、候选-延迟曲线和定性图；
4. 与 BIM 主表分开汇总，并在 limitations 明确逐场景训练成本。

## 11. 预计工作量和优先级

| 工作 | 预计时间 | 是否阻塞现有两场景论文结果 |
|---|---:|---|
| 场景来源/许可与只读体检 | 0.5-1 天 | 否 |
| 通用转换器和 Sponza smoke | 1-2 天 | 否 |
| Sponza 正式采样、关系、三 seed 和评价 | 2-4 天，取决于硬件采样 | 否 |
| 每个大型室外场景 | 3-6 天，取决于转换和 Color-ID 采样 | 否 |
| 三档单位粒度敏感性 | 2-4 天 | 否 |

优先顺序为：先完成现有 HKUST/IFCBench 的冻结 HZB、图像和 streaming 主实验；标准场景管线并行开发，但不得占用正式 HZB 的独占浏览器 GPU 窗口。标准场景首先交付 Sponza 端到端结果，再决定是否能在投稿周期内完成三场景目标。

## 12. 正式产物目录

```text
neural_instance_culling/benchmark/out/paper_results/standard_graphics/
  scene_source_audit/
  conversion/
  test_metrics/
  image_metrics/
  runtime/
  hzb/
  granularity/
  figures/
```

每份结果必须记录场景版本、单位协议、实际 unit/resource 数、split 数、候选/GT 语义、冻结阈值来源和真实运行命令。正式浏览器采样和时间继续遵循[硬件 GPU 执行政策](../current/hardware_gpu_execution_policy.md)。

## 13. 引用与核验来源

- Wang et al., [NeuralPVS: Learned Estimation of Potentially Visible Sets](https://arxiv.org/abs/2509.24677), SIGGRAPH Asia 2025。
- [NeuralPVS official rendering repository](https://github.com/DerThomy/NeuralPVS)，包括第三方场景许可清单。
- [NeuralPVS official training repository](https://github.com/windingwind/neuralpvs)。
- [McGuire Computer Graphics Archive](https://casual-effects.com/data/) 和具体场景随附许可；正式场景必须固定到实际下载版本，不能用聚合站点的总说明替代单场景许可。

## 14. 截至 2026-09-11 的进度与后续行动

| 阶段 | 当前事实 | 后续行动 |
|---|---|---|
| G0 来源与资产审计 | Big City 与 Sponza 源文件均已取得并完成 source audit；Sponza 的许可边界已按实际模型文件修正 | 完成或登记 Viking 来源；本地派生场景资产不进入公开结果包 |
| G1 单位转换 | Big City 已生成 `2734` 个、Sponza 已生成 `132` 个一单位一资源 GLB，并写出 conversion/runtime/audit manifests；Sponza MASK 硬件语义核验通过 | Big City 采样后复核完整实例绑定 |
| G2 数据、训练、评价 | Sponza `38,784/38,784` subpose 完成，聚合为 `2,424` view-cell，split 为 `1728/132/276/288`；平均 candidate `44.77`、平均 GT `9.07`，无 candidate 漏正。六层关系覆盖全部 train cell，保留 `26,218` 边和 `376,959` 生存观察。Full 三种子与 AABB 扫描运行中；Big City `129,408` subpose 采样运行中 | 完成 Sponza validation 选择、一次 frozen test、Hi-Z/图像/runtime；随后构建 Big City CSR、关系与训练 |
| G3 表图与敏感性 | 没有标准场景正式指标、图像或性能数字 | 仅在代表场景完成 `64/128/256 KiB` 敏感性，再生成 G1-G3 表、Pareto、延迟和定性图 |

Sponza 当前使用统一入口 `run_standard_graphics_mainline.py`：calibration 冻结每个 checkpoint 的阈值，validation 安全池按 useful cull、balanced accuracy、Occlusion Recall 和 precision 选择成员，只对该成员读取一次 test。AABB 复用 `run_aabb_ray_baseline.py` 的学习率扫描和三种子协议。Big City 在采样完成后使用相同入口，不新建场景专用模型。

标准场景接入检查还修正了 Color-ID sampler 的两项 GT 语义。资源绑定现在只按 `globalGlbId -> componentGlobalIds` 的显式关系完成，并交叉检查 runtime meta 与 GLB index；旧 `glbHash` 关联已删除。HKUST、Big City 和 Sponza 的静态映射核验分别得到 `3273 -> 18831`、`2734 -> 2734` 和 `132 -> 132`，缺失或冲突映射会在渲染前失败，避免生成 component 0 污染的 GT。MASK 的 Color-ID 材质保留原 base-color 纹理 alpha 与 cutoff，但不让纹理 RGB 乘到编码颜色；正式采样前仍需用真实 Sponza MASK primitive 完成硬件像素 smoke。

## 15. 本次方案登记

- 变更目的：确定无原型复用标准场景如何进入当前 PVS 框架，并冻结转换、训练、基线和论文汇报边界。
- 修改文件：本文、论文实验总计划、第三场景/HZB 背景文档和 `docs/README.md`。
- 依赖资源：标准场景源文件及许可、现有硬件 Color-ID sampler、Pose CSR、V4 训练、统一 evaluator 和 WebGPU Batched Geometry-shell Hi-Z。
- 本次运行：完成 Big City 官方资产来源审计、`128 KiB` deterministic conversion 和输出 manifest/audit 核对；没有执行 Big City 相机采样、训练、正式 Hi-Z、test 或性能测量。
- 主线决定：保留 HKUST/IFCBench 为论文核心应用与 streaming 场景；标准场景作为 non-BIM generality 扩展，不修改当前默认 checkpoint 或前端资产。
- 待验证风险：Big City 派生 GLB 的公开分发边界、固定单位粒度下的候选规模、大型室外场景的硬件采样时间，以及后续 Color-ID/Mask 语义核验。
