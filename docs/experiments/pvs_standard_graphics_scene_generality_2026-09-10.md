# 标准图形学场景的 PVS 主实验方案

日期：2026-09-10；2026-09-11 更新执行状态

状态：2026-09-11 完成主协议纠正并重新冻结。旧标准场景数据把 3D box 标签交给只支持
水平圆盘的 V4 查询，并把 `0.75 m` 外接半径错误写成 `0.5 m`，实际后退距离仅
`0.866025 m`；对应训练和中途结果已经停止，不能进入论文。新计划统一使用水平圆盘、
按物理中心分组的随机 split 和由半径计算的后退距离，重新采样后再启动训练。

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

本文采用 NeuralPVS 第 3.2 节的 view-cell 后退构造，而不是复现其完整网络和渲染设置。对真实显示 FOV $\theta=60^\circ$ 和场景登记半径 $r$，后退距离固定为

$$
d_{back}=\frac{r}{\tan(\theta/2)}.
$$

候选/模型 FOV 固定为 `66°`，即比显示 FOV 每侧增加 `3°`。NeuralPVS 官方实验使用 `90°` 几何/PVS FOV、每 cell 1000 个 GT 位置和 `r=0.3/0.6/0.9 m`；本文的 `66°`、32 个 subpose 和实例级输出是自己的固定协议，不能写成对官方设置的逐项复现。

本项目使用的 Sponza 固定为 Khronos glTF 分发，不声称与 NeuralPVS Unity 工程中的 `Sponza_Modular.FBX` 是逐三角形相同版本；因此 Sponza 只提供共同标准场景语境，不用于直接复现或比较 NeuralPVS 论文数字。Big City 则来自 NeuralPVS 官方资产，但在候选、GT 和输出单位未对齐前同样不放入直接数值行。

## 4. 场景选择与许可门

### 4.1 首选矩阵

| 优先级 | 场景 | 作用 | 当前许可/来源状态 |
|---|---|---|---|
| P0 | Sponza | 室内、多层遮挡，先打通完整转换和训练链 | 固定为 Khronos `glTF-Sample-Assets/Models/Sponza` 当前检出版本；模型文件声明 CRYENGINE Limited License，实验只保留本地派生资产，不随论文结果包再分发 |
| P1 | Viking Village | 大型室外村落，与 NeuralPVS 评价场景对齐 | 本地统一输入为用户提供的完整 `VikingVillage.glb`；源审计和许可边界已登记，原 Unity 工程不再保留或参与执行链 |
| P1 | Big City | 大型室外城市，与 NeuralPVS 评价场景对齐 | 已从 NeuralPVS 官方资产取得 `Assets/BigCity/scene.gltf`，来源审计通过，并已完成本项目固定单位转换；派生 GLB 仅作为本地实验产物，不据此声明公开再分发许可 |

NeuralPVS 论文报告的 primitive 规模约为 Sponza `0.3M`、Viking Village `7.2M`、Big City `15.7M`。这些是该论文场景版本的说明；本项目 Big City 的来源审计另记录了 `29` 个源 primitive、`15,711,990` 个源三角形和 `12,855,583` 个源顶点，不能用论文概数替代本项目统计。

Viking Village 已固定为 NeuralPVS 官方渲染仓库配套资产包中的版本，不再寻找同名替代模型。若 Unity 批处理导出暴露不可修复的源资产错误，才按固定顺序替换为 San Miguel、Bistro 或 Power Plant；替代场景必须先记录来源版本、论文使用许可、是否允许发布派生 GLB 和相机轨迹来源，不能为了得到更好的模型结果临时换场景。

### 4.2 Viking Village 来源与转换状态

NeuralPVS 第三方许可清单将该场景标为 Unity Technologies 的 Viking Village URP，
并受 Unity Asset Store EULA 约束；论文可报告统计和结果，但公开 artifact 不分发源
资产或转换后的 GLB。原 Unity 工程只用于早期来源核验，现已删除，不再作为输入、
复现依赖或保留资产。

本项目实际转换输入统一存放在
`dataset/out/standard_graphics_sources/viking_village/VikingVillage.glb`。该文件是由
Khronos Blender glTF I/O `4.1.63` 生成的完整场景 GLB，source audit 实测为 `1,702`
个节点、`112` 个 mesh、`1,264` 个静态 renderable primitive、`4,359,435` 个三角形，
无 animation 或 skin；材质分类为 `1,123 OPAQUE / 133 MASK / 0 BLEND / 8 UNKNOWN`。
场景世界范围为约 `1088.98 x 132.59 x 827.32`，后续相机生成必须依据这个实际范围，
不能引用 Unity 工程或 NeuralPVS 论文中的概数。

该 GLB 保留 scene graph 实例、世界变换、材质、纹理和 `KHR_texture_transform`，因此
直接交给统一 glTF importer，再按固定 `128 KiB` 规则展开为一单位一资源。相邻的
NeuralPVS 官方 Unity 资产只保留用于来源和许可核验，不再进入执行链。由于本项目尚未
证明该 Blender 转换与 NeuralPVS 论文运行时场景逐三角形相同，论文应称其为相同公开
Viking Village 场景来源的本项目转换，并报告上述实测规模；不得把 NeuralPVS 的
`7.2M primitives` 直接写成本项目场景统计。

### 4.3 最小和目标配置

- 最小可发表补充：Sponza 加一个大型室外标准场景；
- 目标配置：Sponza、Viking Village、Big City 三个场景；
- 不在本轮增加 Robot Lab 和 Industrial Set v3.0：前者的官方第三方清单标记 license unknown，后者受 Unity Asset Store EULA 约束且规模较小。

每个场景在转换前输出 `scene_source_audit.json`，只记录来源、版本、坐标单位、格式、材质类别、三角形/节点统计和许可边界。许可证不清楚时停止该场景，不产生可被误用的正式结果。

### 4.4 Big City 已完成产物

Big City 的来源记录为 `neural_instance_culling/dataset/out/standard_graphics_sources/bigcity/source_audit.json`，源文件为 `neural_instance_culling/dataset/out/standard_graphics_sources/bigcity/scene.gltf`。审计结果为 `29` 个静态 `OPAQUE` primitive、`15,711,990` 个三角形、`12,855,583` 个顶点，无动画、skin、BLEND 或排除 primitive。

固定 `128 KiB` 目标的转换产物位于 `neural_instance_culling/dataset/out/standard_graphics_scenes/bigcity_128k/assets/`，并包含 `scene_source_audit.json`、`scene_audit.json`、`conversionManifest.json`、`runtimeVisibilityMeta.json` 和 `glbIndex.json`。转换 manifest 明确 `oneUnitPerResource=true`，生成 `2734` 个 renderable units、`2734` 个 instance 和 `2734` 个 GLB/resource；实际单位 GLB 字节为 p50 `93,464 B`、p95 `116,968.2 B`、最大 `131,088 B`，总计 `260,906,928 B`。这些是来源与资产转换事实，不是 PVS、图像或运行时性能结果。

### 4.5 Sponza 已完成产物

Sponza 固定为 Khronos `glTF-Sample-Assets` 的 `Models/Sponza/glTF/Sponza.gltf` 分发。来源审计记录 `103` 个静态 primitive、`262,267` 个三角形、`192,496` 个顶点，其中 `89` 个 OPAQUE、`14` 个 MASK，没有 BLEND、动画或 skin。该分发的模型许可文件声明 CRYENGINE Limited License，因此只用于本地论文实验，不将源文件、纹理或派生 GLB 纳入可公开结果包。

固定 `128 KiB` 转换产物位于 `neural_instance_culling/dataset/out/standard_graphics_scenes/sponza_128k/assets/`，生成 `132` 个一单位一资源的 renderable units，总 GLB 字节为 `7,188,496 B`，单元字节 p50 `48,476 B`、p95 `121,103.8 B`、最大 `125,244 B`。其中 `110` 个输出单位为 OPAQUE、`22` 个为保留 alpha 纹理的 MASK。AABB 体积碰撞判定会把非闭合建筑 shell 内部全部排除，因此 Sponza 正式计划改用真实三角形最近表面距离；非闭合 shell 不执行内部实体判定，也不把输出宣称为 navmesh。

### 4.6 固定几何表与标准场景 HZB 外壳

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

截至 2026-09-11，Big City 与 Sponza 已完成来源审计和 renderable-unit 转换。旧 box 采样和由其产生的 Pose CSR、train-only 关系、训练 checkpoint 与 HZB test 计划不再有效。新协议仍保留相同中心集合：Big City `674` 个中心、`8088` 个 oriented view-cell；Sponza `202` 个中心、`2424` 个 oriented view-cell。两场景都重新生成圆盘 subpose 和后续正式产物。

### 6.1 相机与 view-cell

- 主协议继续使用真实显示相机 `60 degrees` 和后退候选/模型相机 `66 degrees`；aspect 从实际 viewport 读取。
- 代表相机沿可导航区域自动生成，Sponza 只在建筑可行走空间，室外场景沿道路/开放地面；不得在墙内、地下或几何内部均匀撒点。
- view-cell shape 固定为世界 XZ 水平圆盘；半径根据场景单位尺度预先登记，不能直接套用 HKUST 的 `2 m`。
- 每个 view-cell 的方向固定，subpose 只改变合法位置；GT 是所有成功 subpose 的可见单位并集。
- 高空或不同楼层通过不同 Y 的中心采样，不在单个圆盘内增加垂直扰动。HKUST 同样有 `800` 个 sky 和 `640` 个 far 高空/远景中心，并非没有高空采样。
- 主训练采样数沿用当前可承受的场景协议；另在固定 100 个 validation cells 上补到 128 个嵌套采样点，报告 GT 收敛。不能把 NeuralPVS 的 1000 点数字写成本文已经执行的采样量。

本轮标准场景固定 `r=0.75 m`，中心到几何表面的 clearance 为 `r+0.05=0.80 m`，每 cell 使用 `32` 个面积均匀、确定性的圆盘 subpose。后退距离固定为 `0.75/tan(30°)=1.299038 m`。Big City 和 Sponza 分别需要 `258,816` 和 `77,568` 次 Color-ID 渲染。半径、clearance、后退距离和 subpose 数必须写入 pose-plan audit 与 Pose CSR meta。

### 6.2 Split 与模型

- 按物理相机中心分组，以 seed `20260911` 做确定性随机交错划分；同一中心的 12 个 yaw/pitch 方向和全部 subpose 不得跨 split；
- 先形成约 `80/10/10` 的 train/validation/test，再从初始 train 中取约 `10%` 为 calibration，最终约 `72/8/10/10`，与 HKUST 的有效分组语义一致；
- Sponza 固定为 `146/16/20/20` 个中心，即 `1752/192/240/240` 个 oriented view-cell；Big City 固定为 `486/54/67/67` 个中心，即 `5832/648/804/804`；Viking Village 固定为 `92/10/13/13` 个中心，即 `1104/120/156/156` 个 oriented view-cell；
- Full V4 使用现有 96D 几何、28D 结构化生存场和相同综合损失，不增加标准场景专用头；
- AABB + Ray MLP 使用与 Full 相同的训练步数、三 seed、安全校准规则和候选集合；
- 每个 checkpoint 只在 calibration 冻结阈值，validation 选模型，test 只读取一次；
- 先用 Sponza 完成单 seed 管线 smoke，确认语义后仍需执行登记的三 seed 正式训练，smoke 不能进入论文表格。

### 6.3 当前正式执行状态（2026-09-11）

- Sponza 的 Full V4 与 AABB + Ray MLP 已进入三种子 `40 x 900` 正式训练；中间 validation 只用于健康检查，不提前终止登记训练。
- Viking Village 的 `49152/49152` 次 Color-ID subpose 已完成，16 个关系分片均为硬件 WebGL/ANGLE Vulkan，页面 `gpuGate.hardware=true`，train-only 关系 CSR 覆盖全部 `1104` 个 train view-cell，且 native `66 degrees` candidate audit 通过；Full V4 与 AABB + Ray MLP 已启动。
- Big City 的 `258816/258816` 次 Color-ID subpose、Pose CSR 和 train-only 关系均已完成，`candidateMissVisible=0`，16 个关系分片通过同一硬件门；Full V4 与 AABB + Ray MLP 已完成预检并排入正式训练队列。
- 三个场景的正式 test 均保持关闭；只有各自三种子训练完成、calibration 冻结阈值并由 validation 选定成员后，runner 才执行一次 frozen test。

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
| G0 来源与资产审计 | Big City、Sponza 和 Viking Village 的固定源文件、许可边界及 source audit 均已登记；Viking 以本地 `VikingVillage.glb` 为唯一转换输入 | 保持三份源审计随最终结果包引用；受许可限制的源资产和派生 GLB 不公开分发 |
| G1 单位转换 | Big City、Sponza、Viking 分别生成 `2734/132/1890` 个一单位一资源 GLB，并完成固定几何表、runtime metadata、reference binding 和 lossless shell | 等各场景 Full V4 运行资产冻结后导出 equal-asset shell |
| G2 数据、训练、评价 | 三场景圆盘 Color-ID、Pose CSR 和 train-only 关系均完成且通过硬件/候选审计；Sponza、Viking 的 Full V4 与 AABB 正式训练运行中，Big City 两条 runner 已排在 Sponza 对应任务后自动启动 | 完成三场景三种子 validation 选择和一次 frozen test，再执行 Hi-Z、图像和独占 runtime |
| G3 表图与敏感性 | 没有标准场景正式指标、图像或性能数字 | 仅在代表场景完成 `64/128/256 KiB` 敏感性，再生成 G1-G3 表、Pareto、延迟和定性图 |

三个标准场景使用统一入口 `run_standard_graphics_mainline.py`：calibration 冻结每个 checkpoint 的阈值，validation 安全池按 useful cull、balanced accuracy、Occlusion Recall 和 precision 选择成员，只对该成员读取一次 test。Viking 已按实际 split 数加入该入口。AABB 复用 `run_aabb_ray_baseline.py` 的学习率扫描和三种子协议，不新建场景专用模型。

标准场景接入检查还修正了 Color-ID sampler 的两项 GT 语义。资源绑定现在只按 `globalGlbId -> componentGlobalIds` 的显式关系完成，并交叉检查 runtime meta 与 GLB index；旧 `glbHash` 关联已删除。HKUST、Big City 和 Sponza 的静态映射核验分别得到 `3273 -> 18831`、`2734 -> 2734` 和 `132 -> 132`，缺失或冲突映射会在渲染前失败，避免生成 component 0 污染的 GT。MASK 的 Color-ID 材质保留原 base-color 纹理 alpha 与 cutoff，但不让纹理 RGB 乘到编码颜色；正式采样前仍需用真实 Sponza MASK primitive 完成硬件像素 smoke。

2026-09-11 补充：`run_geometry_shell_hzb_paper.py` 已把 `sponza_128k`、
`bigcity_128k` 和 `viking_village_128k` 加入与 HKUST/IFCBench 相同的校准、冻结
test 和五轮 timing 编排，三个标准场景均使用 `Region66 1/all`，没有场景专用 HZB 算法。
Sponza lossless 外壳已导出，实际启动资产 `1,844,914 B`，包含 `110` 个确定
不透明 primitive、`227,327` 个三角形；`22` 个 MASK primitive 不作为保守不透明
遮挡物。Big City lossless 外壳实际启动资产 `106,166,442 B`，包含 `2,734`
个不透明 primitive、`15,711,990` 个三角形。Equal-asset 外壳必须等对应 Full V4
运行资产冻结后再按实际字节预算生成。

Sponza 新圆盘正式采样于 2026-09-11 完成：`2,424` 个 oriented view-cell、每 cell
`32` 个 subpose，共 `77,568` 次 Color-ID 渲染；16 个分片均为 NVIDIA RTX A6000
Vulkan/ANGLE，`gpu_execution_summary.json` 的 `formalReady=true`。聚合后的 Pose CSR
split 为 `1752/192/240/240`（train/calibration/validation/test），平均候选
`47.8201`、平均 GT 可见 `9.1465`，全部 subpose 成功且
`candidateMissVisible=0`。新 train-only 三角形关系使用每个 train cell 的 5 个固定
空间代表，共 `8,760` 个渲染视点；稀疏合并后保留 `21,904` 条关系边和 `378,757`
条生存观测。对应 Full V4 三种子与 AABB + Ray MLP 正式矩阵已经启动，尚未产生可汇报
的 validation/test 结果。

Big City 新圆盘正式采样也已完成：`8,088` 个 oriented view-cell、每 cell `32` 个
subpose，共 `258,816` 次 Color-ID 渲染，16 个分片全部为 NVIDIA Vulkan/ANGLE，
`formalReady=true`。新 Pose CSR split 为 `5832/648/804/804`，平均候选
`781.8077`、平均 GT 可见 `211.1635`，全部 subpose 成功且
`candidateMissVisible=0`。其中 `988` 个空候选、`1,855` 个零 GT view-cell 按冻结
协议保留。train-only 关系使用 `5,832×5=29,160` 个深度视角；16 个分片均已完成，
页面 `gpuGate.hardware=true` 且首层参考检查零不一致。关系 CSR 覆盖全部 `5,832` 个
train view-cell，native candidate audit 通过，保留 `449,179` 条关系边和
`20,100,507` 条生存观测。Full V4 与 AABB + Ray MLP 已通过不读 test 的预检并排入
正式训练队列。

Viking 新圆盘正式采样已完成：`1,536` 个 oriented view-cell、每 cell `32` 个
subpose，共 `49,152` 次 Color-ID 渲染，16 个分片全部通过 NVIDIA Vulkan/ANGLE
硬件门。新 Pose CSR split 为 `1104/120/156/156`，平均候选 `585.7337`、平均 GT
可见 `134.6706`，全部 subpose 成功且 `candidateMissVisible=0`，没有空候选或空 GT。
train-only 关系使用 `1,104×5=5,520` 个深度视角；16 个分片均已完成且
`formalReady=true`，关系 CSR 覆盖全部 `1,104` 个 train view-cell，native candidate
audit 通过，保留 `110,189` 条关系边和 `1,553,537` 条生存观测。Full V4 与 AABB +
Ray MLP 已通过不读取 test 的预检并启动正式矩阵。

协议修正前 Sponza/Big City 的绑定 summary、Point60 和重复外壳结果已删除。Sponza
绑定清单已从当前 Pose CSR 原地重建并核对后退距离为 `1.299038 m`；Big City 绑定清单
也只从新 Pose CSR 生成。旧错误口径不再保留为兼容结果或 runner 输入。

### 14.1 协议纠正记录

旧标准场景数据存在三项关联错误：标签区域是 `camera_aligned_box`，V4 查询区域是
`horizontal_disk`；外接半径从 `0.75 m` 被改写为 `0.5 m`；主 split 使用连续空间块而
HKUST 使用中心组交错划分。2026-09-11 已停止所有基于该数据的 Full/AABB 训练，删除其
论文资格。正式实验只接受本节冻结的圆盘、`1.299038 m` 后退距离和中心组 split。

实际进入完整流水线的标准场景固定为 Sponza、Big City 和 Viking Village。Viking 的
统一 GLB 输入、source audit 和 `128 KiB` 单位转换已经完成：共 `1,890` 个一单位一资源
的 renderable units，包含 `4,359,435` 个三角形，单位 GLB 总计 `98,214,428 B`，p50
为 `39,568 B`，p95 为 `118,107.4 B`。每单位 `1024` 个点的缓存 `1,890/1,890`
成功解码，零失败、零回退和零空几何；共享冻结编码器导出的固定几何表为
`[1890,96]` float16，数值全部有限。

Viking 不使用全场四层均匀 Y 网格。其代表相机采用已冻结的
`ground_surface_grid`：从相机域中排除 `terrain_far`，在 `24×24` 固定 XZ 网格上向
`terrain_near` 三角形求最高交点，相机中心置于地面上方 `1.7 m`，再以近地形真实表面
和非地形单位 AABB 执行 `0.8 m` clearance。实际得到 `128` 个合法物理中心和 `1,536`
个定向 view-cell，split 为 `1104/120/156/156`，随后生成每 cell `32` 个 subpose，
共 `49,152` 个硬件 Color-ID 待采样相机。每个 cell 仍严格使用半径 `0.75 m` 的水平
圆盘、`66°` 模型 FOV、`60°` 显示 FOV 和 `1.299038 m` 后退距离；地表跟随只改变中心
放置，不改变 PVS 查询契约。

Viking 已同时登记到 `run_standard_graphics_mainline.py`、`run_aabb_ray_baseline.py`
和 `run_geometry_shell_hzb_paper.py`。在硬件 Color-ID、Pose CSR、train-only relation
完成前不能开始模型或 HZB 精度评价；lossless HZB 外壳可独立导出，equal-asset 外壳
仍须等待对应 Full V4 运行资产冻结。

Viking lossless HZB 外壳已经导出：仅 `1,647` 个确定 OPAQUE 单位作为遮挡物，`235`
个 MASK 与 `8` 个材质不确定单位不写遮挡深度；外壳含 `4,037,221` 个三角形，压缩几何
`28,408,601 B`，包含 AABB、映射和 metadata 的完整启动资产为 `30,068,115 B`。
该数字只表示基线预下载成本，不是精度或耗时结果。

## 15. 本次方案登记

- 变更目的：确定无原型复用标准场景如何进入当前 PVS 框架，并冻结转换、训练、基线和论文汇报边界。
- 修改文件：本文、论文实验总计划、第三场景/HZB 背景文档和 `docs/README.md`。
- 依赖资源：标准场景源文件及许可、现有硬件 Color-ID sampler、Pose CSR、V4 训练、统一 evaluator 和 WebGPU Batched Geometry-shell Hi-Z。
- 本次运行：停止旧标准场景 Full/AABB 训练；修正 pose 生成、subpose 采样、后退候选和 split 协议；新正式采样从圆盘计划重新开始。
- 主线决定：HKUST、IFCBench、Sponza、Viking Village 和 Big City 均为论文正式主实验场景；streaming 资源复用实验集中在 HKUST/IFCBench，三个标准图形学场景正式报告可见性、图像、HZB 和运行成本结果。
- 待验证风险：Big City 派生 GLB 的公开分发边界、固定单位粒度下的候选规模、大型室外场景的硬件采样时间，以及后续 Color-ID/Mask 语义核验。

## 16. 2026-09-12 训练状态与 Viking 短微调登记

截至 2026-09-12，Big City 的 Full V4 seed01/02 和 AABB + Ray MLP seed01/02 已经
启动；Sponza Full seed01/02、Sponza AABB 三种子和 Viking AABB 三种子已完成登记的
`40 x 900` 长训，Sponza Full seed03、Viking Full seed01/03 仍在补齐。所有正式 test
继续关闭，已有长训成员和排队任务不得由短微调替代或取消。

Viking 当前最佳安全成员的 validation Occlusion Recall 为 `49.0648%`（seed01）和
`50.6960%`（seed02），对应 weighted-recall LCB 为 `99.0304%` 和 `99.1497%`。
候选覆盖审计为零漏 GT，因此不通过扩大候选集合改善表面指标。新增独立实验
`pvs_v4_viking_region_stability_finetune_v1`，目标是在保持 validation weighted recall
及其单侧 95% LCB 均大于 `0.99` 的前提下，提高 Occlusion Recall 和 Useful Cull，并
降低 calibration 到 validation 的区域波动。

首个成员固定从 test-free 的 seed02 epoch-32 快照初始化，使用新 AdamW、`2e-5` 学习率、
`8 epoch x 450 step`、每批 8 个 pose。训练只读取原 train split；每 2 epoch 用自身
calibration 冻结阈值并在 validation 评价。召回保护权重设为 `0.35`，最差 pose 权重设为
`0.35`，困难边界分离权重设为 `0.25`，困难负例比例设为 `0.02`，实例校准残差正则设为
`0.03`。输入、96D 几何表、关系 CSR、模型结构和运行时 schema 均不改变。

该成员只作为短微调首个成员，不提前终止 Viking 原三种子长训。源 seed01/03 完成长训
后按相同配置补齐对应成员。正式保留条件为 validation 安全门通过，且相对各自源最佳
安全 checkpoint 提高 Occlusion Recall 或 Useful Cull；否则如实记录为无收益微调，论文
继续使用原始 Full V4。

首个成员运行命令：

```bash
CUDA_VISIBLE_DEVICES=3 conda run --no-capture-output -n slm_pvs python -u \
  neural_instance_culling/model/train_pvs.py \
  --dataset-dir neural_instance_culling/dataset/out/pose_csr_viking_village_standard_graphics_128k_fov66_v1 \
  --relation-dir neural_instance_culling/dataset/out/viking_village_standard_graphics_v4_bounded_relation_csr_v1 \
  --runtime-meta neural_instance_culling/dataset/out/standard_graphics_scenes/viking_village_128k/assets/runtimeVisibilityMeta.json \
  --initial-geo-features neural_instance_culling/dataset/out/standard_graphics_scenes/viking_village_128k/instance_geo_features_fp16.bin \
  --glb-index neural_instance_culling/dataset/out/standard_graphics_scenes/viking_village_128k/assets/glbIndex.json \
  --glb-root neural_instance_culling/dataset/out/standard_graphics_scenes/viking_village_128k/assets \
  --init-checkpoint neural_instance_culling/model/out/pvs_mainline_v4_standard_graphics_v1/viking_village_128k/full_seed20260802_e40/checkpoint_epoch_032.pt \
  --output-dir neural_instance_culling/model/out/pvs_v4_viking_region_stability_finetune_v1/seed20260802_from_e032_lr2e-5_e8x450 \
  --experiment-name pvs_v4_viking_region_stability_finetune_v1_seed20260802 \
  --variant full_integrated_visibility_mainline --occlusion-representation survival \
  --survival-rank 4 --relation-source bounded_hierarchical --spectral-mode moment_envelope \
  --instance-calibration-mode residual --loss-variant pose_balanced_rvl_contrastive \
  --epochs 8 --steps-per-epoch 450 --poses-per-batch 8 --observation-batch-size 8192 \
  --eval-every 2 --snapshot-every 2 --max-eval-poses 0 \
  --calibration-bootstrap-replicates 10000 --seed 20260802 --device cuda \
  --learning-rate 0.00002 --weight-decay 0.00001 --survival-loss-weight 0.25 \
  --relation-consistency-weight 0.10 --instance-calibration-regularization-weight 0.03 \
  --instance-calibration-max-abs 4.0 --sparse-instance-penalty 3.0 \
  --instance-calibration-warmup-fraction 0.0 --instance-calibration-ramp-fraction 0.0 \
  --relation-gradient-cap 0.25 --integrated-rvl-recall-guard-weight 0.35 \
  --integrated-rvl-recall-target 0.99 --integrated-rvl-recall-temperature 0.05 \
  --integrated-rvl-pose-cvar-fraction 0.25 --integrated-rvl-pose-cvar-weight 0.35 \
  --integrated-separation-weight 0.25 --integrated-tail-ramp-fraction 0.0 \
  --frontier-positive-mass-fraction 0.005 --frontier-positive-count-cap 64 \
  --frontier-negative-fraction 0.02 --frontier-negative-count-cap 256 \
  --frontier-margin 0.50 --frontier-temperature 0.25 \
  --frontier-positive-importance-floor 0.5 --frontier-positive-importance-power 0.5
```

### 16.1 首个微调结果与第二成员

seed02 首个成员已于 2026-09-12 完成 `8 x 450`，未读取 test。四次 evaluation 的
calibration WR LCB 均约为 `99.18%--99.22%`，但对应 validation LCB 仅为
`97.68%--97.82%`，最终状态为 `no_qualified_safety_workpoint`。其 validation
Occlusion Recall 约为 `48.84%--53.57%`，没有在安全门内超过源 checkpoint，因此该
配置不进入正式结果，也不复制到 seed01/03。

第二成员仍从 seed02 epoch-32 的原始冻结快照初始化，不从失败成员继续。配置改为
`1e-5`、`6 epoch x 450 step`，恢复困难负例比例 `0.01` 和分离权重 `0.20`，将训练期
weighted-recall 目标提高到 `0.995`、召回保护权重提高到 `0.50`、最差 pose 权重提高到
`0.50`。目的不是靠降低阈值增加预测数量，而是优先修复跨物理区域的高权重正例尾部；
正式保留条件仍为 checkpoint 自身 calibration 冻结阈值下 validation WR 与 LCB 均大于
`0.99`，且遮挡召回或有效剔除超过源最佳安全 checkpoint。

第二成员已完整执行 `6 epoch x 450 step`。epoch 2 最接近安全门：validation WR 为
`0.994029`，LCB 为 `0.989959`，aggregate Occlusion Recall 为 `0.508776`，Useful Cull
为 `0.381038`；其后 epoch 4/6 的 LCB 分别降至 `0.977090/0.977657`。因此第二成员仍为
`no_qualified_safety_workpoint`，不复制到 seed01/03，也不替换原始 Full V4。

第三成员 `conservative_seed20260802_from_e032_lr5e-6_e4x450` 于 2026-09-12 启动，仍从
原始 seed02 epoch-32 初始化。它使用 `5e-6` 学习率、`4 epoch x 450 step`、每 epoch
评价，训练期 weighted-recall 目标提高到 `0.997`，召回保护和最差 pose 权重均提高到
`0.75`；困难边界分离权重和困难负例比例保持原始主线的 `0.20/0.01`。该成员必须完整
执行，只有自身 calibration 阈值冻结后的 validation WR 与 LCB 均严格大于 `0.99`，且
Occlusion Recall 或 Useful Cull 超过源 checkpoint 时，才会扩散到另外两个 seed。

同日完成了两项后续流水线接入：统一离线成本汇总现已覆盖 HKUST、IFCBench、Sponza、
Viking Village 和 Big City 共五个场景；标准场景 Full runner 增加独立 `export` 阶段，
可在 validation 选择和一次 frozen test 完成后仅导出所选运行资产，不重复读取 test。
端侧运行时间场景清单也已登记三个标准场景，运行阈值直接读取各自导出资产，不另设
手工阈值常量。

第三成员 epoch 1 已通过 validation 安全门：WR=`0.995464`、LCB=`0.991894`、aggregate
Occlusion Recall=`0.508259`、Useful Cull=`0.380651`。它相对源 seed02 epoch-32 的
安全工作点在安全下界、遮挡召回和有效剔除上均有小幅改善，但仍须完整执行 4 epoch。
Viking 原 `all` runner 的编排父进程已暂停，两个 Full 训练子进程继续运行；这样可以
防止微调三种子 validation 比较完成前读取 test。

若第三成员完成后仍保留安全成员，队列会等待原始 Full seed01/03 和 Sponza 占用的 GPU
任务结束，再从各 seed 自身的 `best_safe.pt` 启动相同 `4 x 450` 配置。三个微调成员全部
存在时，标准场景 runner 才把它们和三个 from-scratch 成员合并为同一个 validation 安全池；
选择规则仍是安全池内优先 Useful Cull、balanced accuracy、Occlusion Recall、precision
和更少预测。选择完成后只运行一次 Viking frozen test。

第三成员已完整执行完毕，最终状态为 `safe`，最佳安全 checkpoint 固定为 epoch 1。其
validation WR/LCB/Occlusion Recall/Useful Cull 分别为
`0.995464/0.991894/0.508259/0.380651`。epoch 2--4 的 Occlusion Recall 一度达到
`0.516998`，但 LCB 仅约 `0.9791`，因此不会被选择。seed01/03 复现和后续六成员
validation 选择已经进入自动队列；在这一步完成前 Viking test 继续关闭。

## 17. 2026-09-12 Big City seed02 查询头塌缩与恢复实验

Big City Full V4 seed02 从 epoch 4 到 epoch 36 的 calibration 冻结阈值始终为 `0`，
validation 的 WR/LCB 虽为 `1/1`，但 Occlusion Recall 和 Useful Cull 均为 `0`。
这不是安全且有效的模型，而是所有候选均保留。阈值扫描显示 `0.56` 时仍保留约
`772.04` 个候选，增至 `0.58` 后只剩约 `0.8` 个且 calibration WR 降至 `0.389852`。

进一步在固定的 32 个 validation view-cell、`44,329` 个候选上检查 epoch-24 输出：
seed02 概率标准差仅 `0.011482`，q01/q50/q99 为
`0.568400/0.568671/0.568808`；seed01/03 的标准差分别为 `0.240049/0.209911`。
seed02 查询隐藏特征逐维标准差均值为 `0.042599`，也低于 seed01/03 的
`0.147935/0.088890`。因此判定为轻量运行时查询头陷入近常数局部最优，而不是 GT、
候选或阈值实现错误。

登记恢复实验 `pvs_v4_bigcity_runtime_head_recovery_v1`。恢复成员从 seed02 完整
`40 x 900` 的 test-free `last.pt` 初始化，保留已经学习的 96D 几何、分层遮挡关系、
生存场和实例校准残差，仅重新初始化 `shared_trunk`、实例可见性头及未参与本阶段损失的
两个任务头。训练入口新增 `--reset-runtime-heads`，运行时网络结构、导出 schema 和前端
算子不变。

两个成员均完整执行 `8 epoch x 600 step`，每 epoch 用自己的 calibration 阈值评价
validation，不读取 test：

- `head_reset_balanced`：学习率 `1e-4`，召回保护 `0.30`，困难边界分离 `0.35`，
  困难负例比例 `0.05`、上限 `512`。
- `head_reset_strong_separation`：学习率 `1e-4`，召回保护 `0.30`，困难边界分离
  `0.60`，困难负例比例 `0.10`、上限 `1024`。

公共运行命令为：

```bash
CUDA_VISIBLE_DEVICES=2 conda run --no-capture-output -n slm_pvs python -u \
  neural_instance_culling/model/train_pvs.py \
  --dataset-dir neural_instance_culling/dataset/out/pose_csr_bigcity_standard_graphics_128k_fov66_v1 \
  --relation-dir neural_instance_culling/dataset/out/bigcity_standard_graphics_v4_bounded_relation_csr_v1 \
  --runtime-meta neural_instance_culling/dataset/out/standard_graphics_scenes/bigcity_128k/assets/runtimeVisibilityMeta.json \
  --initial-geo-features neural_instance_culling/dataset/out/standard_graphics_scenes/bigcity_128k/instance_geo_features_fp16.bin \
  --glb-index neural_instance_culling/dataset/out/standard_graphics_scenes/bigcity_128k/assets/glbIndex.json \
  --glb-root neural_instance_culling/dataset/out/standard_graphics_scenes/bigcity_128k/assets \
  --init-checkpoint neural_instance_culling/model/out/pvs_mainline_v4_standard_graphics_v1/bigcity_128k/full_seed20260802_e40/last.pt \
  --reset-runtime-heads --variant full_integrated_visibility_mainline \
  --occlusion-representation survival --survival-rank 4 \
  --relation-source bounded_hierarchical --spectral-mode moment_envelope \
  --instance-calibration-mode residual --loss-variant pose_balanced_rvl_contrastive \
  --epochs 8 --steps-per-epoch 600 --poses-per-batch 4 --observation-batch-size 8192 \
  --eval-every 1 --snapshot-every 1 --max-eval-poses 0 \
  --calibration-bootstrap-replicates 10000 --seed 20260802 --device cuda \
  --learning-rate 0.0001 --weight-decay 0.00001 --survival-loss-weight 0.25 \
  --relation-consistency-weight 0.10 --instance-calibration-regularization-weight 0.02 \
  --instance-calibration-max-abs 4.0 --sparse-instance-penalty 3.0 \
  --instance-calibration-warmup-fraction 0 --instance-calibration-ramp-fraction 0 \
  --relation-gradient-cap 0.25 --integrated-rvl-recall-guard-weight 0.30 \
  --integrated-rvl-recall-target 0.99 --integrated-rvl-recall-temperature 0.05 \
  --integrated-rvl-pose-cvar-fraction 0.25 --integrated-rvl-pose-cvar-weight 0.25 \
  --integrated-tail-ramp-fraction 0 --frontier-positive-mass-fraction 0.005 \
  --frontier-positive-count-cap 64 --frontier-margin 0.50 --frontier-temperature 0.25 \
  --frontier-positive-importance-floor 0.5 --frontier-positive-importance-power 0.5
```

每个成员分别补充自己的 `--output-dir`、`--experiment-name`、
`--integrated-separation-weight`、`--frontier-negative-fraction` 和
`--frontier-negative-count-cap`。两个成员都必须跑完；安全资格仍要求 checkpoint 自身
calibration 及 validation 的 aggregate WR 和单侧 95% LCB 均严格大于 `0.99`。
合格成员与三个原始 Full 成员进入同一 validation 池，按 Useful Cull、balanced
accuracy、Occlusion Recall、precision、WR LCB 和更少预测选择，之后 Big City test
只读取一次。

### 17.1 正式恢复 runner 与完成状态

三个标准场景的原始 Full V4 三种子均已完成。Big City seed02 原始最佳安全成员仍是
epoch 4 的全保留工作点；seed01/03 最佳安全 validation 的
WR/LCB/Occlusion Recall/Useful Cull 分别为
`0.999864/0.999839/0.503400/0.394889` 和
`0.999417/0.999055/0.499145/0.391552`。

`run_standard_graphics_mainline.py` 新增两个正式执行模式：

```bash
conda run --no-capture-output -n slm_pvs python -u \
  neural_instance_culling/benchmark/run_standard_graphics_mainline.py \
  recover-bigcity --scenes bigcity_128k --gpu-ids 1 2

conda run --no-capture-output -n slm_pvs python -u \
  neural_instance_culling/benchmark/run_standard_graphics_mainline.py \
  finetune-viking --scenes viking_village_128k --gpu-ids 0 3
```

两种模式直接以训练子进程返回码和 `calibration_ready_summary.json` 推进完整训练、
validation 选择与唯一一次 frozen test，不再用训练 PID 存活状态代表任务是否完成。
后续运行资产导出同时等待 `viking_finalize.done` 和 `bigcity_finalize.done`，确保三个
标准场景的模型与阈值均已冻结后才进入 runtime、HZB 和图像评价。
