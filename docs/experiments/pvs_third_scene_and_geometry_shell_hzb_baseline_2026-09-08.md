# 第三场景与纯几何外壳 HZB 基线计划

日期：2026-09-08

状态：方案记录，尚未下载第三场景、导出几何外壳或产生正式 HZB 实测结果。

## 目的

当前论文已有两个主要场景：

- HKUST：真实、多楼栋、校园尺度的详细构件场景；
- Metropolis：由 IFC-Bench v2 多个 IFC 项目组合并实例化得到的合成城市，当前版本包含 41,298 个实例和 3,669 个 GLB 原型。

仅依赖这两个场景仍有两个证据缺口：

1. Metropolis 与 IFC-Bench 来源绑定，不能再选一个 IFC-Bench 单体作为独立来源的第三场景后宣称数据多样性明显增加；
2. 当前缺少一个真实、多专业、室内遮挡和 MEP 遮挡密集的大型 BIM，用来验证方法不仅适用于校园室外和规则化合成城市。

同时，论文需要一个能够代表传统在线遮挡剔除系统的强基线。该基线不应依赖尚未下载的正式 GLB，因此不采用“完整场景几何已驻留”的 Triangle HZB，而采用启动时单独下载的纯几何外壳 HZB。

## 第三场景结论

### 公开数据现状

按照“一个场景同时包含多栋建筑，且每栋建筑都有详细构件级 BIM”的严格条件，目前讨论过的公开数据中没有第二个可以直接替代 HKUST 的现成场景：

| 数据源 | 场景组织 | 详细构件 | 与 HKUST 的差异 |
|---|---|---:|---|
| IFC-Bench v2 | 多个相互独立的 IFC 项目 | 有 | 已用于合成 Metropolis，不能再次作为独立数据来源 |
| BIMData R&D IFC collection | 约 40 个独立项目，部分项目含多专业文件 | 有 | 多专业通常属于同一栋建筑，不是一个真实园区 |
| BIMNet | 25 个真实扫描序列及对应 IFC | 有 | 是多个独立建筑，不在同一个园区坐标系中 |
| ResBIM | 1,000 多个合成住宅 | 有限 | 单体较小且原始格式以 RVT 为主，适合生成住宅区压力场景 |
| 884k IFC products | 从 245 个公开 IFC 提取的构件集合 | 有 | 不是可直接渲染的完整场景 |
| 3D BAG | 城市级多建筑 | 无详细内部构件 | 主要是建筑外壳，不适合细粒度实例遮挡评价 |

真实校园、医院园区和商业综合体的详细 BIM 很少公开，主要限制来自设计知识产权、设施安全、专业模型再分发许可和坐标/版本管理。因此第三场景不应继续等待一个形态完全等同于 HKUST 的公开园区。

### 第三场景的定位

第三场景采用一个**独立于 IFC-Bench 来源的真实大型多专业单体 BIM**。它不替代 HKUST，而是补足以下困难类型：

- 多楼层室内空间；
- 房间和走廊形成的深层遮挡；
- 建筑、结构、暖通、给排水和电气构件共同存在；
- 大量细小 MEP 构件和重复设备；
- 与校园室外视点明显不同的候选/GT 比例。

优先候选顺序：

1. [BIMData Medical Clinic](https://github.com/bimdata/BIMData-Research-and-Development/blob/master/pages/IFC_FILES.md)：聚合列表显示 combined、architecture、electrical、mechanical、MEP、HVAC 和 structure 等多个文件，总列示规模约 730.7 MB；
2. 同一列表中的 LTU AHouse：包含 architecture、structure、air、duct、plumbing、heating 和 cooling 等多专业文件，总列示规模约 553.4 MB；
3. [Nordic BIM4LCA](https://www.nordicsustainableconstruction.com/knowledge/2024/august/bim4lca-files) 的 office building：公开建筑、结构、机电和不同设计阶段文件，可作为前两项许可或转换不合适时的独立来源；
4. [BIMNet](https://github.com/LydJason/BIMNet) 中规模最大的真实建筑：只有在对应 IFC 可以完整下载、许可允许论文使用且构件规模满足要求时采用。

BIMData 页面是资源聚合列表，不等于所有模型拥有相同许可证。正式下载和再分发前必须追溯具体模型的来源页和许可证。

### 选择方法

第三场景先做数据体检，不直接启动完整采样和训练。体检至少输出：

- IFC 是否能由 IfcOpenShell 完整解析和三角化；
- 可渲染构件数、三角形数、场景范围和楼层数；
- 建筑、结构和 MEP 的构件数量及比例；
- IFC 显式类型复用和几何严格去重后的原型数；
- 实例复用率、最大原型实例数和 GLB 资源数；
- 透明构件、异常尺度、无几何产品和转换失败数量；
- 初步视点中的平均候选数、平均可见数和 GT/候选比例；
- 原始许可、论文使用许可和再分发边界。

选择依据是构件规模、遮挡复杂度、专业完整性、实例复用程度和许可，不以 IFC 文件大小单独决定。完成体检后选择相对最合适的一项继续完整数据集构建；不会因为某个任意构件数量门限而取消第三场景实验。

### 训练和论文表述

当前模型依赖场景专属的固定实例特征表，因此第三场景应独立完成：

```text
IFC 转换与实例化
  -> runtimeVisibilityMeta/glbIndex
  -> 合法相机区域与 view-cell 计划
  -> 硬件 Color-ID subpose 采样
  -> train/calibration/validation/test
  -> 场景专属离线关系特征
  -> 场景专属模型训练、校准和评价
```

多个场景分别训练可以证明方法在不同场景结构上稳定工作，但不能表述为一个模型对未见建筑的零样本泛化。若以后研究跨场景模型，需要重新设计实例特征归一化、关系编码和训练协议，不能从当前多场景结果反推。

## 图形学场景的角色

Bistro、Power Plant、San Miguel 等场景不是第三个 BIM 主场景。它们最多作为图形学补充场景，用于和传统遮挡剔除工作建立联系。

图形学场景通常没有 IFC 构件语义。转换时应按以下顺序定义实例：

1. 源 glTF/FBX 场景节点中的独立可渲染对象作为 component；
2. 多个节点引用同一个 Mesh 时直接保留为同一原型的多个实例；
3. 对未显式共享的对象，只在局部几何、拓扑、材质类别和刚体配准严格一致时复用原型；
4. 无法恢复对象边界的大网格只能按空间簇定义 render cluster，论文中不能称为 BIM 构件；
5. 每个原型导出为 `task-*/glb/LOD0/sub_<prototype>.glb`，重复出现位置使用 `EXT_mesh_gpu_instancing`；
6. 为每个出现位置分配独立实例 ID，并生成 `sceneWeb.json`、`glbIndex.json` 和 `runtimeVisibilityMeta.json`。

当前 [`tools/glb_instancer`](../../neural_instance_culling/tools/glb_instancer/README.md) 的推荐模式能够处理“一个 `sub_*.glb` 已经代表一个完整构件”的场景。其单体 GLB 模式只支持一个 indexed triangle primitive，并按连通区域拆分，不能直接作为复杂图形学场景的正式转换器。若引入 Bistro 等补充场景，应先增加保留多节点、多 primitive 和源对象身份的通用 glTF 导入路径。

## 正式基线矩阵

正文不堆叠大量作用重复的基线，保留下列四项：

| 方法 | 输入 | 输出 | 作用 |
|---|---|---|---|
| Frustum/Keep-all | 相同的后退候选和真实视锥 | 保留视锥内全部候选 | 无遮挡剔除的安全与资源下界 |
| AABB + Ray MLP | 实例 AABB、相机到实例的方向、距离和投影关系 | 每个候选的可见分数 | 检验普通包围盒和视角学习是否已经足够 |
| Geometry-only shell HZB | 预下载的详细不透明纯几何外壳、候选 AABB、当前相机 | 当前视角的可见实例及 GLB 队列 | 传统在线几何遮挡剔除的强系统基线 |
| Proposed model | 固定离线特征表、轻量视角查询和后退候选 | view-cell 保守可见实例及 GLB 队列 | 当前论文方法 |

训练集实例可见频率、相机距离和投影 AABB 面积可以放在附录诊断，不作为正文的主要竞争方法。当前 `baseline_aabb_hzb` 历史名称表示 AABB depth proxy，不是真正的 HZB；正式报告必须使用准确名称，不能把它写成 Triangle HZB 或 geometry-shell HZB。

## 纯几何外壳 HZB 定义

### 外壳包含什么

纯几何外壳保留：

- 不透明遮挡三角形的位置；
- 三角形索引；
- 原型到各实例的变换；
- 用于空间分块或 BVH 的边界；
- 必要的 occluder/non-occluder 标志。

纯几何外壳删除：

- 法线、切线和 UV；
- 顶点颜色；
- PBR 材质参数；
- 纹理和其他不参与深度判断的属性；
- 构件业务元数据。

“删除材质”不代表把所有三角形都当成不透明遮挡物。玻璃、透明幕墙、alpha-cutout 和来源不确定的表面必须在离线导出阶段依据原材质分类，运行时外壳只保留确定的不透明遮挡三角形。错误地把透明面加入深度外壳会产生假遮挡和画面缺失；保守地少放遮挡物只会降低剔除率。

### 运行路径

正式实现采用现代 GPU 驱动路径：

```text
空间块/BVH 视锥筛选
  -> 纯几何 depth-only 光栅化
  -> GPU 构建分层深度 mip 链
  -> GPU 批量测试候选 AABB
  -> GPU 压缩最终实例 ID
  -> GPU 按 GLB 聚合
  -> 只读回最终实例集合和 GLB 队列
```

不得把所有候选深度或所有候选概率逐项读回主线程。深度偏置、近裁剪面、反向 Z、双面几何和透明面处理必须固定，并在 calibration 上确定保守参数。

## 为什么不使用完整几何 Triangle HZB

完整几何 Triangle HZB 假定正式场景三角形已经下载并驻留。它适合评价“场景已经加载以后如何减少 draw call”，但不能解决本项目的主要问题：正式 GLB 尚未下载时，浏览器应该先请求哪些资源。

因此新的论文系统基线不把完整场景 Triangle HZB列入主要矩阵。Geometry-only shell HZB 将遮挡几何作为单独的启动资产，能够在正式 GLB 之前工作，才是与神经资产相对应的实际竞争方案。

这里的 geometry-only shell 保留详细形状，不等同于减面外壳。若以后研究减面率和安全性的关系，应作为外壳压缩补充实验，而不是增加另一个名称含糊的主基线。

## 当前资产规模估算

以下数字来自 2026-09-08 对当前 GLB accessor、Meshopt buffer view、实例变换和运行资产的只读统计。它们是外壳实现前的工程估算，不是正式 HZB benchmark 结果。

### HKUST

| 项目 | 当前统计 |
|---|---:|
| 实例数 | 18,831 |
| runtime 登记的全局 GLB 数 | 3,273 |
| 扫描到的 LOD0 GLB 文件数 | 3,283，和 runtime 登记数的差异需在导出前核对 |
| 顶点数 | 94,953,445 |
| 索引数 | 166,907,502 |
| 三角形数 | 约 55,635,834 |
| 当前 LOD0 GLB 文件总大小 | 约 566.1 MB |
| Meshopt 压缩后 POSITION | 约 290.5 MB |
| Meshopt 压缩后 INDEX | 约 58.1 MB |
| 实例变换 | 约 0.2 MB |
| 预计详细纯几何外壳传输大小 | 约 350--365 MB |
| 当前 HKUST 神经运行资产 | 约 5.23 MiB |

当前 GLB 去掉法线、UV、材质和纹理后仍需要约 349 MB 的压缩位置、索引与实例变换；加上容器和场景结构后，预计详细外壳约 350--365 MB。进一步使用局部 16 位位置量化和尽可能多的 16 位索引，预计可能降到约 220--300 MB，但不能在没有实际导出结果时把该范围写成正式结果。

Meshopt 是传输编码，光栅化前通常仍需解码为 GPU 顶点和索引缓冲。当前 Float32 位置与现有索引完全展开约为 1.77 GB；合理量化和局部索引后，预计仍需约 0.9--1.2 GB GPU 几何内存。1080p 单份 R32Float HZB mip 链约 11 MB，不是主要内存来源。

### Metropolis

| 项目 | 当前统计 |
|---|---:|
| 实例数 | 41,298 |
| GLB 原型数 | 3,669 |
| 顶点数 | 4,526,291 |
| 索引数 | 25,946,499 |
| 三角形数 | 约 8,648,833 |
| 当前实例化 LOD0 GLB 文件总大小 | 约 183.3 MB |
| 当前未进一步压缩的纯位置、索引和实例变换 | 约 159.7 MB |
| 预计 Meshopt/量化后详细外壳 | 约 60--100 MB，需实际导出确认 |

现有 HKUST proxy 总计约 0.53 MB，主要是极粗粒度盒体；Metropolis proxy 为约 24.84 MB、495,576 个三角形，基本对应每个构件的盒体。二者都不是详细几何外壳，不能用其大小或准确率代表本基线。

## 性能预估与实测要求

HZB mip 构建和候选测试本身不是主要成本。以 HKUST 约 2 万实例和 1080p 为例：

| 阶段 | 桌面独显估算 | 集成或移动 GPU 估算 |
|---|---:|---:|
| HZB mip 构建 | 0.1--0.6 ms | 0.5--2 ms |
| 约 2 万个候选 AABB 测试与压缩 | 小于 0.5 ms | 0.5--2 ms |
| 直接绘制全部详细外壳深度 | 8--25 ms | 30--150 ms，可能受内存限制 |
| 空间块/BVH 预剔除后的外壳深度 | 3--12 ms | 10--50 ms，强依赖视点 |

这些范围只用于决定是否值得实现。正式论文只能报告硬件时间戳查询得到的实测值，并同时给出 GPU、分辨率、浏览器、相机轨迹和 p50/p95。

350 MB 外壳的理想纯传输下限约为：

- 100 Mbps：约 28 秒；
- 20 Mbps：约 140 秒。

当前约 5.23 MiB 神经资产对应的理想纯传输下限约为 0.44 秒和 2.2 秒。正式结果还必须加入连接建立、缓存、解码和上传时间。

## 准确性和公平评价

Geometry-only shell HZB 对当前相机通常会比神经模型更接近真实遮挡，但不能预先写成“准确率必然更高”：

- AABB 遮挡测试会因包围盒覆盖过大而保留一些实际被挡住的对象，形成 FP；
- 深度偏置、近裁剪面和透明面处理错误可能造成 FN；
- 单个当前相机 HZB 不会输出整个 view-cell 内任一合法位置的可见并集。

同一套 HZB 实现需要报告两个明确区分的评价模式，而不是建立两个方法名称：

1. **Current-view runtime**：当前真实 60 度相机只运行一次，评价前端当前画面、下载队列和每帧成本；
2. **View-cell offline union**：对数据集中完全相同的 subpose 集合运行 HZB 后取并集，用于和 view-cell GT 做同语义集合比较；该模式只用于离线公平评价，不能写成一次前端查询的运行成本。

所有方法必须使用相同候选、实例 ID、GT 和 split。学习方法只在 calibration 冻结阈值；HZB 的深度偏置和保守参数同样只能由 calibration 确定。正文至少同时报告：

- pose-macro 和 aggregate precision、recall、accuracy、balanced accuracy、specificity、F1；
- weighted recall 及其单侧置信下界；
- useful cull、bad cull、平均预测实例数和预测/候选比例；
- image PER、miss-pixel、wrong-ID pixel 和 extra-pixel；
- GLB 数量/字节削减及相同图像效用下的下载字节；
- 外壳或神经资产传输大小、解码后 GPU 内存、冷启动时间；
- depth-only、HZB mip、候选测试、压缩/回读和总调度 p50/p95；
- 桌面独显、集成 GPU 和真实移动设备结果。

完整指标定义继续以[统一 PVS 论文评价协议](../evaluation/unified_pvs_metrics_evaluation.md)为准。正式浏览器性能必须遵循[硬件 GPU 执行政策](../current/hardware_gpu_execution_policy.md)。

## HZB 相关工作边界

论文应引用但暂不数值复现：

- Greene、Kass 和 Miller 的 [Hierarchical Z-Buffer Visibility](https://research.adobe.com/publication/hierarchical-z-buffer-visibility)，作为 HZB 基础；
- Mattausch、Bittner 和 Wimmer 的 [CHC++](https://www.cg.tuwien.ac.at/research/publications/2008/mattausch-2008-CHC/mattausch-2008-CHC-draft.pdf)，作为利用空间和时间连贯性的层次遮挡查询代表；
- Lee 等人的 [Hierarchical Raster Occlusion Culling](https://cg.skku.edu/pub/papers/2021-lee-eg-hroc-crc.pdf)，作为对象 BVH 和 GPU 批量遮挡测试的现代代表。

本论文的创新不在改进 HZB 算法，因此不需要重做一组 HZB 论文方法。HROC 和 CHC++主要优化几何已经驻留后的查询组织、层次遍历和时间连贯性，没有消除流式冷启动时获取遮挡几何的成本。它们用于相关工作和实现合理性说明；正式系统对照只保留一个经过合理 GPU 优化的 Geometry-only shell HZB，避免使用低效 CPU readback 或逐对象查询构造弱基线。

如果目标投稿的审稿意见明确要求现代在线遮挡算法的数值比较，再评估 HROC。该决定不能影响已经登记的第三场景和 Geometry-only shell HZB 完整实测。

## 实施顺序

1. 对 Medical Clinic、LTU AHouse 和 BIM4LCA office 做许可与 IFC 转换体检，选择独立来源的第三场景；
2. 完成第三场景实例化、运行元数据、硬件采样、数据划分和场景专属训练；
3. 编写不透明纯几何外壳导出器，先在 HKUST 输出实际传输大小和解码 GPU 大小；
4. 实现 GPU depth-only、HZB mip、候选 AABB 测试和实例/GLB 压缩；
5. 完成 current-view runtime 与 view-cell offline union 两种口径评价；
6. 在 HKUST、Metropolis 和第三场景上统一报告模型与 HZB 的准确性、图像、资源和运行成本。

## 代码和资源边界

预期复用的现有入口：

- `neural_instance_culling/tools/glb_instancer/`：严格实例原型识别与 SLM 资产布局；
- `neural_instance_culling/dataset/build_scene_runtime_meta.mjs`：构建 `glbIndex.json` 和 `runtimeVisibilityMeta.json`；
- `neural_instance_culling/sampler/run_scene_viewcell_colorid_sampling.mjs`：硬件 view-cell Color-ID 采样；
- `neural_instance_culling/benchmark/evaluate_pvs.py`：统一实例集合指标；
- `neural_instance_culling/benchmark/evaluate_viewcell_image_per.py`：真实 60 度图像评价。

本计划在产生正式结果前不修改当前默认 checkpoint、冻结阈值、前端场景资产或已有 HKUST/Metropolis 结论。
