# 标准图形学场景 PVS 正式实验协议

日期：2026-09-10；2026-09-15 按 Connected-SAH 主线重写

状态：执行中。本文档只描述当前正式协议与当前正式产物。

## 1. 实验目的

Sponza、Viking Village 和 Big City 是论文正式测试场景，不只是“证明不依赖 BIM”的附加
例子。三个场景用于回答：

1. Full V4 在普通图形场景上能否满足画面安全门并有效剔除遮挡单位；
2. 相同单位、候选、GT 和 split 下，Full 是否优于 Keep-All 与 AABB + Ray MLP；
3. 相比 Geometry-shell HZB，Full 能否以更小的启动资产和更低的区域查询成本得到有用 PVS；
4. 模型延迟随候选单位数的变化是否适合桌面和移动 WebGPU/WASM。

标准场景不要求 GLB 原型复用。每个输出 renderable unit 对应一个 component ID 和一个
resource ID：

```text
renderable unit -> componentGlobalId -> resource ID -> 96D fixed geometry row
```

模型是场景专属 visibility codec，不宣称跨场景 zero-shot。HKUST/IFCBench 仍承担真实 GLB
复用与 progressive streaming 主实验；标准场景主要报告剔除、图像、启动资产和运行成本。

## 2. 场景与来源

| 场景 | 类型 | 固定来源 | 许可边界 |
|---|---|---|---|
| Sponza | 室内多层遮挡 | Khronos glTF Sample Assets 的 Sponza glTF | 仅本地实验，不分发源纹理和派生资产 |
| Viking Village | 大型室外村落 | `standard_graphics_sources/viking_village/VikingVillage.glb` | 论文报告统计和结果，不公开分发受限资产 |
| Big City | 大型室外城市 | NeuralPVS 官方资产中的 `BigCity/scene.gltf` | 本地实验；不据此声明派生资产可再分发 |

每个场景必须保留 `scene_source_audit.json`，记录实际源版本、节点、primitive、三角形、材质、
坐标单位、场景范围和许可。NeuralPVS 论文中的 primitive 概数不能替代本项目源审计。

## 3. Renderable-unit 协议

正式转换名称为 `standard_graphics_connected_sah_128k_v1`，partition schema 为
`connected-sah-pack-v2`。单位化只读取 scene graph、几何、材质和压缩字节，不读取标签、
模型分数或 HZB 结果。

1. 每个源 node、primitive、材质和 alpha mode 独立处理；不跨 primitive 或材质打包。
2. 先按共享顶点求三角形连通分量。
3. 小连通分量作为原子项，用确定性 16-bin SAH 按空间紧凑性打包到接近 `128 KiB`。
4. 超大连通分量按三角形质心做确定性 SAH 递归拆分。
5. 输出经固定 meshopt 参数编码；实际编码几何超过 `128 KiB` 时继续拆分。
6. 单三角形是唯一不可再分的超限例外，必须在 manifest 中登记。
7. 每个单位内部重索引但不减面；按固定源顺序分配连续 ID。
8. manifest 记录 `componentCount`、`sourceComponentCount`、`partitionReason`、
   `oversizedComponentSplit`、AABB、三角形数和编码字节。

正式统计必须同时报告单位数、源连通分量数、每单位分量数、超大分量拆分数、三角形分布、
编码字节 p50/p95/max 和 AABB 尺度。主结果按 unit 统计，并补 triangle/byte cull，避免单位
粒度本身抬高剔除率。

材质政策固定为：OPAQUE 可写关系和 HZB 深度；MASK 在 reference Color-ID 中保留 alpha
test，但不作为无纹理外壳的确定不透明遮挡物；BLEND 不进入静态遮挡剔除；未知材质按非
遮挡物处理。HZB 关闭背面剔除以保持保守。

## 4. 相机、View-cell 与 Split

三个场景使用同一空间契约：

- 显示相机 FOV 为 `60 degrees`；模型/候选相机 FOV 为 `66 degrees`；aspect 使用实际值；
- view-cell 是世界 XZ 水平圆盘，半径 `r=0.75 m`；
- 每 cell 使用 32 个确定性面积均匀 subpose，方向固定，只改变合法位置；
- 表面 clearance 为 `0.80 m`；
- 后退距离为 `r/tan(30 degrees)=1.299038 m`；
- GT 为全部成功 subpose 可见单位的并集，`visible_weight` 为单位跨 subpose 的最大屏幕覆盖；
- 浏览器运行时只做一次后退相机查询，不展开 32 个 subpose。

高空或不同楼层通过不同 Y 的 view-cell 中心表达，不在单个圆盘中加入垂直扰动。候选必须
覆盖整个 cell，正式数据要求 `visible_ids` 是 `candidate_ids` 的子集。

Split 按物理中心分组，使用固定 seed 做确定性随机划分。同一中心的全部方向和 subpose
不得跨 split。顺序统一为 train/calibration/validation/test；calibration 只选阈值，
validation 只选成员，test 在全部方法冻结后读取一次。

## 5. 当前正式数据

| 场景 | 源连通分量 | 单位 | 96D 表 | 成功 subpose | Train / Cal / Val / Test | 平均候选 | 平均 GT |
|---|---:|---:|---:|---:|---:|---:|---:|
| Sponza | 8,091 | 129 | `[129,96] FP16` | 213,504 | 4800 / 528 / 672 / 672 | 44.1667 | 7.9519 |
| Viking Village | 141,456 | 1,763 | `[1763,96] FP16` | 86,784 | 1944 / 216 / 276 / 276 | 544.4108 | 125.9956 |
| Big City | 652,925 | 2,861 | `[2861,96] FP16` | 514,560 | 11580 / 1284 / 1608 / 1608 | 748.5629 | 155.0919 |

三场景均无编码超限单位，`candidateMissVisible=0`。Color-ID 使用 NVIDIA Vulkan/ANGLE
硬件路径并保存 GPU evidence。固定几何特征由同一个冻结几何编码器从每单位 1024 个归一化
表面点导出；几何编码器不在目标场景上更新。

训练关系只使用 train split。每个稠密深度 shard 完成后立即压缩成稀疏遮挡边、生存观察和
深度矩，再删除 ID/depth payload；不构建全场景稠密缓存。当前 K=8 关系结果为：

| 场景 | 深度 shard | Train render rows | 关系边 | 生存观察 | 未覆盖 train pose |
|---|---:|---:|---:|---:|---:|
| Sponza | 32 | 43,200 | 23,280 | 1,739,324 | 0 |
| Viking Village | 16 | 17,496 | 128,997 | 4,922,117 | 0 |
| Big City | 64 | 104,220 | 438,877 | 62,247,879 | 0 |

## 6. Full 训练协议

三个场景使用同一 V4 架构和配置，从随机初始化训练：

| 项目 | 固定值 |
|---|---:|
| Epoch x step | `40 x 900` |
| Learning rate | `5e-5` |
| Poses per batch | `8` |
| Pose sampling | `ambiguity_balanced` |
| Hard pose fraction / quantile | `0.35 / 0.65` |
| Survival rank / relation K | `4 / 8` |
| Weighted-recall target | `0.99` |
| Recall guard final weight | `0.30` |
| Tail separation final weight | `0.30` |
| Positive tail / negative frontier | `0.01 / 0.02` |
| Positive importance floor / power | `0.25 / 0.5` |
| Calibration bootstrap | `10,000` |

Epoch 1-4 关闭尾部分离和召回保护；epoch 5-12 逐步加入尾部分离；epoch 13-40 再逐步加入
召回保护。固定 `p=0.5` 不作为安全门，也不使用 bias/temperature 移动阈值伪装效果。

每个 checkpoint 使用 calibration 的实际 float32 score change-points，预测规则为
`score >= threshold`。选择满足 weighted recall 及单侧 95% bootstrap LCB 都严格大于
`0.99` 的最高安全阈值；validation 只重放该阈值。安全成员依次比较 Useful Cull、CNOR、
balanced accuracy、Occlusion Recall、precision 和平均保留数。

## 7. Scene-first 调度

Full 固定按 seed round 执行：

1. Sponza、Viking Village、Big City 的 seed20260801 同时训练；
2. 首轮稳定后，如果单卡显存和利用率仍有余量，同时启动三个场景 seed20260802；
3. 再按同一资源门决定是否同时启动三个场景 seed20260803；
4. 三种子 validation 完成后，冻结每场景唯一 Full 成员与阈值。

不能只给一个场景提前启动额外种子；增加并发时必须整轮覆盖三个场景。GPU 参数表示并发
slot，允许重复 GPU ID，但启动前必须按实际显存、利用率和吞吐判断。同卡并发不能与正式
Color-ID、HZB、WebGPU timing 同时运行。GPU0 上的外部任务不得终止或抢占。

正式 runner：

```bash
conda run --no-capture-output -n slm_pvs python -u \
  neural_instance_culling/benchmark/run_standard_graphics_connected_sah_full.py \
  train --seeds 20260802 --gpu-slots 1 2 3
```

## 8. 基线与执行顺序

所有方法使用相同 Connected-SAH units、`66 degrees` candidate CSR、区域 GT 和 split。

| 方法 | 作用 |
|---|---|
| Keep-All | 绝对安全、零剔除参照 |
| AABB + Ray MLP | 检验包围盒、投影和相机射线特征是否足够 |
| Geometry-shell HZB | 几何外壳已经预下载后的现代 GPU 遮挡基线 |
| Full V4 | 本文紧凑场景专属 PVS 表示 |

执行阶段门固定为：

```text
Full 三场景三种子并冻结
  -> AABB MLP 学习率扫描和三种子
  -> lossless/equal-asset HZB calibration、test、timing
  -> Full/AABB/HZB 的一次 frozen test、图像和端侧 runtime
```

AABB 使用与 Full 相同的训练步数、三种子、安全校准和候选，不扩充到 96D 输入；HKUST 的
核心消融已隔离 Full 各模块贡献，而 AABB 基线的定义就是只使用几何包围盒与视角信息。

HZB 唯一算法为：确定不透明外壳深度光栅化、max-depth pyramid、候选 projected AABB 的
保守测试。仅当 `HZBMax + bias < candidateNear` 时剔除；近裁剪面、非有限投影、相机位于
AABB 内和其他不确定情况全部保留。

- Point60：只测试当前真实相机点，是单点在线遮挡能力参照；
- Region66：对登记 subpose 做 HZB 并取可见并集；多次查询的总耗时全部计入；
- Lossless shell：保留全部确定不透明 Connected-SAH 外壳单位；
- Equal-asset shell：以冻结神经运行资产字节为上限，从同一输出单位按 train-only 投影收益/
  压缩字节选择完整遮挡单位。它是预算敏感性基线，不宣称保守或最优简化。

论文只声称实现现代批量 Geometry-shell HZB，不冒称完整复现 HROC。相关工作包括
Hierarchical Z-Buffer Visibility、HROC 的 GPU 批组织，以及基于平面截面的遮挡物简化。

## 9. 指标

| 类别 | 必须报告 |
|---|---|
| 排序 | pose-macro PR-AUC、pose prevalence、AP lift；aggregate AP 作为辅助 |
| 画面安全 | Visible Recall、False Occlusion Rate、weighted recall、WR 单侧 95% LCB |
| 遮挡能力 | Occlusion Recall、CNOR、Useful Cull、Bad Cull |
| 分类诊断 | Precision、instance accuracy、balanced accuracy、平均候选/GT/保留数 |
| 图像 | PER、miss-pixel、wrong-ID、extra-pixel 的 aggregate/mean/p95 |
| 系统成本 | 启动传输、解码内存、纯查询 p50/p95、候选规模拟合、冷启动 |

其中：

```text
Occlusion Recall = TN / (TN + FP) = specificity
False Occlusion Rate = FN / (TP + FN) = 1 - Visible Recall
Useful Cull = TN / candidate
Bad Cull = FN / candidate
CNOR = sum_p(TN_p / C_p) / sum_p((TN_p + FP_p) / C_p)
```

CNOR 先按每个 pose 的候选数归一化剔除机会，避免宏平均被极少负样本 pose 放大，也避免
aggregate 被超大候选 pose 主导。它不替代 weighted-recall 安全门或 Useful Cull。

## 10. 论文表图

`Table G1` 报告场景范围、源三角形、源连通分量、单位数、单位字节分布、split、候选和 GT。

`Table G2` 报告每场景 Keep-All/AABB/HZB/Full 的 pose AP、prevalence、Visible Recall、
WR/LCB、Occlusion Recall、CNOR、Useful/Bad Cull 和平均保留数。

`Table G3` 报告神经资产、equal-asset shell 和 lossless shell 的传输字节、解码内存、
Point/Region 查询次数与 p50/p95、冻结安全点 Useful Cull 和图像 miss p95。

核心图包括：

1. 启动资产大小（对数轴）与安全门下 Useful Cull 的 Pareto；
2. 候选单位数与桌面/移动端延迟；
3. 每场景 median、p95、worst miss 的 Reference/Prediction/Difference；
4. 若时间允许，在一个代表场景补 `64/128/256 KiB` 单位粒度敏感性。

标准场景的一单位一资源 streaming 若补做，只单独报告 score ranking 的 units/bytes-to-
visible-weight coverage，不与 HKUST/IFCBench 的资源复用主表混合。

## 11. 当前状态与下一步

截至 2026-09-15：

- 三场景转换、固定几何表、硬件 Color-ID、Pose CSR、深度分片和 K=8 关系均完成；
- 三场景 preflight 与 seed20260801 smoke 均通过，且没有读取 test；
- seed20260801、seed20260802 和 seed20260803 均已按 Sponza/Viking/Big City 完整三场景
  round 启动；每张训练卡并发三个种子；
- 三任务同卡后 GPU 1/2/3 显存约 `1.6/3.3/7.8 GiB`，利用率约 `99-100%`，功耗约
  `123/131/130 W`；后续不再增加训练任务，只监控总吞吐、数值和首次 validation；
- AABB、HZB、标准场景 test、图像和 runtime 尚未启动，等待 Full 冻结；
- HKUST/IFCBench checkpoint 不重训，已完成精确 calibration/validation 重校准；新阈值下
  的 test、图像、streaming 和前端资产尚未重放。

训练监控必须检查进程存活、epoch/step、loss 数值有限、step/s、ETA、显存和 GPU 利用率。
中间 validation 只用于健康诊断，不提前取消登记的完整矩阵。

## 12. Big City 剔除能力专项优化

日期：2026-09-15。

Big City 原 `128 KiB / K=8` 三种子长训在 epoch 20-22 主动停止，只保留为训练诊断，
不进入论文正式成员池。停止时三个种子均未通过 validation 的 `weighted recall LCB > 0.99`
安全门，CNOR 为 `0.2101/0.2232/0.2958`。该结果不能继续消耗约 14 小时/种子的剩余训练时间，
因为数据和训练入口存在以下结构性瓶颈：

1. Big City 的 652,925 个源连通片被打包为 2,861 个单位，每单位连通片数均值 228.2、
   p50 227、p95 476；一个 1024 点、96 维几何表征需要概括过多离散碎片；
2. `K=8` 关系单元有 74.0% 发生截断，关系质量 q01 只有 0.5196；
3. 训练集约 22.6% view-cell 没有 GT 可见单位，其中候选非空的纯负例 pose 未进入旧
   `ambiguity_balanced` 采样，模型没有直接学习这批重要剔除机会；
4. seed02 epoch20 的正样本 q01 为 0.464973、负样本 q99 为 0.476780，安全阈值落在重叠
   尾部内，说明低 CNOR 来自分数不可分，而不是候选负例不足。Big City 平均约 736 个候选、
   160 个 GT，可剔除机会约占 78%。

### 12.1 第一阶段：固定 128 KiB 单元上的训练专项扫描

实验名固定为 `pvs_v4_bigcity_connected_sah_occlusion_opportunity_v1`。只读取
train/calibration/validation，不读取 test。每个 pilot 从随机初始化训练 `8 x 450`，使用
相同 V4 架构和精确 float32 calibration。最终单因素矩阵为：

| 配置 | 关系 K | 学习率 | 纯负例 pose | 困难边界 pose | 普通可见 pose |
|---|---:|---:|---:|---:|---:|
| P0 控制 | 8 | `5e-5` | 0% | 35.0% | 65.0% |
| P1 只增关系容量 | 16 | `5e-5` | 0% | 35.0% | 65.0% |
| P2 只增纯负例 | 8 | `5e-5` | 25.0% | 37.5% | 37.5% |
| P3 联合配置 | 16 | `5e-5` | 25.0% | 37.5% | 37.5% |
| P4 低学习率 | 16 | `2e-5` | 25.0% | 37.5% | 37.5% |
| P5 更大关系容量 | 24 | `5e-5` | 25.0% | 37.5% | 37.5% |

这里的“纯负例增强”不是修改相机或标签，而是从 train split 中显式重采样
`candidate_count > 0 && visible_count == 0` 的合法 view-cell。每个 8-pose batch 按表中固定
配额组成；困难边界池仍由 train-only 实例可见频率和 subpose 边界命中率构造。困难负例尾部
比例由 `0.02` 提到 `0.04`，召回保护从总步数 10% 后开始、30% 时离开慢速段，避免旧协议
到 epoch 13 后才有效约束安全尾部。

完整六项 pilot 不由中间指标提前取消。选择时先取通过 calibration 和 validation 安全门的
成员，再比较 Useful Cull、CNOR、balanced accuracy、Occlusion Recall 和平均保留数；若无
安全成员，按 validation LCB、weighted recall、CNOR、Useful Cull 的顺序选相对最优配置，
如实标记为诊断配置。选定配置必须从头完成三种子 `40 x 900`，不能从已停止 checkpoint
续训，也不能用 test 选参数。

关系重建与训练 smoke 已完成：`K=16` 的截断 cell 为 30,986、保留质量 q01 为 0.7327；
`K=24` 为 21,266 和 0.8456。两者均来自同一 train-only 深度缓存。修正后的两步 smoke
每批实际进入损失的 pose 数为 8，纯负例配额没有在训练张量构造阶段被丢弃，loss 数值有限。

短训只缩短实际运行步数，不能压缩训练课程。pilot 的 3,600 个实际 optimizer step 必须按
完整 36,000 步课程计算学习率、实例校准、尾部分离和召回保护进度，即只观察完整训练的前
10%。2026-09-15 首次启动错误地把课程也压缩为 3,600 步，导致末步学习率变成 0、召回
保护接近满权重，ROC-AUC 退化到随机水平；该批结果不具参数选择含义，删除后按本段口径
原地重跑。第二次启动虽然使用了正确总 horizon，但把尾部分离提前到完整训练 5% 处，仍然
污染了基础分类比较；因此最终矩阵恢复到 10% 后启动尾部分离，并加入 P0/P1/P2 控制项，
分别隔离关系容量和纯负例增强的影响。前两批错误配置结果均删除，不进入实验汇总。

### 12.2 第二阶段：单元粒度重建触发条件

若第一阶段相对最优配置仍未通过 `weighted recall LCB > 0.99`，或安全工作点 CNOR 仍低于
`0.50`，则必须执行场景无关的细粒度规则：目标压缩大小 `64 KiB`，且每个单位最多包含
128 个源连通片。两项限制都只使用源几何和编码字节，不读取标签或模型结果。单位 ID 改变后
必须重建 GLB、1024 点/96D 几何表、硬件 Color-ID、Pose CSR、train-only depth/relation，
并用同一 split 中心划分协议重新训练。正式论文同时报告单位数、候选数、模型资产、运行时
和 byte/triangle cull，避免只靠更细单位获得好看的 unit-level 指标。

第一阶段完成前不启动 AABB MLP、HZB 或 test；这些基线必须使用最终冻结的 Big City 单位、
候选、GT 和 split。

2026-09-16，六项第一阶段 pilot 已全部完成。唯一通过 validation 安全门的是 K24、
`lr=5e-5`、25% 纯负视点的 P5：`WR=0.994473`、`LCB=0.991721`，但其
`CNOR=0.053696`、Useful Cull `0.009193`，平均预测 `824.98/834.38`，实质上接近
Keep-All。其余配置均未通过安全门，最高的非安全 CNOR 也没有形成安全工作点。因此汇总文件
`pilot_selection.json` 明确给出 `partitionRebuildRequired=true`，按预登记规则进入第二阶段，
不把 P5 当作论文合格结果，也不读取 test。

第二阶段正式数据键为 `bigcity_64k`，输出到
`dataset/out/standard_graphics_connected_sah_64k_v1/bigcity_64k`。通用 Connected-SAH
分割器新增显式 `maxComponentsPerUnit`，转换命令固定使用 `64 KiB` 与最多 `128` 个源连通片；
转换清单和 preflight 同时记录并校验这两个字段。该变体复用冻结的 Big City V2 view-cell
pose plan 与 split 数量，但因单位 ID 改变，GLB、1024 点、96D 特征、硬件 Color-ID、
Pose CSR、train-only depth 和 relation 全部重新生成。默认预处理场景仍只有原三个 128 KiB
场景，`bigcity_64k` 必须显式指定，避免覆盖现有资产。

2026-09-16，Big City 64 KiB 的 64 个 train-only 深度分片与 64 份 NVIDIA Vulkan 硬件
证据全部完成，两个 stderr 均为空；K=8 关系 CSR 和生存观测随后生成成功。正式 Full
实验名为 `pvs_v4_bigcity_connected_sah_64k_full_v1`，入口
`benchmark/run_bigcity_64k_full.py`，三种子各自从头执行固定 `40 x 900` Full V4 协议。
该实验不复用 128 KiB 模型名，不读取 test；Full 冻结前不启动 Big City AABB/HZB。

## 13. Sponza 与 Viking 安全性损失微调

日期：2026-09-16。实验名为 `pvs_v4_standard_graphics_safety_refinement_v1`。Sponza 三种子
已完成但 validation LCB 为 `0.9773/0.9780/0.9812`，均未过安全门；Viking 当前只有 seed01
通过安全门，另外两个种子的最近 LCB 为 `0.9886/0.9860`。两场景不修改单位、GT、split、
关系 K、学习率主协议或候选集合，先从各自现有最佳 checkpoint 用 fresh AdamW、`lr=1e-5`
执行 `4 x 450` 的 train-only 安全 refinement：

| 配置 | 尾部分离 | 召回保护 | 最差 pose 权重 | 生存场 | 关系一致性 |
|---|---:|---:|---:|---:|---:|
| R1 | 0.15 | 0.30 | 0.25 | 0.25 | 0.10 |
| R2 | 0.15 | 0.45 | 0.35 | 0.25 | 0.10 |
| R3 | 0.10 | 0.45 | 0.35 | 0.15 | 0.05 |

R1 只降低过强的困难尾部分离；R2 再增强加权召回和最差 pose 保护；R3 同时降低结构辅助
损失，检验生存/关系监督是否在训练后期继续压制最终可见性分类。refinement 的前 5% 步数
线性恢复目标权重，之后保持全权重；不重置运行时头，不读取 test。

每场景先在 validation 安全池中最大化 Useful Cull、CNOR 和 balanced accuracy；没有安全
成员时按 LCB、weighted recall、CNOR 选相对最优。选定后，对该场景三个原始种子分别执行
同一配置的 `4 x 900` refinement 和 10,000 次 calibration bootstrap，作为正式三种子候选。
Viking 原始三种子必须先完成，不能用未完成 checkpoint 进入正式 refinement。

2026-09-16，六个 pilot 已按登记矩阵全部完成，且均未读取 test。Sponza 的 R1/R2/R3
validation `weighted recall LCB / CNOR` 分别为 `0.982012 / 0.610289`、
`0.982096 / 0.611349`、`0.980356 / 0.625331`；三者均未过严格安全门，因此按诊断池的
LCB 优先规则选择 R2 进入正式三种子 refinement。Viking 的对应结果为
`0.987277 / 0.879463`、`0.987265 / 0.878485`、`0.987308 / 0.887478`，同样没有安全
成员，按 LCB、weighted recall、CNOR 的既定顺序选择 R3。这里的“选择”只表示下一阶段
相对最优配置，不表示论文安全结论；正式三种子仍须独立通过 `LCB > 0.99`。

正式 runner 支持用 `--scenes` 分阶段执行。Sponza 的原始三种子已经完成，可先在空闲 GPU
上运行；Viking 必须等待三个原始长训成员均生成最终 calibration summary 后再运行，防止
从未冻结的中途 checkpoint 启动 refinement。默认不传 `--scenes` 时仍覆盖两个场景。

Sponza R2 正式三种子已于 2026-09-16 使用 GPU 1 启动，配置为 `4 x 900`、fresh AdamW、
`lr=1e-5` 和 10,000 次 calibration bootstrap。Viking 仍等待原始三种子完成后再启动。

同日，Sponza 正式 refinement 已完成。seed01/02/03 的 validation
`weighted recall LCB / CNOR / Useful Cull` 分别为
`0.984861 / 0.438623 / 0.414399`、`0.982582 / 0.461547 / 0.408474`、
`0.981572 / 0.615240 / 0.553772`，三个成员状态均为
`no_qualified_safety_workpoint`，且 `testRead=false`。R2 对 seed01 的安全性有改善，但没有达到
严格 `LCB > 0.99`；因此 Sponza 当前仍没有可进入冻结 test 的安全模型。该结果保留为正式
负结果，下一步不继续围绕相同损失权重堆叠微调，待 Viking 与 Big City 主线完成后统一决定
是否调整 Sponza 数据/单位协议。

Viking 原始 40 epoch 三种子随后全部完成。seed01/02 的 validation
`LCB / CNOR` 为 `0.991276 / 0.867268`、`0.990626 / 0.846276`，均通过安全门；seed03
为 `0.986638 / 0.837853`，未通过。R3 正式 refinement 已在 GPU 1/2/3 各启动一个种子：
seed01/02 从各自 `best_safe.pt` 继续，seed03 从 `best_diagnostic.pt` 继续，均为 `4 x 900`
和 10,000 次 calibration bootstrap，test 保持关闭。

Viking R3 正式 refinement 随后完成，seed01/02/03 的 validation
`LCB / CNOR / Useful Cull` 分别为 `0.987469 / 0.881627 / 0.700522`、
`0.984181 / 0.849112 / 0.668840`、`0.988815 / 0.847689 / 0.658310`，三个成员均未过
安全门。微调没有改善原始安全模型，因此最终保留原始 seed01/02 的 `best_safe.pt`；原始
seed03 作为未过门的三种子诊断成员保留。R3 不覆盖原始结果，也不触发 test 读取。

Viking 后续目标同时要求 validation `LCB > 0.99` 和 `CNOR >= 0.80`。seed01/02 原始安全
checkpoint 已满足该目标；seed03 checkpoint rescue 为
`WR/LCB/CNOR=0.992659/0.987740/0.806078`，剔除合格但安全不足。专项实验
`pvs_v4_viking_seed03_safe_cnor_refinement_v1` 从该 checkpoint 比较 calibration margin
`0.995/0.997`，使用 `lr=1e-6`、16 pose/batch、75% 困难 pose，并保留 `0.15` 尾部分离与
`0.005` 负尾比例。只有同时满足安全门和 CNOR 目标的成员才可替换 seed03；test 保持关闭。

## 14. Sponza 与 Viking checkpoint 安全补救

日期：2026-09-16。实验名为 `pvs_v4_standard_graphics_checkpoint_rescue_v1`，入口为
`benchmark/run_standard_graphics_checkpoint_rescue.py`。上一轮 R2/R3 说明继续以
`lr=1e-5` 增加普通召回损失会造成安全排序漂移。本轮只处理尚未达标的 Sponza 三种子和
Viking seed03，不重训已安全的 Viking seed01/02，不读取 test。

Sponza 的另一个具体问题是上一轮从早期最佳诊断 checkpoint 初始化。seed01/02 的来源
checkpoint 位于实例校准课程早期，逐实例残差融合系数仍接近零；checkpoint 续训会继承该
系数，因此上一轮 refinement 实际没有获得完整的逐单元校准能力。本轮改从已有长训中
融合系数已接近 `1` 且 validation LCB 相对最好的 checkpoint 初始化：seed01 epoch24、
seed02 epoch12、seed03 epoch16。Viking seed03 从原始 epoch32 最优诊断点初始化。四个来源
均为 `testRead=false`。

两项短续训配置均使用 `8 poses/batch`、50% train-only 困难 pose、较低的生存场/关系辅助
权重和完整启用的召回保护课程：

| 配置 | LR | 召回保护/目标 | 最差 pose 比例/权重 | 尾部分离 | 正尾质量 | 负尾比例 |
|---|---:|---:|---:|---:|---:|---:|
| guarded | `2e-6` | `0.75 / 0.997` | `0.35 / 0.75` | `0.15` | `0.01` | `0.005` |
| tail-protected | `5e-6` | `1.00 / 0.998` | `0.50 / 1.00` | `0.10` | `0.02` | `0.0025` |

扫描固定为 `2 x 450` updates 和 2,000 次 calibration bootstrap。Sponza 用同一配置覆盖三个
种子，先比较安全种子数，再比较最差/平均 LCB、WR、CNOR 和 Useful Cull；Viking 只选择
seed03 配置。即使扫描没有安全成员也选择相对最优配置，随后完成 `4 x 450`、10,000 次
bootstrap 的正式续训。正式成员仍须独立通过 validation `WR > 0.99` 且 `LCB > 0.99`；
未过门时保留为诊断结果，不打开 test。

同日，两项扫描的 8 个登记成员全部完成。Sponza 的 guarded 配置三种子 validation
`WR LCB / CNOR` 为 `0.985681 / 0.598150`、`0.978377 / 0.633747`、
`0.984378 / 0.575611`；tail-protected 对应为 `0.983799 / 0.633778`、
`0.978148 / 0.636481`、`0.982935 / 0.588948`。两项均无安全成员，按预登记的最差/平均
LCB 优先规则选择 guarded。Viking seed03 的 guarded 与 tail-protected LCB 分别为
`0.986755`、`0.987683`，后者的 `WR/CNOR/Useful Cull` 为
`0.992693/0.812336/0.614368`，因此选择 tail-protected。正式续训已按选择启动；扫描结果
只用于 validation 配置选择，不读取 test。

checkpoint rescue 的 Sponza 正式三种子随后完成，`WR LCB / CNOR` 为
`0.985783 / 0.596611`、`0.978539 / 0.631816`、`0.984479 / 0.576166`，仍无安全成员。
进一步检查确认：早期最佳诊断 checkpoint 的实例校准融合系数为零，而原续训入口会固定
继承该值。训练器因此新增显式 `--init-instance-calibration-blend`，仅用于 checkpoint 续训
并同时记录来源值和生效值，默认行为不变。

下一轮独立实验为 `pvs_v4_sponza_recall_first_refinement_v1`。三个种子从各自原始
`best_diagnostic.pt` 初始化并将融合系数设为 `1`，比较 `lr=1e-6/2e-6` 两项召回优先配置；
两项均使用 16 pose/batch、75% train-only 困难 pose、召回目标 `0.999`，降低尾部分离和
负尾压力。扫描为 `2 x 450`，三种子统一选择后正式续训为 `4 x 450`；validation 只用于
配置选择，test 保持关闭。

该扫描六个成员已完成。`lr=1e-6` 三种子 validation LCB 为
`0.989492/0.986695/0.984990`，`lr=2e-6` 为
`0.989683/0.986649/0.985206`；均未过安全门。按预登记的最差 LCB 优先规则选择
`lr=2e-6` 完成三种子 `4 x 450` 正式续训。Viking checkpoint rescue 的 seed03 正式结果
为 `WR/LCB/CNOR=0.992659/0.987740/0.806078`，仍未过门；Viking 继续保留原始 seed01/02
安全 checkpoint，test 不读取。

Sponza recall-first 正式三种子随后完成。seed01 的
`WR/LCB/CNOR=0.992162/0.990110/0.422597`，成为首个严格安全成员；seed02/03 的 LCB 为
`0.986583/0.984718`，仍未过门。三者 calibration LCB 只有
`0.990995/0.990142/0.990190`，说明最高安全阈值缺少跨 split 余量。

下一阶段 `pvs_v4_sponza_calibration_margin_refinement_v1` 将 calibration 阈值选择目标与最终
validation 安全门分离：前者扫描 `0.995/0.997`，后者继续固定为 `0.99`。扫描只使用尚未
达标的 seed02/03，各从 recall-first 最佳诊断 checkpoint 用 `lr=5e-7` 继续 `2 x 300`；
按安全成员数、最差 LCB、Useful Cull 和 CNOR 选择统一 margin 后，再对三个种子执行
`2 x 450` 与 10,000 次 bootstrap。具体阈值仍只由 calibration 冻结，validation 不反向
搜索阈值，test 保持关闭。

margin 选择规则补充为：先最大化安全成员数；若某个配置使全部登记种子安全，则在全安全池
中按平均 Useful Cull、CNOR、最差 LCB 排序。LCB 超过安全门后的额外余量不能优先于实际
剔除收益，避免选择接近 Keep-All 的工作点。

`0.995/0.997` margin 扫描完成后，seed02 的 validation LCB 为
`0.988343/0.988829`，seed03 为 `0.986687/0.989019`；两项均未过门，但更严格 margin
持续提高安全性。扩展实验 `pvs_v4_sponza_calibration_margin_extension_v1` 因此固定比较
`0.999/0.9995`，其余 checkpoint、训练和选择协议不变。原 margin 扫描完整保留，不用扩展
结果覆盖旧含义。

扩展扫描四项均通过 validation 安全门。`0.999` 的 seed02/03
`WR/LCB/CNOR` 为 `0.995417/0.991439/0.200793`、
`0.997859/0.996548/0.420289`；`0.9995` 虽有更高 LCB，但 CNOR 降至
`0.125575/0.386284`。按全安全池最大化 Useful Cull/CNOR 的规则选择 `0.999`。

`0.999` 正式三种子 `2 x 450`、10,000 次 bootstrap 已完成，seed01/02/03 的 validation
`WR / LCB / CNOR / Useful Cull` 分别为
`0.999158 / 0.998847 / 0.176069 / 0.192766`、
`0.995422 / 0.991499 / 0.195047 / 0.171581`、
`0.997860 / 0.996592 / 0.418873 / 0.368343`。三个成员均为 `safe` 且
`testRead=false`。该结果只保留为安全诊断：其 CNOR 明显低于论文所需的安全—剔除折中，
不作为 Sponza 最终冻结模型。

Sponza 下一阶段改用 `sponza_64k`：目标压缩单位 `64 KiB`、每单位最多 64 个源连通片，
复用同一 view-cell pose plan、中心级随机 split、32 subpose GT 和 66 度候选协议，但重新
生成 GLB、1024 点、96D 特征、Color-ID、Pose CSR、train-only 深度与 K=8 关系。目标不是
靠更低阈值过门，而是在 validation `LCB > 0.99` 时达到 `CNOR >= 0.80`；若仍达不到，必须
分析单位粒度与表征上限，不能把 margin-only 结果写成最终主结果。

2026-09-16，`sponza_64k` 预处理已经完成。转换得到 303 个独立 renderable units；1024 点、
96D 几何特征、16 个硬件 Color-ID 分片、Pose CSR、32 个 train-only 三角形深度分片及其
NVIDIA Vulkan 证据、K=8 关系 CSR 均生成成功，相关 stderr 为空。训练 preflight 确认 split
为 `4800 train / 528 calibration / 672 validation / 672 test`，test 尚未读取；两步 smoke
通过并产生安全 checkpoint。正式实验使用
`pvs_v4_sponza_connected_sah_64k_full_v1`，三种子均从头执行固定 `40 x 900` 协议，正式
验收同时要求 validation `weighted recall LCB > 0.99` 和 `CNOR >= 0.80`。seed01 已在 GPU0
启动，seed02/03 在 Viking 短程微调释放 GPU 后继续。

Viking seed03 的 `0.995/0.997` calibration margin 扫描已完整结束。两项 validation LCB
分别为 `0.999517/0.999981`，但 CNOR 只有 `0.596124/0.421267`，说明过强安全裕量通过
降低阈值换取了近 Keep-All 的结果，均不进入 final。随后对原始
`WR/LCB/CNOR=0.992659/0.987740/0.806078` checkpoint 做不改权重的精确分数变化点审计：

| Calibration floor | Validation LCB | Validation CNOR | Avg pred |
|---:|---:|---:|---:|
| 0.9905 | 0.988065 | 0.795412 | 180.64 |
| 0.9910 | 0.988542 | 0.778917 | 193.49 |
| 0.9915 | 0.989015 | 0.757935 | 210.46 |
| 0.9920 | 0.989207 | 0.746278 | 219.03 |

该审计只用 calibration 冻结阈值并在 validation 回放，不读取 test。结果证明现有排序没有
同时满足 `LCB > 0.99` 和 `CNOR >= 0.80` 的阈值交集，不能继续靠调低阈值补救。专项实验
`pvs_v4_viking_seed03_tail_separation_refinement_v1` 因此从原始高 CNOR checkpoint 出发，
固定 calibration floor `0.99`，将全局召回保护降到 `0.10`、共享困难尾部分离提高到 `1.0`，
比较 `2e-7/5e-7` 两个小学习率；目标是抬高低分重要正例并压低高分负例，而不是整体抬高
可见概率。只有双门槛同时通过时才运行正式 10,000 次 bootstrap 成员。

首轮尾部分离扫描完成后，`2e-7/5e-7` 的 validation `LCB / CNOR` 分别为
`0.987468 / 0.819580` 和 `0.987316 / 0.826272`。两项都提高了剔除能力，但安全尾部没有
改善，因此不进入 final。登记扫描扩展为两个中间折中点：`balanced_guard050` 使用召回保护
`0.50`、目标 `0.997`、尾部分离 `0.30`；`balanced_guard080` 使用召回保护 `0.80`、目标
`0.998`、尾部分离 `0.25`。两项均使用 `lr=2e-7`，用于定位召回优先导致 CNOR 塌缩与分离
优先导致 LCB 不足之间的可行区间，test 仍关闭。

中间保护扫描的 `0.50/0.80` 两项最终 `LCB / CNOR` 为
`0.987379 / 0.822950` 与 `0.987437 / 0.821748`，仍未改变安全尾部，故不进入 final。逐 pose
回放进一步定位到：validation pose `1401/1402/1403` 位于同一中心，只漏掉 unit `1639`，
该单元权重分别为 `344598/249844/134223`，模型分数为 `0.6884/0.6849/0.6854`，略低于
冻结阈值 `0.7018`。unit `1639` 在 train 中可见 801 次且最大权重为 `1,000,000`；这不是
未见实例，而是空间遮挡边界上的高视觉风险尾部。

逐 pose 回放说明安全尾部主要受少数高权重可见单元影响。这类误差作为模型局限和定性案例
报告。Viking 的预测集合严格由 Full V4 分数与 calibration 阈值决定，统一运行契约仍是
`66° AABB 候选 -> Full V4 分数 -> calibration 冻结阈值 -> 60° 渲染过滤`。

sampling-v2 的纯模型按 2026-09-17 新资格规则重新冻结 calibration 阈值：优先满足
`WR/LCB > 0.99`；validation LCB 未达目标但平均 WR 达标的成员仍保留。结果为：

| Seed | WR | WR LCB | CNOR | Useful Cull | Avg pred | Qualification |
|---:|---:|---:|---:|---:|---:|---|
| 20260801 | 0.992606 | 0.991289 | 0.866635 | 0.693639 | 89.51 | LCB 目标达成 |
| 20260802 | 0.992468 | 0.990656 | 0.846497 | 0.668437 | 107.47 | LCB 目标达成 |
| 20260803 | 0.992705 | 0.987887 | 0.804466 | 0.608423 | 173.20 | 平均 WR 达标 |

seed03 的三处高权重漏检会显著拉低单侧置信下界，但不应被解释为模型整体没有剔除能力：
三个种子的 validation WR 和 CNOR 均超过 `0.99/0.80`，seed01/02 还达到 LCB 目标。

模型、阈值和 split 冻结后，三个纯 Full V4 成员在 sampling-v2 的同一 276-pose test 上得到：

| Seed | WR | WR LCB | CNOR | Useful Cull | Bad Cull | Pose PR-AUC | Avg pred |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 20260801 | 0.992563 | 0.991133 | 0.839233 | 0.672193 | 0.128270 | 0.891669 | 111.76 |
| 20260802 | 0.993848 | 0.992719 | 0.816053 | 0.645458 | 0.119556 | 0.875141 | 131.62 |
| 20260803 | 0.994121 | 0.992392 | 0.782926 | 0.601489 | 0.071351 | 0.830266 | 183.24 |
| mean | 0.993511 | 0.992081 | 0.812737 | 0.639713 | 0.106392 | 0.865692 | 142.21 |

三个 test 成员的平均 WR 和 LCB 都超过 `0.99`；seed03 test CNOR 为 `0.782926`，三种子
均值为 `0.812737`。正式产物位于
`benchmark/out/paper_results/pvs_v4_recall_target_policy_v1/viking_village_128k`，包含每个
seed 的 calibration、validation、test、逐 pose sidecar 和日志。

## 15. 三个标准场景统一为 64 KiB 分割粒度

日期：2026-09-17。

为避免正文主表中 Sponza、Big City 使用 `64 KiB` 而 Viking Village 使用 `128 KiB`，新增
`viking_village_64k`。三个场景的跨场景控制变量统一为 meshopt 压缩后目标单位大小
`64 KiB`；Connected-SAH 仍以源 primitive/material 内的连通片为原子，超过目标的连通几何
继续内部切分。`maxComponentsPerUnit` 只是限制单个资源聚合过多微小碎片的结构保护，不替代
也不改变 `64 KiB` 这一正式分割粒度。Viking 与 Big City 均取 128，Sponza 取 64；论文场景
统计表同时报告实际单位数和实际压缩字节分布，不能只报告名义目标。

Viking 64 KiB 复用已冻结 sampling-v2 的 `viewcell_pose_plan.jsonl` 和
`1944 train / 216 calibration / 276 validation / 276 test` 划分；不改变相机中心、方向、
32 个 subpose、66/60 度 FOV 或 GT 并集语义。单位改变后，GLB、1024 点几何输入、96D 固定
几何特征、Color-ID、Pose CSR、train-only 深度层和 K=8 关系表必须全部重建，不能复用
128 KiB 的单位级资产。

预处理入口：

```bash
conda run -n slm_pvs python neural_instance_culling/benchmark/run_standard_graphics_connected_sah_preprocessing.py \
  convert --scenes viking_village_64k
```

之后依次执行 `geometry`、`color-id`、`pose-csr`、`depth-manifest`、`shards` 和 `relation`。
Color-ID 与深度分片继续执行 NVIDIA Vulkan 硬件门。正式模型实验名为
`pvs_v4_viking_connected_sah_64k_full_v1`，入口为
`benchmark/run_viking_64k_full.py`，三个种子均从头训练 `40 x 900`，训练和阈值冻结期间不读取
test。选择时优先以 validation 平均 WR 与 LCB 达到 `0.99`、CNOR 达到 `0.80` 为目标；平均
WR 大于 `0.99` 但 LCB 或 CNOR 未达目标的成员仍保留并如实报告，不以 CNOR 单项提前终止。
最终标准场景主表使用三个 64 KiB 版本；Viking 128 KiB 结果降为分割粒度对照，不能与 64 KiB
主表混写成同一控制条件。

同日该流水线已经完整执行。转换得到 3,132 个 renderable units，相比 128 KiB 版本的 1,763
个单位更细；141,456 个源连通片全部进入 Connected-SAH，几何分区清单的 oversize unit 为
0。单位 GLB 文件总计 99,751,512 bytes，均值 31,849 bytes、median 34,184 bytes、p95
63,032 bytes；最大 GLB 文件 68,076 bytes，包含容器开销，因此实际文件字节分布与清单中的
64 KiB 压缩几何目标必须分别报告。

3,132 个单位的 1024 点表全部成功解码，无 fallback 或空几何；固定几何特征为
`[3132, 96] FP16`。16 个 Color-ID 分片全部满足 `formalReady=true`、NVIDIA Vulkan 硬件门和
同窗口 GPU 证据，随后生成 2,712 个 view-cell 的 Pose CSR。16 个 train-only 深度分片覆盖
17,496 个代表子视点，全部返回 `formalReady=true`；K=8 关系表只读取 train split，最终训练
preflight 再次确认 `[3132,96]`、K=8 与 `1944/216/276/276`，且 `testRead=false`。

两步 smoke 在 GPU0 通过并生成安全 checkpoint，stderr 为空。正式 seed20260801 已在
`tmux:pvs_viking64_seed01` 上用 GPU0 启动；seed20260802 等待 Big City seed03 释放 GPU3，
seed20260803 等待 Viking seed01 完成后使用 GPU0，两项均已登记为独立 tmux 队列。正式输出
目录为 `model/out/pvs_v4_viking_connected_sah_64k_full_v1`，日志位于同名 paper-results
目录；三个成员均从头训练，不复用 128 KiB checkpoint。

### 15.1 Sponza 与 Big City 64 KiB 定向微调

日期：2026-09-17。实验名为 `pvs_v4_standard_graphics_64k_targeted_refinement_v1`，入口为
`benchmark/run_standard_graphics_64k_targeted_refinement.py`。Viking 64 KiB 尚未完成，不进入
本轮参数设计，也不因中间结果提前微调。

Sponza 64 KiB 不是相对 128 KiB 的整体退化：128 KiB 三成员的最佳 CNOR 为 `0.571935`，
64 KiB seed02/03 在 epoch36 已分别达到 `0.619564/0.608013`；其中 seed03 的
`WR/LCB=0.992093/0.989818`，主要缺口是少量安全尾部，而不是剔除能力不足。seed01 的冻结
安全工作点 CNOR 只有 `0.403419`，说明不能再次使用强召回或 calibration margin 把预测推向
Keep-All。因此 Sponza pilot 固定从 seed03 完整长训后的 `best_diagnostic.pt` 出发：S0 使用
`lr=1e-6` 的温和安全尾部保护；S1 使用 `lr=2e-6` 和更强困难正负边界分离。两项均不重采样
纯负 pose，因为 Sponza train 中不存在 `candidate>0 && GT=0` 的 view-cell。

Big City 64 KiB 相比 128 KiB 已将 CNOR 从约 `0.21-0.30` 提高到当前约 `0.30-0.47`，但
WR/LCB 与剔除效果仍没有形成合格交集。其 train split 有 1,157 个候选非空纯负 view-cell，
占 train 的 `9.99%`，默认 ambiguity-balanced 采样没有充分利用这类直接剔除监督。Big City
pilot 固定从 seed01 完整长训后的 `best_diagnostic.pt` 出发：B0 每个 16-pose batch 目标含
12.5% 纯负 pose，B1 提高到 25%；两项同时提高高权重正例保护、最差 pose CVaR 和困难负例
边界分离，避免纯负增强以牺牲安全为代价。第一阶段保持 K=8，禁止 runtime-head reset；历史
head reset 已产生 WR=1、Useful Cull=0 的 Keep-All 结果。只有两项 K=8 pilot 都无法改善
安全与 CNOR 折中，才登记 K=16 第二阶段。

| Scene | 配置 | LR | 困难 pose | 纯负 pose | 召回保护 | 目标 WR | CVaR 权重 | 边界分离 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Sponza | S0 温和安全桥接 | `1e-6` | 50% | 0% | 0.40 | 0.995 | 0.35 | 0.40 |
| Sponza | S1 尾部分离桥接 | `2e-6` | 65% | 0% | 0.35 | 0.993 | 0.30 | 0.60 |
| Big City | B0 平衡纯负 12.5% | `2e-6` | 50% | 12.5% | 0.55 | 0.995 | 0.50 | 0.55 |
| Big City | B1 保护纯负 25% | `2e-6` | 50% | 25% | 0.70 | 0.997 | 0.65 | 0.65 |

四项 pilot 必须全部完成 `2 x 450`，不能由中间结果提前取消。两场景原始三种子
`calibration_ready_summary.json` 全部存在前，preflight 必须失败，防止用未冻结 checkpoint
启动微调。资格先分为 `confidence_target_met`、`mean_target_met` 与
`mean_target_not_met`；LCB 是优先置信目标而非硬否决门，CNOR 在同一资格层内用于选优而非
单独硬门。每场景选定配置后，从各自三个原始 checkpoint 执行 `4 x 450` 正式微调；test、
AABB、HZB 和图像评价继续关闭，直到模型与阈值冻结。

扫描已登记到 `tmux:pvs_64k_sponza_bigcity_refinement_queue`：队列每 10 分钟检查上述四个尚未
完成的 Full summary，全部存在后依次执行 preflight、四项 scan 和 validation 汇总；不会自动
启动三种子 final，必须先审阅完整 pilot 结果和选择文件。

选择池必须包含 `control_no_refinement`，即代表种子的原始冻结 checkpoint。2026-09-17，
Sponza 两项 pilot 已完成：S0 的 `WR/LCB/CNOR=0.995359/0.993925/0.393599`，S1 为
`0.995390/0.993859/0.382034`；原始 seed03 最终冻结结果为
`0.992341/0.990114/0.594584`。两项微调虽然增加安全余量，却显著损失剔除能力，因此 Sponza
选择 `control_no_refinement`，不进入三种子 final 微调。该结论也说明不能把更高 LCB 单独
解释为更好的 PVS 工作点。

Big City 第一轮 B0/B1 已完成。原始 seed01 的 `WR/LCB/CNOR` 为
`0.988996/0.987694/0.486438`；B0 为 `0.989218/0.987424/0.434592`，没有改善安全且损失
剔除；B1 为 `0.991188/0.989210/0.386395`，达到平均 WR 保留门但召回保护过强。B1 暂不
扩展到三种子。第二轮增加 B2/B3，两项都从原始 seed01 重新开始，使用 `lr=1e-6`、召回保护
`0.62`、目标 WR `0.996`、CVaR 权重 `0.55`，并把困难边界分离提高到 `0.80`、困难负例比例
提高到 `0.07`；二者只比较 12.5% 与 25% 纯负 pose。目标是在保持平均 WR 大于 `0.99` 时
将 CNOR 恢复到约 `0.45`，仍不改 K=8。

B2/B3 最终 `WR/LCB/CNOR` 为 `0.988133/0.986475/0.454071` 与
`0.989109/0.987311/0.412262`，更强边界分离恢复了部分 CNOR，但两项平均 WR 都未达到
`0.99`。calibration 审计确认 B0-B3 均使用精确 float32 变化点，且 calibration LCB 已压在
`0.990000x` 的最高安全点，不能通过重新选阈值解决。

因此按登记条件进入单因素 K=16 第二阶段。K=16 只重建 train-only 关系表，不改变单位、GT、
split 或模型架构；其截断 cell 从 K=8 的 `131,657/176,628` 降到
`94,546/176,628`，关系保留质量 q01/q05 从 `0.4443/0.5906` 提高到
`0.6429/0.7960`。B4 完全复用 B1 的损失、学习率、困难 pose 和 25% 纯负 pose 配置，只把
关系 K 从 8 改为 16，用于隔离关系容量是否是 Big City 的剩余瓶颈。

B4 得到 `WR/LCB/CNOR=0.991182/0.989200/0.388683`，与 K=8 B1 的
`0.991188/0.989210/0.386395` 基本相同；关系容量不是当前主要瓶颈，K=16 不进入主线。最后
登记一个 B5 插值点：保持 25% 纯负 pose，使用 `lr=1.5e-6`、召回保护 `0.67`、目标 WR
`0.9965`、CVaR 权重 `0.60`、边界分离 `0.72`。B5 用于检验 B1 与 B3 之间是否存在平均
WR 达标且 CNOR 更高的窄区间；若不能优于 B1，则停止继续扫描并保留 B1 的平均 WR 工作点，
不再增加 K 或堆叠损失权重。

B5 最终为 `WR/LCB/CNOR=0.988642/0.986817/0.436463`，没有达到平均 WR。完整扫描至此停止。
B4 相对 B1 的 CNOR/Useful Cull 只提高 `0.002288/0.002486`，属于实际等价；同资格层内若
高 K 对 CNOR 和 Useful Cull 的提升都不足 `0.01`，选择规则优先 K=8。因此 Big City 最终
选择 B1 `b1_guarded_neg25`，并从三个原始 Full checkpoint 执行 `4 x 450` 正式微调；B4
只作为关系容量负结果保留。

Big City B1 三种子正式微调已于 2026-09-17 完成，stderr 均为空，且全过程保持
`testRead=false`：

| Seed | 原始 WR / LCB / CNOR | B1 WR | B1 LCB | B1 CNOR | B1 Useful Cull | B1 Bad Cull | 资格层 |
|---:|---:|---:|---:|---:|---:|---:|---|
| 20260801 | `0.988996 / 0.987694 / 0.486438` | 0.991214 | 0.989288 | 0.387687 | 0.410047 | 0.014443 | mean target met |
| 20260802 | `0.988260 / 0.986372 / 0.394600` | 0.991295 | 0.988604 | 0.346361 | 0.332753 | 0.004330 | mean target met |
| 20260803 | `0.987914 / 0.986410 / 0.302162` | 0.987449 | 0.985961 | 0.261445 | 0.259950 | 0.024451 | mean target not met |

B1 把 seed01/02 提升到平均 WR 大于 `0.99` 的第二资格层，但三个种子都未达到 LCB 目标；
同时 seed01/02 的 CNOR 分别下降 `0.098751/0.048239`，seed03 的安全与剔除均未改善。三种子
B1 平均 `WR/LCB/CNOR/Useful Cull=0.989986/0.987951/0.331831/0.334250`，相对原始 Full
三种子平均 `0.988390/0.986825/0.394400/0.428559`，表现为小幅增加加权安全、明显减少实际
剔除。因而 B1 不能作为统一主线替换三个原始 Full checkpoint；按资格分层记录时可保留
seed01/02 的平均 WR 工作点，seed03 继续保留原始 Full。该结果终止 Big City 64 KiB 的损失
权重扫描：继续增加召回保护预计只会进一步压低 CNOR，而 K=16 已证明关系容量不是瓶颈。
正式 test 仍保持关闭，待最终 checkpoint 选择与论文报告规则统一冻结后再读取一次。

### 15.2 Viking 64 KiB 定向微调

Viking 64 KiB 三种子从头长训已于 2026-09-17 完成，均为严格安全成员，且未读取 test：

| Seed | WR | WR LCB | CNOR | Useful Cull | Bad Cull | Pose Recall | Avg pred |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 20260801 | 0.994775 | 0.991956 | 0.746930 | 0.620034 | 0.084277 | 0.785759 | 286.11 |
| 20260802 | 0.997703 | 0.994952 | 0.546569 | 0.410887 | 0.002228 | 0.961597 | 567.88 |
| 20260803 | 0.999511 | 0.998869 | 0.499540 | 0.361610 | 0.001329 | 0.969112 | 616.43 |
| mean | 0.997330 | 0.995259 | 0.597680 | 0.464177 | 0.029278 | 0.905489 | 490.14 |

seed01 证明当前表示可以达到较高 CNOR，但其普通 pose recall 和 Bad Cull 明显弱于另外两个
种子；seed02/03 的均匀实例覆盖较健康，却保留了过多负例。Viking train split 不存在
`candidate>0 && GT=0` 的 view-cell，因此不使用 Big City 的纯负 pose 重采样。微调仅在
seed03 这个安全余量最大、CNOR 最低的代表成员上先运行两个 `2 x 450` pilot，均从
`best_diagnostic.pt` 初始化：

| 配置 | LR | 困难 pose | 召回保护 | 目标 WR | CVaR 权重 | 边界分离 | 困难负例比例 |
|---|---:|---:|---:|---:|---:|---:|---:|
| V0 平衡分离 | `1e-6` | 65% | 0.25 | 0.992 | 0.25 | 0.60 | 5% |
| V1 强分离 | `1e-6` | 75% | 0.20 | 0.990 | 0.20 | 0.90 | 7% |

两项都不重置运行时头、不改变单位、关系 K、split 或 calibration 规则，也不读取 test。控制组
`control_no_refinement` 必须进入选择池；仍先按 `confidence_target_met / mean_target_met /
mean_target_not_met` 分层，再在同层比较 CNOR、Useful Cull、LCB 与 WR，并同时审阅普通 pose
recall、Bad Cull 和平均保留数。pilot 必须完整结束后才能决定是否把同一配置扩展到三种子；
不允许根据中间 epoch 提前取消，也不以固定 `CNOR=0.80` 单项否决结果。

V0/V1 已在无竞争的 GPU0 上顺序完整结束，stderr 均为空。此前 GPU1/2 的未完成启动因同机
其他用户训练造成严重算力争用而中止；当时只生成 run manifest，没有 checkpoint，正式结果
全部来自随后从源 checkpoint 重新开始的完整运行。

| 配置 | WR | WR LCB | CNOR | Useful Cull | Bad Cull | Pose Recall | Avg pred |
|---|---:|---:|---:|---:|---:|---:|---:|
| control | 0.999511 | 0.998869 | 0.499540 | 0.361610 | 0.001329 | 0.969112 | 616.43 |
| V0 平衡分离 | 0.999937 | 0.999909 | 0.372079 | 0.262970 | 0.000962 | 0.978371 | 712.23 |
| V1 强分离 | 0.999945 | 0.999922 | 0.360749 | 0.254852 | 0.000880 | 0.979412 | 720.16 |

三项都处于严格安全层，但 V0/V1 通过保留更多实例换取额外安全余量，CNOR 分别下降
`0.127461/0.138791`，Useful Cull 也下降 `0.098640/0.106758`。因此选择
`control_no_refinement`，不启动 Viking 三种子正式微调。该结果说明当前瓶颈不能靠短程增加
边界分离或调整召回损失权重解决；继续同类扫描只会增加 validation 调参自由度。Viking 64 KiB
正文模型保持原始 Full 三种子，后续直接进入冻结 test、AABB/HZB 和图像/运行时评价。

### 15.3 三场景视觉效用损失再平衡

日期：2026-09-18。上一轮 Sponza 安全桥接、Big City 纯负增强和 Viking 尾部分离都表明，
单纯提高召回保护或困难边界权重会增加平均保留数并压低 CNOR。当前 pose-balanced BCE 对正例
使用压缩后的视觉权重，但最低权重固定为 `0.25`；因此即使一个可见单元像素贡献很低，它仍
至少获得最高权重正例四分之一的训练权重。这与论文的 weighted recall 安全目标不完全一致，
也是模型在安全阈值下趋向保留大量低效用单元的一个可检验原因。

本轮不改变 Full V4 架构、单位、关系 K、split、候选、GT 或校准规则，只调整现有综合损失：

| 配置 | 正例最低权重 | 权重幂 | 召回保护 | 目标 WR | 边界分离 | Survival / Relation |
|---|---:|---:|---:|---:|---:|---:|
| U0 视觉效用 | 0.05 | 1.0 | 0.25 | 0.992 | 0.30 | 0.10 / 0.05 |
| U1 视觉效用剔除 | 0.01 | 1.0 | 0.20 | 0.990 | 0.45 | 0.05 / 0.02 |

降低正例最低权重只影响逐 pose BCE；正式 weighted recall guard 仍按原始 `visible_weights`
保护高视觉效用正例。降低 continuation 阶段的 survival/relation 辅助权重用于减少已收敛离线
表征漂移，让短程更新集中在可见性排序。Sponza 与 Viking 从各自 Full seed03、Big City
从已达到平均 WR 的 B1 seed01 出发。Big City 的 U0/U1 分别保留
12.5%/25% 合法纯负 pose；另外两个场景没有纯负 pose，不启用该采样。

三个场景各两项 pilot 均固定为 `2 x 450`、`lr=5e-7`，必须完整运行且不读取 test。选择池
继续包含控制组及已有合法 pilot，先按安全资格层分组，再比较 CNOR、Useful Cull、LCB、WR、
Bad Cull、普通 pose recall 与平均保留数。若 U0/U1 都不能在同一资格层提高 CNOR，本轮停止
损失权重扫描，不能继续用更多 validation 配置追逐结果。

## 16. 引用边界

- Wang et al., NeuralPVS: Learned Estimation of Potentially Visible Sets, SIGGRAPH Asia 2025。
- Greene et al., Hierarchical Z-Buffer Visibility。
- NeuralPVS official rendering/training repositories 仅用于场景来源与方法关系说明。

Sponza 版本、输出空间、view-cell、GT 采样和训练目标未与 NeuralPVS 官方协议完全对齐，
因此不能直接摘录其论文数值放入同协议结果表。若以后加入 NeuralPVS 数值行，必须先固定
同场景、同相机、同 GT、froxel 到 renderable unit 的映射和统一指标重算协议。
