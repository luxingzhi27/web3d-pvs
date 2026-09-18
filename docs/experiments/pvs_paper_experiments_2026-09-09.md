# PVS 论文核心实验与一周执行计划

日期：2026-09-09；2026-09-10、2026-09-11、2026-09-13、2026-09-15 更新正式执行状态

状态：执行中。截至 2026-09-15，HKUST/IFCBench 的冻结 V4 checkpoint 保持不变，只用
统一精确 calibration 阈值重放现有评价。三个标准图形学场景使用 Connected-SAH 128 KiB
单位，先完成并冻结 Full，再执行同单位 AABB、HZB、图像和 runtime。
本文是论文实验、图表、结果目录和执行顺序的唯一计划。

## 论文需要证明的内容

论文围绕五个问题组织：实例 PVS 是否安全且有效；分层关系、生存场和区域频谱查询为何有效；神经启动资产相比几何外壳是否更小；模型能否在桌面和手机实时运行；连续可见性分数能否让正确 GLB 更早到达。

论文重点突出三个系统优势：昂贵遮挡关系在离线完成，前端只查询固定实例表；一次查询覆盖整个 view-cell；相对 geometry-shell HZB，启动传输和运行内存更小。所有优势必须分别由 test 指标、真实设备时间和冷缓存下载实验支撑。

## 当前证据与场景

| 项目 | HKUST | IFCBench/Metropolis |
|---|---:|---:|
| 实例 / GLB | 18,831 / 3,273 | 41,298 / 3,669 |
| Train / calibration / validation / test | 5926 / 659 / 730 / 684 | 19647 / 2183 / 2712 / 2710 |
| 完整 V4 | 三种子 `40 x 900` 已完成 | 三种子 `40 x 900` 已完成 |
| 核心消融 | 六项三种子已完成 | 不重训；只重做精确阈值复评 |
| 生存场 rank | 2/4/8/12 已完成 | rank 4 固定 |
| Validation 图像 | seed 2 已完成 | seed 2 已完成 |

论文正式评价使用 HKUST、IFCBench/Metropolis、Sponza、Big City 和 Viking Village 五个场景。三个标准图形学场景采用一 renderable unit 对应一 resource 的协议；一个单位可以包含同一源 node/primitive/material 内经 SAH 紧凑打包的多个小连通分量，但不跨 primitive 或材质。它们正式报告可见性、图像、HZB、资产和运行成本。Streaming 资源复用与下载排序实验集中在 HKUST/IFCBench。完整转换、训练和表图要求见[标准图形学场景方案](pvs_standard_graphics_scene_generality_2026-09-10.md)。

## 统一评价协议

- 实例任务使用 `66` 度区域候选和区域可见并集；图像与即时显示使用真实 `60` 度相机及每个 pose 的实际 aspect。
- 每个 checkpoint 只在 calibration 冻结阈值。优先选择 aggregate weighted recall 与单侧 95% bootstrap LCB 均大于 `0.99` 的最高阈值；若 LCB 目标不可达，则选择平均 WR 仍大于 `0.99` 的最高阈值并保留结果。LCB 单独报告，不作为硬否决门。
- Validation 选择配置；模型、阈值和评价结构冻结后，每个 test 样本只读取一次。零 GT pose 保留并单独统计。
- 同时报告 pose-macro 和 aggregate precision、recall、F1、Jaccard、accuracy、balanced accuracy、specificity、PR-AUC、正样本比例、AP lift，以及候选归一化遮挡召回率 CNOR、useful cull、bad cull、预测数量和 GLB 字节。论文主表的 PR-AUC 固定为 pose-macro AP，并紧邻 pose 正样本比例与 AP lift；aggregate AP 只作为完整汇总的辅助口径。CNOR 是跨 pose 的主要遮挡效率指标，但不替代 weighted recall、LCB 或 Useful Cull。
- AP 使用合并同分值的 Average Precision。Pose-macro AP 只对有 GT 的 pose 计算并报告有效 pose 数；零 GT pose 的 FP 仍进入集合和资源指标。
- 三种子报告 mean 和 sample standard deviation。核心差值使用按 seed 聚类、seed 内配对 pose 重采样的 10,000 次 bootstrap。
- 图像统计以 view-cell 为重采样单位，报告 PER、miss 和 wrong-ID 的 aggregate、mean、median、p95；extra 使用全图像素分母。

## 正文方法与基线

| 方法 | 输入和用途 |
|---|---|
| Keep-All | 保留相同候选集合，作为绝对安全和零剔除参照 |
| AABB + Ray MLP | 18 维 AABB、投影和视角特征，小型 MLP，不使用固定几何表或遮挡关系 |
| Geometry-only | HKUST 已有无生存场消融，保留现有查询和损失，用于隔离结构化遮挡表示 |
| Geometry-shell HZB | 预下载不透明纯几何外壳，GPU depth/HZB/AABB 测试 |
| Full V4 | 分层关系校准生存场、区域矩包络频谱查询和综合损失 |

AABB MLP 正式版使用与 Full 相同的逐 pose 平衡、weighted-recall 保护和困难边界损失。先扫描学习率 `2e-4/1e-3`，每项 `6 x 300` updates，再以选定配置完成三种子 `40 x 900`。所有方法独立校准阈值；标准场景 AABB 扫描和长训必须等三个场景 Full 全部冻结后开始。

现有核心消融继续按逐项移除解释，不能改写成未经训练的累加模型。Generic-28 是与结构化生存场同容量的可训练实例记忆对照。

## Geometry-shell HZB

正式主基线 `geometry_shell_hzb_lossless` 保留 LOD0 中确定不透明的 POSITION、INDEX 和实例变换，删除法线、UV、颜色、纹理和无关材质，采用 meshoptimizer 传输压缩。关闭背面剔除；透明、玻璃、alpha-cutout 和材质不确定表面不写遮挡深度。

附加 `geometry_shell_hzb_equal_asset`：资产字节限制为同场景冻结神经资产大小，从与 Full 相同的输出 renderable units 中按固定 128 个 train 中心视点的投影面积累计值/压缩字节选择完整 occluder 单位，不移动顶点。该项用于资产敏感性，不宣称保守或最优简化，也不能从旧源 primitive 外壳复用结果。

当前唯一 HZB 算法为 `opaque depth -> max pyramid -> conservative projected AABB test`：只将确定不透明外壳写入深度，构建 max pyramid，再对同一 candidate CSR 中的每个候选执行保守 projected AABB test。四套外壳的 `shell_meta.json` 使用修正后的 `geometry-shell-hzb-v2` schema。投影矩形向外取整并检查覆盖 mip 的全部 texel；仅当 `HZBMax + bias < candidateNear` 时剔除。近裁剪面、非有限投影、相机位于 AABB 内和其他不确定情况全部保留。HZB 论文口径只保留这条算法路径。

HKUST/IFCBench 已导出外壳的交接资产统计如下；这些是启动资产和解码内存数字，不是 HZB 可见性或性能结果。标准场景外壳须在其 Full 资产冻结后按 Connected-SAH 单位重新导出：

| 场景 | 变体 | 传输 B | 展开几何 MiB | 展开运行时 MiB | prototype | prototype 三角形 | 展开三角形 | occluder 实例 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| HKUST | lossless | 399,312,852 | 1,697.30 | 1,697.80 | 3,042 | 54,771,324 | 55,537,431 | 18,566 |
| HKUST | equal-asset | 5,147,576 | 17.78 | 18.29 | 81 | 682,664 | 780,713 | 1,257 |
| IFCBench | lossless | 59,924,084 | 153.30 | 154.40 | 3,669 | 8,648,833 | 25,556,160 | 41,298 |
| IFCBench | equal-asset | 11,819,713 | 25.43 | 26.54 | 1,405 | 1,342,146 | 6,290,848 | 31,909 |

正式 HZB 编排固定为 `36 tasks`：`2 preflight + 16 calibration + 2 select + 12 frozen test + 4 timing`。其中 calibration 为 `512x288 / 1024x576` × depth bias `0.01/1/10/100 m`，每场景 8 个 lossless 配置、每配置一次 invocation；按安全条件下 useful cull 最高选取，平局选择耗时更低配置。投影矩阵逐 pose 使用真实 aspect，HZB 纹理分辨率作为基线自身预算固定，不冒充前端图像分辨率。

完整 calibration 和 frozen test 每个 pose 只执行一轮可见性查询，避免把精度矩阵重复五遍；独立的 120-pose 性能任务才对每个 pose 执行五轮并统计 p50/p95。两类任务都必须通过同一硬件 GPU 和无并发计算证据门。

八个 lossless calibration task 由 `select_geometry_shell_hzb.py` 冻结配置：安全池要求 weighted recall 及其单侧 95% 下界均严格大于 `0.99`，安全池内按 useful cull、balanced accuracy、specificity、precision 和延迟排序；没有安全成员时仍输出诊断配置，但不得标成安全 HZB。

公平比较分为：

- Point60：canonical center 的真实相机和同点 GT。
- Region66：相同区域候选，对多个真实 subpose 做 HZB 后取并集。HKUST 比较 `1/5/9/all`；IFCBench 比较 `1/all`。子集由中心最近点开始做最远点空间覆盖，不读取可见标签。
- 全 subpose HZB 是区域质量参照，多次查询的总耗时全部计入；有限点结果不宣称连续区域保证。

主要相关工作包括 Greene 等人的 Hierarchical Z-Buffer Visibility、HROC 的 GPU 批量组织方式，以及 Occluder Simplification using Planar Sections。论文只数值实现一个现代 GPU HZB 管线，不冒称复现这些完整系统。

## 论文实验与图表

| 实验 | 动作 | 论文产物 |
|---|---|---|
| 场景统计 | 统计范围、原型/展开三角形、GLB 字节分布、split、候选和 GT 分布；标准场景另报划分 schema、源连通分量、每单位分量数、超大分量拆分数和编码字节分布 | Table 1 |
| Test 可见性 | Keep-All、AABB MLP、HZB、Full V4 全量 test | Table 2 |
| Test 图像 | Full、AABB MLP、选定 HZB 的真实60度 Color-ID | 图像表、Fig. 7 |
| 核心消融 | 复用六项三种子和 Generic-28，统一重新汇总 | Table 3 |
| 容量与资产 | 复用 rank 2/4/8/12，补同候选延迟 | Pareto 图、Table 4 |
| 端侧性能 | 两场景 WebGPU/WASM，A6000、M2、vivo | Table 4、Fig. 5 |
| HZB 系统成本 | 两种外壳、Point/Region、冷启动和分阶段 GPU 时间 | Table 4 |
| Progressive streaming | 全 test 冷缓存离线模拟和固定子集真实调度 | Table 5、Fig. 6 |
| 安全效率曲线 | Validation 的 WR-useful cull 和 WR-GLB bytes | Fig. 4 |
| GT 收敛 | 每场景100个分层 validation cell，嵌套采样到128点 | 附录图 |
| 离线成本 | 采样、关系、编码、训练、校准、导出时间与峰值资源 | 附录表 |
| 标准图形学场景主实验 | Sponza、Big City 与 Viking Village 均使用目标压缩大小 64 KiB 的 renderable units，一单位一资源；同单位比较 Keep-All、AABB MLP、Hi-Z 与 Full | 正式场景结果表、资产-剔除 Pareto、定性图 |

正文保留五张表：场景统计、test 可见性、消融、资产与运行时间、streaming。核心图为系统总览、模型架构、生存场可视化、安全效率曲线、候选数量延迟曲线、下载覆盖曲线和定性对比。研究图输出 PDF、SVG、PNG 和源 CSV；架构图保留可编辑源文件。

定性图每场景选择 PER 中位数附近、p95 附近、最大错误和一个预登记细构件/遮挡边界视角，报告 pose ID。不能只挑低误差画面。

## 端侧性能

扩展现有 HTTPS 测试站，使场景配置提供模型、实例数、pose 数和候选文件，并允许 WebGPU/WASM 选择。

- 每个场景和后端使用 50 次 warmup、5 个 session，纯模型前向遍历该场景完整 test 候选负载。
- WebGPU 报告 kernel 和 submit-completion；WASM 使用一次批量模型调用。二者排除资产下载、初始化和候选构建。
- 候选分桶为 `0-1k/1-2k/2-5k/5-10k/10-15k/15k+`，必要时为 IFCBench增加高候选桶，报告 count、mean、p50、p95和区间。
- 拟合 `T(N)=a+bN` 并报告拟合误差；不强行宣称完全线性。
- HZB 和完整 PVS 管线在每场景120个按候选规模分层的固定 test pose 上做5次重复，记录压缩和回读总成本。
- 冷启动另外计入资产 fetch、解码、GPU upload/init、首次查询和首个结果。

## Progressive streaming

每个 test pose 都从空缓存开始。结果明确分成三类且不混合：`threshold filtering` 使用冻结阈值形成预测资源集合；`threshold-free ranking` 在完整 candidate GLB 集合上排序，不读取阈值；`scheduler replay` 单独重放 urgent/warm/speculative 调度状态机并计入资源处理过程。

过滤实验使用 calibration 冻结的阈值，报告候选到预测 GLB 的数量/字节、最终 visible-weight coverage 上限以及漏掉的必需资源。

排序实验对所有方法使用相同完整 candidate GLB set，不应用阈值。神经 GLB 分数先按 `p_g=max_i(p_i)` 聚合；成本感知分数为 `p_g/Bytes_g^alpha`，`alpha` 只能在 validation 从 `{0,0.5,1}` 选择并写入冻结 manifest，test 只读取该 manifest。比较 Full、AABB MLP、原始顺序、距离、投影面积、投影面积/字节、20 个固定随机种子、HZB visible-first 和 GT-informed utility/byte oracle。Distance 与 projected AABB area 必须先逐候选实例计算，再按 GLB 聚合为 `distance=min`、`area=max`；不使用合并 GLB AABB 的投影结果。

每个 GLB 完整到达后才累计 coverage。主要指标为 Bytes@95/99/99.9/100、99%前无用字节、必需资源平均排名；时间按 `10/25/50/100 Mbps` 换算，正文展示25和50 Mbps。Oracle 按 GT utility/byte 贪心排序，并另给分数背包字节下界，不能把贪心值称为所有目标的全局最优。

全 test 的主覆盖字段明确为 visible-weight coverage：使用数据集提供的 `visible_weights`，在实例映射到 GLB 后累计可见权重；Color-ID 屏幕覆盖权重按 view-cell subpose 最大值解释，不解释为隐藏表面总覆盖。若补充 reference-frontmost screen utility，则单独用于四个下载阶段的真实重渲染核验，核验前景移除后露出的 wrong-ID，不替代 visible-weight coverage。

真实调度在每场景12个固定 test pose、25/50 Mbps、每项3次，调用现有 urgent/warm/speculative 状态机，取消与固定首页位置相关的 startup-100 前缀。计入方法资产、初始化、GLB 下载、解析、挂载和首个正确画面。无法达到目标时报告未达到和覆盖上限。

## HKUST/IFCBench 精确重校准

HKUST 与 IFCBench 不重新训练。对每个现有 checkpoint，在 calibration 保存 float32 分数、
标签、可见权重和 pose offsets；阈值候选使用实际分数变化点，预测规则固定为
`score >= threshold`。复用固定的 10,000 组 pose bootstrap 索引，找到最高安全阈值，
再只在 validation 比较成员。精确重校准不会改变 AP，收益必须单独记录。模型和阈值冻结后
才允许重放一次 test、图像、streaming 和前端资产。

## 必要接口

- 评价入口增加冻结 test 模式，并将 test 结果明确标为 `testRead=true`。训练与 calibration 继续拒绝 test。
- 大型分数使用按pose offsets对齐的二进制sidecar；JSON只保存小型指标和路径。图像manifest每个view-cell只保存一次预测集合。
- 测试站移除HKUST固定684/18831限制，改为场景清单驱动，并增加WASM测量。

## 并行执行和结果目录

Full 冻结后的独立工作可从同一提交创建不同 worktree：

| 执行者 | 责任 |
|---|---|
| AABB 工作树 | 三场景 AABB 扫描、三种子训练与 validation 冻结 |
| HZB 工作树 | 同单位外壳导出、WebGPU calibration/test/timing |
| 评价工作树 | 冻结 test、图像评价、端侧 runtime 与统一指标表 |
| Streaming 工作树 | HKUST/IFCBench 全 test 模拟、真实 scheduler replay 和论文图 |
| 主工作树 | 公共接口、阶段门、结果集成与最终审阅 |

GPU 1-3用于当前训练和评分；GPU 0 的外部任务不干预。正式浏览器采样和计时另等独占窗口，
所有长任务写 stdout/stderr 日志。

统一结果目录为 `neural_instance_culling/benchmark/out/paper_results/`，包含 `scene_statistics.csv`，以及 `test_metrics/`、`image_metrics/`、`ablation/`、`rank_sweep/`、`mobile_runtime/`、`hzb/`、`streaming/`、`threshold_curves/`、`gt_convergence/`、`preprocessing/` 和 `figures/`。每组结果记录实际样本数、冻结配置、来源和复现命令。

执行顺序不按日历压缩：先完成三个标准场景 Full 三种子并冻结成员和阈值；随后并行执行
AABB 与 HZB；最后统一打开 frozen test，完成图像、端侧 runtime、HKUST/IFCBench
streaming、GT 收敛和论文数据包。任何中间指标都不能提前绕过阶段门。

验收覆盖同分AP、零GT统计、阈值边界、checkpoint和固定表一致性、HZB深度/近裁剪面、区域并集、GLB原子到达、覆盖不可达、像素直方图与重渲染一致性，以及硬件计时边界。禁止新增兼容层、前端历史模型路径或手工哈希步骤。

## CNOR 评价口径

新增候选归一化遮挡召回率：

\[
\mathrm{CNOR}=\frac{\sum_p TN_p/C_p}{\sum_p(TN_p+FP_p)/C_p}.
\]

该指标从正式 `perPose` 混淆计数或 pose-aligned score sidecar 直接重算，不由 aggregate
specificity 近似。Full 与 AABB 的三种子结果报告 mean +/- sample std；只有一份逐 pose
资产时明确标成单成员结果。

HKUST 核心消融已从 730 个 validation pose 的逐 pose 混淆计数重新汇总：Full 的三种子
CNOR 为 `0.903764`；去除分层关系、去除生存场、通用 28 维、去除矩包络、去除召回保护、
去除困难边界的 CNOR 依次为 `0.875568/0.838316/0.894385/0.895451/0.882723/0.897642`。
生存场 rank `2/4/8/12` 的 CNOR 分别为
`0.893121/0.906264/0.890525/0.905261`。这些结果仍是 validation 消融，不与 frozen test
主表混用。

核心消融和容量汇总命令为：

```bash
conda run -n slm_pvs python neural_instance_culling/benchmark/summarize_core_ablation.py \
  --root neural_instance_culling/benchmark/out/pvs_v4_integrated_visibility_mainline_v1 \
  --output neural_instance_culling/benchmark/out/pvs_v4_integrated_visibility_mainline_v1/paper_core_ablation_summary.json \
  --bootstrap-replicates 10000 --seed 20260823

conda run -n slm_pvs python \
  neural_instance_culling/benchmark/run_survival_rank_capacity.py summarize
```

## 执行状态（2026-09-15）

- 不修改当前 V4 的 96D 几何表、结构化生存场和运行时查询头；HKUST/IFCBench 不重训。
- 标准场景统一使用 `connected-sah-pack-v2`：先求共享顶点连通分量；小分量只在同一源
  node/primitive/material 内用确定性 16-bin SAH 打包；超大分量按三角形 SAH 递归拆分；
  meshopt 实际编码字节执行 `128 KiB` 最终上限。
- Full 固定使用 `lr=5e-5`、`poses_per_batch=8`、`hard_pose_fraction=0.35`，尾部分离权重
  升至 `0.30`，正/负尾部比例为 `0.01/0.02`，正样本重要性下限为 `0.25`；采用先分类与
  生存场、再尾部分离、最后召回保护的课程。
- 三场景 GLB、1024 点/96D 几何表、硬件 Color-ID、Pose CSR 和 train-only K=8 关系均已
  完成。Sponza/Viking/Big City 单位数为 `129/1763/2861`，split 分别为
  `4800/528/672/672`、`1944/216/276/276`、`11580/1284/1608/1608`。
- seed20260801、seed20260802 和 seed20260803 已分别按完整三场景 round 在 GPU 1/2/3
  启动。每张训练卡并发三个种子，后续只监控总吞吐、数值稳定性和首次 validation。
- calibration 阈值统一改为实际 float32 分数变化点；HKUST/IFCBench 对现有 checkpoint
  重做 calibration/validation 阈值冻结和派生结果，不用 test 选阈值、不重新训练。
- 精确重校准已完成。HKUST 三种子均达到 validation LCB 目标，选择 seed02；其
  WR/LCB/Useful Cull/CNOR 为 `0.996847/0.994558/0.904606/0.915014`。IFCBench seed03
  达到 validation LCB 目标，对应 `0.991350/0.990509/0.559296/0.648690`；seed01/02
  的 validation LCB 为 `0.989853/0.989395`，但平均 WR 为 `0.991020/0.990323`，按当前规则
  同样保留并标注为平均 WR 达标。当前尚未用新阈值
  重放 test、图像、streaming 或前端资产。
- 标准场景执行顺序固定为 `Full -> AABB MLP / HZB -> frozen test/image/runtime`，后两类
  基线不得提前占用 Full 的采样、训练和浏览器 GPU 资源。
- Sponza 与 Viking 使用 `pvs_v4_standard_graphics_safety_refinement_v1` 做快速损失调参：
  固定数据和 K，从现有最佳 checkpoint 比较降低尾部分离、增强召回/CVaR，以及降低生存场
  与关系辅助权重三种配置；pilot 为 `4 x 450`，validation 选定后对三个原始种子执行
  `4 x 900` 正式 refinement。六个 pilot 均未过严格安全门；诊断池分别选择 Sponza R2
  和 Viking R3。正式 runner 可按场景分阶段启动，Sponza 先利用空闲 GPU，Viking 必须等
  原始三种子完成后再读取冻结来源 checkpoint。Sponza R2 正式三种子已经完成，但 LCB
  只有 `0.984861/0.982582/0.981572`，均未过安全门。Viking R3 正式三种子 LCB 为
  `0.987469/0.984181/0.988815`，同样没有改善原始结果；Viking 最终保留原始 seed01/02
  两个安全 checkpoint，seed03 保留为诊断成员。两场景 test 均保持关闭。
- 第二轮安全补救使用 `pvs_v4_standard_graphics_checkpoint_rescue_v1`，不再从 Sponza 的
  早期低融合 checkpoint 继续。Sponza 三个种子分别从 epoch24/12/16 的完整实例校准点
  初始化，Viking 只补救未达标的 seed03（epoch32）；比较 `2e-6` 保守保护与 `5e-6`
  强正尾保护两个 `2 x 450` 配置，再用场景级统一选择完成 `4 x 450` 正式续训。该实验只用
  train/calibration/validation，test 继续关闭。8 个扫描成员已完成：Sponza 两配置都未
  过门，按最差/平均 LCB 选择 `2e-6` 保守配置；Viking seed03 选择 `5e-6` 强正尾配置，
  扫描得到 `WR/LCB/CNOR=0.992693/0.987683/0.812336`。正式续训已经启动。
- Sponza checkpoint rescue 正式三种子仍未过门，LCB 为
  `0.985783/0.978539/0.984479`。后续 `pvs_v4_sponza_recall_first_refinement_v1` 从原始
  最佳诊断 checkpoint 继续，显式启用此前早期 checkpoint 中为零的实例校准融合，并比较
  `1e-6/2e-6` 的召回优先配置；不改变数据、split 或 test 状态。六个扫描成员已完成，
  `2e-6` 三种子 LCB 为 `0.989683/0.986649/0.985206`，按最差 LCB 选择后已启动正式续训。
  Viking seed03 checkpoint rescue 正式结果为 `WR/LCB/CNOR=0.992659/0.987740/0.806078`，
  仍未过门。后续专项从该 checkpoint 扫描 `0.995/0.997` calibration margin，同时保留较强
  分离损失，正式资格固定为 validation `LCB > 0.99` 且 `CNOR >= 0.80`。
- Sponza recall-first 正式 seed01 已以 `WR/LCB/CNOR=0.992162/0.990110/0.422597`
  过门，seed02/03 LCB 为 `0.986583/0.984718`。后续 calibration-margin 实验只在
  calibration 比较 `0.995/0.997` 安全余量，最终 validation 门仍为 `0.99`，不使用
  validation 反向选择具体阈值。
- Sponza calibration margin `0.995/0.997` 扫描仍未过门；seed02 LCB 为
  `0.988343/0.988829`，seed03 为 `0.986687/0.989019`。独立扩展实验继续比较
  `0.999/0.9995`，不改写已完成实验的名称和含义。
- Sponza margin 扩展最终选择 `0.999`：正式 seed01/02/03 的 validation
  `WR/LCB/CNOR` 为 `0.999158/0.998847/0.176069`、
  `0.995422/0.991499/0.195047`、`0.997860/0.996592/0.418873`，三种子全部严格安全。
  该结果仅作为安全诊断，未冻结为论文最终模型。Sponza 将按 `64 KiB + 每单位最多 64 连通片`
  重建并重训，正式目标为 validation `LCB > 0.99` 且 `CNOR >= 0.80`；test 继续关闭。
- Sponza 64 KiB 已完成 303 个单位的转换、1024 点/96D 几何特征、16 个硬件 Color-ID
  分片、Pose CSR、32 个 train-only 硬件深度分片和 K=8 关系表。preflight 与 smoke 均通过，
  split 为 `4800/528/672/672`，正式三种子实验名为
  `pvs_v4_sponza_connected_sah_64k_full_v1`；seed01 已启动，test 保持关闭。
- Viking seed03 的 calibration margin `0.995/0.997` 虽分别得到 validation
  `LCB=0.999517/0.999981`，CNOR 仅为 `0.596124/0.421267`，不进入 final。对原始高 CNOR
  checkpoint 的精确 calibration-only 审计显示，floor `0.9905/0.9910/0.9915/0.9920`
  对应 validation `LCB=0.988065/0.988542/0.989015/0.989207`，CNOR 则从
  `0.795412` 下降到 `0.746278`，不存在靠阈值同时通过双门槛的工作点。后续
  `pvs_v4_viking_seed03_tail_separation_refinement_v1` 固定 calibration floor `0.99`，
  降低全局召回保护并增强困难正负尾部分离，直接改善排序间隔。
- Viking 尾部分离首轮 `2e-7/5e-7` 得到 validation
  `LCB/CNOR=0.987468/0.819580` 与 `0.987316/0.826272`，剔除合格但安全仍不足。登记扫描
  继续比较中等召回保护 `0.50/0.80` 与较弱尾部分离 `0.30/0.25`，寻找双门槛交集；所有
  结果仍为 validation 配置选择，test 未读取。
- Viking 中等召回保护 `0.50/0.80` 仍只有 `LCB=0.987379/0.987437`，逐 pose 回放定位到
  同一中心三个 pose 漏掉 unit `1639`。该现象保留为高视觉权重尾部误差分析，不改变预测集合；
  Viking 正式方法只使用 Full V4 模型分数和 calibration 冻结阈值。sampling-v2 validation
  中，seed01/02 的 `WR/LCB/CNOR` 为 `0.992577/0.991257/0.867249` 和
  `0.992474/0.990662/0.846276`，均通过安全门；seed03 checkpoint rescue 为
  `0.992660/0.987841/0.806092`，CNOR 合格但 LCB 未过门。
- 2026-09-17 起，LCB `0.99` 改为优先置信度目标而非硬否决门：validation 平均 WR
  `>0.99` 的成员均保留，LCB 是否达标单独标注。Viking sampling-v2 已按该规则完成三种子
  276-pose test，平均 `WR/LCB/CNOR/Useful Cull/Pose PR-AUC` 为
  `0.993511/0.992081/0.812737/0.639713/0.865692`。该结果替代此前不同 split 的 Viking
  test 数字，正式产物位于 `pvs_v4_recall_target_policy_v1/viking_village_128k`。
- Big City 原 `128 KiB / K=8` 三种子在 epoch 20-22 停止并降级为诊断结果：validation
  CNOR 仅为 `0.2101/0.2232/0.2958`，且都未满足 `weighted recall LCB > 0.99`。专项实验
  `pvs_v4_bigcity_connected_sah_occlusion_opportunity_v1` 固定完成六项 train-only 单因素
  pilot：K8 旧采样控制、只增 K、只加 25% 纯负例、K16+纯负例、低学习率和 K24+纯负例。
  pilot 只运行 3,600 步，但所有课程按完整 36,000 步计算，且尾部分离在 10% 后才启动，
  防止短训压缩或提前损失课程。纯负例增强只重采样现有合法 train view-cell，不修改 GT。若相对最优配置
  仍未通过安全门或安全 CNOR 低于 `0.50`，则重建 `64 KiB + 每单位最多 128 连通片` 数据。
  六项 pilot 已完成；P5 虽以 `LCB=0.991721` 过安全门，但 `CNOR=0.053696`、Useful Cull
  `0.009193`，已触发 `partitionRebuildRequired=true`。正式 `bigcity_64k` 重建复用冻结 pose
  plan/split，但重建全部单位相关资产。该专项仍遵守
  `pilot 全部完成 -> validation 判定 -> 64 KiB 重建与三种子从头长训 -> 基线/test` 顺序。
- Big City 64 KiB 的几何、Color-ID、Pose CSR、64 个硬件深度分片和 K=8 关系资产均已完成；
  `pvs_v4_bigcity_connected_sah_64k_full_v1` 将在 margin 扩展释放 GPU 后并行启动三种子
  `40 x 900` 从头训练。test、AABB MLP 与 HZB 继续等待 Full 冻结。
- Viking Village 新增 `viking_village_64k`，复用 sampling-v2 的固定 pose plan 与
  `1944/216/276/276` split，只把 Connected-SAH 的压缩后目标单位从 128 KiB 改为 64 KiB。
  单位相关的 GLB、1024 点/96D 几何、Color-ID、Pose CSR、train-only 深度与 K=8 关系必须
  全部重建。正式三种子入口为 `run_viking_64k_full.py`，仍执行 `40 x 900`；最终标准场景主表
  使用 Sponza、Big City、Viking 三个 64 KiB 版本，Viking 128 KiB 只保留为粒度对照。
  该重建已经完成：3,132 个单位、16 个硬件 Color-ID 分片、2,712 个 view-cell、16 个硬件
  train-only 深度分片和 K=8 关系表均通过 preflight；smoke 通过，seed01 已在 GPU0 启动，
  seed02/03 已按现有训练释放顺序排队。test 保持关闭。
- Sponza 与 Big City 64 KiB 的后续微调登记为
  `pvs_v4_standard_graphics_64k_targeted_refinement_v1`。Sponza 比较温和安全尾部保护与更强
  正负边界分离，不做纯负 pose 采样；Big City 比较 12.5%/25% 合法纯负 pose 重采样，并
  同时增强加权正例保护和困难边界。四项 pilot 均为 `2 x 450`，必须等待两场景原始三种子
  全部结束后运行；先保持 K=8，不做已知会退化成 Keep-All 的 runtime-head reset。选择后再
  对三种子执行 `4 x 450`，LCB/CNOR 均按当前资格分层规则报告，test 保持关闭。
  Sponza S0/S1 已得到 `LCB/CNOR=0.993925/0.393599` 与 `0.993859/0.382034`，均明显差于
  原始 seed03 的 `0.990114/0.594584` 剔除折中；选择池加入 `control_no_refinement`，Sponza
  保留原始 Full，不启动正式微调。Big City 仍等待两项 pilot 完成后按同一规则选择。
  Big City B0/B1 最终 `WR/LCB/CNOR` 为 `0.989218/0.987424/0.434592` 与
  `0.991188/0.989210/0.386395`；B1 只达到平均 WR，CNOR 退化明显，暂不扩展。第二轮 B2/B3
  使用中间召回保护 `0.62`、更强边界分离 `0.80` 和 `lr=1e-6`，继续隔离 12.5%/25% 纯负
  pose 配额；两项仍须完整运行后再决定是否进入 K=16。
  B2/B3 最终为 `WR/LCB/CNOR=0.988133/0.986475/0.454071` 与
  `0.989109/0.987311/0.412262`，均未达到平均 WR。Big City 因此进入 B4 单因素 K=16：保持
  B1 的损失和采样不变，只提高 train-only 关系容量；关系质量 q01/q05 已从
  `0.4443/0.5906` 提高到 `0.6429/0.7960`。
  B4 最终 `WR/LCB/CNOR=0.991182/0.989200/0.388683`，与 K=8 B1 基本一致，K=16 不进入
  主线。最后运行 B5 插值点（25% 纯负、召回保护 `0.67`、`lr=1.5e-6`、边界分离 `0.72`）；
  若无法在平均 WR 达标时提高 CNOR，则终止 Big City 参数扫描并保留 B1。
  B5 得到 `0.988642/0.986817/0.436463`，未达到平均 WR。扫描停止并选择 K=8 B1；K=16
  对 CNOR/Useful Cull 的提升都不足 `0.003`，按实际等价时优先简单配置的规则不进入主线。
  B1 三种子正式 `4 x 450` 随后完整结束。seed01/02 的 `WR/LCB/CNOR` 为
  `0.991214/0.989288/0.387687` 与 `0.991295/0.988604/0.346361`，只进入平均 WR 达标层；
  seed03 为 `0.987449/0.985961/0.261445`，没有改善。B1 三种子平均 CNOR 从原始 Full 的
  `0.394400` 降至 `0.331831`，因此不统一替换原始 Full；保留 seed01/02 的平均 WR 工作点
  作为安全—剔除折中结果，seed03 保留原始 checkpoint，不再追加 Big City 损失权重扫描。
  三项正式微调均未读取 test。
- Viking 64 KiB 三种子 Full 已全部完成，validation 均值
  `WR/LCB/CNOR/Useful Cull=0.997330/0.995259/0.597680/0.464177`。三个成员均通过严格安全门，
  但 CNOR 从 seed01 的 `0.746930` 降到 seed02/03 的 `0.546569/0.499540`；seed01 的普通
  pose recall 仅 `0.785759`，说明当前主要问题是安全权重、均匀实例覆盖与剔除之间的跨种子
  不稳定。已在统一 64 KiB runner 中登记 V0/V1 两项 `2 x 450` pilot：不使用不存在的纯负
  pose，只降低召回保护并把困难边界分离提高到 `0.60/0.90`。先在 seed03 完整运行，控制组
  同池选择，未审阅完整 validation 前不扩展三种子，test 保持关闭。
  V0/V1 最终 `WR/LCB/CNOR` 为 `0.999937/0.999909/0.372079` 与
  `0.999945/0.999922/0.360749`，均比控制组 `0.999511/0.998869/0.499540` 更接近
  Keep-All；平均保留数也从 `616.43` 增至 `712.23/720.16`。因此选择
  `control_no_refinement`，不执行 Viking 三种子正式微调，也不继续相同类型的损失权重扫描。
  Viking 64 KiB 后续使用原始 Full 三种子进入冻结 test 和基线评价。
- 2026-09-18 应用户要求登记最后一轮三场景统一损失再平衡。核心改动不是继续增加召回保护，
  而是把 pose-balanced BCE 的低视觉效用正例最低权重从 `0.25` 降为 `0.05/0.01`，让 weighted
  recall guard 负责保护真正重要的正例；同时把 continuation 的 survival/relation 辅助权重
  降为 `0.10/0.05` 或 `0.05/0.02`。Sponza、Big City、Viking 各运行 U0/U1 两项
  `2 x 450` pilot，控制组同池选择，完整矩阵结束前不扩展三种子且不读取 test。若同一安全
  资格层内 CNOR 没有提高，则停止继续按 validation 扫描损失权重。
- 完整单位契约和重建步骤以
  [标准图形学场景方案](pvs_standard_graphics_scene_generality_2026-09-10.md)第 3-11 节为准。
