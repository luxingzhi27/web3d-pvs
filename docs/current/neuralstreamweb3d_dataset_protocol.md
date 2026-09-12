# NeuralStreamWeb3D 数据集与采样协议

更新时间：2026-09-13

本文定义当前训练数据如何从场景资产生成，以及每个二进制文件的语义。正式 V4 主线的 view-cell 统一为固定相机朝向和高度的世界 XZ 水平圆盘，在圆盘内采样多个位置不同、方向相同的子相机，最终可见集合取这些子相机结果的并集。参考论文：[NeuralPVS](https://arxiv.org/abs/2509.24677)。

## 1. 相机口径

新的统一 Color-ID 主线采用三种明确口径：

- **采样相机**：用于实际光栅化采样，垂直视场角 66°；
- **模型后退相机**：前端预测所使用的后退相机，垂直视场角 66°，与采样口径一致；
- **真实渲染相机**：浏览器实际显示画面，垂直视场角 60°。

后退相机可以沿当前视线反向移动指定距离，并使用 66° 模型视场角覆盖位置扰动带来的潜在可见实例。模型学习的是后退相机候选上的保守可见性；真实 60° 视锥负责最终实例级安全过滤。候选相机的视场角由当前协议统一给出。

后退距离采用 NeuralPVS 第 3.2 节的构造。设真实显示垂直 FOV 为 $\theta=60^\circ$，水平圆盘半径为 $r$，则

$$
d_{back}=\frac{r}{\tan(\theta/2)}.
$$

候选相机为 $c'=c-d_{back}f$，其中 $c$ 是圆盘中心，$f$ 是单位前向。候选和模型 FOV 固定为 `66°`，比显示 FOV 每侧多 `3°`。HKUST 的 `r=2 m` 对应 `d=3.464102 m`；Sponza、Big City 和 Viking Village 的 `r=0.75 m` 对应 `d=1.299038 m`。半径可以按场景尺度登记，但公式、FOV 和候选语义不得按结果调整。

采样宽高默认 512×288，宽高比会写入每条 pose。代码不应只根据垂直视场角推导横向视场角而忽略 aspect；数据构建同时保存 `tan_x` 和 `tan_y`，候选 AABB 计算也使用这两个量。

## 2. View-cell 与 subpose

一个代表性相机行描述一个 view-cell 中心、前向方向、FOV、aspect、类别和数据集划分。`build_neuralpvs_viewcell_pose_plan.mjs` 为每个 view-cell 生成 K 个 subpose：

1. 保持中心相机的前向方向、yaw、pitch、FOV 和 aspect 不变；
2. 在世界 XZ 平面内对面积均匀采样水平圆盘，Y 保持中心高度；
3. 高空、楼层和远景通过不同高度的 view-cell 中心表达，不在单个 cell 内增加垂直扰动；
4. 第一个 subpose 保留 view-cell 中心，便于保留代表点；
5. 每个 subpose 写出独立世界坐标，但共用 `viewcell_id` 和方向信息。

不同场景可以使用不同半径，但 shape 固定为 `horizontal_disk`，并必须以实际 pose plan 和数据集 meta 为准。HKUST 使用 `r=2 m`，普通 cell 使用 `32` 个 subpose，sky/far 使用 `48` 个；其中 sky `800` 个、far `640` 个中心说明 HKUST 包含高空/远景采样，只是每个 cell 内没有垂直扰动。Sponza、Big City 和 Viking Village 使用 `r=0.75 m`、每 cell `32` 个 subpose。前端 `CameraPredictionGate` 使用同一世界 XZ 位移契约，浏览器不展开 subpose。

Viking Village 的场景完整包围盒被远山扩大到约 `1.1 km`，因此中心放置使用固定的
`ground_surface_grid`，而不是全包围盒多层 Y 网格：相机域排除 `terrain_far`，XZ 网格
向 `terrain_near` 三角形求交，相机高度为交点上方 `1.7 m`，并继续执行 `0.8 m` 表面
clearance。冻结的 V1 数据使用 `24×24` 网格；与旧中心不重合的 sampling V2 使用
`32×32` 网格。该规则只决定圆盘中心放在哪里；圆盘半径、subpose、FOV、后退候选和
split 规则与其余标准场景一致。

相机中心到几何表面的安全距离只用于保证整个圆盘不穿过几何，定义为 `radius + 0.05 m`。因此标准场景为 `0.80 m`。该 clearance 不是后退距离；后退距离始终由上式独立计算。

采样计划中的类别用于保证空间分布覆盖，包括街道缝隙、建筑近旁、广场、外围、天空俯视和远景等。采样点需要在场景空隙或可行走区域，避免大面积落在实体模型内部；如果需要建筑内部采样，必须在实验说明中单独声明。

### 2.1 统一 split 约定

主实验按物理相机中心分组划分，同一中心的全部 yaw、pitch 和全部 subpose 必须进入同一 split。冻结 V1 使用 seed `20260911`，sampling V2 使用 seed `20260913`；二者都做确定性随机交错分配：先按中心组形成约 `80/10/10` 的 train/validation/test，再从初始 train 中取约 `10%` 为 calibration，最终约为 `72/8/10/10`。这与 HKUST 的有效语义一致：HKUST 的 `737` 个重复中心中没有任何中心跨 split；其历史 validation/test 保持冻结，calibration 从历史 train 抽取，因此实际计数仍为 `5926/659/730/684`。

标准场景不得再使用连续 Morton 空间块作为主 split。空间块 holdout 回答的是未见区域外推问题，会显著改变正样本比例；它与本文的场景专属可见性压缩主问题不同。所有方法在一个场景内共享完全相同的中心组 split；test 只在模型、阈值和方法选择冻结后读取一次。

## 3. Color-ID 光栅化

正式数据集默认使用 Three.js Color-ID 采样器在浏览器硬件 GPU 上进行离屏光栅化：每个实例分配一个可解码颜色，GPU 将颜色写入离屏缓冲，CPU 只负责资源加载、相机组织和颜色统计。采样器默认启用硬件 GPU 门，必须回报非软件 WebGL 后端；检测到 SwiftShader、llvmpipe、softpipe、swrast 或无法确认后端时直接失败。rvcServer 的 `component_weights` 只作为特定历史数据来源，权重语义按第 6 节解释。

每条原始采样记录通常包含：

- `pose_index`、`viewcell_id`、`subpose_id`；
- `camera_pos`、`camera_forward`、`fov_y`、`aspect`；
- `visible_component_ids`；
- `component_weights`，即屏幕覆盖率的 parts-per-million；
- 可选的错误、渲染耗时和类别信息。

Color-ID 的权重可以用于视觉重要性监督，但它不是深度缓冲，也不能表达实例之间的遮挡深度。当前方向遮挡证据由后续离线几何投影步骤构建，而不是从 Color-ID 颜色计数直接推断。

正式采样的 Chrome 启动参数、WebGL/WebGPU 后端核验、GPU evidence 文件和软件路径边界见[硬件 GPU 执行政策](hardware_gpu_execution_policy.md)。本协议只定义数据语义；正式 view-cell 采样还必须确认每个分片的行数、`pose_index` 覆盖和 `gpu_execution_summary.json` 的 `formalReady=true`。

## 4. 从 subpose 聚合到 view-cell

`build_rvc_viewcell_pose_csr.py` 负责把同一 view-cell 的多个 subpose 聚合成一个训练 pose。它的行为是：

1. 丢弃标记为采样失败的 subpose；
2. 要求成功 subpose 数达到 `min_success_subposes`；
3. 对所有成功 subpose 的可见实例编号取并集；
4. 对同一实例的权重取最大值，命中次数单独保存；
5. 使用单个后退 `66°` 相机生成 AABB 候选，与浏览器运行时一致；
6. 正式模式不补入可见正样本；如果可见并集不属于候选并集，直接使数据构建失败，并保存漏正样本诊断。只有显式的探索性开关才允许补入；
7. 保存 view-cell 中心作为模型查询相机。

构建器要求每条正式 raw row 显式携带 `train/validation/calibration/test/guard` 之一；不再随机补 split，也不接受旧 `val` 名称。它同时写出 `query_center_world` 和 `candidate_camera_world = query_center_world - forward * pvs_back_offset`，其中前者进入区域查询，后者与 `poses.camera_world`、MVP 和候选相机语义一致。

候选集合来自单个后退 `66°` 相机；可见集合来自整个圆盘的 subpose 可见并集。正式数据必须直接证明 `visible_ids ⊆ candidate_ids`，不能用标签补入制造这个关系。该检查同时验证后退距离、FOV、AABB 和实例编号是否足以支持浏览器的一次查询。

这些 subpose 不进入浏览器运行时。前端以当前相机建立一个 view-cell 预测锚点，通过一次后退扩展候选和一次模型批查询输出整个区域的保守潜在可见集；相机仍在该 cell 的空间与方向门限内时复用结果，越界后才建立新锚点并重新查询。真实 `60` 度视锥随后只对保守集合做当前帧实例级过滤。因而，只在某个边缘 subpose 可见的实例是合法正例，不是中心点的误报。

脚本支持将一个源 view-cell 划分为 4 个空间子组。划分时按 subpose 相对中心在右向和前向的正负侧分配象限，从而减少一个 cell 过大造成的过度并集；只有在数据量和候选规模需要时才启用，不能把它解释为新的相机扰动语义。

## 5. 候选集合的计算

候选计算使用实例 AABB 与后退 `66°` 相机视锥的保守相交测试。对 AABB 中心和半尺寸投影到相机前向、右向、上向后，检查：

- 前向深度加包围半径是否超过 near；
- 水平中心距离减水平半径是否落在水平视锥范围内；
- 垂直中心距离减垂直半径是否落在垂直视锥范围内。

正式候选文件由该单相机 AABB 算法独立产生，数据契约要求：

```text
visible_ids ⊆ candidate_ids
```

如果不满足，正式构建直接失败并记录 `candidateMissVisible`；训练/评测批构造器也会拒绝继续。历史数据可以通过 `--allow-candidate-visible-union` 进行探索性复现，但必须把它标为非正式结果。这样 AABB、相机口径、实例编号映射或颜色采样坐标的错误不会被候选补丁隐藏。

## 6. Pose CSR 二进制格式

CSR（压缩稀疏行）用一个 offsets 数组描述每个 pose 的连续 ID 区间。当前数据目录包括：

| 文件 | 类型 | 语义 |
|---|---|---|
| `poses.bin` | 固定 64 字节结构 | 归一化相机位置、世界相机位置、前向、`tan_x/tan_y`、split、类别 |
| `mvp.bin` | float32[pose,16] | 与后退候选相机一致的 66 度保守投影矩阵 |
| `query_center_world.bin` | float32[pose,3] | view-cell 的规范查询中心 |
| `candidate_camera_world.bin` | float32[pose,3] | 单次模型查询使用的后退 66 度候选相机 |
| `viewcell_radius_m.bin` | float32[pose] | 区域空间半径 |
| `visible_offsets.bin` | uint64 | 每个 pose 的可见 ID 起止位置 |
| `visible_ids.bin` | uint32 | pose 级 GT 可见实例编号 |
| `visible_weights.bin` | float32 | 与 `visible_ids` 对齐的权重 |
| `visible_hit_counts.bin` | uint16 | 一个实例在多少个成功 subpose 中命中 |
| `candidate_offsets.bin` | uint64 | 每个 pose 的候选 ID 起止位置 |
| `candidate_ids.bin` | uint32 | 单个后退 66° 相机的 AABB 候选实例编号 |
| `dataset_meta.json` | JSON | schema、相机口径、统计、原始候选语义、文件语义和 split |

`visible_weights` 必须在报告中说明来源：Color-ID 数据是屏幕覆盖率 parts-per-million；历史 rvcServer 数据是 `component_weights`，只能按可见重要性权重解释，不能宣称为严格像素覆盖率。

## 7. 训练数据与运行时资源的一致性

训练前由 `benchmark/run_pvs.py preflight` 和 `train_pvs.py` 检查：

- 数据集元数据、pose、visible 和 candidate 文件存在；
- visible/candidate ID 不越过运行时实例数量；
- 如果 candidate offsets 存在，则逐 pose 检查 visible 是否为 candidate 子集；
- 点云缓存的实例行数覆盖运行时实例数；
- 运行时元数据的 AABB 和实例到 GLB 映射可读；
- 遮挡证据的方向单元、深度层和来源数量有效。

改变 view-cell 半尺寸、subpose 数量、相机 FOV、aspect 或后退距离后必须重建数据集。只修改前端 FOV 或训练参数而继续复用旧候选集合，会使候选安全边界和模型输入语义不一致。

## 8. 当前数据集的状态边界

仓库中的 Pose CSR 数据统一按当前相机协议解释：

| 数据集 | 采样来源 | 模型/采样 FOV | 前端真实 FOV | 权重语义 |
|---|---|---|---|---|
| `pose_csr_hkust_v3_main_stratified_calibration_fov66_v1` | HKUST Color-ID view-cell + 显式 split | 66° Y | 60° Y | 屏幕覆盖率 parts-per-million |
| `pose_csr_ifcbench_fantasy_metropolis_main_stratified_calibration_fov66_v1` | IFCBench Metropolis Color-ID view-cell + 显式 split | 66° Y | 60° Y | 屏幕覆盖率 parts-per-million |
| `pose_csr_sponza_standard_graphics_128k_fov66_v1` | Sponza 128 KiB renderable units + 中心组随机 split | 66° Y | 60° Y | 屏幕覆盖率 parts-per-million |
| `pose_csr_bigcity_standard_graphics_128k_fov66_v1` | Big City 128 KiB renderable units + 中心组随机 split | 66° Y | 60° Y | 屏幕覆盖率 parts-per-million |
| `pose_csr_viking_village_standard_graphics_128k_fov66_v1` | Viking Village 128 KiB renderable units + 地表中心组随机 split | 66° Y | 60° Y | 屏幕覆盖率 parts-per-million |

正式训练数据使用上表中带显式 split 的目录。`build_color_id_pose_csr.py` 只把每条 JSONL 记录作为一个 pose 打包，不会聚合 subpose；需要 NeuralPVS view-cell 并集时使用 `build_rvc_viewcell_pose_csr.py`，实际数据来源以 `sourceSampler` 和 `dataset_meta.json` 为准。

## 9. 推荐复现顺序

以下是当前 Linux/conda 口径的最小流程，具体场景路径按 `docs/current/current_instance_pvs_versions.md` 替换：

```bash
conda run -n slm_pvs node neural_instance_culling/sampler/build_neuralpvs_viewcell_pose_plan.mjs \
  --input neural_instance_culling/sampler/out/<scene>/representative_pose_plan.jsonl \
  --output neural_instance_culling/sampler/out/<scene>/viewcell_pose_plan.jsonl \
  --scene <scene> --viewcell-shape horizontal_disk --radius <scene-radius> \
  --subposes-per-viewcell 32 --fov-y 66

conda run -n slm_pvs node neural_instance_culling/sampler/run_scene_viewcell_colorid_sampling.mjs \
  --scene <scene> --assets-dir <scene>/assets \
  --pose-plan neural_instance_culling/sampler/out/<scene>/viewcell_pose_plan.jsonl \
  --output-dir neural_instance_culling/sampler/out/<scene>/color_id \
  --parallel 4 --shards 16 --fov-y 66

conda run -n slm_pvs python neural_instance_culling/dataset/build_rvc_viewcell_pose_csr.py \
  --raw-dir neural_instance_culling/sampler/out/<scene>/color_id \
  --output-dir neural_instance_culling/dataset/out/<scene>_viewcell_colorid \
  --runtime-meta <scene>/assets/runtimeVisibilityMeta.json \
  --source-sampler three_color_id \
  --default-fov-y 66
```

构建后确认 `rawRows`、`viewcellCount`、`successSubposeCount`、`avgCandidate`、`avgVisible` 和 `candidateMissVisible`。正式训练和 benchmark 只能引用明确命名的输出目录，不能直接读取 sampler 的临时 JSONL。

重建任何场景时，采样和模型输入都必须使用 66°，真实前端保持 60°；不能从不同 FOV 的 Pose CSR 行中恢复当前数据集。
