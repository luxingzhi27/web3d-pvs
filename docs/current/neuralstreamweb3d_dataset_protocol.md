# NeuralStreamWeb3D 数据集与采样协议

更新时间：2026-09-09

本文定义当前训练数据如何从场景资产生成，以及每个二进制文件的语义。核心目标是让采样语义和 NeuralPVS 的 view-cell 思路一致：一个 view-cell 固定相机朝向和视场，在局部空间盒内随机生成多个位置不同但方向相同的子相机，最终可见集合取这些子相机结果的并集。参考论文：[NeuralPVS](https://arxiv.org/abs/2509.24677)。

## 1. 相机口径

新的统一 Color-ID 主线采用三种明确口径：

- **采样相机**：用于实际光栅化采样，垂直视场角 66°；
- **模型后退相机**：前端预测所使用的后退相机，垂直视场角 66°，与采样口径一致；
- **真实渲染相机**：浏览器实际显示画面，垂直视场角 60°。

后退相机可以沿当前视线反向移动指定距离，并使用 66° 模型视场角覆盖位置扰动带来的潜在可见实例。模型学习的是后退相机候选上的保守可见性；真实 60° 视锥负责最终实例级安全过滤。候选相机的视场角由当前协议统一给出。

采样宽高默认 512×288，宽高比会写入每条 pose。代码不应只根据垂直视场角推导横向视场角而忽略 aspect；数据构建同时保存 `tan_x` 和 `tan_y`，候选 AABB 计算也使用这两个量。

## 2. View-cell 与 subpose

一个代表性相机行描述一个 view-cell 中心、前向方向、FOV、aspect、类别和数据集划分。`build_neuralpvs_viewcell_pose_plan.mjs` 为每个 view-cell 生成 K 个 subpose：

1. 保持中心相机的前向方向、yaw、pitch、FOV 和 aspect 不变；
2. 根据前向、右向和上向构成相机局部基；
3. 在相机对齐盒中独立随机采样右向、前向和上向偏移；
4. 第一个 subpose 保留 view-cell 中心，便于保留代表点；
5. 每个 subpose 写出独立世界坐标，但共用 `viewcell_id` 和方向信息。

不同场景可以使用不同 view-cell 尺寸，但必须以实际 pose plan 和数据集 meta 为准。当前 HKUST view-cell 采样计划 `hkust_v3_viewcell_fov66` 使用水平圆盘，半径为 `2 m`、垂直扰动为 `0 m`，每个 cell 平均约 `34.88` 个成功 subpose。前端 `CameraPredictionGate` 已按同一契约判断：使用世界 XZ 平面位移、拒绝 Y 方向扰动、要求四元数朝向固定；越过边界时不受最小间隔抑制。显示相机保持 `60°`，Worker 的模型查询相机保持 `66°`，浏览器不展开 subpose。其他场景的尺寸必须单独登记，不能把一个场景的 cell 尺寸直接套到另一个场景。

采样计划中的类别用于保证空间分布覆盖，包括街道缝隙、建筑近旁、广场、外围、天空俯视和远景等。采样点需要在场景空隙或可行走区域，避免大面积落在实体模型内部；如果需要建筑内部采样，必须在实验说明中单独声明。

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
5. 以成功 subpose 位置集合生成候选 AABB 并集；
6. 正式模式不补入可见正样本；如果可见并集不属于候选并集，直接使数据构建失败，并保存漏正样本诊断。只有显式的探索性开关才允许补入；
7. 保存 view-cell 中心作为模型查询相机。

构建器要求每条正式 raw row 显式携带 `train/validation/calibration/test/guard` 之一；不再随机补 split，也不接受旧 `val` 名称。它同时写出 `query_center_world` 和 `candidate_camera_world = query_center_world - forward * pvs_back_offset`，其中前者进入区域查询，后者与 `poses.camera_world`、MVP 和候选相机语义一致。

因此，候选集合不是“中心相机一次视锥的候选”，而是所有位置扰动 subpose 的候选并集；可见集合也不是某个 subpose 的可见集合，而是整个 view-cell 内潜在可见集合的并集。正式数据必须直接证明 `visible_ids ⊆ candidate_ids`，不能用标签补入制造这个关系；这样训练标签与 NeuralPVS 的 from-region PVS 语义一致，也能把候选生成错误暴露出来。

这些 subpose 不进入浏览器运行时。前端以当前相机建立一个 view-cell 预测锚点，通过一次后退扩展候选和一次模型批查询输出整个区域的保守潜在可见集；相机仍在该 cell 的空间与方向门限内时复用结果，越界后才建立新锚点并重新查询。真实 `60` 度视锥随后只对保守集合做当前帧实例级过滤。因而，只在某个边缘 subpose 可见的实例是合法正例，不是中心点的误报。

脚本支持将一个源 view-cell 划分为 4 个空间子组。划分时按 subpose 相对中心在右向和前向的正负侧分配象限，从而减少一个 cell 过大造成的过度并集；只有在数据量和候选规模需要时才启用，不能把它解释为新的相机扰动语义。

## 5. 候选集合的计算

对每个 subpose，候选计算使用实例 AABB 与相机视锥的保守相交测试。对 AABB 中心和半尺寸投影到相机前向、右向、上向后，检查：

- 前向深度加包围半径是否超过 near；
- 水平中心距离减水平半径是否落在水平视锥范围内；
- 垂直中心距离减垂直半径是否落在垂直视锥范围内。

多个 subpose 的候选 ID 合并去重。正式候选文件由 AABB 算法独立产生，数据契约要求：

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
| `candidate_ids.bin` | uint32 | 后退/子 pose AABB 候选实例编号 |
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
| `pose_csr_sponza_standard_graphics_128k_fov66_v1` | Sponza 128 KiB renderable units + 显式空间 split | 66° Y | 60° Y | 屏幕覆盖率 parts-per-million |

正式训练数据使用上表中带显式 split 的目录。`build_color_id_pose_csr.py` 只把每条 JSONL 记录作为一个 pose 打包，不会聚合 subpose；需要 NeuralPVS view-cell 并集时使用 `build_rvc_viewcell_pose_csr.py`，实际数据来源以 `sourceSampler` 和 `dataset_meta.json` 为准。

## 9. 推荐复现顺序

以下是当前 Linux/conda 口径的最小流程，具体场景路径按 `docs/current/current_instance_pvs_versions.md` 替换：

```bash
conda run -n slm_pvs node neural_instance_culling/sampler/build_neuralpvs_viewcell_pose_plan.mjs \
  --input neural_instance_culling/sampler/out/<scene>/representative_pose_plan.jsonl \
  --output neural_instance_culling/sampler/out/<scene>/viewcell_pose_plan.jsonl \
  --scene <scene> --subposes-per-viewcell 16 --fov-y 66

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
