# NeuralPVS 式 View Cell 数据集采样协议

日期：2026-06-23

> 当前协议：前端真实渲染使用 60°，模型采样、后退候选相机和推理使用 66°（60° × 1.1）。实际运行时不读取旧数据中的候选 FOV 字段。本文早期的“rvcServer 或软件回退”描述不再适用于正式采样；正式浏览器采样必须通过硬件 GPU 门，详见 [`hardware_gpu_execution_policy.md`](hardware_gpu_execution_policy.md)。

## 目的

本文件单独定义当前多场景神经可见性预测数据集的采样语义，避免把“单相机姿态瞬时可见集”和“后退相机 view cell 潜在可见集”混用。

本协议参考 NeuralPVS 的 from-region 潜在可见集思想：预测对象不是单个相机姿态下的瞬时可见集合，而是一个局部视点区域内所有可能可见对象的集合。NeuralPVS 使用 viewcell 表示局部相机活动区域，并用后退相机扩大 frustum 来覆盖 viewcell 内可能出现的可见对象；训练监督来自 viewcell 内多个采样视点可见集合的并集。论文页面见：https://arxiv.org/abs/2509.24677。

## 核心定义

**真实渲染相机**：前端实际用于显示画面的相机。当前约定垂直视场角为 `60°`。

**模型查询相机**：用于模型输入、候选生成和训练采样的相机。它沿真实相机朝向反向后退，并使用更宽的垂直视场角 `66°`。这个相机不是直接显示画面，而是用于给真实相机局部移动留安全余量。

**View cell**：以模型查询相机为代表的局部相机区域。一个 view cell 表示真实相机在短时间内可能落入的一小块空间范围，朝向通常保持与代表相机一致，或只允许很小扰动。

**潜在可见集**：在同一个 view cell 内，对多个扰动相机位置分别渲染 color-id 后得到的可见实例并集。它是训练可见性标签，应覆盖真实相机在该 view cell 内移动时可能看到的实例。

**候选集**：用模型查询相机的 `66°` 扩展视锥对实例 AABB 做粗剔除得到的实例集合，并强制并入潜在可见集。模型只在候选集内预测可见性、可见性分数和下载优先级。

## 当前实现差距

当前 `threejs_colorid_*_pose_csr_fov66_v1` 数据集还不是严格的 view cell 潜在可见集数据集。当前构建脚本 `neural_instance_culling/dataset/build_color_id_pose_csr.py` 的语义是：

- 每一条 color-id JSONL 记录保留为一个训练 pose。
- `visible_ids` 是该单个 pose 的 color-id 可见实例。
- `candidate_ids` 是该同一个 pose 的 `66°` AABB 视锥候选，再并入该 pose 的 `visible_ids`。
- 没有把同一个后退 view cell 内多个相机位置的可见实例取并集。

因此，当前数据集更接近“按 view cell 分布采样的大量单 pose 数据”，而不是 NeuralPVS 式“每个 view cell 一个潜在可见集合”。后续正式多场景训练应重建为 viewcell-level CSR。

## 目标采样流程

1. 在场景内生成可用相机区域。

   采样区域应优先覆盖楼间空隙、道路、广场、近建筑边界、远景视点和天空/俯视视角，同时避免相机落在模型占据体内部。每个场景先根据 AABB 或可行走区域估计主要模型分布，再生成 view cell 中心。

2. 为每个 view cell 生成代表相机。

   代表相机使用模型查询相机定义：从预期真实相机位置沿视线反向后退，垂直 FOV 使用 `66°`。代表相机保存 `camera_world`、`camera_forward`、`tanHalfFovX`、`tanHalfFovY` 和 view cell 尺寸。

3. 在 view cell 范围内随机采样子视点。

   对每个 view cell 生成 `K` 个子视点。位置不是围绕代表相机做临时 jitter，而是在该 view cell 的空间范围内随机采样；首版使用相机局部坐标系对齐的盒状 view cell，即沿相机右方向、前方向和竖直方向分别给出半尺寸。所有子视点朝向默认与代表相机相同。若后续需要模拟手持转头，可加入很小 yaw/pitch 扰动，但必须在 meta 中记录扰动范围，并重新计算后退扩大视锥能否覆盖该范围。

4. 用当前登记的 GPU 光栅化入口渲染每个子视点。

   每个子视点以 `66°` FOV 渲染，输出可见实例 id 和屏幕覆盖权重。当前正式入口是 Three.js Color-ID 浏览器采样器，并且必须确认 Chrome 使用硬件 Vulkan/NVIDIA 后端。历史上尝试过通过 Wine 启动 rvcServer，但这不是当前正式数据集的默认依赖；不能因为 rvcServer 不可用就静默退回软件光栅化。

5. 聚合潜在可见集。

   对同一个 view cell 内 `K` 次渲染的可见实例取并集，写入 `visible_ids`。`visible_weights` 建议记录每个实例在 view cell 内的最大屏幕占比，另可记录出现频率和平均屏幕占比，便于后续 weighted recall、视觉效用和下载优先级监督。

6. 构建候选集。

   使用代表模型查询相机的 `66°` 扩展视锥对所有实例 AABB 做候选筛选。候选集必须再并入潜在可见集，保证 `visible_ids ⊆ candidate_ids`。数据集构建阶段不应限制候选数量；训练和快速验证可以通过参数裁剪候选，但正式评估必须报告裁剪口径。

7. 写入 viewcell-level CSR。

   每个 CSR row 表示一个 view cell，而不是一个单独 pose。建议 schema 使用 `viewcell-csr-color-id-fov66-v1`，meta 中必须写清：

   - `viewcellCount`
   - 每个 view cell 的代表相机语义
   - view cell 位置扰动半径或盒大小
   - 每个 view cell 的内部扰动采样数 `K`
   - 是否有 yaw/pitch 扰动
   - `visible_ids` 是扰动采样并集
   - `visible_weights` 的聚合方式
   - `candidate_ids` 是否为完整候选或训练裁剪候选

## 推荐参数

首版重建建议采用保守参数，优先保证安全召回：

- 真实前端相机 FOV：`60°`
- 模型查询/采样 FOV：`66°`
- 每个 view cell 内扰动位置数：`K = 16` 起步；HKUST 可追加到 `K = 32` 做高风险区域补采。
- view cell 尺寸：按场景尺度设定，不能三个场景共用一个全局半径。当前首版采用相机局部坐标系对齐盒：
  - `hkust-v3`：右/前/上半尺寸约 `4.0 / 4.0 / 1.5`。
  - `ifcbench_fantasy_metropolis_instanced_v2`：右/前/上半尺寸约 `2.5 / 2.5 / 1.0`。
  - 上述值来自三个场景包围盒尺度差异和当前前端触发范围的保守估计，正式报告必须记录实际使用参数；若发现子视点落入模型内部或潜在可见集过宽，应先调整 view cell 尺寸，而不是在训练端截断候选。
- 朝向扰动：首版默认不扰动朝向，只聚合相同朝向下不同位置的潜在可见集。若前端触发门限允许较大 yaw/pitch 变化，再单独扩展。
- 候选集：构建阶段保留完整 `66°` AABB 候选，训练阶段可使用 hard-negative 裁剪，例如 `4096`，但正式报告必须同时给出完整候选评估或与前端一致的候选上限评估。

## 与前端运行逻辑的对应关系

前端运行时应与该数据语义保持一致：

1. 根据当前真实相机生成后退模型查询相机。
2. 使用模型查询相机做扩展视锥候选。
3. 模型预测该 view cell 的潜在可见实例、可见性分数和下载优先级。
4. 下载调度使用模型输出的优先级。
5. 最终显示仍使用真实相机视锥做实例级过滤，避免把后退 view cell 的预取实例全部渲染出来。

这样训练标签、模型查询和前端显示三者的语义是一致的：模型负责预测局部 view cell 的潜在可见性，前端负责把潜在可见集合进一步筛到当前真实画面。

## 后续实施任务

- 新增 viewcell-level pose plan 生成脚本，输出代表相机和 `K` 个同朝向子视点。
- 正式使用 Three.js Color-ID 硬件 GPU 采样器；每个分片保存 `gpuBackend`、`gpuGate` 和 Chrome 日志，并在检测到软件后端时失败。
- rvcServer/Wine 仅作为另行登记的对照实现，不能替代当前正式采样协议。
- 新增 `build_rvc_viewcell_pose_csr.py`，不要复用当前单 pose CSR 名称。
- 对三个场景重建 `viewcell-csr-rvc-fov66-v1` 数据集。
- 重建遮挡证据表，因为 `visible_ids` 语义从单 pose 可见集变为 view cell 潜在可见集。
- 重新训练并报告完整候选与训练裁剪候选两套指标。
