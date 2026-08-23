# NeuralStreamWeb3D 当前架构与技术说明

更新时间：2026-08-10

本文描述当前可运行的 NeuralStreamWeb3D 主线。它面向大规模、由大量可下载构件组成的三维场景，把离线几何分析、视点区域可见性学习、浏览器端推理和资源调度组织为一条可复现链路。当前部署仍使用方向遮挡代理模型；论文模型尚未替换前端资产，其架构和 validation 结果见 [`pvs_mainline_2026-08-23.md`](pvs_mainline_2026-08-23.md)。

## 1. 系统目标与边界

系统要同时完成两个粒度不同的决策：

1. **实例级显示决策**：对后退扩大视锥中的每个候选实例输出可见性分数，前端再用真实相机视锥做安全过滤，得到最终显示的实例集合。
2. **GLB 级资源决策**：把实例级结果按所属可下载模型文件聚合，使用模型输出的下载优先级决定哪些 GLB 先请求、先解码和进入缓存。

实例是场景中可以单独显示/隐藏的构件记录，具有唯一实例编号、世界坐标轴对齐包围盒和所属 GLB 编号。GLB 是下载和解码粒度，不是显示粒度：一个 GLB 中的某个实例可见，不能导致同一 GLB 的全部实例都被显示。

当前实现的主要边界如下：

- 训练端可以运行较重的点云编码和方向遮挡证据编码；浏览器端不运行 PointNet++、图传播、dynamic-pool、Triplane 或动态图邻居传播。
- 浏览器端只读取离线导出的固定实例特征、包围盒和轻量推理头，并在 Web Worker 中执行 WebGPU 推理。
- 当前统一协议规定采样相机、模型后退相机和模型推理使用 66° 垂直视场角，真实渲染相机使用 60°；模型输出的保守预取范围最后由真实视锥收紧。唯一配置源是 `neural_instance_culling/config/neuralpvs_viewcell_protocol.json`，前端运行元数据只保留 `frontendRenderFovYDeg=60` 和 `modelInputFovYDeg=66`，不再回读旧的训练/推理别名或历史 Pose CSR 候选 FOV。
- 视锥候选仍由 Worker 中的 Three.js `Frustum.intersectsBox` 完成，当前不是 GPU AABB 剔除。WebGPU 只负责候选实例的神经推理。

### 1.1 正式浏览器采样必须使用硬件 GPU

Three.js Color-ID 采样、实例级图像评价和浏览器三角形 HZB 构建的正式结果，必须来自 Chrome 的 NVIDIA Vulkan/ANGLE 硬件 WebGL 光栅化路径。脚本不能因为页面成功创建 WebGL 上下文、成功输出 JSON/PNG，或 `nvidia-smi` 中出现 Chrome 进程，就把任务标记为硬件 GPU 执行。页面必须回报非空的实际 `gpuBackend`，并满足 `gpuGate.required=true`、`gpuGate.hardware=true`；`SwiftShader`、`llvmpipe`、`softpipe`、`swrast` 或其他软件后端会使正式任务失败并停止汇总。

正式入口必须显式使用 `--require-hardware-gpu`，保留 `--enable-gpu`、`--enable-webgl`、`--use-angle=vulkan` 和 `--disable-software-rasterizer` 等启动约束，同时保存 Chrome 日志、页面后端信息以及同一执行窗口的 `nvidia-smi`/`nvidia-smi pmon` 证据。缺少任一项时，结果只能写作“浏览器渲染完成”，不能写作硬件 GPU 采样、图像性能或移动端性能结果。完整规则和证据文件格式见 [`hardware_gpu_execution_policy.md`](hardware_gpu_execution_policy.md)。

`--allow-software-gpu` 只允许用于独立命名的小规模语义调试目录，不能重建正式数据集、填充 GPU 延迟、覆盖硬件输出，或支撑论文中的性能结论。WebGL 硬件门也不能替代 WebGPU 硬件门；WebGPU 推理和 WGSL 性能必须单独核验 `navigator.gpu.requestAdapter()` 的适配器，当前若回报 SwiftShader，只能记录为 WebGL 硬件通过、WebGPU 硬件性能门失败。

## 2. 端到端数据流

```text
场景源模型 / 实例化 GLB
        │
        ├─ 运行时元数据：实例 AABB、实例到 GLB 映射、场景边界
        ├─ GLB 点云缓存：每个实例的离线几何采样
        └─ view-cell 采样：同方向、同 FOV 的位置扰动子相机
                 │
                 ├─ 可见实例集合
                 ├─ 可见权重与命中次数
                 └─ 后退视锥候选集合
        │
        ├─ 实例点云编码器 → 视角无关几何特征
        └─ 方向遮挡证据编码器 → 上下文特征 + 遮挡代理特征
                 │
                 └─ 可见性模型训练与阈值校准
                            │
                            └─ FP16 前端资产
                                      │
相机状态 → 后退相机 AABB 候选 → Web Worker/WebGPU 推理
                                      │
                    可见性分数 + 视觉效用 + GLB 下载优先级
                                      │
                 真实 60° 视锥实例过滤 → 实例级渲染
                                      │
                     GLB 聚合、预算排序、下载与缓存
```

每个阶段有明确的数据契约，阶段之间不通过隐式的全局状态传递结果。这样可以单独替换采样器、模型或部署方式，而不改变实例编号和 GLB 映射的基本语义。

## 3. 场景与资源组织

当前保留场景为：

| 场景 | 实例数 | GLB 数 | 模型目录 |
|---|---:|---:|---|
| HKUST v3 | 18,831 | 3,273 | `pvs_directional_occlusion_proxy_encoder_rvl_strong_v2_full40_best` |
| IFCBench Fantasy Metropolis 实例化 v2 | 41,298 | 3,669 | `pvs_directional_occlusion_proxy_ifcbench_fantasy_metropolis_instanced_v2_k4_full40_best` |

场景源资产与前端运行资产分离。源场景目录保留构件 GLB、场景树和转换信息；前端部署目录只携带场景元数据、代理几何、神经运行资产和配置，体量较大的 GLB 本体使用独立的场景包或远端地址提供。

运行时元数据至少包含：

- `componentRecords`：实例编号与世界 AABB；
- `globalGlbId` 或等价映射：实例到可下载 GLB 的映射；
- `sceneBounds`：场景中心与尺寸；
- 场景树、GLB 路径和可选的分组浏览信息。

模型资产的实例顺序必须与运行时元数据一致。导出脚本会检查实例数量、特征行数、AABB 数量和映射关系；这些检查失败时不能继续生成前端资产。

## 4. 研究主线的架构取舍

之前的 dynamic-pool 会在运行时为每个候选实例反复查询多个邻居并计算遮挡修正。消融实验表明它对离线指标有帮助，但前端计算量随候选数量和邻居传播次数快速增加，不适合作为当前浏览器主线。当前方案把方向相关的遮挡证据在训练/导出阶段编码为每个实例的固定遮挡代理，运行时只使用相机到实例的视线方向和少量射线空间标量选择代理信息。

这个取舍保留了遮挡建模的论文叙事，同时把前端复杂度从“候选实例之间动态传播”降为“固定表查询 + 小型查询头”：

- 实例几何与上下文不随相机位置重新编码；
- 相机只作为查询条件，不作为场景记忆的唯一索引；
- 遮挡邻居不是固定的最近邻居，而是离线根据方向和深度层组织的证据来源；
- 前端可以继续按照实例显示，GLB 只承担资源聚合。

## 5. 运行时安全约束

模型在各自数据集记录的后退相机候选集上训练和推理，因此模型输出包含预取意图，不能直接当作真实画面的最终显示集合。当前安全链路是：

1. 使用后退相机和实例 AABB 生成候选实例。
2. 只对候选实例执行神经推理。
3. 根据校准阈值保留模型预测实例，同时保留下载优先级和视觉效用。
4. 使用真实 60° 相机再次做实例 AABB 视锥过滤。
5. 把过滤后的实例编号写入实例化网格的可见矩阵/数量，不能只按 GLB 整体开关。

预测触发门控和渲染状态更新是两件事。相机没有跨过预测门控阈值时，不重新发起模型预测；但当前渲染集合仍可以在资源到达、真实视锥变化或实例状态变化时更新。当前版本不会通过较长的历史驻留时间把已经显示的实例永久保留。

## 6. 评价与验收

评价必须区分画面安全、有效剔除、资源节省和运行成本。普通逐实例准确率、precision 和 F1 只能作为诊断指标，不能单独证明 PVS 系统有效。

- 画面安全：pose-level recall、weighted recall、图像漏检率；
- 有效剔除：`useful cull = TN / candidate`；
- 错误剔除：`bad cull = FN / candidate` 以及 `FN / GT`；
- 资源效率：平均预测数、GLB 数量/字节削减、预算内 GLB utility recall；
- 运行成本：Worker 推理延迟、总调度延迟、固定特征大小和实际绘制实例数。

正式阈值和 checkpoint 选择统一为：先要求 `weighted recall > 0.99` 及其单侧 95% 置信下界 `> 0.99`，再在合格工作点中选择最高 `pose precision`。普通 pose recall、F1、accuracy、useful cull、bad cull、GLB 字节削减和图像级指标继续完整报告，但不改变这条主选择规则；本阶段暂不使用 bad-cull 置信区间上界否决路线。

裸的 `1 - average(prediction) / average(candidate)` 不能替代有效剔除，因为它把正确剔除的不可见实例和错误剔除的可见实例混在一起。正式 test split 必须遍历全部唯一测试 view-cell，并在报告中写明候选集合、可见权重语义和阈值工作点。

## 7. 代码边界与复现入口

| 模块 | 主要入口 | 责任 |
|---|---|---|
| 模型结构 | `neural_instance_culling/model/directional_occlusion_proxy_encoder_model.py` | 离线几何/上下文/遮挡代理编码与运行时查询头 |
| 训练 | `neural_instance_culling/model/train_directional_occlusion_proxy_encoder.py` | 资源校验、pose 集合损失、训练、阈值扫描、特征导出 |
| 采样计划 | `neural_instance_culling/sampler/build_neuralpvs_viewcell_pose_plan.mjs` | 生成 view-cell 内同方向 subpose |
| Color-ID 采样 | `neural_instance_culling/sampler/run_scene_viewcell_colorid_sampling.mjs` | 并行调用 Three.js 光栅化采样器 |
| CSR 数据集 | `neural_instance_culling/dataset/build_color_id_pose_csr.py`、`build_rvc_viewcell_pose_csr.py` | 打包 pose、可见集合、候选集合和权重 |
| 前端 Worker | `slm2viewer/src/LightweightPVSWorker.js` | 后退视锥候选、WebGPU 调用、结果回传 |
| 前端调度 | `slm2viewer/src/LightweightPVSDispatcher.js`、`InstancePVS.js` | 预测请求、结果解析、GLB 级下载计划 |
| 实例渲染 | `slm2viewer/src/RenderVisibilitySystem.js`、`slm2viewer/slm2/SLM2Loader.js` | 真实视锥过滤和实例化网格更新 |
| 导出 | `neural_instance_culling/model/export_directional_occlusion_proxy_frontend.py` | 生成 FP16 二进制和运行元数据 |
| 打包 | `slm2viewer/scripts/package_deploy.mjs`、`package_scene_glb.mjs` | 按场景打包前端和独立 GLB 包 |

模块化说明见：模型与训练、数据集协议、前端运行时、导出部署四份专题文档。仓库清理边界见 `docs/current/repository_layout_2026-07-31.md`。

## 8. 当前风险

- 当前前端 AABB 候选过滤仍在 Worker 中使用 Three.js，尚未迁移为 GPU 计算；候选很多时，CPU/Worker 侧筛选仍可能成为瓶颈。
- 正式图像 PER、miss pixel rate 和 wrong-ID pixel rate 需要以同位姿的完整渲染对比补齐，现有模型评测摘要中的 `imageEvaluation` 仍标记为未运行。
- `visible_weights.bin` 的语义依数据源不同：Three.js Color-ID 是屏幕覆盖率 parts-per-million，历史 rvcServer 数据是 `component_weights`，不能统一宣称为真实像素覆盖率。
- 跨场景泛化仍需要合成 view-cell 预训练和留一场景验证，当前两个模型是按场景训练/导出的，不应把场景内 test 结果写成通用泛化结论。
