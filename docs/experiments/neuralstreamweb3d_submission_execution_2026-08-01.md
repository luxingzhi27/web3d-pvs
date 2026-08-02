# NeuralStreamWeb3D 投稿计划执行记录

日期：2026-08-01
依据：[neuralstreamweb3d_submission_plan_2026-09.md](neuralstreamweb3d_submission_plan_2026-09.md)
状态：执行中，尚未形成可投稿结论

## 记录规则

本文件只记录已经由当前工作区、日志、模型产物或可复核脚本证明的状态。计划中的目标不作为实验结果；探索性结果、失败结果和正式结果分别标记。任何阈值、checkpoint、候选集合、图像评价和设备结论都必须能追溯到具体文件、命令和数据哈希。

正式 test 遵循固定 validation、独立 calibration、冻结参数、test one-shot 协议。test 之后不回调阈值；若安全门失败，重新训练或如实保留失败结果，不覆盖原产物。

## 初始审计

### 环境与工作区

- 当前分支：`main`，基线提交：`8aba381`。
- 远端：`origin` 指向 `git@github.com:luxingzhi27/web3d-pvs.git`。
- 工作区包含上一阶段尚未提交的 M10/M12 前端、benchmark、数据集脚本和文档修改；本轮不覆盖或删除这些修改。
- 当前使用 conda 环境 `slm_pvs`、Linux CUDA 训练口径；长时间任务写入独立 stdout/stderr 日志。
- 桌面浏览器 M12 使用 SwiftShader，不能作为硬件 WebGPU 或 Android 性能证据。

### 阶段门控基线

| 阶段 | 当前判断 | 证据与缺口 |
|---|---|---|
| M0 | 子门通过，总门需复核 | 固定 split、独立 calibration、frozen manifest 和 HKUST one-shot 产物已存在；Metropolis 需要复核 test 是否真正完成及摘要是否完整。 |
| M1 | 通过 | HKUST/Metropolis 正式资源审计存在；旧候选补正样本失败记录不得进入主表。 |
| M2 | 通过但有风险 | 空间 split manifest 存在；Metropolis 稀疏天空/类别分布需要在泛化报告中单独说明。 |
| M3 | 正式产物存在，贡献门未通过 | 两场景 intervention 文件存在；方向代理的独立 useful-cull 增益仍需结合 M4 三种子结果判断。 |
| M4 | 运行中 | 三种子、输入/上下文/代理/RVL 消融矩阵正在 GPU 0-2 上运行，不能提前读取为正式结论。 |
| M5 | 明确 No-Go | HKUST validation 图像管线已运行，但 miss-pixel 长尾超过计划 p95 门槛；需要改进模型/采样/安全策略后重新冻结，而不是 test 调阈值。 |
| M6 | 子门通过，总门未通过 | AABB、bitset、学习型 AABB+ray 和三角形 HZB 产物存在；完整 NeuralPVS 公平适配和同信息主比较仍缺失。 |
| M7 | 子门通过，总门未通过 | model-free 与独立 RankNet 结果存在；冷/温真实解码、统一轨迹和联合级联的 paired 比较仍缺失。 |
| M8 | 未通过 | 尚无完整冷/热缓存与受控网络导航效用曲线。 |
| M9 | 子门进行中 | 前端静态审计、空间页和浏览器 smoke 存在；运行时分项计时、真实视锥更新和 Cold-0 请求证据仍需统一。 |
| M10 | No-Go/设备证据缺失 | 已固化 Android 测试方案；没有实体 Android、ADB 和硬件 WebGPU，不生成移动端性能数字。 |
| M11 | 运行中/未通过 | HKUST 方向留出训练已在 GPU 3 启动；Metropolis 和零/少样本跨场景实验尚未完成。 |
| M12 | 数值一致性子门通过，硬件性能门未通过 | 参考已按前端后退相机口径修复；16 个 case 的 FP16/WGSL 阈值翻转率为 0，SwiftShader 不能作为硬件性能结论。 |
| M13 | 未开始 | 三种子正式重训、冻结阈值、one-shot test、图像、调度和运行主表均未冻结。 |

## 当前后台任务

- M4：`formal_m4_matrix` 与 `formal_m4_eval`，使用固定 HKUST 空间数据和预注册三种子矩阵。
- M11：`m11_metropolis_fewshot_retry3`，实验名为 `pvs_m11_fewshot_1pct_metropolis_yaw20_rvl_strong_v2_full40_seed20260801_retry3`，GPU 3，Metropolis 1% 少样本适配。

后台任务完成前不重用其输出目录、不修改其 checkpoint、不把中间 epoch 写入正式主表。

## 执行顺序

1. 复核 M0 frozen test、M1/M2 资源和 split manifest，补齐不可变 artifact manifest。
2. 等待并审计 M4 三种子结果，决定方向代理路线 A 或路线 B；若未达到预注册效应量，删除代理有效性主张。
3. 修复 M5 图像安全问题，优先分析大型构件漏像素的几何/候选/阈值来源，重新训练或改进安全约束并在 validation/calibration 重冻结。
4. 完成同信息 M6/M7/M8 基线与端到端流送成本，严格扣除特征页、权重和元数据启动字节。
5. 完成 M9 空间分页、GPU compact、分项计时和 Cold-0 请求审计；M12 只有在真实 WebGPU 或明确标记 exploratory 的情况下使用。
6. 按设备可得性执行 M10；设备缺失时保留测试协议和“证据缺失”，不得放宽为桌面 SwiftShader 结论。
7. 完成 M11 空间、方向和跨场景泛化，随后进行 M13 三种子 one-shot test。
8. 最终更新指标报告、失败实验记录、artifact manifest、README 和投稿表格，执行逐项质量审计。

## 质量门调整规则

本轮不预先放宽任何门槛。若某阶段失败，最多先做十次有明确假设、独立命名和可复核指标的改进尝试；每次必须记录原因、代码、数据、命令和结果。十次仍失败时，只有在论文主张同步收缩、替代指标能反映真实画面/传输体验且统计证据充分的情况下，才允许进入下一阶段，并在本文件记录放宽理由。测试污染、实例级图像安全失败或代理无因果贡献不能通过改写指标名称消除。

## 后续记录模板

每次实验追加以下信息：

- 日期、实验名称、所属阶段和假设；
- 修改文件、依赖数据和 SHA-256；
- 完整运行命令、conda 环境、GPU、seed、stdout/stderr；
- validation/calibration/test split 与阈值来源；
- 画面安全指标、有效剔除指标、资源/调度指标和运行时指标；
- 质量门判断、失败原因、是否保留、下一步；
- 若放宽门槛：尝试次数、论文主张收缩范围和统计依据。

## 2026-08-02 M0 归档与 M5 根因分析

### M0 frozen artifact 归档

HKUST 和 Metropolis 的 frozen test 均已核对为完整唯一 test pose、单一 calibration 冻结阈值、`testEvaluationCount=1`。关键 checkpoint、固定特征、校准摘要、数据集元数据、运行时元数据和 GLB 索引的字节数与 SHA-256 已写入：

`docs/evaluation/m0_frozen_artifact_manifest_2026-08-01.json`

归档结果：HKUST 722 个 test pose，阈值 `0.02`，weighted recall `0.997047`；Metropolis 2,340 个 test pose，阈值 `0.32`，weighted recall `0.992120`。这些仍是集合级 frozen test 证据，不等价于图像近似无损。

### M5 失败原因

对完整 HKUST validation 的图像结果进行了逐实例回查。row 41 中三个高贡献实例 15226、15230、15231 均存在于 66°后退候选集合，但模型分数分别为 `0.017711477`、`0.017812831`、`0.002496877`，低于冻结阈值 `0.02`。本地 GLB 清单完整，实例绑定预检通过，参考渲染 self-consistency PER 为 `0`，因此错误定位为模型在空间留出区域的高贡献长尾漏检，而不是候选构建或浏览器绑定问题。

M5 失败分析和修复准则见 `docs/evaluation/m5_hkust_image_failure_analysis_2026-08-01.md`。在 M4 训练完成前不启动新修复训练，避免占用其预定 GPU 槽；修复必须使用新实验名、validation/calibration 冻结和完整图像指标，不能用前端白名单或 test 后阈值调整解决。

## 2026-08-02 当前执行核验与 M5 修复预注册

### 当前资源状态

- M4 三种子消融矩阵仍在运行；五个变体中已有 `10/15` 个成员生成
  `calibration_ready_summary.json`。第三个 seed 的 `aabb_ray`、`geometry_ray` 和
  `geometry_context_ray` 正在训练，完整矩阵汇总尚未生成，因此不读取中间产物作为 M4 最终结论。
- M11 当前运行的是 Metropolis 1% 少样本 `retry3`，约处于 `32/40` epoch，非有限 loss/gradient 跳过计数均为
  `0`；5% 和 10% 适配尚未启动，不能把该训练状态当作泛化结果。
- GPU 槽位仍由 M4/M11 占用，新 M5 修复训练尚未启动，避免改变预注册矩阵的资源和代码环境。

### 回归检查

2026-08-02 执行并通过：

- `slm2viewer`: `npm test`，当前场景/模型完整性 smoke；
- benchmark Python unittest，最新回归为 39 项通过；
- `node --check` 检查 `InstancePVS.js`、`LightweightPVSDispatcher.js`、`LightweightPVSWorker.js`；
- `evaluate_proxy_interventions.py --self-test`。

这些检查只证明代码契约和探针入口没有回归，不改变 M4、M5、M10 的科学质量门判断。

### M5 修复协议

针对当前 `visible_weights` 的 `log1p`/`1024` 截断无法区分高覆盖构件的问题，新增 `docs/experiments/m5_visual_safety_repair_protocol_2026-08-02.md`。协议预注册了线性视觉质量损失、视觉质量加上高贡献正例 margin、软化权重和当前损失控制变量四个变体。损失直接近似 pose 内漏掉可见覆盖的比例，并保留 RVL 误报项、预算项和 calibration 安全规则；它不是降低阈值，也不把 Color-ID 权重宣称为精确真实像素覆盖率。

M5 修复只有在 M4 核心路线确定后启动。所有变体都必须使用固定 validation、独立 calibration、真实 60°实例级图像评价和 test one-shot 规则；当前 M5 仍为 No-Go。

## 2026-08-02 M7 成本索引与验证集回放

### 运行目的

在不读取正式 test split、不重新选择阈值的前提下，补齐 HKUST 的真实浏览器 GLB 解码/上传成本索引，并用同一条确定性 validation pose 回放比较当前联合级联和独立 RankNet。该回放用于检查 M7/M8 工具链和排序顺序，不能替代真实用户轨迹、真实网络测量或移动设备测试。

### 成本采集结果

命令：

```bash
node neural_instance_culling/benchmark/measure_glb_decode_upload_costs.mjs \
  --glb-index hkust-v3/assets/glbIndex.json \
  --glb-root hkust-v3/assets \
  --output neural_instance_culling/benchmark/out/m7_glb_decode_upload_costs_hkust_20260802.json \
  --chrome-exe /usr/bin/google-chrome \
  --timeout-ms 7200000
```

结果为 3,273/3,273 个 GLB 成功，失败 0 个，实际文件总大小 `563,269,492` bytes。每个条目包含网络读取、glTF 解析、渲染提交和总解码/上传时间。浏览器使用的是 headless Chrome，当前适配器不能证明使用了硬件 GPU；该索引可以作为桌面浏览器成本证据，不能外推为移动设备性能。

采集器现将总成本写入 `totalDecodeUploadMs`；回放器同时兼容早期已生成的 `decodeUploadMs` 字段。回放命令也允许只注册独立排序器，不再强制提供一个可见性模型。相关修改位于 `measure_glb_decode_upload_costs.mjs`、`evaluate_visual_utility_metrics.py` 和 `evaluate_download_trajectory.py`。

### 确定性 validation 回放

轨迹由 `pose_csr_hkust_v3_spatial_raw_subpose_aabb_fov66_v1` 的 664 个 validation pose 按原始索引升序生成，网络模型为 5 MB/s、50 ms 请求延迟、6 个下载并发和 2 个解码/上传并发，初始缓存为空。轨迹文件和两个结果文件为：

- `neural_instance_culling/benchmark/out/m7_m8_hkust_validation_replay_20260802.json`
- `neural_instance_culling/benchmark/out/m7_m8_hkust_validation_cascade_20260802.json`
- `neural_instance_culling/benchmark/out/m7_m8_hkust_validation_ranknet_20260802.json`

| 方法 | 50 MB 弱效用召回 | 5 s 弱效用召回 | 平均缺失弱效用比例 | 首个有用完成时间 |
|---|---:|---:|---:|---:|
| 当前联合级联 | 0.019998 | 0.007831 | 0.1547% | 113.75 ms |
| 独立 RankNet | 0.018891 | 0.004672 | 1.7456% | 112.14 ms |

这里的弱效用是 `log1p(visible_weights)`，不是像素覆盖率。两种方法在完整 331.5 s 回放结束时都请求并上传了全部 3,273 个 GLB，无效下载字节均为 `79,165,748` bytes，占总资源约 `14.05%`。因此结果只说明当前级联在该固定回放的早期时间/字节工作点提供了更高的弱效用，不支持“减少总下载量”或“联合级联已经优于独立排序器”的投稿结论。

### 阶段判断

- 成本索引子门：通过，资源完整且回放器能够严格读取，不做缺失成本填补。
- 回放工具子门：通过；单元测试 5 项通过，Python 编译检查和 `git diff --check` 通过。
- M7/M8 正式质量门：仍未通过。当前缺少真实网络轨迹、冷/温缓存配对、多轨迹置信区间和移动/硬件浏览器端到端测量；弱效用也不能替代 M5 的像素级图像效用。
- M10：继续保持 `No-Go / 设备证据缺失`，仅保留 [移动设备测试方案](../frontend/m10_device_benchmark_2026-08-01.md)，不生成移动端 p50/p95/p99 数据。

## 2026-08-02 M9 空间索引正确性修复

### 目的

强制空间索引审计在短远裁剪面下发现候选集合漏失。该问题不是实例 AABB 数据错误，而是完整 AABB 的
`Frustum.intersectsBox` 保守误报在空间切桶后不能保持：完整大盒体可能通过，任意一个子桶却都不通过。继续使用
旧索引会让候选集合依赖索引路径，存在画面漏检风险。

### 修改与验证

- 修改 `slm2viewer/src/InstancePVS.js`：跨越任意空间桶边界的实例统一进入溢出列表；仅完全包含在单桶内的 AABB
  放入桶索引；溢出项保持原有精确盒体测试。
- 新增并归档 `neural_instance_culling/benchmark/out/m9_spatial_aabb_index_far100_20260802.json`、
  `far500`、`far1000`、`far2000` 四份结果，并将正式 far=2000 摘要写入
  `docs/evaluation/m9_spatial_aabb_index_audit_2026-08-02.json`。
- 验证命令：

  ```bash
  npm --prefix slm2viewer run test:m9
  node --check slm2viewer/src/InstancePVS.js
  node slm2viewer/scripts/benchmark_m9_spatial_index.mjs --far 100 --samples 128
  node slm2viewer/scripts/benchmark_m9_spatial_index.mjs --far 500 --samples 128
  node slm2viewer/scripts/benchmark_m9_spatial_index.mjs --far 1000 --samples 128
  node slm2viewer/scripts/benchmark_m9_spatial_index.mjs --far 2000 --samples 128
  ```

四组均为 `0/128` 集合差异，索引实例数为 `16,053`，溢出实例数为 `2,778`。far=2000 时索引 p50 为
`12.20 ms`，全量扫描 p50 为 `0.69 ms`；因此空间索引只通过正确性子门，未形成性能收益主张，默认查询仍会在
桶数量过大时回退全量扫描。`npm run test:m9` 与 Node 语法检查通过。

### 质量判断

该修复保留为当前前端代码，原因是它消除了索引路径对候选集合的错误影响。性能子门仍为 No-Go；移动端方案继续
保持“设备证据缺失”，不以桌面或 headless Chrome 结果替代真实 Android 测量。

## 2026-08-02 M4 队列可复现性修复

### 变更目的

M4 队列脚本原先把等待中的外部训练任务写成固定进程号。这些进程号只对应一次服务器运行，换机器、重启或发生进程号复用时，脚本可能无意义地等待，或者把无关进程当成实验依赖。该问题影响实验编排的可复现性，但不影响已经越过等待阶段的当前训练进程。

### 修改与验证

- 修改 `neural_instance_culling/benchmark/run_formal_m4_ablation_matrix.sh`：删除硬编码 PID；改为仅在启动器显式提供 `SLM_M4_HKUST_MAINLINE_PID`、`SLM_M4_METROPOLIS_MAINLINE_PID` 或 `SLM_M4_AABB_RAY_PID` 时等待对应任务。
- 已有的 `calibration_ready_summary.json` 产物门控保持不变，注册的 AABB+ray 结果仍必须存在，不能通过跳过等待伪造矩阵完整性。
- 当前 `formal_m4_matrix` 已在修改前进入训练阶段，未重启、覆盖或改变其 GPU 任务；本次修改只影响后续启动。
- 待当前队列完成后执行 `bash -n neural_instance_culling/benchmark/run_formal_m4_ablation_matrix.sh`、`git diff --check`，并在下一次启动时验证空 PID 不会阻塞、显式 PID 仍能正确等待。

### 质量判断

该变更保留为正式实验基础设施修复。它不改变数据、模型、损失、阈值或评测口径，也不改变当前 M4 的科学结果；后续矩阵运行不再依赖本机历史进程号。

### M4 路线判定的预注册口径

在 M4 结果生成前已冻结路线规则：以 `full - geometry_context_ray` 的三种子分层配对 bootstrap 为方向代理的主增量比较；两个变体各自使用 calibration 冻结的安全工作点。只有 useful-cull 平均增益至少 `+2` 个百分点、95% 配对区间下界大于 `0`，且没有明显 bad-cull 或 miss-pixel 恶化时，才保留路线 A 的“方向代理具有独立贡献”主张。否则采用路线 B，以 `geometry_context_ray` 为主架构，方向代理降级为辅助消融。该规则不读取 test，也不允许通过阈值、候选集合或前端规则补救。

## 2026-08-02 M9 三次有效主体资源 smoke

### 目的与命令

本地前端直接访问远端站点根路径时得到 404，第一次复测没有初始化 viewer，已作为排除记录保留。随后按照 M9 文档的正确入口，启动本地 Parcel，并通过 `glbResourcesBaseUrl` 指向已部署的 HKUST 主体 GLB，执行三次独立 headless Chrome 观测。原始报告位于 `neural_instance_culling/benchmark/out/m9_remote_hkust_smoke_20260802/`，汇总由 `slm2viewer/scripts/summarize_m9_runtime_runs.mjs` 生成。

每次命令的核心参数为：

```bash
node slm2viewer/scripts/benchmark_webgpu_runtime.mjs \
  --start-server --scene hkust-v3 --port <34361|34362|34363> \
  --url 'http://127.0.0.1:<port>/?scene=hkust-v3&glbResourcesBaseUrl=https%3A%2F%2Fwww.liteweb3d.com%2Fdata%2Fhkust-v3%2F' \
  --duration-ms 30000 --settle-ms 10000 --wait-for prediction \
  --executable-path /usr/bin/google-chrome
```

### 结果与判断

- 三次均为 `completed-observation`，最终后端为 `worker-webgpu`，页面、加载和请求错误均为 0。
- Cold-0 三次通过：首个目标 GLB 请求均发生在首轮预测完成之后。
- 三次均观测到实例级显示、真实相机移动后的独立视锥刷新和 `renderOutsideRaw=0`；FOV 观测为真实 `60°`、模型/后退 `66°`。
- 推理总耗时 p50/p95/p99 为 `1507.1/1607.2/1616.1 ms`；WebGPU 推理 p50/p95/p99 为 `1495.8/1597.4/1606.4 ms`。
- 三次候选数均为 `5,959`，原始预测实例均为 `5,087`，最终实例均为 `4,928`。

该结果使 M9 的有效主体资源、实例级过滤和 Cold-0 顺序子门获得重复证据，但仍不是多轨迹、候选规模分桶或移动设备 benchmark。服务器 headless Chrome/SwiftShader 的耗时不进入移动端性能表，M9 总门继续保持未通过。

## 2026-08-02 M12 后退相机口径修复与 WGSL 一致性

### 变更目的

M12 首轮 parity 对比出现较大的 logit 差异。逐层审计发现，Python 参考使用数据集的规范 view-cell 中心，而当前前端按照
`predictionCameraMode=viewcell-back-camera` 从规范位置沿主视线后退 `3.464101552963257 m` 后执行模型查询。参考和前端并非同一相机位置，因此首轮差异不能用于判断 WGSL 实现。

### 修改内容

- 修改 `neural_instance_culling/benchmark/prepare_m12_webgpu_parity.py`：读取导出的 `instance_model_meta.json`，在 cases 中保存 `predictionPosition` 和后退相机参数；PyTorch 参考使用实际查询位置。
- 修改 `slm2viewer/src/InstancePVS.js`：保留 M12 专用的 18-word 调试输出，记录射线、固定特征和门控中间量；正常运行仍为 4-word raw 输出，不改变生产模型协议。
- 保留 `slm2viewer/src/LightweightPVSWorker.js`、`LightweightPVSDispatcher.js` 和 `benchmark_m12_webgpu_parity.mjs` 的 M12 opt-in 调试入口；该入口不被正常调度器调用。

### 复核命令与结果

固定输入为 HKUST validation 的 16 个 case、7,845 个候选实例、模型查询 FOV `66°`、真实渲染 FOV `60°`、当前导出权重和固定实例特征。产物目录为：

`neural_instance_culling/benchmark/out/m12_webgpu_parity_hkust_strong_v2_back_camera_20260802/`

验证包括：

```bash
node --check slm2viewer/src/InstancePVS.js
node --check slm2viewer/src/LightweightPVSWorker.js
node --check slm2viewer/scripts/benchmark_m12_webgpu_parity.mjs
npm --prefix slm2viewer test -- --runInBand
```

结果：页面错误 `0`，WebGPU probe `16/16` 完成，阈值翻转率相对 FP16 为 `0`；可见性 logit 平均绝对误差/p99/最大值为
`0.000507/0.001489/0.001709`，下载 logit 为 `6.51e-5/2.02e-4/2.89e-4`，下载排序 Spearman/top-10% Jaccard 为
`0.999996/1.000000`。浏览器适配器是 `google/swiftshader`，所以只通过数值一致性子门，不能作为硬件 WebGPU 或移动端性能证据。

### 阶段判断

M12 的“前端半精度资产、WGSL 权重布局、射线查询、实例级可见性输出和 GLB 下载排序输出与 FP16 参考一致性”子门通过。首轮较大 logit 差异作为参考口径错误的失败证据保留，不再引用为实现缺陷。M10 仍保持 `No-Go / 真实设备证据缺失`；移动设备测试方案、候选规模矩阵、冷/温缓存协议和预注册正确性门限已写入
`docs/frontend/m10_device_benchmark_2026-08-01.md`，未用 SwiftShader 数字填充移动端 p50/p95/p99。

### 2026-08-02 M8 多轨迹离线回放补录

为避免单条固定轨迹支撑过强的下载调度结论，本轮从两个场景的 validation split 各生成三条、每条 128 个
pose 的确定性轨迹。轨迹使用相同的 66°模型查询相机和 60°真实渲染相机，起始偏移分别为 HKUST
`0/180/360`、Metropolis `0/600/1200`；网络模型固定为 5 MB/s、50 ms 请求延迟、6 个下载并发和 2 个
解码/上传并发。轨迹、成本索引和运行结果均保存在：

`neural_instance_culling/benchmark/out/m8_trajectories_20260802/` 和
`neural_instance_culling/benchmark/out/m8_<scene>_track_<a|b|c>_<cascade|ranknet>_20260802.json`。

Metropolis 成本索引覆盖 `3,669/3,669` 个 GLB，失败 `0`；HKUST 沿用覆盖 `3,273/3,273` 个 GLB 的完整索引。
两份索引都来自 headless Chrome/SwiftShader，只能作为离线解码成本输入，不能外推到移动 GPU。

| 场景 | 方法 | 缺失弱效用 | 最终弱效用召回 | 首个有用画面 ms |
|---|---|---:|---:|---:|
| HKUST | 当前联合级联 | `2.25% ± 0.39%` | `0.9707 ± 0.0103` | `103.4 ± 9.0` |
| HKUST | 独立 RankNet | `10.68% ± 1.25%` | `0.8542 ± 0.0175` | `107.9 ± 7.3` |
| Metropolis | 当前联合级联 | `6.04% ± 0.66%` | `0.9346 ± 0.0184` | `78.1 ± 1.0` |
| Metropolis | 独立 RankNet | `32.06% ± 5.14%` | `0.6700 ± 0.0227` | `69.4 ± 13.8` |

“弱效用”严格指 `log1p(visible_weights)`，其中 `visible_weights` 是构件重要性权重，不是真实像素覆盖率。
结果支持当前级联在这组固定离线条件下的早期效用诊断，但不满足 M8 正式质量门：没有真实网络轨迹、真实
设备解码/上传、多种子 paired bootstrap 或像素级图像效用；因此不将其写成端到端下载收益，也不改变 M7/M8
和 M10 的 No-Go 状态。详细事件、字节指标和轨迹文件见
[M7/M8 回放报告](m7_m8_trajectory_replay_2026-08-01.md)。

### 2026-08-02 M5 视觉安全修复矩阵排队

M5 validation 图像门的失败证据显示，漏像素主要来自少数高屏幕贡献实例，而现有 `visible_weights` 在
`log1p`/`1024` 截断后无法区分这些正例。已把预注册协议中的视觉安全损失正式合入训练器：它对每个 pose
按屏幕覆盖代理计算软漏视觉质量项，并可选地对权重最高的正例施加最低 logit margin；RVL、预测预算和 hard
negative 排序保持不变。默认 `visual_safety_loss_weight=0`，因此当前主线行为没有改变。

本轮冻结四个变体和三个 seed：线性视觉质量、线性质量加 top-8 margin、平方根软化质量加 top-8 margin，以及
不启用新项的独立 seed 控制。参数表和语义见
[M5 修复协议](m5_visual_safety_repair_protocol_2026-08-02.md)，训练入口为
`neural_instance_culling/benchmark/run_m5_visual_safety_repair.sh`。回归测试 `test_visual_safety_loss.py`
和训练控制测试均通过。

为避免改变正在运行的 M4/M11 资源和 GPU 调度，矩阵已在 `m5_visual_repair` tmux 会话排队，日志为
`neural_instance_culling/benchmark/out/m5_visual_safety_repair_queue.log`；当前状态是等待正式会话退出，尚未
生成修复 checkpoint，也没有读取 test 或重新选择阈值。

为避免正式图像评价对每个修复模型重复加载完整 HKUST GLB 清单，新增
`neural_instance_culling/benchmark/run_m5_visual_safety_image_evaluation.py`。该入口先对四个变体、三个
seed 的 validation/calibration 生成 schema-only 的完整实例级 manifest，再通过
`run_m5_component_image_batch.py` 在同一个浏览器页面中顺序处理 24 个批次，完整 `3,273` 个 GLB 只加载一次。
批次之间使用带变体、seed 和 split 的样本前缀，避免不同模型的相同 view-cell 样本 ID 冲突；脚本不读取 test、
不扫描阈值、不改候选集合，也不产生前端兼容降级。

队列已在 `tmux m5_visual_image` 中启动，日志为
`neural_instance_culling/benchmark/out/m5_visual_safety_image_evaluation_queue.log`。截至本记录更新时，
12 个修复 checkpoint 均尚未完成，因此队列只在等待，不占用 GPU；完成后输出 validation/calibration 图像汇总，
若浏览器渲染失败则保留原始日志并保持 M5 No-Go，不以缩小样本替代正式评价。

## 2026-08-02 Benchmark 回归复核

在不改变正在运行的 M4/M11 训练、评测队列和任何正式输出目录的前提下，重新执行当前 benchmark
Python 回归套件：

```bash
PYTHONDONTWRITEBYTECODE=1 conda run --no-capture-output -n slm_pvs \
  python -m unittest discover \
  -s neural_instance_culling/benchmark/tests -p 'test*.py' -v
```

结果为 `39 tests, OK`。覆盖范围包括下载轨迹冷/温缓存语义、固定测试入口和阈值来源、实例级
Color-ID schema、三角形 HZB 缓存查询、M4 输入/抑制消融序列化、M5 视觉安全损失及训练校准控制。
该结果只证明代码契约和回归测试通过，不替代正式 validation/calibration 图像质量、泛化、移动设备
性能或 M13 one-shot test；相关质量门状态保持不变。

## 2026-08-02 M11 跨场景资源修正与零样本边界

首次启动 Metropolis 少样本适配时，训练器发现方向数据集使用 41,298 个原始构件记录，而队列脚本传入
的 `glb_points_v3_formal_metropolis_fov66.bin` 只有 3,669 行实例化原型缓存，因而在训练前以
`GLB point cache row count 3669 < runtime GLB count 41298` 拒绝运行。该失败输出目录保留，未用补零、
截断或更换实例编号来绕过检查。

已修正 `neural_instance_culling/benchmark/run_m11_metropolis_fewshot.sh`：方向数据集现在使用
`ifcbench_fantasy_metropolis_instanced_v2_glb_points_v3.bin`，其元数据为 41,298 行，与
`ifcbench_fantasy_metropolis_source/assets/runtimeVisibilityMeta.json` 和方向数据集一致；重跑输出追加
`retry2` 后缀。该修正仅恢复数据语义一致性，不改变训练、阈值或评价规则。

同时完成 HKUST 到 Metropolis 的零样本迁移，输出为
`model/out/m11_transfer_hkust_directional_to_metropolis_yaw20_seed20260801_retry2/summary.json`。
目标 calibration 冻结阈值为 `1.7782794e-07`；Metropolis test weighted recall `0.9999826`，但
agg precision `0.0911073`，平均预测 `11,481.48`、平均候选 `11,481.53`，有效剔除近乎为零。
这证明零样本参数迁移只能保留安全召回，不能保留目标场景的剔除能力；M11 不提出通用零样本主张。
少样本适配的第一次资源修正运行 `retry2` 在 1% 训练的第 1 个 epoch 因单步上下文传播显存不足而失败，
失败目录和 stderr 保留。随后 `run_m11_metropolis_fewshot.sh` 增加了单 pose 批次、较小离线特征导出批次、
默认 AMP 和可扩展显存分段配置，使用新输出后缀 `retry3` 在 GPU 3 重新启动 1%、5%、10% 队列；完成前
不记录正式适配指标。

### 2026-08-02 M11 少样本显存边界与移动测试边界

Metropolis 方向数据集按原始构件粒度包含 `41,298` 个实例，训练时上下文证据会把当前 pose 的目标实例和
证据来源同时展开。`retry2` 使用两 pose 批次，在第一个 epoch 中触发 `torch.OutOfMemoryError`，当时
可见显存不足约 `2.61 GiB`；这不是通过减少候选、修改 GT 或替换点云缓存解决的问题。新队列配置仅将
批处理和冻结 test 评测均改为默认逐 pose，并降低离线特征导出峰值；如果 retry3 仍失败，将保留日志并把 M11 少样本适配降级为
“资源边界未建立”，不伪造跨场景收益。

移动端目前没有实体 Android 设备、ADB 和可核验的硬件 WebGPU 适配器，因此 M10 仍只保留测试方案和
`No-Go / 真实设备证据缺失` 状态。方案已经固定了高性能/中端两档设备、256 至 16k 候选桶、冷/温缓存、
Wi-Fi/受控 4G、三条轨迹、每条件至少 30 次有效运行、p50/p95/p99、分项计时、内存/温度和 FP16/WGSL
正确性 smoke；没有真实样本的桶必须报告缺失，不能用 SwiftShader 结果填充移动端结论。

### 2026-08-02 M8 当前主线回放 runner 组合修正

六个当前主线离线回放初次启动后被评估器拒绝：命令同时注册当前模型和独立 RankNet，却指定统一的
`current-cascade`；独立 RankNet 没有下载头，不能提供该分数模式。原始 stdout/stderr 保留在
`benchmark/out/m8_formal_current_*_20260802.{stdout,stderr}.log`，不把失败输出计入 M8。

新增 `neural_instance_culling/benchmark/run_m8_formal_trajectory_replay.sh`，按 runner 粒度拆成
当前级联和独立排序两组，分别使用 `current-cascade` 与 `independent-utility`，并在相同轨迹、成本索引和
预算下运行。新结果使用 `m8_formal_*_20260802_retry1` 命名，避免覆盖旧失败目录和历史 `w042` replay。

回放已完成。固定离线条件下，HKUST 当前级联的最终弱效用召回为 `0.8749 ± 0.0153`，独立 RankNet 为
`0.8542 ± 0.0175`；Metropolis 当前级联为 `0.9059 ± 0.0170`，独立 RankNet 为 `0.6700 ± 0.0227`。
20 MiB 工作点的对应弱效用召回为 HKUST `0.1982/0.1948`、Metropolis `0.3004/0.1509`（当前级联/RankNet）。
这些数字来自三条确定性离线轨迹的均值和样本标准差，不是移动设备或真实网络测量；M8 仍保留 No-Go，
因为缺少多种子 paired bootstrap、真实 4G/Wi-Fi trace、图像效用和硬件解码/上传成本。

### 2026-08-02 M4/M5/M11 队列状态复核

本次复核确认正式队列仍按预注册依赖运行，没有使用中间 checkpoint 或通过阈值调整提前推进质量门。M4
输入/遮挡代理消融矩阵共有 `15` 个变体（5 种输入变体 × 3 个 seed），其中 `7` 个已经生成
`calibration_ready_summary.json`；第二批的 `geometry_context_ray`、
`geometry_context_proxy_ray_no_inhibition` 和 `full` 仍在 40 epoch 训练中，最近核验约为第 `17`、
`19` 和 `16` 个 epoch，训练记录中的非有限 loss/gradient 跳过计数均为 `0`。M4 validation evaluator
继续等待全部矩阵，尚未生成配对 bootstrap 汇总，因此方向遮挡代理的路线 A/B 尚未判定。

M11 Metropolis 1% 少样本适配 `retry3` 约完成第 `7/40` 个 epoch，非有限 loss/gradient 计数为 `0`；
5% 和 10% 队列尚未启动，当前不记录少样本指标。M5 视觉安全修复矩阵和共享浏览器图像评价均继续等待
正式 M4/M11 会话退出，尚未生成修复 checkpoint 或图像结果。M4/M5 队列脚本通过 Bash 语法检查，
相关 Python 入口通过编译和帮助命令 smoke；该检查只证明执行入口可用，不改变各阶段的 No-Go 状态。

截至本记录，工作区无未提交代码修改，默认模型、前端资产、测试阈值和 FOV 口径均未改变。下一步仍是
等待 M4 完整矩阵，随后只在固定 validation/calibration 规则下执行汇总，再按预注册门限决定是否保留
完整方向遮挡代理路线。

### 2026-08-02 M5/M11 队列会话口径修正

复核 M5 视觉安全修复队列时发现，脚本等待列表使用了旧的无后缀会话名
`m11_metropolis_fewshot`，而当前正式少样本任务实际运行在
`m11_metropolis_fewshot_retry3`。tmux 的唯一前缀匹配使现有进程暂时仍能等待到该任务，但这一行为依赖
会话命名唯一性，不适合作为可复现的实验门控。已将 `run_m5_visual_safety_repair.sh` 改为显式等待
`m11_metropolis_fewshot_retry3`，并同步更新 M5/M11 协议文档和当前状态说明。

本次修改只修正队列资源依赖，不停止或重启 M4/M11，不改变 GPU 分配、数据、训练参数、阈值或输出目录。
已执行 `bash -n neural_instance_culling/benchmark/run_m5_visual_safety_repair.sh` 和
`git diff --check`；修改已提交并推送，提交为 `4efff64`（队列脚本）和 `917c071`（M11 文档）。

### 2026-08-02 M4/M11 长任务进度更新

后续轮询显示 M4 已完成第二个正式 seed 的全部五个变体，当前共有 `10/15` 个
`calibration_ready_summary.json`。第三个 seed 已启动第一波三个变体：`aabb_ray` 约第 `20` 个 epoch，
`geometry_ray` 和 `geometry_context_ray` 约第 `6` 个 epoch；其余两个变体尚未启动。训练进程的 GPU
计算负载和 checkpoint 更新时间正常，当前没有非有限 loss、梯度异常或 OOM。M4 配对 validation 评估器
已从等待第二个 seed 转为等待第三个 seed 的第一个校准产物，尚未读取不完整结果。

M11 Metropolis 1% 少样本 `retry3` 已推进到约第 `30/40` 个 epoch，非有限计数仍为 `0`；完成后队列
才会按固定协议进入 5% 和 10% 适配。M5 修复训练和图像评价继续等待 M4/M11 依赖会话，尚未生成修复模型
或图像汇总。该进度更新只记录执行状态，不改变 M4 路线判定、M5/M10 No-Go 或默认模型。

### 2026-08-02 工具链回归与 M13 清单工具复核

在不改变任何训练进程、模型输出、候选集合、阈值或前端默认资产的条件下，重新执行了投稿收尾所需的
工具链检查。`neural_instance_culling/tools/build_artifact_manifest.py --self-test` 通过，覆盖单文件
哈希、目录树哈希、字节数统计和重叠路径拒绝；该工具只记录路径、字节数和 SHA-256，不复制数据集、GLB
或权重。当前 benchmark 回归套件完整执行 `39 tests, OK`，覆盖冷/温缓存回放、冻结 test 入口、实例级
Color-ID schema、三角形 HZB 缓存、M4 输入消融序列化、视觉安全损失和训练校准控制。

上述结果只证明工具链和代码契约可复现，不替代 M4 的配对 validation 统计、M5 的图像质量门、M10 的真实
移动设备证据或 M13 的 one-shot test。当前 M4/M11 长任务仍按原队列运行，正式质量门状态不变。

### 2026-08-02 M11 AMP 梯度跳过审计

复核 `pvs_m11_fewshot_1pct_metropolis_yaw20_rvl_strong_v2_full40_seed20260801_retry3` 的完整
`train_history.json` 后，纠正此前只查看最后一个 epoch 所造成的“非有限梯度为 0”表述。训练器中的
`trainSkippedNonFiniteGrad` 是当前 epoch 的计数，不是全程累计计数；截至 epoch `36/40`，累计跳过
`16` 个梯度步，`trainSkippedNonFiniteLoss` 累计为 `0`。这些跳过发生在 AMP 反向传播和梯度裁剪之后、优化器
更新之前，训练器会清零梯度并调用 scaler 更新，不会把该步的非有限梯度写入模型参数。

按已完成的 `36 × 900 = 32,400` 个训练 step 计算，当前跳过比例约为 `0.049%`。该现象不是静默忽略：
原始 stdout、stderr、逐 epoch 计数和 checkpoint 均保留。当前 retry3 不中途停止，先完成预注册的 40 epoch
和安全 calibration；最终报告必须同时给出跳过总数、比例和是否存在非有限 loss。如果安全工作点或正式 test
失败，再以独立输出目录执行关闭 AMP 的 FP32 重跑，不能覆盖本次证据或将其改写为无异常训练。

### 2026-08-02 M4 可用成员预评估与回归复核

在不改变 M4 训练参数、候选集合、校准阈值或输出目录的前提下，利用空闲的 GPU 0 顺序运行了已经生成
`calibration_ready_summary.json` 的 `11/15` 个 M4 成员。评估输出使用
`m4_formal_baseline_<experiment>_validation/interventions.json` 的正式队列命名，严格读取 validation
split，未读取 test，也未重新选择阈值；剩余四个成员中的两个仍在 seed `20260803` 训练，另外两个等待其
训练完成，因此没有提前生成矩阵汇总或路线结论。

本轮工具复核结果为：benchmark 回归套件 `39 tests, OK`，`build_artifact_manifest.py --self-test`
通过，M4/M5/M11 队列脚本 Bash 语法检查和 `git diff --check` 通过。M11 少样本 `retry3` 已记录到
epoch `37/40`，当前 epoch 的非有限 loss 和梯度跳过均为 `0`；M4 的两个 seed-20260803 运行成员仍
保持正常 GPU 计算。上述检查只缩短后续评估等待，不改变 M4、M5、M10 或 M13 的质量门状态。

### 2026-08-02 M4 阈值来源校验修复

对已生成的 M4 validation 输出做结构化审计时发现，汇总器原先把 `thresholdSource` 序列化后进行
`"test"` 子串匹配。合法的 `calibration_ready_pre_test` 记录本来就包含该单词，导致正式汇总会错误拒绝
所有 calibration 阈值，形成实验契约层面的阻断。现已在
`neural_instance_culling/benchmark/summarize_formal_m4_matrix.py` 中改为检查结构化字段：协议必须是
`calibration_ready_pre_test`、`testEvaluationCount` 必须为 `0`，且不得启用 test threshold override；
真正的一次性 test 协议会被拒绝。

新增 `test_formal_m4_summary.py` 覆盖合法 pre-test 记录和三种非法 test 来源。新增测试与既有协议测试共
`7 tests, OK`，并对现有 `11` 个 M4 validation 输出执行了同一 `read_input` 审计。该修复只影响汇总器的
来源判定，不改变模型、阈值、候选集合或 test 数据。

### 2026-08-02 M11 少样本冻结协议修复与移动端测试边界复核

少样本模型的冻结 test 首次执行被 split 校验拒绝，原因是原生数据集的完整训练划分为 `20,454` 个 pose，
而 checkpoint 记录的 `trainFitCount=205` 是按固定种子抽出的 1% 适配子集。该错误发生在推理前，不是模型预测或
候选集合错误。已修改 `neural_instance_culling/benchmark/evaluate_frozen_test.py`，并新增少样本协议回归用例：
校验完整 `originalTrainCount`，按 `trainFitSelectionFraction/Seed` 重建训练子集，严格校验子集 digest；
validation、calibration、test 仍保持完整划分和一次性冻结规则。相关测试共 `6 tests, OK`。

旧失败 manifest 和输出目录没有覆盖；新的 manifest 和 test 输出使用 `protocolfix` 后缀。

Metropolis directional 1% 适配使用 `2,271` 个唯一 test pose、查询 FOV `66°`、真实渲染 FOV `60°` 和严格存储候选集合。
阈值为 calibration 冻结的 `0.05000000074505806`，test 只执行一次。结果为：weighted recall `0.993608`、pose
recall `0.945077`、pose precision `0.156026`、useful cull `0.279348`、bad cull `0.004880`，平均候选/GT/预测
为 `11,481.53/1,046.06/6,908.04`。该比例满足重要构件召回安全约束，但效率和普通 recall 不足以支持高效跨场景泛化结论。

5% 和 10% 适配已在 `m11_metropolis_fewshot_5_10` 会话中按同一 protocol 启动，日志为
`neural_instance_culling/benchmark/out/m11_metropolis_fewshot_5_10_queue.log`。

移动端仍没有实体 Android、ADB 连接和可确认的硬件 WebGPU adapter，因此 M10 继续保持
`No-Go / 真实设备证据缺失`，不生成移动端 p50/p95/p99，也不把桌面 SwiftShader 数字外推到移动端。预注册方案见
`docs/frontend/m10_device_benchmark_2026-08-01.md`：拿到设备后按高性能/中端两档、三个固定轨迹、冷/温缓存、
受控 Wi-Fi/4G、`256/512/1k/2k/4k/8k/10k/16k` 候选桶和每条件至少 30 次有效重复执行；先通过 256 候选 schema
smoke 与 M12 FP16 parity，再进入完整矩阵。缺失设备或缺失候选桶只报告缺失，不能补造统计数字。

### 2026-08-02 11:37 M4/M5/M11 持久任务复核

本次复核以当前文件和进程状态为准，没有停止、重启或修改正在运行的正式任务。M4 消融矩阵已有
`11/15` 个成员写出 `calibration_ready_summary.json`；seed `20260803` 的
`geometry_ray` 和 `geometry_context_ray` 仍在 40 epoch 训练，最近分别约为第 `22` 和第 `21` 个 epoch，
GPU 计算与 checkpoint 更新时间正常，已记录的非有限 loss/gradient 计数为 `0`。M4 validation evaluator
继续等待这两个成员，尚未生成矩阵汇总，因此方向遮挡代理的路线 A/B 仍未判定。

M11 Metropolis 少样本 `retry3` 的 1% 适配已完成 40 epoch、校准和冻结 test；5% 适配正在第 `2/40` 个 epoch，
10% 适配尚未启动。5% 当前没有非有限 loss，训练过程仍使用预注册的 AMP 参数；最终报告会保留每个 epoch 的
梯度跳过统计，不把单个 epoch 的计数误写成全程累计值。M5 修复训练和共享浏览器图像评价分别等待正式 M4/M11
会话结束，尚未生成新的修复 checkpoint 或图像汇总。

本轮重新执行 benchmark 回归套件，结果为 `43 tests, OK`；M4/M5/M11 队列 Bash 检查和 `git diff --check`
通过。该检查只证明当前协议与工具链没有回归，不改变 M4、M5、M10 或 M13 的质量门状态。工作区保持干净，
默认模型、前端资产、阈值和 `66°` 模型查询 / `60°` 真实渲染 FOV 口径均未改变。

### 2026-08-02 M4 路线判定工具补齐

为避免在三种子矩阵完成后凭人工阅读选择贡献路线，新增
`neural_instance_culling/benchmark/decide_m4_route.py`。它只读取完整的
`m4_formal_matrix_validation_summary.json`，要求摘要明确标记为 validation-only、包含三种子和完整变体，
然后按预注册规则检查完整模型相对 `geometry_context_ray` 的 useful-cull 增益是否至少 `0.02` 且 paired
bootstrap 95% CI 下界大于 `0`。M4 摘要当前尚未完整生成，因此本节不提前写入路线 A/B 结果。

工具同时明确记录图像/字节替代门尚未包含在 M4 矩阵中，不能从 useful-cull 数字推断下载收益；它拒绝 test-derived
摘要，输出摘要 SHA-256、阈值来源边界和 `testRead=false`。新增 3 个单元测试通过，formal M4 evaluator 已
接入该工具作为矩阵汇总后的只读步骤。

### 2026-08-02 NeuralPVS 官方实现事实更正

复核临时审计副本 `946088616cad18de81cde12fecd6ab204e52eac9` 后，修正文档中“公开资料不足以复现”的表述。
官方仓库实际包含 OACNN/VNet、三维交错模块、数据集读取器、训练与推理入口，以及 Dice/focal/
repulsive/no-guess 类损失；数据格式为 `gv/*.bin.gz` 和 `pvv/*.bin.gz` 的 bit-packed 三维体素网格。

这项更正不改变实验结果或 M6 门控。当前 `slm_pvs` 缺少 `spconv`/`cupy`，项目也没有完成真实深度/实例 ID
到 froxel 的转换、froxel-to-instance 保守映射、严格 candidate CSR runner、同 FOV/view-cell 的公平评测和
冷启动资源核算。因此已有 AABB depth proxy、三角形 HZB warm-cache 和实例级 MLP 仍不能作为 NeuralPVS
结果；M6 保持 `No-Go / adaptation not implemented`。详细审计见
`docs/experiments/m6_neuralpvs_baseline_audit_2026-08-01.md`。
