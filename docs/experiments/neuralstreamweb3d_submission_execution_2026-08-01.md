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
| M12 | 探索性 | 16 个 case 的 SwiftShader WGSL parity 已生成，但不能作为硬件性能结论，且 logit 差异仍需定位。 |
| M13 | 未开始 | 三种子正式重训、冻结阈值、one-shot test、图像、调度和运行主表均未冻结。 |

## 当前后台任务

- M4：`formal_m4_matrix` 与 `formal_m4_eval`，使用固定 HKUST 空间数据和预注册三种子矩阵。
- M11：`m11_hkust_directional`，实验名为 `pvs_m11_directional_yaw20_rvl_strong_v2_full40_hkust_fov66_seed20260801`，GPU 3，方向留出训练。

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

- M4 三种子消融矩阵仍在运行。已生成的 `aabb+ray`、`geometry+ray`、`geometry+context+ray` 和“无显式抑制头”目录均只作为 calibration-ready 中间产物；完整模型与第二个正式 seed 尚未完成，因此不读取其指标作为 M4 最终结论。
- M11 HKUST 航向留出训练仍在持久会话中运行；其 validation 摘要不能替代 M11 的方向留出、跨场景和少样本完整实验。
- GPU 槽位仍由 M4/M11 占用，新 M5 修复训练尚未启动，避免改变预注册矩阵的资源和代码环境。

### 回归检查

2026-08-02 执行并通过：

- `slm2viewer`: `npm test`，当前场景/模型完整性 smoke；
- benchmark Python unittest，31 项通过；
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
