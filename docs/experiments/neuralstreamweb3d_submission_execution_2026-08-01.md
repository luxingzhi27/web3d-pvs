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
