# Geometry-shell HZB 论文执行编排

日期：2026-09-09

状态：正式 `all` 已于 2026-09-10 启动；WebGPU 大 buffer 问题与结果协议问题已修正，当前从 HKUST calibration 恢复执行，尚未形成完整 HZB 结论。

## 2026-09-10 外壳实例可见性语义修正

恢复运行后，HKUST lossless `512x288` 的三个 depth-bias 配置均保留约 `99.55%` 候选。代码审查确认这不是正常的 HZB 精度/效率权衡：旧查询把所有 `instance_occluder=1` 的外壳实例无条件加入可见集合。HKUST lossless 外壳覆盖 `18,566/18,831` 个实例，IFCBench lossless 覆盖 `41,298/41,298` 个实例，因此旧实现会结构性退化为 Keep-All。该规则是本项目 MVP 的保守扩展，不是 Greene 等人的经典 HZB 或 Lee 等人的 HROC 所要求的对象可见性判定。旧正式矩阵已停止，其 calibration 数字作废，不进入论文表格、选择或 streaming。

论文基线采用与冷启动任务匹配的 WebGPU geometry-shell HZB，而不宣称完整复现依赖连续帧、对象 BVH、fragment-ray traversal 和完整驻留几何的 HROC。修正后的唯一语义为：

1. 外壳实例流为每个实例变换保存真实 `componentGlobalId`；
2. depth-only pass 同时写正线性深度和最近表面的 component ID；
3. GPU 将 ID attachment 压缩成外壳实例可见 bitset；
4. 已选外壳实例只有在真实贡献最近深度像素时才保留；
5. 未选入外壳的候选继续执行保守 AABB/max-depth-HZB 测试；
6. Region66 仍对同一 view-cell 的登记 subpose 分别执行并取实例并集。

修正会提升 runtime schema，不保留旧 selected-self 兼容路径。重新导出 lossless/equal-asset 外壳后，仍只在 calibration 选择分辨率和 depth bias；最终 HZB 与 Full V4 必须使用同一冻结 test split、同一 candidate CSR、同一 Region66 GT 和同一指标实现。HROC 作为现代层次 GPU 遮挡查询相关工作引用，不作为当前代码的复现名称。

### 基线定位与对象 BVH 决策

正式名称固定为 **WebGPU Batched Geometry-shell Hi-Z**。它使用图像空间 max-depth pyramid，并对已经由同一 `66°` candidate CSR 完成视锥筛选的实例执行一线程一个 AABB 的 GPU 批量查询。它不是 HROC 的完整复现；没有实现 HROC 的连续帧可见集、对象 BVH、occludee-group extraction、fragment-ray traversal 或 indirect multidraw，因此论文不得用 HROC 名称或直接对照 HROC 论文时间。

不增加对象 BVH 是本实验的明确设计，而不是遗漏：神经模型和该基线都接收相同候选并具有 `O(N_candidate)` 的轻量逐实例查询，能够直接输出 component ID。旧执行的时间只能用于诊断阶段构成、不能用于可见性结论；其 HKUST `512x288` 单 subpose 中值约为 depth raster `19.08 ms`、HZB build `4.43 ms`、AABB test `4.42 ms`、压缩/回读 `7.67 ms`。对象 BVH直接影响的 AABB 阶段约占总时间 `12%`，主要成本仍是外壳光栅、mip 和结果传输。正式 v2 timing 将重新测量这些数字，并给 candidate-count scaling。

为避免几千个 prototype 的 JavaScript draw 编码把几何基线人为做弱，正式实现使用预录 WebGPU `RenderBundle` 复用固定外壳 draw commands。论文同时报告各阶段时间和 `depth + visible-ID + mip` 的查询无关成本，使读者可以判断即使采用更激进的对象层次查询，外壳预下载与深度构建成本仍有多少。若描述相关工作，Greene 等人的 HZB 是基础，CHC++ 与 HROC 是完整几何驻留条件下的现代层次查询代表；本实验只声称实现经过 GPU 批处理的实际 Web Hi-Z 基线。

### 遮挡指标显示名称

本项目以可见为正类。HZB 与神经模型同表时，论文使用 `Visible Recall = recall`、`Occlusion Recall = specificity` 和 `False Occlusion Rate = 1 - recall`。`usefulCull = TN/candidate`、`badCull = FN/candidate` 继续单列；二者不能分别替代 `specificity` 和 `1-recall`。所有字段同时输出 pose-macro 与 aggregate，正式表头及分母以统一指标协议为准。

## 2026-09-10 正式启动与 WebGPU limit 修正

首次正式 HKUST lossless calibration 在资产上传阶段失败。外壳解码后的 POSITION 和 INDEX buffer 分别约为 `1.12 GB` 与 `657 MB`，而 `GeometryShellHZB` 仍以 WebGPU 默认 `maxBufferSize=256 MB` 请求 device；A6000 adapter 实际声明支持约 `4 GB`。该失败发生在首个 pose 前，`formalReady=false`，没有进入 evaluator。

`GeometryShellHZB.init()` 现根据 shell metadata 的顶点、索引、变换和实例表实际大小计算所需 `maxBufferSize`，确认不超过 adapter limit 后通过 `requestDevice({ requiredLimits.maxBufferSize` 显式申请。修改不改变外壳内容、HZB 算法、候选集合、深度偏置或评价口径。失败目录保留为 `h288_b0p0001.failed_default_max_buffer_2026-09-10`，新正式任务从原路径重新执行；完整前端测试已通过。

修正 buffer limit 后，HKUST lossless `512x288 / 0.0001 m` 已由 A6000 硬件完成全部 `659` 个 calibration view-cell，耗时约 `14` 分钟。浏览器结果顶层正确记录了 `mode=Region66`，但写入嵌套 `workload` 摘要时字段白名单漏掉 `mode`，编排器因此在浏览器任务完成后拒绝接收，尚未生成 `metrics.json` 和 `task.json`。该目录改名保留为 `h288_b0p0001.failed_missing_nested_mode_2026-09-10`，不得作为正式 calibration 结果。

生产端 `attachWorkloadProvenance()` 已补齐 `workload.mode`，并新增测试确认 `mode` 和 `split` 会同时进入正式结果。修正仅补齐既有 workload 元数据，不放宽编排器校验，也不改变 HZB 输出。`slm2viewer npm test` 与 Python HZB 编排器测试均已通过，正式矩阵从空出的原任务路径重新执行。

## 目的与入口

新增入口：`neural_instance_culling/benchmark/run_geometry_shell_hzb_paper.py`。

它只编排现有的 Geometry-shell HZB 浏览器 runner、HZB evaluator 和 calibration selector，不改变四套已生成 shell、浏览器实现或既有评价入口。默认读取：

- `geometry_shell_hzb_lossless_hkust/`、`geometry_shell_hzb_equal_asset_hkust/`；
- `geometry_shell_hzb_lossless_ifcbench/`、`geometry_shell_hzb_equal_asset_ifcbench/`；
- 两场景 Pose CSR、Region66 subpose 数据、runtime metadata、`glbIndex.json` 和 120-pose plan。

默认新结果根目录为：

`neural_instance_culling/benchmark/out/paper_results/hzb/geometry_shell_hzb_paper_2026-09-09/`

入口模式为 `preflight`、`calibrate`、`select`、`test`、`timing` 和 `all`。`all` 的顺序固定为 preflight、calibrate、select、test、timing。

## 正式矩阵

| 阶段 | HKUST | IFCBench | 规则 |
|---|---:|---:|---|
| preflight | 1 | 1 | CPU 只检查 schema、shape、split、候选/GT 语义和资源存在性 |
| calibration | 6 | 6 | 仅 lossless；高度 `288/576` × depth bias `0.0001/0.001/0.01 m` |
| select | 1 | 1 | 只读取本场景六个 lossless Region66 calibration metrics |
| frozen test | `1/5/9/all × lossless/equal-asset` | `1/all × lossless/equal-asset` | 全 test split，使用 selection 冻结的高度和 depth bias |
| timing | `120 poses × lossless/equal-asset` | `120 poses × lossless/equal-asset` | 使用已有显式 plan，Region66 `all`，每项 5 轮 |

Calibration 每个配置只有一次扫描任务，不重复运行配置；calibration 和 frozen test 均传 `--timing-rounds 1`，每个 pose 只生成一次用于准确率评价的可见集合。只有独立的 120-pose timing 任务传 `--timing-rounds 5`，用于统计各运行阶段的 p50/p95。硬件证据、完整 split、冻结配置与 `formalReady=true` 仍是 calibration/test 的正式门，不能把 `formalReady=false` 的结果送入 evaluator。

测试区域计数使用 runner 的语义：`0` 表示 `all` source subposes，HKUST 还运行 `1/5/9`，IFCBench 运行 `1`。selection 只决定 resolution/depth bias；equal-asset 测试复用相同冻结工作点，不重新校准。

## 产物和恢复规则

每个任务有独立目录，例如：

- `calibration/hkust/lossless/h288_b0p0001/`；
- `select/hkust/`；
- `test/hkust/equal_asset/region5/`；
- `timing/ifcbench/lossless/`。

浏览器任务要求 `result.json`、`geometry_shell_hzb_gpu_evidence.json`、`geometry_shell_hzb_workload.json`、候选二进制、`metrics.json`（test/calibration）以及 `stdout.log`、`stderr.log`。成功任务还保存 `command.json` 和 `task.json`，记录参数、阶段、测试读取状态和产物路径。

- 完整且 schema/语义有效的已有任务直接跳过；
- 已存在但缺少任一正式产物、日志、硬件证据或指标的非空目录直接失败，不覆盖、不拼接旧结果；
- runner result 的 `formalReady`、`executionClass=formal-hardware-gpu`、NVIDIA adapter、硬件 gate 和无并发计算证据必须全部通过；
- 任何 `formalReady=false` 结果只保留为失败诊断，永远不会调用 `evaluate_geometry_shell_hzb.py`；
- test/timing 直达模式首先读取并校验 selection，selection 不存在或不完整时不会创建 test/timing 目录；
- 正式 test 只在 calibration selection 已冻结后执行；每个 test task 只用冻结配置对 test split 读取一次，不参与任何后续选择。

## Point60 边界

Point60 不进入本次 Region 主链。preflight 只登记独立的 raw Color-ID GT handoff：每个已有 Point60 plan JSONL 对应一个 `run_sampler.mjs --point60-gt --require-hardware-gpu` 命令及其独立 stdout/stderr 路径；raw GT 齐备后才可给 HZB evaluator 传 `--point-gt-raw-dir`。该登记不启动采样，也不阻塞 Region66 calibration、selection、test 或 timing。

## 评价口径

Region66 evaluator 继续报告 aggregate 和 pose-macro 的 precision、recall、F1、Jaccard、accuracy、balanced accuracy、specificity、useful cull、bad cull，以及 `visible_weights` 加权 recall 和单侧 bootstrap 下界。`useful cull` 是正确剔除的候选比例，`bad cull` 是漏剔除 GT 的候选比例；二者不能用裸预测数量替代。GLB 数量/字节来自显式 `glbIndex.json` 和源 GLB 文件。

本 orchestrator 不生成图像 PER、miss pixel、wrong-ID pixel、Color-ID 图像 manifest 或移动设备结果；这些仍需在正式 Region test 和独立图像/设备入口完成后报告。本文件不把现有并发 smoke 或 CPU shell 导出数字提升为正式论文结果，Geometry-shell HZB 仍是基线对照，不替换当前神经 PVS 主线。

## 命令与验证

只查看正式任务矩阵、不启动 GPU：

```bash
conda run --no-capture-output -n slm_pvs \
  python neural_instance_culling/benchmark/run_geometry_shell_hzb_paper.py \
  all --dry-run
```

实际独占硬件窗口中，去掉 `--dry-run` 后运行 `all`；入口会把每个子进程 stdout/stderr 写入对应任务目录，并沿用现有 runner 的 Chrome Vulkan hardware gate。正式 `all` 正在恢复执行；已完成的 preflight 会直接复用，失败诊断目录不会进入选择和汇总。

本次已通过：

- `python -m py_compile neural_instance_culling/benchmark/run_geometry_shell_hzb_paper.py neural_instance_culling/benchmark/tests/test_run_geometry_shell_hzb_paper.py`；
- `python -m unittest neural_instance_culling.benchmark.tests.test_run_geometry_shell_hzb_paper`；
- 真实 worktree 资源上的 `all --dry-run`：`32` 个计划任务，即 2 preflight、12 calibration、2 select、12 test、4 timing，全部可计划；
- 真实资源上的 CPU `preflight`：HKUST 与 IFCBench 均通过，确认 calibration 为 `512x288 / 1024x576`、每配置一次 invocation 和单轮可见性，timing 为 120 pose 五轮。

新增文件：

- `neural_instance_culling/benchmark/run_geometry_shell_hzb_paper.py`；
- `neural_instance_culling/benchmark/tests/test_run_geometry_shell_hzb_paper.py`；
- 本执行记录。
