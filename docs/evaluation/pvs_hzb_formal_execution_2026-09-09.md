# Geometry-shell HZB 论文执行编排

日期：2026-09-09

状态：窄范围 orchestrator 与 CPU preflight 已完成；两场景输入、两档分辨率和 120-pose 计划均通过。尚未启动 Chrome、WebGPU 或 GPU 正式实验，也没有产生新的 HZB visibility、图像或性能结论。

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

实际独占硬件窗口中，去掉 `--dry-run` 后运行 `all`；入口会把每个子进程 stdout/stderr 写入对应任务目录，并沿用现有 runner 的 Chrome Vulkan hardware gate。正式 `all` 尚未执行；不占 GPU 的 `preflight` 已执行。

本次已通过：

- `python -m py_compile neural_instance_culling/benchmark/run_geometry_shell_hzb_paper.py neural_instance_culling/benchmark/tests/test_run_geometry_shell_hzb_paper.py`；
- `python -m unittest neural_instance_culling.benchmark.tests.test_run_geometry_shell_hzb_paper`；
- 真实 worktree 资源上的 `all --dry-run`：`32` 个计划任务，即 2 preflight、12 calibration、2 select、12 test、4 timing，全部可计划；
- 真实资源上的 CPU `preflight`：HKUST 与 IFCBench 均通过，确认 calibration 为 `512x288 / 1024x576`、每配置一次 invocation 和单轮可见性，timing 为 120 pose 五轮。

新增文件：

- `neural_instance_culling/benchmark/run_geometry_shell_hzb_paper.py`；
- `neural_instance_culling/benchmark/tests/test_run_geometry_shell_hzb_paper.py`；
- 本执行记录。
