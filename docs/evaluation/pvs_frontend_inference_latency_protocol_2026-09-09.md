# PVS V4 前端推理耗时评价协议

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: plan
- Origin Date: 2026-09-09
- Verification Status: UNVERIFIED
- Version Label: pvs_frontend_latency_protocol_v1

## 目标

本实验测量当前 PVS V4 模型在桌面浏览器和真实移动设备上的运行成本，为论文提供可复现的前端耗时数据。主指标是用户从主线程发起一次完整 view-cell 预测，到 Worker 返回最终实例集合和 GLB 队列的耗时。模型初始化、GPU 内核、缓存重过滤和场景显示更新分别报告，不能混成一个数字。

实验只测当前生产模型和当前冻结阈值，不重新选择 checkpoint 或阈值。运行时关闭逐候选诊断，不加载场景 GLB，不运行材质解析和渲染，以免公网下载、Three.js 绘制和模型推理互相污染。完整页面调度成本另列为系统指标。

## 测量边界

一次完整预测按以下路径计时：

```text
主线程生成相机快照
  -> postMessage 到 LightweightPVSWorker
  -> 后退 66° AABB 候选过滤
  -> V4 关系条件、频谱矩、生存场和 MLP
  -> 冻结阈值筛选
  -> 真实 60° 视锥过滤
  -> 实例与 GLB 压缩/回读
  -> Worker 后处理和 postMessage
  -> 主线程收到结果
```

| 指标 | 定义 | 论文用途 |
|---|---|---|
| `queryE2EMs` | 主线程调用 `predict()` 前到 Promise 完成 | 前端推理主指标 |
| `workerTotalMs` | Worker 收到请求到发送结果 | 排除主线程消息等待后的后端总成本 |
| `backendTotalMs` | `InstancePVSRuntime.predict()` 的完整耗时 | 候选、模型、压缩和结果解码 |
| `backendComputeMs` | WebGPU compute pass 的 GPU timestamp；设备不支持时为空 | 诊断 GPU 计算本身 |
| `workerPostMs` | Worker 总耗时减去后端耗时 | 位图、排序和消息构造成本 |
| `messageMs` | `queryE2EMs - workerTotalMs` | Worker 排队、结构化克隆和线程唤醒成本 |
| `refilterE2EMs` | cell 内移动时只做真实 60° 缓存重过滤的往返耗时 | 系统交互成本，不算模型 forward |
| `coldInitMs` | 新浏览器上下文、禁用 HTTP cache，从 init 到 Worker ready | 首次使用启动成本 |
| `warmInitMs` | 浏览器缓存已有运行资产时的 init 到 ready | 后续页面启动成本 |

现有 Worker 字段 `inferenceMs` 包含后端完整调用，不应在论文中命名为纯 GPU kernel 时间。正式 benchmark 输出将它规范为 `backendTotalMs`，保留原字段只作源码定位。GPU timestamp 只在 adapter 支持 `timestamp-query` 时作为补充；主结论使用同步到最终回读后的 `queryE2EMs`，因为它对应用户实际等待。

计时使用单调的 `performance.now()`，符合 [High Resolution Time](https://www.w3.org/TR/hr-time-3/) 语义。WebGPU timestamp 按 [WebGPU timestamp query](https://www.w3.org/TR/webgpu/#timestamp-query) 单独采集。开启 timestamp-query 的诊断轮次不能替代不带插桩的主轮次。

## 固定工作负载

### 标准跨设备工作负载

输入来自：

```text
neural_instance_culling/dataset/out/
pose_csr_hkust_v3_main_stratified_calibration_fov66_v1/
```

使用全部 `684` 个冻结 test view-cell 的位置和朝向。test 仅提供固定、未调参的运行负载，不参与模型选择。所有设备统一使用垂直 FOV `60°`、模型候选 FOV `66°` 和固定 `16:9` aspect，保证硬件之间处理相同候选问题。

benchmark 从 `poses.bin`、`split_manifest.json` 和相机语义生成只读回放文件。回放文件保存 pose ID、位置、四元数、FOV、aspect、near/far 和预期候选数量，不复制 GT，也不改变候选算法。

每次查询都直接调用 `LightweightPVSDispatcher.predict()`，强制执行完整模型。不能经过 `CameraPredictionGate`，否则相邻 pose 可能只运行缓存重过滤，使结果低估模型耗时。查询严格串行，前一个结果返回后才提交下一个。

### 候选规模分层

每条样本记录运行时实际 `candidateCount`。汇总时按标准工作负载的候选数量四分位分为 small、medium、large、very-large，并额外选取候选数最接近 `10,000` 的实际 pose 子集。

现有移动端工程目标定义为：约 `10k` 候选时 `queryE2EMs p95 < 50 ms`。10k 子集要求每个 session 至少有 30 个候选数在 9k--11k 内的实际 pose。如果样本不足，不构造虚假候选；改为报告最大候选组的范围和 p95，并将 10k 结果标记为不可用。

### 设备原生工作负载

跨设备主表使用固定 `16:9`。附录再以每台设备真实前台视口的 aspect 回放同一批 pose，说明手机竖屏或宽屏导致的候选变化。原生视口结果不能与标准工作负载混合求平均。

## 设备矩阵

最低可发表配置包含一台桌面设备和一台真实 Android 设备。推荐配置再加入消费级集成 GPU 笔记本和第二台中端 Android 手机，避免结论只覆盖服务器显卡或单一移动 SoC。

| 层级 | 设备要求 | 后端 |
|---|---|---|
| 必测桌面 | 当前 Linux 工作站，RTX A6000，记录 CPU、RAM、驱动和 Chrome 版本 | WebGPU、WASM SIMD |
| 必测移动 | Android 12 以上、Qualcomm Adreno 或 ARM Mali、Chrome Stable 的真实手机 | WebGPU、WASM SIMD |
| 推荐桌面 | Apple M2/M 系列或 Intel/AMD 集成 GPU 笔记本 | WebGPU、WASM SIMD |
| 推荐移动 | 与第一台不同档位和 GPU 厂商的中端手机 | WebGPU、WASM SIMD |
| 可选平台 | iPhone/iPad Safari；只在当前 V4 Worker 和资产可直接运行时加入 | WebGPU 或 WASM SIMD，分后端报告 |

Chrome 从 Android 121 起默认支持 Android 12 以上的 Qualcomm 和 ARM GPU，设备范围见 [Chrome WebGPU Android 说明](https://developer.chrome.com/blog/new-in-webgpu-121)。模拟器、桌面 DevTools 设备模拟和远程云手机不能代替真实移动设备。

每台设备必须记录：设备型号、SoC/GPU、RAM、操作系统、浏览器版本、后端、adapter 信息、是否 fallback、屏幕状态、供电状态和测试前后温度。论文不能只写“desktop”或“mobile”。

## 浏览器与资产条件

- 使用 production build 和正式 FP16 运行资产，`debugLogging=false`、`neuralDebugLogs=false`。
- 模型资产由同机 localhost 或 Android `adb reverse` 提供，不经过公网；查询阶段所有资产已驻留内存。
- 不加载任何场景 GLB、纹理、AO/SMAA 或调试面板。
- WebGPU 固定请求 `powerPreference: 'high-performance'`。
- 桌面无头 Chrome 使用 `chrome_gpu_flags.mjs`，必须通过 `npm run probe:webgpu-hardware`。
- 移动浏览器保持前台、屏幕常亮，不启用 CPU throttling、低电量模式或 DevTools 性能模拟。
- WebGPU 和 WASM 使用同一模型资产、相机序列和输出语义。

桌面正式运行保存 adapter、WebGL renderer、Chrome 参数和同窗口 `nvidia-smi/pmon`。移动端保存 adapter 的 `vendor`、`architecture`、`isFallbackAdapter`（浏览器支持时）和 Chrome GPU 信息；任何 SwiftShader 或 fallback adapter 结果只能作为软件诊断。

## 执行过程

### 正确性预检

每台设备先选固定的 10 个 pose，比较 WebGPU 和 WASM 返回的候选数、模型可见实例数、真实视锥实例数和 GLB 数。集合不一致时停止计时并修复实现。预检可以开启诊断；正式计时必须重新启动浏览器并关闭诊断。

### 预热与正式测量

每个“设备 × 后端”执行五个独立 session：

1. 启动新的浏览器上下文并初始化模型，记录一次 init 分解。
2. 用覆盖四个候选规模层的 50 个查询预热，不计入结果。
3. 按预先生成的固定随机顺序串行执行全部 684 个 pose。
4. 一次性把内存中的 JSONL 结果传回主机，不能每个 pose 经 CDP 拉取数据。
5. 关闭浏览器。移动设备等待温度恢复后再开始下一 session。

五个 session 共得到 `3420` 条正式查询。WebGPU/WASM 的执行顺序采用 AB/BA 交替，防止后运行的后端持续受热。任何 session 因页面进入后台、设备热保护、adapter 丢失或浏览器崩溃而无效时，整轮重跑；不能只删除慢样本。

冷启动单独执行。桌面每个后端至少 10 次全新 context，移动端至少 5 次；每次禁用 HTTP cache，但仍使用本地服务器。warm init 使用已有浏览器缓存，次数相同。初始化样本不与 3420 条 warm query 混合。

### 移动端连接

Android 推荐使用 USB 和 Chrome remote debugging：

```bash
adb reverse tcp:8080 tcp:8080
adb forward tcp:9222 localabstract:chrome_devtools_remote
```

手机在前台打开 `http://localhost:8080/runtime-benchmark.html`。页面内部完成预热和 684 pose 回放，主机只负责开始、结束和接收最终 JSON。远程调试方法参考 [Chrome Android remote debugging](https://developer.chrome.com/docs/devtools/remote-debugging/local-server)。

正式 session 前关闭其他应用和省电模式，固定屏幕亮度，保持浏览器前台。手机在不充电状态下开始计时；每轮记录 `adb shell dumpsys thermalservice` 和 `adb shell dumpsys battery`，达到严重 thermal throttling 时等待冷却并重跑整轮。

## 统计分析

论文主表对每个设备和后端分别报告，禁止把桌面和手机样本合并：

- `queryE2EMs` 的 p50、p95 和 95% bootstrap CI；
- `backendTotalMs` 的 p50、p95；
- `messageMs` 和 `workerPostMs` 的 p50、p95；
- 10k 候选子集的 p50、p95 和样本数；
- `refilterE2EMs` 的 p50、p95，明确标注“不运行模型”；
- cold/warm init 的 p50、p95；
- 平均、p95 候选数，平均预测实例数和固定回读字节；
- runtime asset 大小、峰值 Worker 内存（平台可获取时）。

置信区间使用 10,000 次分层 bootstrap：先重采样五个 session，再在 session 内重采样 pose。WebGPU 与 WASM 比较使用相同 pose 和 session 序号的配对差值，报告 `WebGPU - WASM` 的 p50 差值、p95 差值、95% CI 和加速比。候选数量与延迟的 Spearman 相关系数及四分位结果放在附录。

不裁剪最慢 1% 样本，不使用“稳定后最好 100 次”，不把均值最低的一轮挑作论文结果。发生 GC、系统调度或频率变化造成的长尾属于前端实际成本，应进入 p95。只有协议明确判定整个 session 无效时才能重跑。

## 论文表格

主表建议格式：

| Device | Backend | Assets MiB | Candidates mean/p95 | Query p50 | Query p95 | Backend p50 | Backend p95 | 10k p95 | Cold init p50 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| RTX A6000 desktop | WebGPU | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| RTX A6000 desktop | WASM SIMD | 同上 | 同上 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| Android device A | WebGPU | 同上 | 同上 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| Android device A | WASM SIMD | 同上 | 同上 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |

系统补充表报告缓存重过滤、主线程应用可见实例差量和完整页面帧时间。不能把无 GLB 的纯推理结果写成完整场景 FPS，也不能把完整页面下载卡顿写成模型 forward。

## 输出和实现计划

实验名称固定为：

```text
pvs_v4_frontend_inference_latency_v1
```

输出目录：

```text
neural_instance_culling/benchmark/out/pvs_v4_frontend_inference_latency_v1/
  workload.json
  desktop/<device>/<backend>/run_*.jsonl
  mobile/<device>/<backend>/run_*.jsonl
  init/<device>/<backend>.jsonl
  summary.json
  paper_table.csv
```

需要实现的入口：

| 文件 | 责任 |
|---|---|
| `slm2viewer/scripts/build_pvs_runtime_workload.py` | 从正式 CSR 导出 684 pose 回放，不读取 GT 做选择 |
| `slm2viewer/src/PVSRuntimeBenchmark.js` | 直接驱动生产 Dispatcher，收集一次 session 的内存结果 |
| `slm2viewer/runtime-benchmark.html` | 无 GLB、无渲染的桌面/移动统一 benchmark 页面 |
| `slm2viewer/scripts/run_pvs_runtime_benchmark.mjs` | 桌面 Playwright 硬件门、三轮回放和日志 |
| `slm2viewer/scripts/run_android_pvs_runtime_benchmark.mjs` | Android CDP 控制、设备与温度元数据 |
| `slm2viewer/scripts/summarize_pvs_runtime_benchmark.py` | 分位数、分层 bootstrap、配对后端比较和论文表格 |

实现时先补 `queryE2EMs`，再增加可选 GPU timestamp。timestamp-query 不支持不能阻止正式端到端评价。所有长任务写日志；无论是否达到 50 ms 目标，都完成全部设备、后端和 session，并如实报告。

## 验收条件

- 桌面和至少一台真实移动设备均完成五个 session；
- 每个正式组合有 `5 × 684 = 3420` 条有效查询；
- timed run 关闭诊断概率和中间特征回读；
- WebGPU adapter 通过各平台硬件门，WASM 明确标记 SIMD；
- 同一 pose 的 WebGPU/WASM 输出集合通过预检；
- 主表同时给出候选规模、p50、p95 和置信区间；
- 10k p95 不可用时明确写出实际候选范围，不能插值或伪造；
- 移动端 p95 是否低于 50 ms 只决定工程目标是否通过，不取消实验或隐藏结果。

## 当前待办

当前桌面 NVIDIA WebGPU 硬件门已经可用，但正式 684 pose benchmark 尚未实现。移动设备型号和连接方式需要在执行前填入设备登记表。本文只冻结测试方法，不包含任何正式论文耗时结果。
