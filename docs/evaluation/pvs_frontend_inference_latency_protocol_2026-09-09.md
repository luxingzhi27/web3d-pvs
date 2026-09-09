# PVS V4 浏览器模型推理耗时评价协议

## 目标与口径

本实验测量当前 PVS V4 模型在桌面和真实移动设备上的 WebGPU 前向耗时。论文主指标是：一个 pose 的候选实例 ID 已经确定并上传 GPU 后，V4 网络对这些候选完成可见性分数计算所需的时间。

主指标不包含以下工作：

- 页面、工作负载和模型资产下载；
- WebGPU adapter/device、权重和固定实例表初始化；
- 后退 66 度 AABB 候选生成；
- 候选 ID 上传 GPU；
- 冻结阈值筛选、真实 60 度视锥过滤；
- 实例编号压缩、GLB 聚合和结果读回；
- Worker 消息、GLB 下载、解析、挂载和 Three.js 渲染。

因此，该数字只能写成“模型前向耗时”，不能写成完整 PVS 调度耗时或场景帧时间。生产路径的候选生成、筛选、压缩和 Worker 往返可在系统实验中另表报告。

## 测量实现

测试页复用生产 V4 的固定实例特征、模型权重和 WGSL 模型计算函数，不维护第二套网络公式。正式 dispatch 以当前 pose 的固定候选 ID 为索引，只执行以下过程：

```text
已上传的候选 ID
  -> 读取候选实例的 124 维固定运行特征
  -> 根据 pose 生成视点区域频谱矩与关系条件
  -> 查询逐实例校准遮挡生存场
  -> 共享可见性 MLP
  -> 写出每个候选的可见性分数
```

不在该 dispatch 中执行 AABB 相交、阈值比较、真实视锥过滤或 GLB 聚合。输出分数只写入 GPU storage buffer，正式计时不读回。

| 字段 | 定义 | 使用方式 |
|---|---|---|
| `gpuKernelMs` | WebGPU timestamp-query 记录的 compute pass 起止时间 | adapter 支持时的首选模型前向指标 |
| `submitCompletionMs` | `queue.submit()` 到 `queue.onSubmittedWorkDone()` 完成 | timestamp-query 不可用时的模型 dispatch 墙钟指标 |
| `modelInferenceMs` | `gpuKernelMs` 可用时取前者，否则取 `submitCompletionMs` | 页面显示和设备内汇总字段 |
| `modelAssetAndPipelineInitMs` | 下载资产、上传权重和创建管线 | 单独记录，不进入前向时间 |
| `workloadDownloadMs` | 下载 pose 元数据和候选 ID | 单独记录，不进入前向时间 |

论文表格必须注明每台设备的 `timingSource`。不能把 timestamp-query 与 submit-to-completion 混成同一组置信区间；需要跨设备统一比较时，同时给出所有设备都有的 `submitCompletionMs`。

## 固定工作负载

工作负载来自：

```text
neural_instance_culling/dataset/out/
pose_csr_hkust_v3_main_stratified_calibration_fov66_v1/
```

使用全部 `684` 个冻结 test view-cell。test 这里只提供固定性能负载，不选择模型、checkpoint 或阈值。每条记录包含：

- 原始 pose ID；
- view-cell 查询中心和归一化观察方向；
- 固定 `60` 度垂直 FOV、`16:9` aspect、near/far；
- 原数据集已有的后退 `66` 度候选实例 ID。

候选列表从 CSR 原样导出，不补入 GT，不由测试页重新生成。当前工作负载共 `3,174,148` 个候选引用，每 pose 平均 `4,640.57` 个，最少 `8` 个，最多 `18,831` 个。

生成命令：

```bash
cd slm2viewer
npm run build:benchmark-workload
```

## 自助测试网站

正式入口：

```text
https://139.196.34.161/pvs-runtime/
```

手机和电脑直接打开该 HTTPS 页面，不需要 ADB、CDP 或安装应用。页面流程如下：

1. 填写设备名称和设备型号/GPU。
2. 选择 `5` 轮论文正式测试并开始。
3. 页面先下载工作负载和模型资产，初始化 WebGPU；这些时间单列。
4. 每轮先执行覆盖候选规模的 `50` 个预热 pose，不计入结果。
5. 按固定种子打乱顺序，串行执行全部 `684` 个 pose。
6. 页面完成本机 p50/p95 汇总后，一次性上传全部样本。
7. 上传失败时点击“下载结果”，保留同一 JSON 供人工回收。

页面必须始终处于前台。进入后台、锁屏或切换应用会停止当前测试，避免把浏览器降频后的无效 session 混入正式结果。页面会申请 screen wake lock，但是否生效取决于浏览器和系统设置。

## 硬件门

- 页面必须由可信 HTTPS 上下文提供，局域网 IP 的普通 HTTP 不能作为移动端 WebGPU 正式入口。
- adapter 通过 `requestAdapter({ powerPreference: 'high-performance' })` 获得。
- 保存 `vendor`、`architecture`、`device` 和 `description`。
- adapter 信息为空，或包含 `SwiftShader`、`llvmpipe`、`softpipe`、`swrast`、`software` 时停止测试。
- WebGL renderer、Chrome 进程和 `nvidia-smi` 不能替代 WebGPU adapter 证据。
- 真实手机关闭省电模式，保持浏览器前台；正式设备记录型号、SoC/GPU、系统和浏览器版本。

Linux 桌面无头自动化继续使用 `chrome_gpu_flags.mjs` 的 NVIDIA Vulkan 路径；移动端正式数据由同一网页在真实设备本地运行。

## 重复次数与统计

每个设备执行五轮，每轮 `50` 次预热和 `684` 次正式查询，共 `3,420` 个正式样本。不得裁掉最慢 1%，不得只挑最快一轮。浏览器崩溃、adapter 丢失或页面进入后台时整轮作废。

每个设备至少报告：

- 候选数 mean/p95 和范围；
- `modelInferenceMs` p50/p95；
- `gpuKernelMs` p50/p95（支持 timestamp-query 时）；
- `submitCompletionMs` p50/p95；
- 9k 到 11k 候选子集的样本数和 p95；
- 模型运行资产大小；
- 模型/管线初始化耗时，明确不计入前向。

置信区间使用 `10,000` 次分层 bootstrap：先重采样 session，再在选中的 session 内重采样 pose，报告 p50/p95 的 95% 置信区间。移动端约 10k 候选 `p95 < 50 ms` 是工程目标，不是取消实验的门控。

汇总命令：

```bash
conda run -n slm_pvs python slm2viewer/scripts/summarize_pvs_runtime_benchmark.py \
  --input-dir neural_instance_culling/benchmark/out/pvs_v4_frontend_inference_latency_v1/browser_uploads
```

## 服务端结果

nginx 使用 `139.196.34.161` 的同名 IP SSL 证书提供静态页面，并把两个同源接口转发到只监听 `127.0.0.1` 的接收服务：

```text
POST /api/pvs-runtime-session
POST /api/pvs-runtime-results
```

服务端签发短期上传 token，限制请求来源和 8 MiB payload，校验 schema、设备名称、`684` pose 数量、候选范围和数值有限性。文件名与 receipt ID 只由服务端生成，不接受客户端路径。正式部署结果保存在 `/var/lib/pvs-runtime-benchmark/results/`，页面显示回执编号。

## 当前验证

2026-09-09 在本机 RTX A6000、NVIDIA Vulkan WebGPU adapter 上完成一轮实现 smoke：`684` 个 pose 全部执行并上传成功，GPU timestamp-query 的模型前向 `p50=0.237 ms`、`p95=0.287 ms`。模型资产和管线初始化约 `344 ms`，已单独记录且未计入前向指标。该轮用于验证代码和计时边界，不替代按设备登记完成的五轮论文正式结果。

同日完成 MacBook Air M2 和 vivo X200 Pro mini 的正式设备测试。每台设备执行 `5 × 684 = 3,420` 次 pose 查询，均使用硬件 WebGPU adapter 和 GPU timestamp-query，未发现候选数或输出集合契约错误。

| 设备 | Kernel p50 | 95% CI | Kernel p95 | 95% CI | 约 10k 候选 p95 |
|---|---:|---:|---:|---:|---:|
| MacBook Air M2 | 2.163 ms | [2.097, 2.163] | 13.242 ms | [12.390, 13.959] | 10.617 ms |
| vivo X200 Pro mini | 3.473 ms | [3.080, 3.801] | 18.350 ms | [17.826, 18.747] | 13.261 ms |

这里的 p50 是一半查询不超过的中位延迟，代表典型响应；p95 是 95% 查询不超过的尾部延迟，代表卡顿风险。两者都比算术平均值更不容易被少数极端样本混淆，论文应同时报告 p50 和 p95，而不能只给平均耗时。
