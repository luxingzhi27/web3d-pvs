# M10 设备 Benchmark 审计与执行方案

日期：2026-08-01  
状态：桌面 headless smoke 通过；真实设备质量门未通过。

## 当前可复核环境

- Chrome 146 和 Playwright 可运行。
- WebGPU API 能初始化，但 headless Chrome 实际使用 SwiftShader，不能代表 RTX A6000 的硬件 WebGPU。
- 当前桌面 smoke 使用 worker WebGPU，页面错误为 0，Cold-0 请求顺序正确，目标 GLB 响应有效。
- HKUST 5,959 候选的一次 smoke 记录 WebGPU 推理约 1,504 ms、总预测约 1,515 ms。
- Metropolis 34,206 候选的一次 smoke 记录 WebGPU 推理约 4,922 ms、总预测约 4,930 ms。

证据文件：`/tmp/m10_desktop_hkust_current_20260801.json`、`/tmp/m10_desktop_metropolis_webgpu_20260801.json`。它们是单次 smoke，不是 p50/p95/p99 分布。

## 缺失项

- 没有硬件独显 WebGPU 与集成 GPU 的可比浏览器测量；服务器上的 RTX A6000 没有被当前 headless Chrome 使用。
- 没有 Android 设备、ADB、Android SDK 或两个性能档位。
- 没有正式冷缓存/温缓存、多轨迹、候选规模分桶和真实网络轨迹。
- 现有采集器没有稳定分离空间候选生成、GPU dispatch、readback、GLB 解码/上传、矩阵更新、主线程附加时间、帧时间、峰值内存和能耗。
- `playwright` 在当前 node_modules 可用，但尚未作为 fresh clone 的明确依赖冻结。

## 门控判断

当前数据只能证明“桌面 SwiftShader 浏览器路径可启动并完成一次 smoke”。不能据此声称移动端实时或硬件 WebGPU 性能，也不能用它判断 10k 候选是否满足移动端延迟目标。M10 保持 No-Go，后续需要在真实设备上按固定 commit、冻结模型、固定 FOV 和固定轨迹采集 p50/p95/p99；没有真实样本的候选规模桶必须标记缺失，不能补齐。

## 移动设备执行方案

本节是后续真实设备到位后的预注册测试方案。当前不产生移动端数值，也不使用桌面 SwiftShader 结果填充移动端分布。

### 1. 设备与软件冻结

准备两个性能档位，每档至少一台真实 Android 手机：

| 档位 | 设备要求 | 用途 |
|---|---|---|
| 高性能 | 近两代旗舰 SoC，硬件 WebGPU 可用 | 估计高端移动设备上限 |
| 中端 | 中端 SoC，硬件 WebGPU 可用 | 检查实际用户下限 |

每台设备记录型号、SoC、GPU、Android 版本、Chrome 版本、WebGPU adapter、屏幕分辨率、刷新率、电量、温度和是否省电模式。测试固定一个 git commit、模型导出目录、场景资产版本、查询 FOV `66°`、真实渲染 FOV `60°` 和 viewport；任何版本变化都建立新的测试批次。

设备通过 USB 调试连接，使用 Chrome 远程调试或 Android Chrome 的 DevTools Protocol。开始测试前必须确认页面实际使用 `worker-webgpu` 和硬件 adapter，并记录没有走 CPU/AABB fallback；无法确认时该批次只作为环境 smoke，不进入性能表。

### 2. 缓存、网络和导航轨迹

每个场景准备三条固定、可回放的相机轨迹，覆盖近建筑、开阔区域和快速转向。每条轨迹包含固定时间戳、相机位置、朝向和 view-cell 边界穿越事件。

每条轨迹分别运行：

- 冷缓存：清除 Service Worker、HTTP 缓存、IndexedDB、GPU 资源和页面状态后启动；
- 温缓存：保留上一轮已下载、解码和上传资源，只重载页面；
- 网络档位：固定 Wi-Fi 与受控 4G trace，记录吞吐、RTT、丢包和请求并发。

冷缓存统计必须把模型权重、实例元数据、首个特征页和必要索引计入启动字节；不能只统计 GLB 字节。轨迹和网络 trace 文件应保存 SHA-256，确保不同方法使用完全相同输入。

### 3. 候选规模与重复次数

通过固定场景/空间页和受控候选截取，覆盖 `256、512、1k、2k、4k、8k、10k、16k` 候选桶。截取只能从严格后退视锥候选中按预注册规则进行，不能改动 GT 或补入不可执行的实例；没有真实样本的桶报告为缺失。

每个设备、场景、缓存状态、网络档位、轨迹和候选桶至少运行 30 次有效重复。前 5 次作为 warm-up，不进入统计；剩余运行计算 p50、p95、p99 和 95% bootstrap 区间。设备温度过高、页面错误、WebGPU adapter 改变、请求返回 HTML 或发生兼容降级的运行必须标记无效并保留原因，不能静默删除。

### 4. 分项计时和系统观测

页面和 Worker 统一写入带 commit、设备、轨迹和运行编号的 JSONL。至少记录：

1. 空间候选生成与特征页命中；
2. Worker 到 WebGPU 的 buffer 更新；
3. compute dispatch、GPU 查询和结果 compact/readback；
4. CPU 实例级结果处理、GLB 聚合和 top-k 队列；
5. 实例矩阵更新、首个可见决策和首个有用画面；
6. GLB 下载、解码、GPU 上传及取消请求；
7. 总调度延迟、主线程附加时间、帧时间和掉帧；
8. JS heap、GPU/纹理内存、设备温度、电量和功耗（系统允许时）。

同时保存 Chrome trace/Perfetto、`adb logcat`、`dumpsys meminfo`、`dumpsys gfxinfo`、温度/降频信息和网络请求日志。首轮 dispatch 前不得出现目标 GLB 三角形、材质或纹理请求；该规则通过请求日志和 GPU buffer 清单双重检查。

### 5. 通过与报告规则

正式报告按设备和场景分别给出冷/温缓存的 p50/p95/p99，不把不同设备或不同候选桶平均成一个数字。系统目标为 10k 候选下移动端 WebGPU 查询 p95 `<50 ms`、主线程附加工作 `<2 ms`，且没有明显帧停顿；图像安全仍必须满足独立的 weighted recall 和 miss-pixel 约束。

若设备不支持硬件 WebGPU、只能连接模拟器，或无法取得 Android 设备，本阶段只报告“设备证据缺失”，保留桌面 smoke 和完整采集协议；不得据此声称移动端实时。拿到设备后先运行小规模 schema/adapter smoke，再启动上述完整矩阵。

### 6. 拿到设备后的固定执行顺序

测试不直接从完整矩阵开始，必须按以下顺序逐级放行：

1. **冻结构建。** 记录 `git rev-parse HEAD`、前端构建产物的 SHA-256、模型权重/特征表/运行元数据的 SHA-256、场景资产版本、模型阈值、查询 FOV `66°`、渲染 FOV `60°` 和 viewport。随后只使用这一份构建完成整个批次。
2. **连接设备。** 设备打开 USB 调试并保持非省电模式，记录电量和初始温度。典型检查命令为：

   ```bash
   adb devices -l
   adb shell getprop ro.product.model
   adb shell getprop ro.board.platform
   adb shell getprop ro.build.version.release
   adb forward tcp:9222 localabstract:chrome_devtools_remote
   curl http://127.0.0.1:9222/json/version
   ```

   `adb`、远程调试端点或硬件 WebGPU 任一项不可用时，只做环境记录，不进入性能统计。
3. **硬件适配器预检。** 用 Android Chrome 打开固定 URL，通过 DevTools Protocol 读取 `navigator.gpu.requestAdapter()`、适配器信息、页面暴露的 PVS backend 和模型初始化状态。必须确认 backend 为 `worker-webgpu`/`webgpu`，适配器为真实移动 GPU，且没有 CPU、AABB 或其他降级路径；同时检查页面错误、WebGPU device lost、shader 编译错误和资源请求错误。
4. **小规模 schema smoke。** 每个场景先执行 3 次、每次 256 个候选的预测，检查输出长度、实例 ID 范围、分数是否有限、冻结阈值下结果是否可复现、页面是否无错误，以及首轮模型 dispatch 前没有目标 GLB 的三角形/材质/纹理请求。此步骤失败则停止该设备，不执行大矩阵。
5. **功能一致性 smoke。** 使用 M12 固定 cases 对比 PyTorch FP16 参考与设备 WebGPU 输出，记录可见性和下载优先级的最大绝对误差、阈值翻转率、实例集合 Jaccard 和 GLB 排序相关性。误差门限先由桌面真实 WebGPU 的 M12 结果冻结；在门限尚未冻结前只能标记为 exploratory，不能写成通过。
6. **正式矩阵。** 依次执行冷缓存、温缓存和网络 trace 条件；每个条件按三条轨迹和候选规模桶运行，保留完整原始记录。不要在同一批次中更新页面代码、模型、资产、Chrome 或设备系统。

Android Chrome 的自动化可以通过 `adb forward` 暴露的 DevTools Protocol 连接；后续实现的采集器应支持类似
`chromium.connectOverCDP('http://127.0.0.1:9222')` 的连接方式，而不是在服务器上启动桌面 Chromium 代替真实设备。

### 7. 原始数据格式与有效运行判定

每次运行写一份 JSONL，推荐目录为：

```text
benchmark/out/m10_mobile/<commit>/<device>/<scene>/<cache>/<network>/<trajectory>/run-001.jsonl
```

每条记录至少包含以下字段：

| 字段 | 含义 |
|---|---|
| `schema`, `commit`, `runId` | 记录格式、代码提交和运行唯一标识 |
| `device`, `android`, `chrome`, `adapter` | 设备、系统、浏览器和硬件 WebGPU 适配器 |
| `scene`, `trajectory`, `cache`, `network` | 场景、轨迹、缓存状态和网络条件 |
| `candidateCount`, `candidateHash`, `featurePageHitRate` | 后退视锥候选数量、候选集合校验值和特征页命中率 |
| `timings` | 候选生成、buffer 更新、dispatch、readback、实例聚合、调度、首个有用画面和 GLB 解码/上传耗时 |
| `frame` | p50/p95 统计前的逐帧耗时、掉帧和长帧计数 |
| `memory`, `temperature`, `battery` | JS heap、GPU/纹理内存、温度、电量及可取得的功耗信息 |
| `errors`, `deviceLost`, `fallback` | 页面错误、设备丢失和任何降级路径 |
| `valid`, `invalidReason` | 是否进入统计以及被排除的明确原因 |

有效运行必须同时满足：硬件 adapter 未变化、无页面错误和 device lost、模型输出有限且候选集合哈希匹配、无兼容降级、请求没有返回 HTML/错误页面、场景没有被浏览器强制回收。温度过高、设备降频、网络超时和 OOM 不能静默删除，必须作为无效原因单独统计。

### 8. 统计、质量门和交付物

对每一个“设备 × 场景 × 缓存 × 网络 × 轨迹 × 候选桶”单元，先去除预注册的 warm-up，再对有效运行计算 p50、p95、p99 和 95% bootstrap 区间。bootstrap 的重采样单位是完整运行，不是单个 dispatch 或单帧；有效运行少于 25 次时只报告缺失，不补造分位数。

质量门分为三类：

- **兼容性门：** 硬件 WebGPU、Worker 推理、实例级过滤和 GLB 调度均可运行，无降级、无 device lost、无 OOM。
- **正确性门：** M12 与参考输出的误差、阈值翻转、实例集合和 GLB 排序达到已冻结的 parity 门限；同一冻结阈值下仍满足模型独立报告中的 weighted recall 与 miss-pixel 安全约束。移动端不得为了性能改变阈值或候选语义。
- **性能门：** 10k 候选时 WebGPU 查询 p95 `<50 ms`，主线程附加工作 p95 `<2 ms`，不出现连续长帧；同时报告首个有用画面、GLB 字节、峰值内存和温度变化。该数值是当前建议目标，不是已完成的设备实测结论。

最终交付两份文件：一份不删减的原始 JSONL/trace 清单，一份按设备和场景分开的结果报告 `m10_mobile_results_<date>.md`。报告必须明确区分“实测通过”“设备证据缺失”和“未达到目标”，不得把桌面 SwiftShader、模拟器或单次 smoke 填入移动端 p50/p95/p99。
