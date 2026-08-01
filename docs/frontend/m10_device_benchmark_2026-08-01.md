# M10 设备 Benchmark 审计

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

当前数据只能证明“桌面 SwiftShader 浏览器路径可启动并完成一次 smoke”。不能据此声称移动端实时、硬件 WebGPU 性能或 10k 候选满足 `p95 < 20 ms`。M10 保持 No-Go，后续需要在真实设备上按固定 commit、冻结模型、固定 FOV 和固定轨迹采集 p50/p95/p99；没有真实样本的候选规模桶必须标记缺失，不能补齐。

