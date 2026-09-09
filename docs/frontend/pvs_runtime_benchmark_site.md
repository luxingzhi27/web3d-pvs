# PVS V4 自助推理测试站

## 用途

该站点让桌面和真实移动设备直接在浏览器中执行两个论文场景的 PVS V4 WebGPU/WASM SIMD 候选前向，并自动回传结果。它不加载场景 GLB，不需要 ADB，不修改当前 Viewer 默认模型和调度路径。

正式地址：`https://139.196.34.161/pvs-runtime/`

## 组成

| 文件 | 责任 |
|---|---|
| `runtime-benchmark.html` | 设备登记、进度、结果和上传状态界面 |
| `src/PVSRuntimeBenchmark.js` | 选择场景和后端，驱动候选专用前向，汇总并上传 |
| `scripts/build_paper_runtime_workloads.py` | 从 HKUST/IFCBench 冻结 test CSR 生成场景清单和负载 |
| `scripts/build_pvs_runtime_workload.py` | 导出一个场景的 pose、实际 aspect 与候选 ID |
| `scripts/run_paper_runtime_benchmark.mjs` | 用登记的 NVIDIA 无头 Vulkan 参数自动执行桌面测试并留存硬件证据 |
| `scripts/runtime_benchmark_server.mjs` | 静态调试服务、短期上传会话和结果校验/落盘 |
| `scripts/summarize_pvs_runtime_benchmark.py` | session 聚类 bootstrap 和论文表格导出 |

WebGPU 候选专用 dispatch 与生产推理共用模型计算函数；WASM 使用 `benchmark_candidates_v4` 对预先写入线性内存的同一候选 ID 执行相同网络。两条 benchmark 路径均不执行候选生成、阈值筛选、GLB 聚合或结果回读。

## 本地运行

```bash
cd slm2viewer
npm run build:benchmark-workload
npm run build
PVS_BENCHMARK_PORT=8321 npm run serve:runtime-benchmark
```

本机 Chrome 可访问 `http://127.0.0.1:8321/runtime-benchmark.html`。其他设备必须使用可信 HTTPS 部署地址，否则 `navigator.gpu` 通常不可用。

## 部署

生成独立部署目录：

```bash
cd slm2viewer
npm run package:runtime-benchmark
```

目录 `public_runtime_benchmark/` 只包含测试页、页面 bundle、两个 V4 模型资产、WASM 内核和固定工作负载，不包含场景 GLB。远端 nginx 将 `/pvs-runtime/` 映射到该目录，将 `/api/pvs-runtime-session` 和 `/api/pvs-runtime-results` 反向代理到本机 `8321` 端口。接收服务以 systemd 运行，结果目录为 `/var/lib/pvs-runtime-benchmark/results/`。

仓库中的正式服务配置位于 `scripts/deployment/pvs-runtime-benchmark.service` 和 `scripts/deployment/nginx-139-pvs-runtime.conf`。nginx 配置复用 `/etc/ssl/139.196.34.161/` 中的 IP 证书，并保留根路径现有的 FileBrowser 代理。

部署后检查：

```bash
curl -fsS https://139.196.34.161/pvs-runtime/
curl -fsS https://139.196.34.161/api/pvs-runtime-health
```

回收服务端测试结果：

```bash
rsync -av -e 'ssh -i ~/.ssh/id_ed25519_slm_deploy -o IdentitiesOnly=yes' \
  root@139.196.34.161:/var/lib/pvs-runtime-benchmark/results/ \
  neural_instance_culling/benchmark/out/pvs_v4_frontend_inference_latency_v1/browser_uploads/
```

WebGPU 模式必须显示真实 adapter 并通过软件 adapter 拒绝检查；WASM 模式必须报告 `wasm-simd-v4`。上传成功后页面会显示服务端 receipt ID；上传失败时由“下载结果”按钮回收 JSON。
