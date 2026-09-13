# 端侧性能论文产物（2026-09-13）

## 口径

新增 `slm2viewer/scripts/generate_pvs_runtime_paper_outputs.py`，读取现有 `pvs-v4-browser-runtime-result-v1` 浏览器上传 JSON。只有完整 test workload、每轮 50 个 warmup、恰好 5 个 session、无页面后台违规且通过 WebGPU/WASM 后端门的上传进入 `formal`；一轮自动化结果和文件/设备名带 `smoke` 的结果明确排除。统计对象是页面记录的 `modelInferenceMs`，不是下载、初始化、候选构造、过滤、GLB 聚合或消息往返。

正式输入中，MacBook Air M2 与 vivo X200 Pro mini 各有 HKUST WebGPU 的 `5 × 684`
次 pose 查询；RTX A6000 已完成 HKUST、IFCBench、Sponza、Big City 和 Viking
Village 全部 frozen test workload 的 WebGPU/WASM 测量。IFCBench 移动端组合仍标为
`unavailable`，没有用桌面结果补齐。

均值、p50、p95 均按 session 重采样、再在选中 session 内按 pose 重采样，使用 10,000 次 bootstrap 给出 95% CI。候选桶为 `0-1k`、`1-2k`、`2-5k`、`5-10k`、`10-15k`、`15k+`。每个 formal 组合另拟合 `T(N)=a+bN`，报告截距、每候选斜率、RMSE、MAE、R² 和最大绝对误差。

## 结果

| 设备 | 场景 | 后端 | 轮数 | mean ms (95% CI) | p50 ms (95% CI) | p95 ms (95% CI) |
|---|---|---|---:|---:|---:|---:|
| MacBook Air M2 | HKUST | WebGPU | 5 | 4.183 (4.010, 4.353) | 2.163 (2.097, 2.163) | 13.242 (12.390, 13.966) |
| vivo X200 Pro mini | HKUST | WebGPU | 5 | 6.466 (6.031, 6.891) | 3.473 (3.080, 3.801) | 18.350 (17.826, 18.809) |
| RTX A6000 | HKUST | WebGPU kernel | 5 | 1.661 | 1.568 | 3.555 |
| RTX A6000 | HKUST | WASM SIMD | 5 | 25.565 | 3.700 | 100.500 |
| RTX A6000 | IFCBench | WebGPU kernel | 5 | 1.726 | 1.385 | 3.590 |
| RTX A6000 | IFCBench | WASM SIMD | 5 | 57.379 | 46.800 | 143.200 |
| RTX A6000 | Sponza | WebGPU kernel | 5 | 1.249 | 1.602 | 1.686 |
| RTX A6000 | Sponza | WASM SIMD | 5 | 1.027 | 0.900 | 2.400 |
| RTX A6000 | Big City | WebGPU kernel | 5 | 1.267 | 1.585 | 1.668 |
| RTX A6000 | Big City | WASM SIMD | 5 | 4.854 | 1.400 | 15.900 |
| RTX A6000 | Viking Village | WebGPU kernel | 5 | 1.182 | 1.434 | 1.663 |
| RTX A6000 | Viking Village | WASM SIMD | 5 | 4.222 | 2.350 | 11.900 |
| 移动设备 | IFCBench | WebGPU/WASM | 0 | unavailable | unavailable | unavailable |

CSV 中同时保留三项 CI 的上下界；上表不复制长浮点数，避免把摘要和源数据混成另一份口径。生成图展示正式样本的候选桶 p50/p95 和线性拟合散点。

两组全候选拟合为：M2 `a=1.3869 ms, b=0.0006026 ms/candidate, RMSE=1.3885 ms, R²=0.8843`；vivo `a=2.6334 ms, b=0.0008259 ms/candidate, RMSE=2.7469 ms, R²=0.7857`。这些是模型前向的拟合误差，不是完整前端帧时间。

## 产物与复现

在 Linux CPU 环境运行，不启动浏览器或 GPU：

```bash
conda run -n slm_pvs python slm2viewer/scripts/generate_pvs_runtime_paper_outputs.py \
  --input-dir neural_instance_culling/benchmark/out/paper_results/mobile_runtime/a6000_formal \
  --input-dir /mnt/sda/rhyang/slm/neural_instance_culling/benchmark/out/paper_results/mobile_runtime/browser_uploads \
  --input-dir /mnt/sda/rhyang/slm/neural_instance_culling/benchmark/out/pvs_v4_frontend_inference_latency_v1/browser_uploads \
  --output-dir neural_instance_culling/benchmark/out/paper_results/mobile_runtime
```

输出目录包含 `runtime_paper_source.csv`、`runtime_paper_summary.csv`、`runtime_paper_candidate_bins.csv`、`runtime_paper_summary.json` 以及同名 `runtime_paper_latency.pdf`、`.svg`、`.png`。源上传中的 smoke 记录只在汇总 JSON 的排除清单中登记，不进入源 CSV、统计或图形。

测试：

```bash
conda run -n slm_pvs python slm2viewer/scripts/test_generate_pvs_runtime_paper_outputs.py
```
