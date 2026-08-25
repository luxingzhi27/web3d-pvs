# PVS V4 前端运行与部署

更新时间：2026-08-25

本文是当前前端神经剔除的唯一运行说明。浏览器只接受 V4 运行包；旧方向代理、dynamic-pool、相机哈希、空间分页和二阶段后端升级接口已经从当前代码与部署包移除。

## 当前模型

HKUST 使用 `pvs_mainline_v4`，运行 schema 为：

```text
pvs-bounded-relation-prior-instance-calibrated-moment-runtime-v4
```

校准阈值为 `0.6800000071525574`，由 checkpoint 自己的 calibration split 冻结。该工作点的 aggregate weighted recall 为 `0.9978366`，单侧置信下界为 `0.9963827`。当前导出对应 seed `20260802` 的无对比尾部分离消融成员，最佳安全 checkpoint 位于 epoch 36；前端不得自行重选阈值。

运行时每个实例保存 `96` 维几何特征和 `28` 维已融合的逐实例生存场系数，共 `124` 维 FP16 固定特征。离线分层关系网络、逐实例校准、点云编码和遮挡证据生成均不在浏览器执行。

## 运行资产

目录：

```text
slm2viewer/assets/neural_instance_culling/pvs_mainline_v4
```

文件包括：

| 文件 | 用途 |
|---|---|
| `model_meta.json` | schema、阈值、FOV、维度和校准信息 |
| `instance_runtime_features_fp16.bin` | 逐实例 124 维固定特征 |
| `instance_aabb_fp32.bin` | 逐实例世界 AABB |
| `instance_to_glb_uint32.bin` | 实例到下载 GLB 的映射 |
| `query_weights_fp16.bin` | V4 轻量查询网络权重 |
| `frequency_cycles_fp32.bin` | 16 组视点区域频率 |
| `chi_table_fp32.bin` | 视点区域矩的固定查表数据 |

六个二进制文件合计约 `5.03 MiB`，加元数据约 `5.48 MB`。运行包不包含 checkpoint、训练点云、关系边、子视点、邻居表或场景 GLB。

## 单次预测数据流

1. 主线程把真实相机位置、旋转、宽高比和裁剪范围发送给 Worker。
2. Worker 从真实相机建立后退 `3.4641 m`、垂直 FOV `66°` 的候选相机，并使用导出的实例 AABB 生成候选集合。
3. V4 WebGPU 查询以真实相机为视点区域中心，对全部候选执行一次批量推理。浏览器不会展开离线 subpose，也不会在线查询邻居。
4. 模型以阈值 `0.68` 产生后退视点区域内的实例级可见集合。
5. Worker 再用真实 `60°` 相机过滤实例 AABB，得到当前帧允许显示的实例编号。
6. 主线程只把这组实例编号写入实例化渲染状态。GLB 编号只用于下载、解码和缓存聚合，不能控制同一 GLB 内全部实例同时显示。

未达到阈值但分数不低于预取阈值 `0.04` 的实例可以参与 GLB 预取。当前 checkpoint 没有独立下载头，GLB 优先级由所属实例的最高可见性概率聚合得到；这属于当前部署实现，不应描述成模型已经学习了独立资源效用。

## 代码边界

| 模块 | 责任 |
|---|---|
| `src/InstancePVS.js` | V4 资产校验、AABB 候选、WGSL 查询和概率解析 |
| `src/LightweightPVSWorker.js` | 66° 候选相机、一次批量推理、60° 最终过滤和 GLB 聚合 |
| `src/LightweightPVSDispatcher.js` | 相机快照、Worker 生命周期和请求串行号 |
| `src/neuralCullingBackendMode.js` | 只为有 V4 资产的场景启用神经模式 |
| `slm2/SLM2Loader.js` | 实例级显示状态和 GLB 下载队列接入 |
| `scripts/test_current.mjs` | 当前单模型静态契约检查 |
| `scripts/capture_v4_frontend_parity.mjs` | 从真实 V4 页面采集一次候选、概率和 WebGPU 后端证据 |
| `scripts/verify_v4_frontend_parity.py` | PyTorch 与 WebGPU 同位姿数值比较 |

HKUST 映射到 `pvs_mainline_v4`。Metropolis 尚未导出 V4 权重，因此明确使用实例 AABB 视锥模式；不得复用 HKUST 权重或回退到旧神经模型。

## 数值一致性

WGSL 必须与训练端依次对齐九维中心视角、`9×2` 视点区域轴、64 维频谱矩和最终概率。频谱方差采用数值稳定的等价公式，避免 GPU 三角函数微小误差在相近大数相减时被放大。

2026-08-25 的同位姿检查覆盖 `6015` 个候选：

| 项目 | 结果 |
|---|---:|
| 平均概率绝对误差 | `0.0000042` |
| 最大概率绝对误差 | `0.0009083` |
| 阈值判定差异 | `0` |
| Worker 可见集合差异 | `0` |

功能 smoke 可以使用软件 WebGPU，但不能报告为硬件性能。正式 WebGPU 延迟必须读取 adapter 信息并通过 NVIDIA/Vulkan 硬件门；WebGL 硬件证据不能替代 WebGPU adapter 证据。

## 验证与打包

```bash
cd slm2viewer
npm test
npm run build
npm run package:deploy -- --scene hkust-v3
```

同位姿数值检查分为页面采集和 PyTorch 对照两步：

```bash
node slm2viewer/scripts/capture_v4_frontend_parity.mjs \
  --viewer-dir slm2viewer/public \
  --out /tmp/pvs_v4_capture.json \
  --allow-software-gpu

conda run -n slm_pvs python slm2viewer/scripts/verify_v4_frontend_parity.py \
  --checkpoint <best_safe.pt> \
  --asset-dir slm2viewer/assets/neural_instance_culling/pvs_mainline_v4 \
  --capture /tmp/pvs_v4_capture.json
```

正式硬件性能采集必须把 `--allow-software-gpu` 换成 `--require-hardware-gpu`，且只有 capture
中的 WebGPU adapter、WebGL renderer 和同窗口 NVIDIA 证据共同通过时才可报告硬件耗时。

生产构建必须只携带 `pvs_mainline_v4`。部署脚本会检查运行 schema、文件集合、实例数和场景元数据；不再打包旧模型目录。
