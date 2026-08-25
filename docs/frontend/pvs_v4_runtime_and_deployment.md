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

源码配置中的 HKUST GLB 根地址固定为 `https://www.liteweb3d.com/data/hkust-v3/`；默认场景和显式 `hkust-v3` 入口使用同一地址。`resourcesBaseUrl` 仍指向本地的小型场景元数据目录，只有实际 GLB、纹理等大型构件资源从该远端根地址下载。

## 单次预测数据流

1. 主线程把真实相机位置、旋转、宽高比和裁剪范围发送给 Worker。
2. Worker 从真实相机建立后退 `3.4641 m`、垂直 FOV `66°` 的候选相机，并把候选相机和真实 `60°` 相机的十二个视锥平面写入 WebGPU 常量缓冲。
3. 第一个 GPU 计算阶段以线程编号作为实例编号，对全部实例执行后退视锥 AABB 测试；只有通过测试的实例才运行 V4 查询网络。浏览器不会展开离线 subpose，也不会在线查询邻居。
4. 同一 GPU 线程完成阈值判断、后退区域可见实例压缩和真实 `60°` 视锥过滤，并以原子最大值把实例可见性分数聚合到所属 GLB。
5. 第二个 GPU 计算阶段遍历 GLB 聚合表，压缩得到下载队列及其最高可见性分数。
6. 普通运行只回读后退区域可见实例编号、真实视锥最终实例编号和 GLB 下载队列，不回读逐候选概率。Worker 只做队列排序与数量限制，主线程只更新实例化渲染状态。

未达到阈值但分数不低于预取阈值 `0.04` 的实例可以参与 GLB 预取。当前 checkpoint 没有独立下载头，GLB 优先级由所属实例的最高可见性概率聚合得到；这属于当前部署实现，不应描述成模型已经学习了独立资源效用。

HKUST 普通模式的固定回读布局为 `4 + 2×18831 + 3×3273` 个 32 位字，即 `189940 bytes`。其中只预留最终实例列表和 GLB 队列容量，不包含逐候选概率。开启 `neuralDebugLogs=true` 时才额外回读候选编号、概率和中间特征，用于数值 parity；调试模式的传输量和延迟不能代表生产运行。

## 冻结结果检查

2026-08-25 修正了调试面板的冻结语义。冻结按钮保存当前一次 GPU 查询经过真实 `60°` 相机视锥过滤后的最终实例编号和最终 GLB 编号。实例编号决定实际显示，GLB 编号只决定需要下载哪些资源；冻结期间不再请求全场 GLB，也不再显示全部已驻留对象。

冻结后即使移动检查相机，Loader 仍保持这份实例级快照。冻结前已经开始但尚未返回的预测会被丢弃，后续才下载完成的实例化 GLB 也会立即按照同一份冻结实例编号压缩实例矩阵。解除冻结后恢复自动缓存调度并强制发起一次新预测。该修正涉及 `src/viewer.js`、`slm2/SLM2Loader.js`、`src/RenderVisibilitySystem.js` 和 `scripts/test_current.mjs`，不修改模型、阈值、运行特征表或场景元数据。

静态契约要求冻结入口只能读取 `renderComponentIds` 和 `renderGlbIds`，并禁止冻结分支调用全驻留显示或读取后退视锥原始实例集合。验证命令为 `cd slm2viewer && npm test && npm run build`。

HKUST 页面功能 smoke 中，冻结前的最终集合为 `3769` 个实例和 `1268` 个 GLB；冻结后页面快照、Loader 快照和实例矩阵过滤集合完全一致，下载工作集保持 `1268/3273` 个 GLB。检查相机平移 `1000 m` 后实例集合仍未变化。该 smoke 没有记录 WebGPU 硬件性能数据。

## 代码边界

| 模块 | 责任 |
|---|---|
| `src/InstancePVS.js` | V4 资产校验、GPU AABB 候选、WGSL 查询、GPU 压缩和 GLB 聚合 |
| `src/LightweightPVSWorker.js` | 构造 66°/60° 相机、排序 GPU 下载队列并传递最终结果 |
| `src/LightweightPVSDispatcher.js` | 相机快照、Worker 生命周期和请求串行号 |
| `src/neuralCullingBackendMode.js` | 只为有 V4 资产的场景启用神经模式 |
| `slm2/SLM2Loader.js` | 实例级显示状态和 GLB 下载队列接入 |
| `scripts/test_current.mjs` | 当前单模型静态契约检查 |
| `scripts/capture_v4_frontend_parity.mjs` | 从真实 V4 页面采集一次候选、概率和 WebGPU 后端证据 |
| `scripts/verify_v4_frontend_parity.py` | PyTorch 与 WebGPU 同位姿数值比较 |

HKUST 映射到 `pvs_mainline_v4`。Metropolis 尚未导出 V4 权重，因此明确使用实例 AABB 视锥模式；不得复用 HKUST 权重或回退到旧神经模型。

## 数值一致性

WGSL 必须与训练端依次对齐九维中心视角、`9×2` 视点区域轴、64 维频谱矩和最终概率。频谱方差采用数值稳定的等价公式，避免 GPU 三角函数微小误差在相近大数相减时被放大。

2026-08-25 的 GPU 融合同位姿检查覆盖 `5959` 个候选：

| 项目 | 结果 |
|---|---:|
| 平均概率绝对误差 | `0.0000039` |
| 最大概率绝对误差 | `0.0005391` |
| 阈值判定差异 | `0` |
| Worker 可见集合差异 | `0` |
| GPU 候选与 Three.js CPU 视锥集合差异 | `0` |
| GPU 60° 最终集合与 Three.js CPU 参考差异 | `0` |

无头 Chrome 的 WebGPU adapter 回报 `google/swiftshader`，因此该次运行只证明数值正确，不构成硬件性能数据。

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

## 渲染位图边界

当前场景渲染使用 Three.js `WebGLRenderer`，神经查询使用 WebGPU。浏览器没有让 WebGL 着色器直接读取 WebGPU storage buffer 的零拷贝互操作接口。把可见性位图从 WebGPU 回读后再上传为 WebGL 纹理不会减少跨设备传输，而且隐藏实例仍会进入顶点阶段；因此本版继续把 GPU 压缩后的最终实例编号交给现有实例矩阵压缩逻辑，它会实际降低 `InstancedMesh.count` 和绘制实例数。

只有把场景渲染整体迁移到 Three.js `WebGPURenderer` 后，渲染着色器才能与可见性计算共享 GPU 位图。该迁移会同时影响材质、后处理、加载器和浏览器兼容性，必须作为独立前端项目验证，不能在当前 WebGL 主线中加入一次 GPU 回读再上传的伪共享路径。
