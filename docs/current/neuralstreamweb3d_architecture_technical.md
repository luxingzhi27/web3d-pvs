# NeuralStreamWeb3D 当前架构

更新时间：2026-09-09

当前系统把重型遮挡关系学习放在离线阶段，浏览器运行路径由固定实例表、一次视角查询、实例级显示和 GLB 级下载聚合组成。HKUST 使用 V4；未配置 V4 资产的场景使用实例 AABB 视锥模式。

## 端到端结构

```text
实例化场景与 GLB 表面点
        │
        ├─ view-cell Color-ID 采样：可见集合、可见权重、66° 后退候选
        ├─ 实例几何编码：96 维视角无关几何
        └─ 分层遮挡关系与逐实例校准：28 维生存场
                         │
                  V4 可见性模型训练
                         │
             calibration 冻结安全阈值
                         │
        124 维固定实例表 + 轻量查询权重
                         │
真实相机 ──> 66°/60° 视锥平面 ──> 一次 Worker/WebGPU 全实例批量查询
                         │
             GPU 候选、阈值与实例压缩
                         │
            GPU 按 GLB 聚合最高可见性
                         │
          实例级渲染 + GLB 下载优先队列
```

## 离线模型

当前 V4 的三个研究模块是：

- 分层遮挡关系先验与逐实例校准生存场：真实遮挡边用于学习共享关系规律，每个实例再保存校准后的 `4×7` 系数。浏览器只读取融合结果，不运行关系网络。
- 视点区域矩频谱查询：九维中心视角和 `9×2` 区域轴经 16 组频率生成 64 维区域频谱矩，表达 view-cell 内位置变化对实例可见性的影响。
- pose 平衡、加权召回保护和困难边界间隔组成的主线损失：阈值由 calibration 冻结，安全门以 weighted recall 及其置信下界为准。

前端固定实例特征为 `96 + 28 = 124` 维 FP16。最终查询头输入为 `130` 维，隐藏层宽度为 `64`。训练端使用的点云、关系边、子视点和校准残差均已在导出时折叠，不进入运行包。

## 浏览器运行契约

候选相机和查询相机承担不同职责：

- 候选相机从真实相机沿反方向后退 `3.4641 m`，垂直 FOV 为 `66°`，只用于确定可能参与查询的实例。
- 查询中心是当前真实相机，渲染 FOV 为 `60°`。模型对整个 view-cell 的保守可见并集打分，最终显示再由真实视锥收紧。
- 浏览器每次预测只执行一个候选批次；多个同方向 subpose 只存在于离线 GT 构建和图像评价中。

Worker 只构造两个相机并提交一次 WebGPU 查询。GPU 直接遍历实例 AABB，依次完成后退视锥候选判断、V4 推理、阈值筛选、真实视锥过滤、实例编号压缩和按 GLB 的最高分聚合。普通运行不回读全部候选编号或概率；Worker 只对压缩后的 GLB 队列排序和限额，主线程只接收最终实例编号与下载计划。

显示粒度始终是实例。一个 GLB 可以被多个实例复用，下载任一 GLB 不会自动显示其全部实例。当前 V4 没有独立下载头，下载优先级暂由实例可见性概率按 GLB 取最大值；独立资源效用学习仍属于后续研究目标。

## 当前代码入口

| 环节 | 入口 |
|---|---|
| 模型结构 | `neural_instance_culling/model/pvs_model.py` |
| 训练 | `neural_instance_culling/model/train_pvs.py` |
| 导出 | `neural_instance_culling/model/export_pvs.py` |
| 评价 | `neural_instance_culling/benchmark/evaluate_pvs.py`、`neural_instance_culling/benchmark/run_pvs.py summarize`、`neural_instance_culling/benchmark/summarize_core_ablation.py` |
| 前端查询 | `slm2viewer/src/InstancePVSBase.js`、`slm2viewer/src/InstancePVSWebGPU.js`、`slm2viewer/src/InstancePVSWebGPUShaders.js`、`slm2viewer/src/InstancePVSWasm.js` |
| Worker | `slm2viewer/src/LightweightPVSWorker.js` |
| 主线程接入 | `slm2viewer/src/LightweightPVSDispatcher.js`、`slm2viewer/slm2/SLM2Loader.js`、`slm2viewer/slm2/SLM2VisibilityRuntime.js`、`slm2viewer/slm2/SLM2GlbPipeline.js` |
| 打包 | `slm2viewer/scripts/package_deploy.mjs` |

## 评价边界

画面安全由 weighted recall 及其 calibration 置信下界判断，普通 pose recall 用于诊断。通过安全门后同时比较 precision、instance accuracy、balanced accuracy、specificity、useful cull、bad cull、平均预测数量、GLB 字节和图像漏检指标。

浏览器数值验证必须比较九维中心视角、区域轴、频谱矩和最终概率。功能 smoke 与硬件性能结论分开：只有 WebGPU adapter 明确回报 NVIDIA/Vulkan 并通过仓库硬件门时，才能记录正式延迟；WebGL 硬件证据不能替代 WebGPU adapter 证据。

## 当前边界

- 当前只有 HKUST 有 V4 运行资产；Metropolis 仅保留 AABB 视锥展示。
- 当前下载优先级复用可见性概率，尚未形成独立的预算感知 GLB 效用头。
- Three.js 场景仍由 WebGLRenderer 渲染，无法零拷贝读取 WebGPU 可见性位图；当前使用 GPU 压缩编号驱动实例矩阵压缩。共享位图需要整体迁移 WebGPURenderer。
- 移动设备硬件 WebGPU 性能尚未实测，必须按既有移动端方案补齐。

前端细节和部署命令见 [`../frontend/pvs_v4_runtime_and_deployment.md`](../frontend/pvs_v4_runtime_and_deployment.md)。
