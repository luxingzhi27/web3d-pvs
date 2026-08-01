# M7/M8：离线轨迹下载回放评估器

**日期：** 2026-08-01  
**状态：** 最小可复现实现完成；已运行 HKUST validation 确定性回放，未运行正式 test split，也未修改任何现有 test 数据。
**目的：** 为 M7 的 GLB 下载排序和 M8 的冷/温缓存导航实验提供一个独立的离线回放入口。该入口不修改训练模型、不修改前端调度代码，也不把弱可见权重冒充真实像素效用。

## 1. 本次变更

本次新增的回放器执行以下数据流：

1. 读取一个离线 `poseIndex` 轨迹，保持轨迹中的 pose 顺序和时间，不重新采样或重排。
2. 对每个 pose 使用数据集保存的后退相机候选实例集合。
3. 调用已有 `model_runners` 接口，得到与候选实例逐行对齐的模型分数。
4. 按 GLB 聚合规则将实例分数变成下载队列优先级。支持 `max`、`sum`、`top-k` 和 `noisy-or`。
5. 按优先级模拟网络下载、带宽并发、网络完成、解码/上传队列和最终完成事件。
6. 只有 GLB 的解码/上传完成后才进入缓存并贡献当前 pose 的可用效用。
7. 输出字节和时间两种资源曲线、缺失效用积分、首个有用画面和无效下载字节。

本次新增文件：

- `neural_instance_culling/benchmark/evaluate_download_trajectory.py`
- `neural_instance_culling/benchmark/measure_glb_decode_upload_costs.mjs`
- `neural_instance_culling/benchmark/tests/test_download_trajectory.py`
- `docs/experiments/m7_m8_trajectory_replay_2026-08-01.md`

没有修改训练脚本、模型权重、数据集文件、前端代码或现有正式 benchmark 输出。

## 2. 输入协议

### 2.1 轨迹 JSON

轨迹文件必须声明 schema：

```json
{
  "schema": "neuralstreamweb3d-pose-index-trajectory-v1",
  "poseIndices": [120, 121, 135, 160],
  "stepMs": 120,
  "network": {
    "bandwidthBytesPerSec": 1250000,
    "requestLatencyMs": 40,
    "maxConcurrentDownloads": 4,
    "maxConcurrentDecodeUploads": 2
  },
  "cache": {
    "mode": "cold",
    "initialGlbIds": []
  }
}
```

也可以用显式时间的形式表达不等间隔导航：

```json
{
  "schema": "neuralstreamweb3d-pose-index-trajectory-v1",
  "poses": [
    {"poseIndex": 120, "timeMs": 0},
    {"poseIndex": 121, "timeMs": 87},
    {"poseIndex": 135, "timeMs": 241}
  ],
  "network": {"bandwidthBytesPerSec": 1250000},
  "cache": {"mode": "warm", "initialGlbIds": [3, 8]}
}
```

时间必须单调不减。`poseIndex` 是 CSR 中的行号，不是相机位置哈希，也不是重新生成的 pose。网络带宽没有默认估计值，必须写在轨迹文件中或通过命令行传入，避免把未知网络条件伪装成实验结果。

### 2.2 GLB 成本

字节成本从 `glbIndex.json` 指向的真实文件大小读取。解码/上传成本从已有的
`neuralstreamweb3d-glb-cost-index-v1` 时间索引读取，例如：

```json
{
  "schema": "neuralstreamweb3d-glb-cost-index-v1",
  "entries": [
    {"globalId": 0, "decodeMs": 2.1, "uploadMs": 1.7},
    {"globalId": 1, "totalDecodeUploadMs": 6.4}
  ]
}
```

所有被场景实例映射使用的 GLB 都必须有正的字节成本和正的解码/上传成本。评估器拒绝重复条目、负值、不一致的总时间和缺失成本，不进行中位数填补。

## 3. 回放语义

### 3.1 候选集合和 runner

保存的后退相机候选集合是唯一候选来源。若一个 pose 的可见实例不在保存的候选集合中，回放器直接报错；它不会把 GT 实例补回候选，也不会截断候选。

已有 runner 的 `score_arrays` 返回一个候选实例对应一个分数。选定的分数模式包括：

- `visibility-only`：使用实例可见性分数；
- `current-cascade`：使用当前模型下载头的分数，并进行概率归一化；
- `visibility-gated`：使用可见性分数乘当前效用头输出，只作为诊断模式；
- `independent-utility`：只有通过 `--independent-ranker-spec name|checkpoint` 显式注册的独立排序器才可用；没有注册时明确返回未实现，不用当前效用头冒充独立模型。

本回放是下载排序评估，不执行可见性阈值扫描。runner 中加载的阈值只作为 checkpoint 元数据记录，不能改变本回放的 GLB 排序。

### 3.2 GLB 聚合

实例到 GLB 的映射来自 runner 的 `instance_to_glb`。同一 GLB 的实例分数按指定规则聚合：

- `max`：最高实例分数；
- `sum`：实例分数总和；
- `top-k`：最高的 `k` 个实例分数之和；
- `noisy-or`：`1 - product(1 - score)`，对多个实例的共同证据进行饱和聚合。

分数相同时用 GLB id 升序作为确定性排序规则。每个 pose 的完整 GLB 排序和弱效用目标会写入输出 JSON，便于复核模型输出和下载队列之间的关系。

### 3.3 缓存和完成事件

支持三种初始缓存状态：

- `cold`：开始时没有 GLB 在缓存中；
- `warm`：由 `initialGlbIds` 明确列出已缓存 GLB；
- `warm-all`：所有运行时 GLB 初始驻留，作为稳态控制条件。

下载请求按当前 pose 的 GLB 排序加入去重队列。已启动的下载不会被后续 pose 取消，尚未启动的队列项可以更新为已经观察到的最高优先级。下载并发数限制网络活动槽位，带宽在活动槽位之间等分：

```text
network duration = request latency + GLB bytes / (bandwidth / concurrent slots)
```

下载完成只表示网络字节已经传输完毕。随后 GLB 进入解码/上传 FIFO 队列，受
`maxConcurrentDecodeUploads` 限制；只有上传完成事件才会将 GLB 加入缓存。该顺序避免把“已经请求”或“下载完成”错误地当成“画面已经可用”。

## 4. 输出指标

输出 schema 为 `neuralstreamweb3d-download-trajectory-replay-v1`。每个模型结果包含每个 pose 的 GLB 排序、runner 信息、成本来源和完整事件记录。

### 4.1 弱视觉效用

当前 CSR 中的 `visible_weights` 是 rvcServer 的构件重要性权重，不是真实像素覆盖率。回放器使用：

```text
weak utility(instance) = log1p(max(visible_weight, 0))
```

同一 GLB 内实例效用相加，同一条轨迹中重复出现的 pose 需求分别计入。所有效用字段都带有：

```text
weak_log1p_visible_weights_not_pixel_coverage
```

因此本实现不输出 miss-pixel、wrong-ID pixel 或真实图像质量结论。真实图像效用仍需 M5 的实例级 Color-ID 渲染管线提供。

### 4.2 `utility@bytes` 和 `utility@time`

- `utilityAtBytes.curve`：横轴是实际完成网络下载的累计字节，纵轴是轨迹需求中已经在对应 pose 窗口内被解码/上传完成的弱效用比例。
- `utilityAtTime.curve`：横轴是从轨迹第一帧开始的毫秒数，纵轴定义相同。
- `budgets`：可选的字节或时间预算查询结果，严格只使用不超过预算的曲线点。
- `area`：效用曲线的梯形积分及按最大资源轴归一化的面积，用于后续曲线比较。

初始温缓存不计入新增下载字节，但会在进入第一个 pose 后立即贡献可用效用。GLB 在网络完成但解码/上传尚未完成时不会贡献效用。

### 4.3 缺失效用积分

回放器在每个 pose 仍处于当前导航窗口时计算：

```text
missing utility(t) = target utility(current pose)
                    - utility of completed GLBs(current pose)
```

输出包括：

- `missingWeakUtilityIntegralMs`：原始弱效用乘时间的积分；
- `missingUtilityRatioIntegralMs`：缺失效用比例乘时间的积分；
- `meanMissingUtilityRatio`：按回放时长归一化的平均缺失比例。

积分在 pose 切换、网络完成和解码/上传完成事件之间分段计算。默认回放在最后一个 pose 时间结束；`--drain-after-last-pose-ms` 可以显式增加尾部排空窗口，尾部窗口的范围会写入输出。

### 4.4 首个有用画面和无效下载字节

`firstUsefulFrame` 是第一个当前 pose 已有正弱效用 GLB 完成解码/上传的时间点。它不是像素级“画面完整”证明，只表示当前弱效用口径下至少有一个目标 GLB 可用。

`invalidDownloadBytes` 的定义是：已经完成网络下载、但在整条轨迹的任何 pose 中都没有正弱可见权重效用的 GLB 字节。它用于诊断预测驱动下载的纯无效资源消耗，不把“晚到但未来仍有用”的 GLB 算成无效下载。输出还保留未完成请求字节、完成下载/上传 GLB 数量和事件明细。

## 5. 运行命令

### 5.1 Self-test

Self-test 只使用内存中的合成计划，不读取项目数据：

```bash
PYTHONDONTWRITEBYTECODE=1 conda run --no-capture-output -n slm_pvs \
  python neural_instance_culling/benchmark/evaluate_download_trajectory.py \
  --self-test
```

### 5.2 新增单元测试

测试使用临时目录和内存 fixture，不修改仓库中的 test 数据：

```bash
PYTHONDONTWRITEBYTECODE=1 conda run --no-capture-output -n slm_pvs \
  python -m unittest neural_instance_culling.benchmark.tests.test_download_trajectory
```

### 5.3 真实离线轨迹回放模板

下面是实际运行模板；该命令用于 validation 确定性回放，不读取正式 test split：

```bash
conda run --no-capture-output -n slm_pvs \
  python neural_instance_culling/benchmark/evaluate_download_trajectory.py \
  --trajectory /path/to/pose_index_trajectory.json \
  --dataset-dir neural_instance_culling/dataset/out/<pose_csr_dataset> \
  --runtime-meta <scene>/assets/runtimeVisibilityMeta.json \
  --glb-index <scene>/assets/glbIndex.json \
  --glb-root <scene>/assets \
  --glb-time-index /path/to/glb_decode_upload_costs.json \
  --models pvs_directional_occlusion_proxy_encoder_rvl_w042_full40_best \
  --score-mode current-cascade \
  --glb-aggregation max \
  --byte-budgets 1048576,5242880,10485760 \
  --time-budgets-ms 100,250,500,1000 \
  --output neural_instance_culling/benchmark/out/m7_m8_trajectory_replay_<run>.json
```

带宽、并发和缓存可以直接写进轨迹 JSON，也可以通过以下参数覆盖：

```text
--bandwidth-bytes-per-sec
--request-latency-ms
--max-concurrent-downloads
--max-concurrent-decode-uploads
--cache-mode cold|warm|warm-all
--initial-cache-glbs 3,8,11
--drain-after-last-pose-ms
```

## 6. 验证结果和质量门控

本次执行结果：

```text
self-test: passed
unit tests: 5 tests, OK
py_compile: passed
git diff --check: passed
HKUST validation replay: 664 poses, cascade and independent RankNet completed
GLB browser cost index: 3273/3273 measured, 0 failed
formal test split: not run
existing test data: unchanged
```

新增实现满足 M7/M8 最小“协议可复现、事件语义清楚、缺失资源不静默填补”的代码质量门。投稿计划中的完整科学门仍未通过，原因是：

1. 独立 RankNet 已注册并完成一条 validation 回放，但尚未完成多种子和多轨迹 paired bootstrap；
2. 本回放使用固定等份带宽槽位模型，不等同于真实浏览器 TCP、HTTP/2 或 HTTP/3 调度；
3. 成本采集来自 headless Chrome，尚未得到硬件移动 WebGPU 的真实 GLB 解码和 GPU 上传时间轨迹；
4. 尚未执行多轨迹、4G/Wi-Fi trace、三种子 paired bootstrap 和真实前端 p95 帧时间；
5. 弱可见权重不能替代 M5 的像素级效用，因此当前结果不能据此宣称图像安全或联合下载头已经优于独立排序器。

所以本文件记录的是 M7/M8 的可复现实验基础设施，不是完整投稿主表结果，也不改变当前主线模型的 Go/No-Go 结论。

## 7. 保留判断和后续风险

该实现保留在当前实验主线上，因为它提供了训练模型和前端之间缺失的、可审计的离线资源回放层，同时保持模型、前端和正式 test 数据不变。正式实验前仍需固定真实轨迹、网络 trace、初始缓存、成本索引和模型 checkpoint，并按相同轨迹配对比较 keep-all、visibility-only、当前级联、独立 ranker 和 oracle。

主要风险是固定带宽槽位模型可能低估或高估真实网络中的头部阻塞和并发收益；真实设备测量完成前，`utility@time` 只能解释为成本索引驱动的离线模拟时间。所有后续结果必须继续保留本文件中的效用警告，不能将其改写为像素级结论。

## 8. 确定性 pose-index 轨迹生成器

新增 `neural_instance_culling/benchmark/build_pose_index_trajectory.py`，用于从数据集已有的
native split 生成可复现的 pose-index 回放输入。它按 split 中的原始 pose 索引升序、固定步长和
可选 stride 生成 `poses`，写入数据集 metadata SHA-256、pose 索引 SHA-256、66°模型视场角、
60°真实渲染视场角以及网络/缓存参数。

该工具输出始终带有：

```text
formal: false
status: deterministic_split_replay_not_live_navigation
network.measurementStatus: assumed_not_device_measurement
```

因此它可以用于协议和回放链路的场景级 smoke，但不能替代真实用户导航轨迹。命令自测已通过：

```bash
PYTHONDONTWRITEBYTECODE=1 conda run --no-capture-output -n slm_pvs \
  python neural_instance_culling/benchmark/build_pose_index_trajectory.py --self-test
```

只有在实测 GLB 解码/上传成本索引和真实网络轨迹准备好之后，输出才可进入 M8 正式比较；当前
没有用该工具生成的假设轨迹宣称投稿质量门通过。

## 9. 浏览器成本采集器（2026-08-02）

新增 `measure_glb_decode_upload_costs.mjs`，使用与 M5 相同的 Three.js `GLTFLoader`，并显式初始化 Meshopt 和 Draco 解码器。它从 `glbIndex.json` 读取完整 GLB 清单，在浏览器中逐个获取二进制、解析 glTF，并通过实际 WebGL render 提交几何后记录：

- 文件字节数；
- fetch 时间；
- glTF 解析/解码时间；
- render 提交时间；
- 从解析开始到提交完成的 `decodeUploadMs`；
- 三角形数、网格数和错误原因。

缺少文件、解码器失败或浏览器异常会使条目保持 `status=error`；只有所有选中条目成功时输出才标记 `status=complete`。采集器不会对失败条目估计时间。

已通过：

```bash
node neural_instance_culling/benchmark/measure_glb_decode_upload_costs.mjs --self-test
node --check neural_instance_culling/benchmark/measure_glb_decode_upload_costs.mjs
```

真实 smoke 使用 HKUST 的 1 个小构件和 3 个最大构件，结果写在 `/tmp`，不作为正式成本索引：1 个小构件成功，约 `2,964` bytes、`12.4 ms` 解析、`31.8 ms` render 提交；3 个约 `41.5 MB` 构件均成功，`decodeUploadMs` 约 `232--529 ms`。这些数值只证明采集链路和大构件路径可运行，尚未覆盖全部 `3,273` 个 GLB，也不代表移动设备性能。

正式 M8 仍需在固定浏览器/后端和完整清单上运行采集，记录浏览器版本、图形适配器、分辨率和温度条件，并将完整成本索引与 SHA-256 一起冻结。SwiftShader smoke 不得冒充硬件 GPU 或 Android 成本证据。
