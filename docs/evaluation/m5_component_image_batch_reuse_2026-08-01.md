# M5 实例级图像评价：GLB 重载审计与批量页面复用

日期：2026-08-01

## 阶段范围

本记录只处理投稿计划中的 M5 子任务：审计真实本地 GLB 与 RGB component-ID
图像评价管线，并为 validation/calibration smoke 提供可复现的批量浏览器运行方式。
本次没有修改 test split、模型阈值、候选集合、可见实例监督或相机口径。

图像语义保持不变：reference 是完整本地场景在真实相机下的实例级深度可见结果，
prediction 只接受模型产生的 `predictionComponentIds`；颜色编码为
`componentGlobalId + 1`，0 表示背景。真实渲染垂直视场角固定为 60°，模型输入和
采样相机字段固定为 66°。

## 重载原因审计

### 单次 evaluator 调用

`evaluate_viewcell_image_per.py` 会先遍历所选 view-cell，生成所有 subpose 的
component-level prediction manifest，然后只调用一次
`render_local_glb_color_id_browser.mjs`。Node 渲染器启动一个临时 Chrome 页面，
在样本循环开始前对完整 `selectedGlbs` 清单逐个执行一次 `GLTFLoader.loadAsync`，
之后才对每个 sample 执行 reference 和 prediction 两次 Color-ID 渲染。因此在同一
manifest、同一浏览器页面内，pose 之间没有重新加载 3273 个 GLB。

### 独立 evaluator 调用

如果外部脚本按 pose、阈值或模型分别调用 evaluator，每次调用都会重新创建 Node
进程、HTTP 服务、临时 Chrome profile 和 Three.js 场景。当前实现没有跨进程的 GLB
对象缓存或持久化页面；临时 profile 在进程结束后删除。因此每次独立调用都会再次
执行完整 GLB load pass。这个边界是此前日志看起来像“每个 pose 重载 3273 个 GLB”
的主要原因。

完整清单不能简单替换为预测 GLB 子集：reference 必须保留全场景实例，才能在
真实深度测试中得到被预测剔除实例的 miss 像素。当前 HKUST 资源预检为：

| 项目 | 数量 |
|---|---:|
| 本地 GLB 清单 | 3,273 |
| componentGlobalId | 18,831 |
| 无 mesh 的空 GLB | 2 |
| reference 语义 | 完整可渲染实例场景 |
| prediction 语义 | 模型给出的实例 ID 集合 |

空 GLB 仍作为显式资源异常保留，不被静默删除；它们不贡献可见像素。

## 本次实现

### 批量 manifest

新增 `local-true-component-id-render-batch-manifest-v1`。它只是在多个既有
`local-true-component-id-render-manifest-v2` 之上增加 `batches` 分组，每个 batch
仍包含原来的 sample 和 `predictionComponentIds`。批量校验会重新调用 v2 校验逻辑，
所以以下约束没有被放宽：

- selected GLB 必须是完整本地 GLB inventory；
- componentGlobalId 到 GLB instance slot 的绑定必须一致；
- 不接受 `referenceGlbs`、`testGlbs` 或 `predictionGlbIds`；
- reference/prediction 仍为实例级 ID buffer；
- 每个 sample 必须使用 60°真实渲染相机和 66°模型输入相机；
- batch 之间不允许重复 sampleId，也不允许混用不同场景绑定表。

### 单页面批量 runner

新增 `run_m5_component_image_batch.py`。它只合并已经生成的 v2 manifest，不运行
模型、不扫描阈值、不修改阈值。调用方式为：

```bash
PYTHONDONTWRITEBYTECODE=1 python \
  neural_instance_culling/benchmark/run_m5_component_image_batch.py \
  --input validation=/path/to/validation_manifest.json \
  --input calibration=/path/to/calibration_manifest.json \
  --output-dir /tmp/m5_component_image_batch_run \
  --chrome-exe /usr/bin/google-chrome
```

只做输入和浏览器 manifest 校验时加 `--validate-only`。这条路径把多个
validation/calibration 结果放进一个页面；页面先加载完整 GLB 清单一次，再顺序
处理所有 batch。它适合批量 smoke 和正式 test 之前的 validation/calibration 图像
检查，不能把自身输出当作 one-shot test 结果。

### 页面内资产和掩码复用

浏览器 summary 新增 `assetReuse`：

- `browserPageCount`：本 manifest 使用的页面数；
- `glbLoadPasses`：页面内完整 GLB 加载遍数；
- `glbLoaderCalls`：实际 GLTFLoader 调用数；
- `reloadsPerPoseWithinPage`：页面内 pose 切换造成的 GLB 重载数；
- `loadElapsedMs`、`renderElapsedMs` 和 `totalPageElapsedMs`：分阶段耗时。

预测掩码也改为增量更新。初始实例可见性值为 0，随后只更新前一个预测集合和
当前预测集合的差集；reference pass 通过关闭掩码渲染完整场景，不再为了 reference
先把全部 18,831 个实例逐项恢复为可见。这个改变只影响浏览器内部更新方式，不改变
最终 ID buffer 的定义。

## 自测与真实 smoke

### 静态与 schema 检查

```bash
node --check neural_instance_culling/benchmark/render_local_glb_color_id_browser.mjs

PYTHONDONTWRITEBYTECODE=1 python -m py_compile \
  neural_instance_culling/benchmark/instance_id_render_schema.py \
  neural_instance_culling/benchmark/evaluate_viewcell_image_per.py \
  neural_instance_culling/benchmark/run_m5_component_image_batch.py

PYTHONDONTWRITEBYTECODE=1 python \
  neural_instance_culling/benchmark/run_m5_component_image_batch.py \
  --self-test
```

结果：Node 语法检查通过，三个 Python 文件编译通过，batch self-test 输出
`M5 component-image batch self-test: PASS`。

未修改的既有 M5 测试也重新运行：

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest -v \
  neural_instance_culling.benchmark.tests.test_instance_id_render_schema
```

结果：5/5 通过。测试文件没有修改。

### Batch schema smoke

使用已有真实 HKUST 单 sample manifest 复制为两个不同 batch，只改变 sampleId，
没有改变相机、component ID 或阈值。`--validate-only` 结果为：

| 项目 | 结果 |
|---|---:|
| batch 数 | 2 |
| sample 数 | 2 |
| selected/loaded GLB | 3,273 / 3,273 |
| binding component 数 | 18,831 |
| render FOV | 60° |
| model input FOV | 66° |
| formal test | 否 |

### 真实浏览器 batch smoke

命令使用两个 batch、320×180 分辨率和完整 HKUST 本地 GLB 清单；输出位于临时
目录 `/tmp/m5_batch_final_20260801`，不作为长期 benchmark 资产：

```bash
PYTHONDONTWRITEBYTECODE=1 python \
  neural_instance_culling/benchmark/run_m5_component_image_batch.py \
  --input validation=/tmp/m5-validation-manifest.json \
  --input calibration=/tmp/m5-calibration-manifest.json \
  --output-dir /tmp/m5_batch_final_20260801 \
  --chrome-exe /usr/bin/google-chrome \
  --timeout-sec 1800
```

浏览器资产复用证据：

| 指标 | 结果 |
|---|---:|
| browserPageCount | 1 |
| glbLoadPasses | 1 |
| glbLoaderCalls | 3,273 |
| loadedGlbCount | 3,273 |
| batch/sample 数 | 2 / 2 |
| reloadsPerPoseWithinPage | 0 |
| GLB load 阶段 | 20.819 s |
| 两 sample 渲染阶段 | 17.521 s |
| 页面总耗时 | 38.500 s |

两个 batch 的图像结果完全一致：

| 指标 | 每个 batch |
|---|---:|
| reference 有效像素 | 36,390 |
| PER | 0.0564716 |
| miss-pixel rate | 0.0453146 |
| wrong-ID pixel rate | 0.0111569 |
| extra pixel rate | 0 |
| self-consistency PER | 0 |

这些数值只是既有探索性单 view-cell smoke 的复现，用来验证 batch 复用和实例 ID
一致性，不是正式 validation/calibration 门控，也不是 test 结论。单页面 batch
避免了第二个独立 Chrome 进程必然产生的第二次完整 GLB load pass；本次没有把不同
机器、不同缓存状态下的墙钟时间差宣称为严格加速比。

## 修改文件

- `neural_instance_culling/benchmark/instance_id_render_schema.py`
  - 增加 batch manifest 校验；复用原 v2 的实例绑定、ID 和 FOV 检查。
- `neural_instance_culling/benchmark/render_local_glb_color_id_browser.mjs`
  - 支持多 batch 单页面渲染；增加加载/渲染/页面复用 telemetry。
  - 增加实例可见性掩码的增量更新和 reference mask pass。
  - 保持 RGB24 componentGlobalId 编码和真实 60°相机。
- `neural_instance_culling/benchmark/run_m5_component_image_batch.py`
  - 新增 validation/calibration 现有 manifest 的单页面批量 runner。
  - 不预测、不扫描、不改变模型阈值。
- `neural_instance_culling/benchmark/evaluate_viewcell_image_per.py`
  - 更新真实 component-ID browser renderer 的描述和限制，反映当前已实现的实例掩码与批量复用入口。

## 剩余性能限制与阶段门

1. 每一个独立 Node/Chrome runner 仍会重新加载完整 3,273 个 GLB；本次只解决同一
   runner 内跨 batch/sample 的复用，没有实现跨进程常驻浏览器或持久化 GLB 对象缓存。
2. reference 为了保持全场景深度语义仍保留全部 GLB。按空间分页减少加载清单需要先
   证明不会改变可见实例和深度遮挡结果，目前不能作为 M5 的隐式优化。
3. 页面仍保留并提交完整已加载场景，每个 sample 仍进行 reference/prediction 两次
   Color-ID 渲染；掩码更新变轻不等于三角形绘制成本消失。
4. 本机 Chrome smoke 输出过 software WebGL fallback 和 `ReadPixels` GPU stall
   警告，因此上述耗时只用于管线分解和复用审计，不能作为移动设备实时性能结论。
5. 正式 validation/calibration 的完整 subpose 图像门控尚未运行；test split 仍未触碰，
   没有生成或修改正式冻结阈值。

本阶段结论：M5 的实例语义和页面级批量复用已经有可复现 smoke 证据，但正式图像
安全门仍待在冻结模型和校准协议完成后运行。方向遮挡代理的因果有效性属于 M3，不能
由本次图像 smoke 代替。
