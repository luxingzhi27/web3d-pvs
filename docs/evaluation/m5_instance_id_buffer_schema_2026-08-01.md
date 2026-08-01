# M5 实例级 60° Color-ID 图像评价管线

日期：2026-08-01

## 目的与范围

M5 要把图像级评价从“一个 GLB 一个颜色”收紧到“一个场景构件实例一个颜色”。这样才能区分同一 GLB 内被预测保留和被预测剔除的实例，并据此形成 reference、prediction、miss、wrong 和 extra 掩码。本次已落实可运行、可审查的 manifest、实例绑定预检、真实 GLB 的 component-level InstancedMesh 渲染和合成 smoke；没有把尚未完成的正式 test split 评价写成论文图像结果。

本次没有修改训练代码、采样代码或大数据文件；后续追加了一次单 view-cell 的真实浏览器
validation smoke，正式 test 仍未启动。

## 当前数据流

1. `runtimeVisibilityMeta.json` 提供 `componentGlobalId` 到 `globalGlbId` 的归属，以及每个 GLB 的 `componentGlobalIds` 顺序。
2. `glbIndex.json` 提供本地 GLB 路径。
3. `instance_id_render_schema.py` 只读取 GLB 的 JSON chunk，检查 GLB 头、mesh node 数量、`EXT_mesh_gpu_instancing` 的实例数量，并验证 metadata 的实例顺序与 GLB accessor 数量一致。
4. 评估脚本把模型输出原样写入 `predictionComponentIds`。它不再把预测实例集合压缩成浏览器的 `testGlbs`，也不再把 `predictionGlbIds` 写进实例级渲染 manifest。
5. 浏览器入口先做严格 manifest 校验。`--validate-only` 生成“schema 已验证、图像未渲染”的摘要；普通浏览器模式通过 `GLTFLoader` 加载 GLB，在 `InstancedMesh` 上绑定实例 ID 与逐实例可见性 attribute，再分别渲染 full-scene reference 和 `predictionComponentIds` prediction；`--synthetic-render-smoke` 只验证内存合成场景的 shader/读回链路。

## Schema 契约

渲染 manifest 使用 `local-true-component-id-render-manifest-v2`。实例绑定表使用 `component-instance-binding-preflight-v1`，其核心关系是：

```text
globalGlbRecords[globalGlbId].componentGlobalIds[instanceIndex]
    == componentGlobalId
```

ID buffer 编码固定为 `componentGlobalId + 1` 的 RGB24 值，0 保留给背景。这样 reference 与 prediction 在同一像素位置的值相等时表示同一实例，掩码定义为：

| 掩码 | 条件 | 含义 |
|---|---|---|
| reference | reference ID 非 0 | 60°真实相机下参考场景中的实例像素 |
| prediction | prediction ID 非 0 | 模型保留实例的渲染像素 |
| miss | reference 非 0 且 prediction 为 0 | 预测漏掉参考实例 |
| wrong | reference、prediction 均非 0 且 ID 不同 | 深度可见实例被另一个实例替换 |
| extra | reference 为 0 且 prediction 非 0 | 预测新增像素，需与剔除效率指标一起解释 |

相机契约固定为：图像 reference/prediction 使用垂直视场角 60°；模型查询和数据集后退相机使用垂直视场角 66°。manifest 和每个 sample 都必须显式写出这两个字段，不能用历史 `fovYDeg` 或默认值隐式推断。

`selectedGlbs` 只表示完整本地场景资产清单，必须覆盖 binding 表中的全部 GLB，不能替换成模型预测的 GLB 子集。渲染预测的唯一语义是 `predictionComponentIds`。因此当前 schema 会拒绝 `referenceGlbs`、`testGlbs` 和 `predictionGlbIds`，防止旧的 GLB 级压缩语义被误当成实例级图像评价。

对当前两套本地资源做了 JSON/header 级只读预检，结果如下。HKUST 的两个空 GLB 被保留为显式异常，没有在预检中静默删除。

| 场景 | GLB 数 | component 数 | 可渲染 component 数 | 空 GLB 数 |
|---|---:|---:|---:|---:|
| HKUST v3 | 3,273 | 18,831 | 18,829 | 2 |
| IFCBench Fantasy Metropolis 实例化 v2 | 3,669 | 41,298 | 41,298 | 0 |

## 已改文件

- `neural_instance_culling/benchmark/instance_id_render_schema.py`
  - 新增 GLB JSON/header 级实例布局检查。
  - 验证 component、GLB、实例槽位的双向关系。
  - 显式保留无 mesh GLB 异常，不静默丢弃。
  - 验证 schema、完整 GLB 清单、60°/66° FOV、预测实例字段和旧字段拒绝规则。
- `neural_instance_culling/benchmark/evaluate_viewcell_image_per.py`
  - 生成 v2 实例 manifest。
  - 写入 `predictionComponentIds` 和完整场景 reference 语义。
  - 在调用 Node 前完成绑定预检和 manifest 校验。
  - `--render-schema-only` 只执行 schema smoke，不宣称产生 PER。
- `neural_instance_culling/benchmark/render_local_glb_color_id_browser.mjs`
  - 增加同口径 Node 校验。
  - `--validate-only` 不查找 Chrome、不检查 Three.js 依赖、不启动浏览器。
  - 普通模式通过 `GLTFLoader` 加载真实 GLB，把 `componentGlobalIds[instanceIndex]` 写入实例 attribute，以 component ID shader 输出 RGB24，并用逐实例可见性 attribute 形成 prediction。
  - 旧的 GLB 级 ID 语义不再作为默认路径；AABB 路径仍需显式选择。
- `neural_instance_culling/benchmark/tests/test_instance_id_render_schema.py`
  - 合成两个 instanced GLB、三个 component 的跨语言 smoke。
- `docs/README.md`
  - 增加本记录索引。

## 验证

已执行以下短时检查：

```bash
node --check neural_instance_culling/benchmark/render_local_glb_color_id_browser.mjs
python -m py_compile \
  neural_instance_culling/benchmark/instance_id_render_schema.py \
  neural_instance_culling/benchmark/evaluate_viewcell_image_per.py \
  neural_instance_culling/benchmark/tests/test_instance_id_render_schema.py
python -m unittest neural_instance_culling.benchmark.tests.test_instance_id_render_schema
```

结果：5 个测试通过。测试覆盖实例槽位绑定、旧 GLB 字段拒绝、错误 66°渲染 FOV 拒绝、Node `--validate-only` 输出 `schema_validated_not_rendered` 摘要并保持 `imageMetrics: null`、真实 Three.js synthetic component-ID smoke，以及真实 GLB 经 `GLTFLoader` 加载后执行 InstancedMesh component-ID 渲染并检出 miss 像素。另有上述两套场景的只读资源预检通过。

## 当前限制与阻塞

- 当前浏览器实现不重排矩阵，而是在原始 `InstancedMesh` 上用逐实例 visibility attribute 丢弃未选实例；这已经能完成实例级显示和 ID buffer，但尚未在完整 HKUST/Metropolis test split 上评估吞吐与显存。
- synthetic smoke 和合成 GLB smoke 都不是正式场景结果；正式 reference/prediction ID buffer、PER、miss pixel rate 或 wrong ID pixel rate 仍未跑完，因此任何 schema-only 或 smoke 摘要都不能作为论文图像指标。
- 现有 GLB AABB Color-ID 路径仍可作为显式调试分支，但它是 GLB 级代理，不属于 M5 实例级正式评价，不能用来替代本管线。
- 完成正式图像评价前，需要在真实场景 test split 上检查所有 GLB 的 mesh/实例布局、资源加载失败、每帧 attribute 更新成本、同一 60°相机下的 reference/prediction 双渲染和像素级自一致性；矩阵重排不再是当前实现的必要条件。

本记录保留当前阻塞，未改变默认训练模型、训练数据或前端运行资产。

## 2026-08-01 多样本真实场景 Validation Smoke

在补充运行时发现，旧命令曾把 Pose CSR-only 目录误传给
`--viewcell-dataset`，入口当时只在 `memmap` 打开阶段报出缺失文件，容易把数据
语义错误误认为渲染器故障。`evaluate_viewcell_image_per.py` 现已在加载前检查完整的
view-cell/subpose 文件集合、声明的 view-cell 数量和偏移数组长度；错误目录会明确要求
使用 `*_viewcell_*_source` 数据集。该修改不改变评价数据或模型结果。

随后使用正确的 HKUST view-cell 源数据和正式候选 CSR 做 validation smoke：

```bash
conda run -n slm_pvs python -u neural_instance_culling/benchmark/evaluate_viewcell_image_per.py \
  --model-name pvs_directional_occlusion_proxy_encoder_rvl_w042_full40_hkust_fov66_best \
  --runtime-meta hkust-v3/assets/runtimeVisibilityMeta.json \
  --viewcell-dataset neural_instance_culling/dataset/out/hkust_v3_viewcell_colorid_fov66_source \
  --pose-csr neural_instance_culling/dataset/out/pose_csr_hkust_v3_spatial_raw_subpose_aabb_fov66_v1 \
  --glb-root hkust-v3/assets --glb-index hkust-v3/assets/glbIndex.json \
  --split val --max-viewcells 4 --subposes-per-viewcell 1 \
  --width 320 --height 180 --device cpu --image-renderer true_glb \
  --threshold 0.64 --preview-samples 2 --skip-raw-subpose-gt \
  --output-dir neural_instance_culling/benchmark/out/m5_hkust_true_glb_validation_component_smoke_20260801_v2
```

本次只选取 4 个 validation view-cell 和 4 个 dense subpose，浏览器完整加载 3,273
个本地 GLB，并按实例 ID 完成 reference/prediction 双渲染。结果为：

| 项目 | 结果 |
|---|---:|
| 真实子姿态数 | 4 |
| 本地 GLB 加载数 | 3,273 |
| 加载阶段 | 21.05 s |
| 双渲染阶段 | 29.04 s |
| 页面总耗时 | 50.24 s |
| self-consistency PER | 0 |
| PER | 1.7695% |
| miss-pixel rate | 1.3648% |
| wrong-ID pixel rate | 0.4047% |
| extra-pixel rate | 0 |

该结果仅证明真实 GLB、实例 ID shader、实例掩码和像素读回链路可运行；它使用历史
探索性模型和显式阈值 `0.64`，样本量不足，不能替代冻结后的 validation/calibration
完整图像评价，也不触碰 test split。当前图像质量门仍为未通过，且全量 GLB 冷启动耗时
为后续 M9 空间分页和 Cold-0 预算实验提供了明确基线。

## 2026-08-01 真实场景 Validation Smoke

为验证端到端编排，使用现有探索性 HKUST 模型和阈值 `0.64` 跑了一个
validation view-cell、一个 subpose，加载完整 3,273 个本地 GLB。该结果不是正式
validation/test 结论，也没有使用空间正式模型或 one-shot test。

```bash
conda run -n slm_pvs python neural_instance_culling/benchmark/evaluate_viewcell_image_per.py \
  --model-name pvs_directional_occlusion_proxy_encoder_rvl_w042_full40_hkust_fov66_best \
  --runtime-meta hkust-v3/assets/runtimeVisibilityMeta.json \
  --viewcell-dataset neural_instance_culling/dataset/out/hkust_v3_viewcell_colorid_fov66_source \
  --pose-csr neural_instance_culling/dataset/out/pose_csr_hkust_v3_viewcell_colorid_fov66 \
  --glb-root hkust-v3/assets --glb-index hkust-v3/assets/glbIndex.json \
  --split val --max-viewcells 1 --subposes-per-viewcell 1 \
  --width 320 --height 180 --image-renderer true_glb \
  --chrome-exe /usr/bin/google-chrome --threshold 0.64 --skip-raw-subpose-gt
```

结果：浏览器加载并渲染 3,273 个 GLB，单 subpose 浏览器渲染耗时约 `11.6 s`，
`selfConsistencyPER=0`；`miss-pixel rate=0.04531`、`wrong-ID pixel rate=0.01116`、
`PER=0.05647`。这次运行证明了真实 `GLTFLoader`、`EXT_mesh_gpu_instancing`、逐实例
可见性和 component-ID 读回链路，但也暴露出全量原型在首轮加载前的成本，以及探索性模型
未达到建议图像安全门的问题。输出保留在
`neural_instance_culling/benchmark/out/m5_hkust_true_glb_validation_smoke_20260801_rerun`，
只能作为 M9 分页优化和 M5 正式运行前的诊断证据。
