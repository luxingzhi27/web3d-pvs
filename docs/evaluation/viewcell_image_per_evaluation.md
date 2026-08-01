# Viewcell Image PER Evaluation

Date: 2026-08-01

图像级评价用于回答：模型在实例集合上满足安全约束后，真实 GLB 渲染画面仍会损失多少像素。它不是新的 PVS 模型，也不能替代集合 precision / recall / useful cull / bad cull。

## Inputs

当前默认输入：

```text
view-cell source: neural_instance_culling/dataset/out/hkust_v3_viewcell_colorid_fov66_source
pose CSR:     neural_instance_culling/dataset/out/pose_csr_hkust_v3_spatial_raw_subpose_aabb_fov66_v1
runtime meta: hkust-v3/assets/runtimeVisibilityMeta.json
glb index:    hkust-v3/assets/glbIndex.json
glb root:     hkust-v3/assets
```

`--viewcell-dataset` 和 `--pose-csr` 是两个不同语义的输入：前者必须包含
view-cell/subpose 文件，后者包含模型推理用的 pose、候选和 MVP CSR。不能把只有
`poses.bin` 的 Pose CSR 目录作为 view-cell source。

当前建议评价模型：

```text
pvs_directional_occlusion_proxy_encoder_rvl_w042_full40_best
```

`visible_weights` 继续解释为 rvcServer 的重要性权重，不是严格像素覆盖率。

validation/calibration 开发运行可以使用 `weighted_precision` 选择工作点；正式 test
必须读取 frozen calibration threshold，不能在 test 上扫描或覆盖阈值。图像 PER 仍然是
画面损失评价，不能替代安全筛选或 useful cull / bad cull 报告。

## Process

1. 从 validation/calibration split 选择 viewcell（正式 test 仍需冻结后单独执行）。
2. 对每个 viewcell，用后退扩大视锥相机和同一候选集合调用模型，得到预测实例集合。
3. 将预测实例保留为 `predictionComponentIds`，不压缩成 GLB 级预测集合。
4. 用真实 60°相机 subpose 渲染完整实例 reference Color-ID 图和 prediction Color-ID 图。
5. reference 使用完整本地 GLB 场景并按深度得到可见 `componentGlobalId`。
6. prediction 在同一场景中只打开模型预测的实例掩码；同一 GLB 内的实例可以独立显示或剔除。
7. 比较两张 ID 图，输出 PER、miss pixel rate、wrong ID pixel rate、extra pixel rate 和主要错误 component。

当前渲染器衡量实例级 component-ID 画面损失；GLB 数量和字节收益仍需由集合/调度指标单独报告，不能从图像 ID 差异中反推 GLB 级结论。

## 批量页面复用

一次 `evaluate_viewcell_image_per.py` 调用会把多个 subpose 放入一个 manifest，
浏览器在样本循环前加载完整 GLB 清单一次。若 validation/calibration 被外部脚本
拆成多个独立 manifest，应使用：

```bash
PYTHONDONTWRITEBYTECODE=1 python \
  neural_instance_culling/benchmark/run_m5_component_image_batch.py \
  --input validation=/path/to/validation_manifest.json \
  --input calibration=/path/to/calibration_manifest.json \
  --output-dir /tmp/m5_component_image_batch \
  --chrome-exe /usr/bin/google-chrome
```

该 runner 只合并已有 component-level prediction，不运行模型、不扫描或修改阈值，
并在一个 Chrome 页面中复用已加载的 GLB。它会在 summary 中记录
`browserPageCount`、`glbLoadPasses`、`glbLoaderCalls`、加载耗时和页面内重载次数。
独立 Node/Chrome 调用之间仍没有常驻对象缓存，完整审计见
`m5_component_image_batch_reuse_2026-08-01.md`。

## Metrics

- `PER`: reference 非背景像素中，test ID 不一致的比例。
- `miss pixel rate`: reference 有 GLB 但 test 是背景的像素比例。
- `wrong ID pixel rate`: reference 和 test 都非背景，但 GLB ID 不一致的像素比例。
- `extra pixel rate`: reference 是背景但 test 画出 GLB 的像素比例。
- `weighted recall`: 按 `visible_weights` 统计 GT 被找回比例；它不惩罚 false positive，不能单独作为主结论。

## Example Command

下面命令仅用于 validation smoke，显式使用历史 exploratory 阈值 `0.64`。正式运行
不得照搬该阈值到 test；应使用完成冻结协议的 `eval_summary.json`，并保持 test
一次性、全量且不扫描阈值。

```bash
conda run --no-capture-output -n slm_pvs python -u neural_instance_culling/benchmark/evaluate_viewcell_image_per.py \
  --model-name pvs_directional_occlusion_proxy_encoder_rvl_w042_full40_best \
  --viewcell-dataset neural_instance_culling/dataset/out/hkust_v3_viewcell_colorid_fov66_source \
  --pose-csr neural_instance_culling/dataset/out/pose_csr_hkust_v3_spatial_raw_subpose_aabb_fov66_v1 \
  --runtime-meta hkust-v3/assets/runtimeVisibilityMeta.json \
  --glb-index hkust-v3/assets/glbIndex.json \
  --glb-root hkust-v3/assets \
  --output-dir neural_instance_culling/benchmark/out/viewcell_image_per_pvs_directional_occlusion_proxy_encoder_full40_best \
  --max-viewcells 256 \
  --subposes-per-viewcell 4 \
  --split val \
  --width 960 \
  --height 540 \
  --device cuda \
  --threshold 0.64 \
  --target-weighted-recall 0.99 \
  > neural_instance_culling/benchmark/out/viewcell_image_per_pvs_directional_occlusion_proxy_encoder_full40_best_stdout.log \
  2> neural_instance_culling/benchmark/out/viewcell_image_per_pvs_directional_occlusion_proxy_encoder_full40_best_stderr.log
```

## Reporting Rule

正式报告必须同时给出：

- 图像 PER / miss pixel / wrong ID pixel。
- pose-level 和 aggregate precision / recall / F1 / Jaccard。
- useful cull 和 bad cull。
- 平均预测实例数、GLB 数量和 GLB 字节削减。
- 前端 Worker/WebGPU 推理耗时和实际 drawn instance 数。
- 浏览器真实 GLB 加载耗时、页面复用范围和每个 sample 的 reference/prediction 渲染耗时。
