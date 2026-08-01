# M5 正式图像评价 split 对齐修复

**日期：** 2026-08-01
**状态：** 评价入口修复并通过资源 smoke；M5 图像质量门仍未通过。
**目的：** 让实例级图像评价使用与正式 Pose CSR 完全一致的空间划分，避免旧 view-cell 随机标签污染 validation、calibration 和 test 结论。

## 发现的问题

`evaluate_viewcell_image_per.py` 原先默认指向不含 view-cell/subpose 元数据的 Pose CSR 目录，并直接使用源数据中的随机 `train/val/test` 标签。正式空间 Pose CSR 的 HKUST 划分为 `train/validation/calibration/test/guard`，两套标签虽然行数相同，但数量为 `2772/213/168/238/4608`，不能混用。原入口默认只取 8 个 view-cell，也不适合作为完整 split 评价。

## 实施内容

- `--split-source pose_csr` 现在为默认值，使用 Pose CSR 的正式空间标签。
- `--split-source viewcell` 保留为显式的历史 exploratory 入口。
- 支持 `validation`、`calibration` 和 `guard`，同时保留 `val` 别名。
- 默认 `--max-viewcells 0`，表示完整 split；小规模 smoke 必须显式传正数。
- 在采用 Pose CSR 标签前逐行校验：
  - Pose CSR 与 view-cell 行数相同；
  - 相机前向量最大绝对误差不超过容差；
  - view-cell 中心必须位于后退相机沿前向量的正向偏移上，中心残差不超过 `1e-3 m`。
- 修复 schema-only 渲染返回空图像指标时汇总器对 `None` 调用 `.get` 的错误。

## 资源与运行命令

正式 HKUST 资源：

```text
view-cell source: neural_instance_culling/dataset/out/hkust_v3_viewcell_colorid_fov66_source
Pose CSR:         neural_instance_culling/dataset/out/pose_csr_hkust_v3_spatial_fov66_v1
FOV:              模型/采样 66°，真实渲染 60°
```

回归测试：

```bash
PYTHONDONTWRITEBYTECODE=1 conda run --no-capture-output -n slm_pvs \
  python -m unittest \
  neural_instance_culling.benchmark.tests.test_viewcell_image_split \
  neural_instance_culling.benchmark.tests.test_frozen_test_entrypoint \
  neural_instance_culling.benchmark.tests.test_training_calibration_control -v
```

结果：8 个测试通过。真实资源对齐结果：`7999` 行，正式 split 为 `2772/213/168/238/4608`；前向量最大误差约 `1.2e-7`，中心残差最大约 `6.1e-5 m`。

schema-only smoke：

```bash
conda run --no-capture-output -n slm_pvs python -u \
  neural_instance_culling/benchmark/evaluate_viewcell_image_per.py \
  --split validation --split-source pose_csr --max-viewcells 2 \
  --subposes-per-viewcell 1 --render-schema-only --device cpu \
  --output-dir /tmp/neuralstream_m5_split_smoke
```

该 smoke 成功生成完整的 `componentGlobalId` manifest，包含全部 `3273` 个本地 GLB；输出明确标记 `formalImageEvaluationReady=false`，没有被计入正式图像结果。

## 质量门判断

本次修复解决了 split 语义和完整遍历入口问题，但没有证明图像质量。M5 仍需在冻结 checkpoint 和 calibration 阈值下，分别完整运行 validation/calibration，并在 test 冻结后只运行一次真实 60°实例 ID 渲染。现有 4 个 view-cell 的旧模型 smoke 仅作为开发证据，不能进入论文主表。
