# M11 航向泛化与跨场景适配协议

日期：2026-08-01
状态：航向数据划分已生成，方向遮挡证据正在重建；正式训练和跨场景适配尚未完成。

## 目的

现有四路空间划分把物理位置隔离开，可以检查相邻视点泄漏，但不能单独回答模型是否能处理训练时没有出现过的观察方向。本实验增加一个正交的航向留出协议：整段相机航向扇区只属于一个 split，保持原始候选、可见实例、可见权重、相机 FOV 和实例特征维度不变。

当前模型的固定实例表是按场景离线生成的，包含场景实例的几何、上下文和方向遮挡代理；它不是一个可以直接跨场景复用的固定 token 表。因此跨场景实验分成两种明确情况：

- **零样本查询头迁移**：只迁移学习到的点云编码器和视线查询头，在目标场景重新计算目标场景的固定实例表，不更新参数；目标 calibration 只用于选择安全阈值，不用于训练。
- **少样本适配**：在目标场景的训练视点中只开放预注册的 1%、5% 和 10% 子集，重新训练或微调同一结构；validation/calibration/test 保持完整且不参与参数更新。

如果目标场景的输入维度、实例映射或资源语义不一致，实验必须失败并记录 schema 不兼容，不能通过补零、复制实例或把一个场景的实例编号强行映射到另一个场景来制造“迁移结果”。

## 航向划分

脚本：`neural_instance_culling/dataset/build_directional_viewcell_split.py`。航向定义为：`yaw=0°` 指向世界坐标 `-Z`，正方向朝 `+X`。默认将 360° 划为 20 个 18° 扇区，整扇区分配为：

| split | 扇区 | 说明 |
|---|---:|---|
| train | 其余 16 个扇区 | 用于参数更新 |
| validation | 14 | 用于 checkpoint 诊断 |
| calibration | 16 | 只用于冻结安全阈值 |
| test | 18 | 冻结后一次评测 |
| guard | 19 | 不参与训练和工作点选择 |

该协议是方向外推测试，不是空间隔离测试；同一物理位置可能在不同航向 split 出现，这是有意保留的变量。俯仰范围和类别分布单独写入 manifest，不能将 yaw 留出结果解释成完整球面方向泛化。

## 已生成资源

| 场景 | 方向 manifest | pose view | 计数（train/validation/calibration/test/guard） |
|---|---|---|---:|
| HKUST v3 | `dataset/out/directional_split_hkust_v3_yaw20_20260801/manifest.json` | `dataset/out/pose_csr_hkust_v3_directional_yaw20_fov66_v1` | 6406/393/420/378/402 |
| Metropolis | `dataset/out/directional_split_metropolis_yaw20_20260801/manifest.json` | `dataset/out/pose_csr_metropolis_directional_yaw20_fov66_v1` | 20454/1149/2271/2271/1107 |

两个 PoseCSR view 通过硬链接复用原始二进制。候选语义仍为“每个成功子姿态独立计算后退 AABB 候选并取并集，不补入 GT 可见实例”；两个新目录的 `candidateMissVisible=0`、`candidateVisibleUnionAdded=0`，FOV 仍为模型 `66°`、真实渲染 `60°`。

## 方向证据与运行命令

方向遮挡代理的离线监督只能使用方向训练 split。若继续复用空间训练证据，会把方向 test 的可见源信息带入固定代理表，因而不具备泛化解释。目前以下两条命令正在独立日志中运行：

```bash
conda run --no-capture-output -n slm_pvs python -u neural_instance_culling/dataset/build_directional_occlusion_evidence.py \
  --dataset-dir neural_instance_culling/dataset/out/pose_csr_hkust_v3_directional_yaw20_fov66_v1 \
  --runtime-meta hkust-v3/assets/runtimeVisibilityMeta.json \
  --output-dir neural_instance_culling/dataset/out/directional_occlusion_evidence_hkust_v3_directional_yaw20_fov66_v1 \
  --splits train --direction-bins 8 --depth-shells 3 --source-k 8

conda run --no-capture-output -n slm_pvs python -u neural_instance_culling/dataset/build_directional_occlusion_evidence.py \
  --dataset-dir neural_instance_culling/dataset/out/pose_csr_metropolis_directional_yaw20_fov66_v1 \
  --runtime-meta ifcbench_fantasy_metropolis_source/assets/runtimeVisibilityMeta.json \
  --output-dir neural_instance_culling/dataset/out/directional_occlusion_evidence_metropolis_directional_yaw20_fov66_v1 \
  --splits train --direction-bins 8 --depth-shells 3 --source-k 8
```

证据完成后，正式方向模型至少需要一个固定 checkpoint、独立 calibration 和冻结 test。报告必须同时给出普通 pose 指标、weighted recall、有效剔除、错误剔除和按航向扇区的长尾结果；方向 test 不能再扫描阈值。

## 跨场景边界

HKUST 和 Metropolis 的实例数、GLB 映射、场景边界和离线特征表不同。当前工程上可行的迁移路径是共享同维度的学习参数，并在目标场景重新运行离线编码器；浏览器仍只接收目标场景自己的固定特征表。若零样本迁移不能达到 weighted-recall 安全门，应报告“场景特定离线预处理”这一真实边界，再比较 1%/5%/10% 少样本适配的收益和适配成本，不把失败包装成通用模型。

## 当前质量门

- 方向划分：通过，manifest、哈希和 split 计数已生成。
- 候选语义：通过，两个方向 PoseCSR view 未发生 GT 补入。
- 方向遮挡证据：进行中，完成前不训练方向模型。
- 方向泛化指标：未执行。
- 零/少样本跨场景迁移：未执行。

本记录只描述协议和已核验资源，不把正在运行的证据生成或后续计划当成结果。
