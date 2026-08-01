# IFC Metropolis 完整构件实例化与可见性训练

## 日期

2026-07-15

## 目的

原始 `ifcbench_fantasy_metropolis_source` 将 41298 个 IFC 构件分别保存为 41298 个 GLB,没有利用场景中大量重复建筑构件。此前按最小网格连通区域恢复实例会把构件进一步拆碎,不符合实例级可见性任务。本次直接使用原始 `sub_*.glb` 作为构件边界,识别完整构件之间的严格刚体复用关系,并在新的实例/GLB 映射上重新训练可见性与下载优先级统一模型。

## 实例化结果

完整构件经过几何、拓扑、顶点颜色指纹筛选后,使用刚体配准逐顶点验证。构件 ID、世界空间 AABB 和训练标签均保持不变。

| 指标 | 结果 | 含义 |
| --- | ---: | --- |
| 实例数 | 41298 | 原始构件 ID 和实例粒度不变 |
| 原型 GLB 数 | 3669 | 前端实际需要管理和下载的唯一几何资产 |
| 复用实例数 | 37629 | 通过原型 GLB 内实例矩阵表示的构件 |
| 实例复用率 | 91.12% | `复用实例数 / 实例数` |
| 最大同型组 | 1362 | 同一个原型包含的最大实例数 |
| 原始 GLB 字节 | 561604720 | 原 41298 个 LOD0 构件 GLB 总字节 |
| 新 GLB 字节 | 183335200 | 3669 个实例化原型 GLB 总字节 |
| GLB 字节削减 | 67.36% | 完整场景几何资产的离线字节削减 |
| 最大单顶点配准误差 | 0.633 mm | 低于 5 mm 拒绝阈值 |

## 粒度解耦修复

训练接入时发现旧实现把两个不同粒度混在一起:

- 实例级点云输入用于提取每个构件自身的几何、方向和上下文特征,必须按实例 ID 读取。
- GLB 级映射用于下载、解码、缓存排序和同资产关系,应按新 `globalGlbId` 聚合。

旧场景中一个实例对应一个 GLB,这个问题不会显现。实例化后多个实例共享一个 GLB,继续用 GLB 编号索引实例点云会丢失实例自身的方向信息,还会错误保留 41298 个 GLB 槽位。修复后模型输入为 41298 行实例点云,下载调度输出为 3669 个 GLB,两个粒度通过 `instance_to_glb` 关系连接。

修改文件:

- `neural_instance_culling/model/directional_occlusion_proxy_encoder_model.py`
- `neural_instance_culling/model/current_pvs_utils.py`
- `neural_instance_culling/model/train_directional_occlusion_proxy_encoder.py`
- `neural_instance_culling/tools/glb_instancer/src/sub_pipeline.mjs`

首次未解耦的 full40 输出已删除,没有进入前端资产或服务器。

## 训练资源

- 数据集:`neural_instance_culling/dataset/out/ifcbench_fantasy_metropolis_instanced_v2_viewcell_colorid_k4`
- 训练 viewcell:21830
- test viewcell:2684
- 平均候选实例:9991.98
- 平均真实可见实例:977.60
- 遮挡证据:`directional_occlusion_evidence_ifcbench_fantasy_metropolis_instanced_v2_k4`
- 实例点云:`ifcbench_fantasy_metropolis_instanced_v2_glb_points_v3.bin`,历史字段 `numGlbs` 在此表示实例对齐的点云行数
- 新运行时元数据:`ifcbench_fantasy_metropolis_instanced_v2/assets/runtimeVisibilityMeta.json`

数据检查确认 `visible_ids` 是 `candidate_ids` 的子集,实例 ID 最大值未越界,新旧运行时元数据的 41298 个构件 AABB 完全一致。

## 训练命令

```bash
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
conda run --no-capture-output -n slm_pvs python -u \
  neural_instance_culling/model/train_directional_occlusion_proxy_encoder.py \
  --dataset-dir neural_instance_culling/dataset/out/ifcbench_fantasy_metropolis_instanced_v2_viewcell_colorid_k4 \
  --evidence-dir neural_instance_culling/dataset/out/directional_occlusion_evidence_ifcbench_fantasy_metropolis_instanced_v2_k4 \
  --glb-points neural_instance_culling/dataset/out/ifcbench_fantasy_metropolis_instanced_v2_glb_points_v3.bin \
  --runtime-meta ifcbench_fantasy_metropolis_instanced_v2/assets/runtimeVisibilityMeta.json \
  --glb-index ifcbench_fantasy_metropolis_instanced_v2/assets/glbIndex.json \
  --glb-root ifcbench_fantasy_metropolis_instanced_v2/assets \
  --output-dir neural_instance_culling/model/out/pvs_directional_occlusion_proxy_ifcbench_fantasy_metropolis_instanced_v2_k4_full40 \
  --experiment-name pvs_directional_occlusion_proxy_ifcbench_fantasy_metropolis_instanced_v2_k4_full40 \
  --loss-profile rvl_strong_v2 \
  --epochs 40 --steps-per-epoch 900 --pose-set-batch-size 1 \
  --eval-every 2 --eval-pose-steps 256 --device cuda
```

40 个 epoch 均无非有限 loss 和非有限梯度跳过。最终 checkpoint 配置为 41298 个实例、3669 个 GLB。

## 是否保留

该实验保留为 metropolis 当前运行版本。图像 PER、miss pixel rate 和 wrong-ID pixel rate 尚未实现本次重评,不能把集合 recall 单独解释为最终画面无损。移动端真实延迟也尚未测试。
