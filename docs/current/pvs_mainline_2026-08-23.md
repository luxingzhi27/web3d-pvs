# 当前 PVS 论文主线

日期：2026-08-23；本页更新于 2026-09-09

## 目标与边界

当前论文模型只处理后退扩大视锥候选上的实例可见性。模型使用同一 view-cell 的保守可见并集作为监督，在 calibration 上为每个 checkpoint 冻结安全阈值；validation 在同一阈值下报告 weighted recall 及其单侧 95% 下界，用于 checkpoint 和配置比较。test 不参与训练、阈值选择或 checkpoint 排名。

当前 HKUST 前端已导出该主线的冻结 V4 成员，运行目录为 `pvs_mainline_v4`，阈值为 `0.6800000071525574`。视觉效用、独立下载优先级和 GLB 字节预算没有进入本轮可见性损失；当前下载调度使用实例可见性概率聚合，不能描述成已训练的独立下载头。

## 固定架构

模型由三部分组成：

1. **分层遮挡关系与逐实例校准生存场**：离线关系编码器读取 train split 的真实三角形遮挡边、实例几何和层次分组，生成每个实例的共享关系先验；逐实例残差再校准该实例的 28 个生存系数。导出时两者融合，因此浏览器只需读取固定实例表，不执行图传播或邻居查询。
2. **视点区域矩包络频谱查询**：浏览器以当前 view-cell 锚点、区域轴和实例深度查询固定生存系数。频谱矩描述区域内方向变化，避免前端展开多个 subpose；每个 view-cell 仍只执行一次候选批量推理。
3. **综合可见性损失**：逐 pose 平衡 BCE 负责普通正负分类，单侧 RVL 项约束重要可见实例的加权召回，共享困难边界项只作用于最终 logit。当前 Full 已关闭表征对比投影头，不再训练或导出该分支。

运行时每个实例保存 `96` 维几何特征和 `28` 维融合生存系数，共 `124` 个半精度值。关系图、层次分组和逐实例校准残差均不进入前端资产。

## 代码入口

| 职责 | 文件 |
|---|---|
| 模型 | `neural_instance_culling/model/pvs_model.py` |
| 固定几何表导出 | `neural_instance_culling/model/prepare_fixed_geometry_features.py` |
| 分层关系编码 | `neural_instance_culling/model/common/relation_encoder.py` |
| 区域频谱查询 | `neural_instance_culling/model/common/spectral_query.py` |
| 综合损失 | `neural_instance_culling/model/common/visibility_loss.py` |
| 训练 | `neural_instance_culling/model/train_pvs.py` |
| 导出 | `neural_instance_culling/model/export_pvs.py` |
| 实验编排 | `neural_instance_culling/benchmark/run_pvs.py` |
| checkpoint 评价 | `neural_instance_culling/benchmark/evaluate_pvs.py` |
| 正式消融汇总 | `neural_instance_culling/benchmark/summarize_core_ablation.py` |

正式 runner 固定使用 `5926 train / 659 calibration / 730 validation / 684 test` 的主 split 和三个随机种子。当前核心比较包括 Full 及六个结构、表示和损失消融。实验输出前缀为 `pvs_v4_integrated_visibility_mainline_v1`，用于读取已完成的 checkpoint 和评价结果。

## 运行命令

```bash
conda run -n slm_pvs python \
  neural_instance_culling/benchmark/run_pvs.py \
  preflight --data-root /mnt/sda/rhyang/slm

conda run -n slm_pvs python \
  neural_instance_culling/benchmark/run_pvs.py \
  summarize --data-root /mnt/sda/rhyang/slm
```

重新训练并汇总完整矩阵：

```bash
conda run -n slm_pvs python \
  neural_instance_culling/benchmark/run_pvs.py \
  all --data-root /mnt/sda/rhyang/slm \
  --gpu-ids 0 1 2 3
```

## 正式 validation 结果

三个种子分别冻结 epoch `32/36/40`，阈值为 `0.42/0.68/0.58`。所有阈值只来自各自 calibration；随后在相同 `730` 个 validation pose 上评价。

| 指标 | Full |
|---|---:|
| Pose precision | 0.48463 |
| Pose recall | 0.97722 |
| Aggregate precision | 0.21353 |
| Aggregate recall | 0.97934 |
| Weighted recall | 0.99729 |
| Weighted recall 单侧 95% 下界 | 0.99535 |
| Aggregate accuracy | 0.91646 |
| Aggregate balanced accuracy | 0.94716 |
| Useful cull | 0.89397 |
| Bad cull | 0.000474 |
| 平均预测实例数 | 501.17 |

消融显示生存场整体贡献最大，分层关系具有稳定贡献，结构化生存场优于同容量通用 28 维实例记忆；视点区域矩包络带来较小但稳定的分类和剔除收益。RVL 保护在各自安全阈值下主要改善边界和有效剔除，困难边界间隔主要改善 balanced accuracy、资源量和分数分布。完整方法和置信区间见 [`../evaluation/pvs_mainline_core_ablation_paper_analysis_2026-08-28.md`](../evaluation/pvs_mainline_core_ablation_paper_analysis_2026-08-28.md)。
