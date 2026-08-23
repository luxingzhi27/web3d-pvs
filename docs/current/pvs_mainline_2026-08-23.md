# 当前 PVS 论文主线

日期：2026-08-23

## 目标与边界

当前论文模型只处理后退扩大视锥候选上的实例可见性。模型使用同一 view-cell 的保守可见并集作为监督，在 calibration 上冻结每个 checkpoint 自己的阈值，并以 validation weighted recall 及其单侧 95% 下界均大于 `0.99` 作为安全门。test 不参与训练、阈值选择或 checkpoint 排名。

本阶段不改变当前部署方向代理模型、前端默认阈值或资产。视觉效用、下载优先级和 GLB 字节预算没有进入本轮可见性损失。

## 固定架构

模型由三部分组成：

1. **分层遮挡关系与逐实例校准生存场**：离线关系编码器读取 train split 的真实三角形遮挡边、实例几何和层次分组，生成每个实例的共享关系先验；逐实例残差再校准该实例的 28 个生存系数。导出时两者融合，因此浏览器只需读取固定实例表，不执行图传播或邻居查询。
2. **视点区域矩包络频谱查询**：浏览器以当前 view-cell 锚点、区域轴和实例深度查询固定生存系数。频谱矩描述区域内方向变化，避免前端展开多个 subpose；每个 view-cell 仍只执行一次候选批量推理。
3. **综合可见性损失**：逐 pose 平衡 BCE 负责普通正负分类，单侧 RVL 项只约束重要可见实例的加权召回，共享困难边界项同时作用于 logit 和训练期表示。对比投影头不导出到运行时。

运行时每个实例保存 `96` 维几何特征和 `28` 维融合生存系数，共 `124` 个半精度值。关系图、层次分组、逐实例校准残差和训练期投影头均不进入前端资产。

## 代码入口

| 职责 | 文件 |
|---|---|
| 模型 | `neural_instance_culling/model/pvs_model.py` |
| 分层关系编码 | `neural_instance_culling/model/common/relation_encoder.py` |
| 区域频谱查询 | `neural_instance_culling/model/common/spectral_query.py` |
| 综合损失 | `neural_instance_culling/model/common/visibility_loss.py` |
| 训练 | `neural_instance_culling/model/train_pvs.py` |
| 导出 | `neural_instance_culling/model/export_pvs.py` |
| 实验编排 | `neural_instance_culling/benchmark/run_pvs.py` |
| checkpoint 评价 | `neural_instance_culling/benchmark/evaluate_pvs.py` |
| 三种子汇总 | `neural_instance_culling/benchmark/summarize_pvs.py` |
| 正式协议重审计 | `neural_instance_culling/benchmark/reaudit_pvs.py` |

正式 runner 固定使用 `5926 train / 659 calibration / 730 validation / 684 test` 的主 split、三个随机种子和五个变体。当前实验输出前缀保持 `pvs_v4_integrated_visibility_mainline_v1`，用于继续读取已完成的数 GB checkpoint 和评价结果；代码文件名不再重复携带架构版本和日期。

## 运行命令

```bash
conda run -n slm_pvs python \
  neural_instance_culling/benchmark/run_pvs.py \
  preflight --data-root /mnt/sda/rhyang/slm

conda run -n slm_pvs python \
  neural_instance_culling/benchmark/reaudit_pvs.py \
  preflight --data-root /mnt/sda/rhyang/slm \
  --bootstrap-replicates 10000
```

如需重新执行正式重审计：

```bash
conda run -n slm_pvs python \
  neural_instance_culling/benchmark/reaudit_pvs.py \
  run --data-root /mnt/sda/rhyang/slm \
  --bootstrap-replicates 10000 --gpu-ids 0 1 2 3
```

## 正式 validation 结果

三个 Full 种子分别冻结 epoch `40/24/32`，阈值为 `0.46/0.60/0.56`。所有阈值只来自各自 calibration；随后在相同 `730` 个 validation pose 上评价。

| 指标 | Full |
|---|---:|
| Pose precision | 0.48291 |
| Pose recall | 0.97543 |
| Aggregate precision | 0.19799 |
| Aggregate recall | 0.98063 |
| Weighted recall | 0.99797 |
| Weighted recall 单侧 95% 下界 | 0.99658 |
| Aggregate accuracy | 0.90827 |
| Aggregate balanced accuracy | 0.94360 |
| Useful cull | 0.88575 |
| Bad cull | 0.000445 |
| 平均预测实例数 | 540.34 |

消融显示分层关系先验具有稳定贡献：相对移除关系先验，aggregate precision 提高 `0.03757`、balanced accuracy 提高 `0.01545`、useful cull 提高 `0.02785`，平均少预测 `131.98` 个实例，相关 95% 区间均不跨零。区域矩包络尚未形成稳定收益；RVL 的主要分类差值跨零；当前对比分离项对 precision、accuracy 和 useful cull 有稳定负面影响，因此不能作为已验证贡献。完整置信区间见 `docs/evaluation/pvs_mainline_validation_2026-08-23.md`。

## 代码清理记录

清理前提交为 `3161453`。本次将主线模型、训练、导出、评价、汇总和 runner 收敛为短名称，并删除已被替代的旧整合网络、尾部探针、独立安全前沿实验及其专用测试和重复计划。以下内容保持不动：

- 当前部署方向代理模型及其训练、导出和前端资产；
- 2026-08-11 正式矩阵、2026-08-12 Fourier 补充矩阵和三角形深度层硬件缓存；
- 通用 Color-ID 图像评价、下载轨迹、冻结 test 和 WebGPU/硬件 GPU 工具；
- 当前论文模型 checkpoint、训练日志和正式 benchmark 输出。

本次只整理源码和文档，不改变数据集、候选、GT、阈值、checkpoint、正式指标或前端默认路径。
