# V4 综合可见性主线训练协议

日期：2026-08-21
当前状态：三种子完整模型和核心消融已完成。

## 目的

本文固定当前论文模型的训练和消融协议，并说明完整矩阵的复现入口。

当前模型只优化实例可见性。视觉效用、下载优先级、GLB 字节预算和资源调度不进入训练损失，也不参与 checkpoint 排名。

## 数据协议

主实验使用 `pose_csr_hkust_v3_main_stratified_calibration_fov66_v1`：

| Split | View-cell 数量 | 用途 |
|---|---:|---|
| train | 5,926 | 参数训练和 train-only 遮挡关系编码 |
| calibration | 659 | 为每个 checkpoint 独立冻结安全阈值 |
| validation | 730 | 选择 checkpoint、模型配置和消融结论 |
| test | 684 | 模型、阈值和结论冻结后执行一次 |

真实显示相机使用垂直 FOV `60°`，采样、后退候选和模型查询使用 `66°`。所有成员使用相同 pose、候选集合、GT、可见权重和关系监督来源，不向候选补入 GT。

Calibration 从主数据按场景类别、候选数、GT 数、观察方向和可见权重分层抽取。Validation 和 test split 固定不变；test 不参与参数扫描、阈值或 checkpoint 选择。

## 固定完整模型

当前 Full 由三个部分组成：

1. 分层遮挡关系先验和逐实例校准生存场；
2. 视点区域矩包络频谱查询；
3. 逐 pose 平衡分类、单侧 RVL weighted-recall 保护和困难边界 logit 间隔。

运行时固定表是 `96` 维几何加 `28` 维生存场，共 `124` 个 FP16 数值。离线关系图、层次聚合、删失监督和逐实例残差训练不进入浏览器运行时。

当前损失固定为：

```text
逐 pose 平衡分类
+ 0.30 × 单侧 RVL weighted-recall 保护
+ 0.20 × 渐入后的困难边界 logit 间隔
```

## 训练配置

| 项目 | 固定值 |
|---|---|
| 随机种子 | `20260801/20260802/20260803` |
| 训练预算 | 每成员 `40 epoch × 900 step` |
| Batch | 每批 `4` 个 pose |
| Checkpoint/评价间隔 | 每 `4` epoch |
| Calibration bootstrap | `10,000` 次 |
| 学习率 | `2e-4` |
| 生存场方向秩 | `4` |
| Test | 训练和 validation 阶段禁止读取 |

每个 checkpoint 只能使用自己的 calibration split 冻结阈值。安全工作点由 calibration 的 aggregate weighted recall 点估计及单侧 95% 下界均大于 `0.99` 判定。Validation 在同一冻结阈值下报告安全指标，并比较 balanced accuracy、precision、accuracy、useful cull 和平均预测数。

## 核心消融

当前正式比较包括：

| 成员 | 唯一变化 |
|---|---|
| Full | 全部当前模块 |
| 去除分层关系 | 用容量匹配的几何路径替代真实关系编码 |
| 去除生存场 | 删除逐实例生存表和相关监督 |
| 通用 28 维 | 用同容量可训练实例表示替代结构化生存场 |
| 去除区域矩包络 | 改为中心视点查询 |
| 去除 RVL 保护 | 召回保护权重设为零 |
| 去除困难边界 | logit 间隔权重设为零 |

所有成员均完成三种子 `40 × 900` 从头训练，并保留完整训练曲线和评价记录。

## 执行入口

```bash
conda run -n slm_pvs python \
  neural_instance_culling/benchmark/run_pvs.py preflight \
  --data-root /mnt/sda/rhyang/slm

conda run -n slm_pvs python \
  neural_instance_culling/benchmark/run_pvs.py all \
  --data-root /mnt/sda/rhyang/slm \
  --gpu-ids 0 1 2 3
```

正式汇总：

```bash
conda run -n slm_pvs python \
  neural_instance_culling/benchmark/run_pvs.py summarize \
  --data-root /mnt/sda/rhyang/slm
```

## 完成结果

当前 Full 三个种子选中 epoch `32/36/40`，calibration 阈值为 `0.42/0.68/0.58`，三个 validation 成员在同一冻结阈值下均满足 weighted-recall 安全要求。当前前端使用 seed `20260802`、epoch `36`、阈值 `0.68`。

完整方法、三种子结果和 10,000 次 paired-bootstrap 消融结论见
[V4 主线模型与核心消融](../evaluation/pvs_mainline_core_ablation_paper_analysis_2026-08-28.md)。机器可读结果为：

```text
neural_instance_culling/benchmark/out/pvs_v4_integrated_visibility_mainline_v1/paper_core_ablation_summary.json
```

当前证据表明：生存场整体和分层关系是主要有效模块，视点区域矩包络有较小但稳定的分类与剔除收益；RVL 保护主要改善安全工作点下的边界和效率，困难边界项主要改善 balanced accuracy。硬件图像评价、浏览器 WebGPU 延迟和真实移动端性能仍需单独回填。
