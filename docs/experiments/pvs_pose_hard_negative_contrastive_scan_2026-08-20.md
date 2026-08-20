# Pose 内困难负例对比学习快速验证

日期：2026-08-20

## 目的

验证训练期对比学习能否改善 HKUST View-cell PVS 的正负分数分离，在保持 weighted recall 的前提下提高 balanced accuracy、accuracy 和 precision。实验不改变候选集合、GT、数据 split、模型运行输入、前端网络或导出资产。

## 方法

使用现有视角条件查询特征和 Fourier 视角特征作为输入，增加一个仅在训练期存在的 `输入维度 -> 64 -> 32` 投影头。每个 pose 独立选择：

- 最多 64 个可见实例，优先保留 `visible_weights` 较高的实例；
- 最多 256 个不可见实例，优先保留当前可见性分数最高的困难负例。

投影向量先做单位化。每个可见实例作为锚点，同 pose 的其他可见实例构成正集合，困难不可见实例进入对比损失分母；高视觉权重正例获得更高锚点权重。不同 pose 之间不构造正对，避免把无关视点的多模态可见状态强行聚合。

对比投影头独立保存在训练 checkpoint 中，不放入模型运行状态和前端导出资产。正式推理仍只运行原可见性网络，因此前端延迟、输入维度和特征表大小不增加。

## 快速扫描

固定条件：

- 数据：`pose_csr_hkust_v3_main_stratified_calibration_fov66_v1`；
- train / validation / calibration / test：5926 / 730 / 659 / 684；
- 随机种子：`20260801`；
- 原始损失：`legacy + evidence RVL，RVL 总权重 0.42`；
- 训练：12 epoch，每 epoch 300 step，pose batch 为 2；
- 温度：0.10；投影维度：32；
- test 不读取，快速扫描不执行 bootstrap。

扫描对比损失权重：`0.02`、`0.05`、`0.10`。历史同口径无对比项 `w042 quick12` 作为零权重对照，不重复训练。

运行命令模板：

```bash
CUDA_VISIBLE_DEVICES=<gpu> conda run -n slm_pvs python -u \
  neural_instance_culling/model/train_directional_occlusion_proxy_encoder.py \
  --dataset-dir neural_instance_culling/dataset/out/pose_csr_hkust_v3_main_stratified_calibration_fov66_v1 \
  --evidence-dir neural_instance_culling/dataset/out/directional_occlusion_evidence_hkust_v3_main_stratified_calibration_fov66_v1 \
  --glb-points neural_instance_culling/dataset/out/glb_points_v3_formal_hkust_fov66.bin \
  --runtime-meta hkust-v3/assets/runtimeVisibilityMeta.json \
  --glb-index hkust-v3/assets/glbIndex.json --glb-root hkust-v3/assets \
  --output-dir neural_instance_culling/model/out/<experiment> \
  --experiment-name <experiment> --epochs 12 --steps-per-epoch 300 \
  --pose-set-batch-size 2 --eval-every 2 --loss-profile legacy \
  --rvl-mode evidence --rvl-loss-weight 0.42 \
  --contrastive-loss-weight <weight> --contrastive-temperature 0.10 \
  --target-weighted-recall 0.99 --calibration-pose-recall-floor 0 \
  --calibration-point-floor 0.9925 --calibration-bootstrap-replicates 0 \
  --seed 20260801 --device cuda --skip-final-test
```

## 评价与判定

每个 checkpoint 的阈值只由 calibration 冻结。首先比较 validation weighted recall，然后同时报告 pose recall、precision、accuracy、balanced accuracy、useful cull、bad cull 和平均预测实例数。

对比学习只有在 weighted recall 不下降的情况下改善 balanced accuracy、accuracy、precision 或有效剔除，才进入后续长训候选。若只增加预测数并降低 precision，则判定为无效；若有效，后续再围绕最佳权重和温度做窄范围扫描。正式 test 只在模型与阈值最终冻结后读取一次。

## 实现文件

- `neural_instance_culling/model/train_directional_occlusion_proxy_encoder.py`
- `neural_instance_culling/model/directional_occlusion_proxy_encoder_model.py`
- `neural_instance_culling/model/tests/test_pose_hard_negative_contrastive_loss.py`

## 初步验证

CUDA smoke 已通过。对比损失、梯度和导出均为有限值；投影头未进入运行特征表，运行特征表仍为 13,257,024 字节。

## 快速扫描结果

以下结果均来自相同 seed、相同 730 个 validation pose，以及各 checkpoint 自己在 calibration 冻结的阈值。每个成员选择 validation weighted recall 最高的 epoch；这些是单种子快速结果，不含 bootstrap，也没有读取 test。

| 成员 | 最佳 epoch | 阈值 | weighted recall | pose recall | precision | accuracy | balanced accuracy | useful cull | bad cull | 平均预测数 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 无对比项 | 8 | 0.030 | 0.98087 | 0.8890 | 0.5549 | 0.8929 | 0.8680 | 0.8022 | 0.0024 | 543.2 |
| 对比权重 0.02，温度 0.10 | 10 | 0.020 | 0.98074 | 0.8749 | 0.6139 | 0.8949 | 0.8624 | 0.8044 | 0.0026 | 551.9 |
| 对比权重 0.05，温度 0.10 | 8 | 0.030 | **0.98404** | 0.8923 | 0.5588 | 0.8917 | 0.8690 | 0.8008 | 0.0024 | 560.8 |
| 对比权重 0.10，温度 0.10 | 8 | 0.030 | 0.98141 | 0.8868 | **0.5860** | **0.8959** | **0.8687** | **0.8052** | 0.0025 | **537.1** |
| 对比权重 0.02，温度 0.20 | 10 | 0.020 | 0.98096 | 0.8771 | 0.6010 | 0.8930 | 0.8624 | 0.8025 | 0.0027 | 565.5 |

权重 0.10、温度 0.10 是当前综合候选。相对无对比项，它的 weighted recall 提高约 0.05 个百分点，precision 提高约 3.11 个百分点，accuracy 提高约 0.30 个百分点，balanced accuracy 提高约 0.07 个百分点，useful cull 提高约 0.30 个百分点，同时平均少预测约 6 个实例。权重 0.05 更偏向召回，weighted recall 提高约 0.32 个百分点，但资源效率略有下降。

温度 0.20 没有优于温度 0.10。它提高了一部分 precision，但 pose recall、balanced accuracy 和平均预测数的综合结果较弱，不进入下一轮候选。

## 更保守 calibration 复评

将 calibration weighted recall 点估计余量提高到 0.995 后，validation 结果如下。历史无对比项使用同一复评口径。

| 成员 | 阈值 | weighted recall | precision | accuracy | balanced accuracy | useful cull | bad cull | 平均预测数 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 无对比项 | 0.010 | 0.98691 | 0.4765 | 0.8768 | 0.8691 | 0.7856 | 0.0020 | 640.3 |
| 对比权重 0.02 | 0.005 | **0.98776** | 0.5215 | 0.8762 | 0.8633 | 0.7849 | **0.0019** | 664.7 |
| 对比权重 0.05 | 0.020 | 0.98497 | **0.5310** | **0.8856** | 0.8692 | **0.7946** | 0.0022 | **599.0** |
| 对比权重 0.10 | 0.010 | 0.98645 | 0.5046 | 0.8781 | **0.8697** | 0.7869 | **0.0019** | 641.6 |

在更保守工作点下，权重 0.10 仍保留 precision、accuracy、balanced accuracy 和 useful cull 的改善，weighted recall 仅低约 0.05 个百分点。权重 0.02 的 weighted recall 最高，但需要扩大预测集合，不能作为综合最优。

## 当前结论

Pose 内困难负例对比学习能够改善模型分数分离，不是单纯通过扩大预测集合换取召回。权重 0.10 在单种子 quick12 中形成了最清楚的综合收益，而且没有增加前端资产或运行成本。

该改进尚未独立解决 validation weighted recall 未达到 0.99 的问题，因此当前只能晋级为长训候选，不能直接替换默认模型。下一步应围绕权重 0.08–0.12 做窄范围扫描，并至少使用三个种子确认收益稳定性；正式安全结论仍需 calibration 单侧置信下界和冻结后的 validation/test 评价。
