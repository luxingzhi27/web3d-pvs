# Pose 内困难负例对比学习快速验证

日期：2026-08-20 至 2026-08-21

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

以下结果均来自相同 seed、相同 730 个 validation pose，以及各 checkpoint 自己在 calibration 冻结的阈值。早期对比学习扫描按 validation weighted recall 观察趋势；后续安全工作点扫描统一改为先过滤 `validation weighted recall > 0.99` 的 epoch，再依次按 accuracy、balanced accuracy、precision、useful cull 和更少平均预测数排序。这些是单种子快速结果，不含 bootstrap，也没有读取 test。

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

## 对比学习阶段结论

Pose 内困难负例对比学习能够改善模型分数分离，不是单纯通过扩大预测集合换取召回。权重 0.10 在单种子 quick12 中形成了最清楚的综合收益，而且没有增加前端资产或运行成本。

该阶段使用较宽松的 calibration 余量时，validation weighted recall 尚未达到 0.99，因此只能说明对比学习有改善分数分离的潜力。后续安全工作点扫描提高 calibration 点估计余量，并在完全不读取 test 的条件下继续调整正例安全项和误报抑制项。

## 安全工作点损失参数扫描

2026-08-20 继续在本实验分支执行。现有对比权重 0.10 checkpoint 的 calibration 余量诊断表明：

| Calibration 点估计下限 | 冻结阈值 | Validation weighted recall | Accuracy | Balanced accuracy | Precision | 平均预测数 |
|---:|---:|---:|---:|---:|---:|---:|
| 0.9975 | 0.002 | 0.99384 | 0.8506 | 0.8711 | 0.3746 | 766.1 |
| 0.9980 | 0.002 | 0.99384 | 0.8506 | 0.8711 | 0.3746 | 766.1 |
| 0.9990 | 0.001 | 0.99596 | 0.8307 | 0.8675 | 0.3167 | 822.5 |

后续扫描固定 calibration 点估计下限为 0.9975。该设置完全由 calibration 冻结阈值，validation 用于损失超参数和 checkpoint 比较，test 不读取。成员必须先满足 validation weighted recall 大于 0.99，再按 accuracy、balanced accuracy、precision、useful cull 和平均预测数比较。

第一阶段固定对比权重 0.10、温度 0.10 以及原始 RVL 参数，只扫描视觉安全损失权重 `0.025/0.05/0.10/0.20`。该损失只提高高 `visible_weights` 可见实例的正类尾部，不直接奖励不可见实例，目标是在保持安全召回时提高可冻结阈值，避免通过全量增加预测来过门。

第一阶段每组仍训练 12 epoch、每 epoch 300 step、seed `20260801`。根据第一阶段相对最优成员，第二阶段再扫描 RVL 假负/假正比例、Tversky 假负系数和预测数量约束；无论中间结果是否达到正式置信下界，扫描均完整运行并保留结果。

### 第一阶段：正例安全权重

固定对比损失权重 0.10、温度 0.10、RVL 总权重 0.42 和其余 legacy 损失参数，只调整高视觉权重正例的安全损失。下表为每个成员所有满足 `validation weighted recall > 0.99` 的 epoch 中 accuracy 最高者。

| 正例安全权重 | epoch | 阈值 | weighted recall | pose recall | precision | accuracy | balanced accuracy | useful cull | bad cull | 平均预测数 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.025 | 12 | 0.002 | 0.99270 | 0.9353 | 0.4273 | 0.8639 | **0.8738** | 0.7721 | **0.00137** | 714.4 |
| 0.050 | 12 | 0.002 | 0.99263 | 0.9365 | 0.4272 | 0.8630 | 0.8733 | 0.7712 | 0.00136 | 712.1 |
| 0.100 | 6 | 0.002 | 0.99199 | 0.9205 | **0.4363** | 0.8639 | 0.8658 | 0.7723 | 0.00156 | 688.4 |
| 0.200 | 8 | 0.005 | 0.99261 | 0.9286 | 0.4308 | **0.8671** | 0.8717 | **0.7755** | 0.00162 | **680.2** |

权重 0.20 在保持 weighted recall 安全要求时获得最高 accuracy、useful cull 和最少平均预测数，因此冻结为第二阶段共同设置。较小权重可以略微提高普通 pose recall 或 balanced accuracy，但会多预测约 8 至 34 个实例，不能作为 accuracy 优先目标下的综合最优。

### 第二阶段：损失分量单因素扫描

以正例安全权重 0.20 为基准，每组只修改一个损失参数。`RVL=0.50` 表示将 RVL 总权重从 0.42 提到 0.50；`RVL FP=1.25` 表示只将 RVL 内假阳性分量系数从 1.00 提到 1.25；计数和排序分别从 0.12、0.50 提到 0.18、0.65。

| 设置 | epoch | 阈值 | weighted recall | pose recall | precision | accuracy | balanced accuracy | specificity | useful cull | bad cull | 平均预测数 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 第一阶段基准 | 8 | 0.005 | 0.99261 | 0.9286 | 0.4308 | 0.8671 | 0.8717 | 0.8149 | 0.7755 | 0.00162 | 680.2 |
| RVL 总权重 0.50 | 8 | 0.005 | **0.99273** | **0.9326** | 0.4194 | 0.8658 | 0.8730 | 0.8134 | 0.7741 | 0.00151 | 685.0 |
| RVL FP 权重 1.25 | 8 | 0.005 | 0.99242 | 0.9299 | **0.4365** | **0.8687** | **0.8735** | **0.8171** | **0.7771** | 0.00162 | **675.2** |
| 集合计数权重 0.18 | 8 | 0.005 | 0.99359 | 0.9379 | 0.4073 | 0.8611 | 0.8731 | 0.8082 | 0.7694 | **0.00147** | 727.1 |
| 集合排序权重 0.65 | 8 | 0.005 | 0.99283 | 0.9320 | 0.4269 | 0.8659 | 0.8731 | 0.8142 | 0.7742 | 0.00146 | 685.7 |

只有 RVL 假阳性权重 1.25 同时提高 accuracy、balanced accuracy、precision、specificity 和 useful cull，并将平均预测数减少约 5 个。提高计数约束没有减少预测数，反而使模型为满足安全阈值多预测约 47 个实例，说明该项在当前高召回校准下与目标不一致。

### 第三阶段：RVL 假阳性权重窄扫

围绕第二阶段最优点继续测试 1.10、1.40、1.60，并增加 `FP=1.40 + 正例安全=0.30` 检查更强正例保护能否补偿过强误报抑制。已有 1.25 结果不重复训练。

| RVL FP 权重 | 正例安全权重 | epoch | 阈值 | weighted recall | precision | accuracy | balanced accuracy | specificity | useful cull | bad cull | 平均预测数 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1.10 | 0.20 | 8 | 0.005 | **0.99374** | 0.4038 | 0.8624 | 0.8737 | 0.8098 | 0.7707 | 0.00151 | 706.2 |
| **1.25** | **0.20** | **8** | **0.005** | 0.99242 | **0.4365** | **0.8687** | 0.8735 | **0.8171** | **0.7771** | 0.00162 | **675.2** |
| 1.40 | 0.20 | 8 | 0.005 | 0.99342 | 0.4185 | 0.8646 | **0.8740** | 0.8122 | 0.7729 | 0.00149 | 705.8 |
| 1.60 | 0.20 | 12 | 0.002 | 0.99295 | 0.4221 | 0.8640 | 0.8722 | 0.8115 | 0.7723 | 0.00149 | 687.1 |
| 1.40 | 0.30 | 8 | 0.005 | 0.99358 | 0.4104 | 0.8623 | 0.8728 | 0.8097 | 0.7706 | 0.00149 | 707.1 |

假阳性权重并非越大越好。1.40 的 balanced accuracy 略高 0.0005，但 accuracy、precision、specificity、useful cull 和平均预测数均弱于 1.25；1.60 还迫使 calibration 将阈值降回 0.002。增加正例安全权重到 0.30 只提高 weighted recall，没有改善分类或剔除效率。

### 第四阶段：局部最优确认

为避免假阳性权重网格过粗，在 1.25 两侧补充 1.20、1.30，并固定假阳性权重 1.25 将正例安全权重微调为 0.15、0.25。该阶段仍完整运行 12 epoch，未根据中间结果提前停止。

| RVL FP 权重 | 正例安全权重 | epoch | 阈值 | weighted recall | pose recall | precision | accuracy | balanced accuracy | useful cull | bad cull | 平均预测数 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1.20 | 0.20 | 8 | 0.005 | 0.99288 | 0.9338 | 0.4285 | 0.8662 | **0.8739** | 0.7746 | 0.00154 | 691.4 |
| **1.25** | **0.20** | **8** | **0.005** | 0.99242 | 0.9299 | 0.4365 | **0.8687** | 0.8735 | **0.7771** | 0.00162 | 675.2 |
| 1.30 | 0.20 | 6 | 0.005 | 0.99195 | 0.9168 | **0.4443** | 0.8673 | 0.8658 | 0.7758 | 0.00170 | **664.1** |
| 1.25 | 0.15 | 12 | 0.002 | 0.99326 | 0.9362 | 0.4227 | 0.8635 | 0.8734 | 0.7717 | **0.00138** | 704.9 |
| 1.25 | 0.25 | 8 | 0.005 | **0.99341** | **0.9365** | 0.4171 | 0.8629 | 0.8733 | 0.7712 | 0.00148 | 710.9 |

1.30 用更低的普通 pose recall 换得更高 precision 和更少预测数，weighted recall 仍通过点估计安全门，但 accuracy 低 0.0015、balanced accuracy 低 0.0077。由于本轮预先登记的是安全后优先最大化 accuracy，再参考 balanced accuracy，因此不能改用 1.30。1.20 以及正例安全权重 0.15、0.25 同样没有超过 1.25/0.20，后者由此确认为当前单种子快速扫描的局部最优。

## 最终快速扫描结论

当前单种子快速扫描的相对最优损失配置为：

- legacy + evidence RVL；
- RVL 总权重 0.42，RVL 假阳性权重 1.25；
- 正例安全损失权重 0.20；
- pose 内困难负例对比损失权重 0.10，温度 0.10；
- 集合计数和排序权重保持 0.12、0.50；
- calibration 点估计下限 0.9975，冻结阈值 0.005。

在 730 个 validation view-cell 上，该配置得到 weighted recall 0.99242、pose recall 0.9299、precision 0.4365、F1 0.5463、accuracy 0.8687、balanced accuracy 0.8735、specificity 0.8171、useful cull 0.7771、bad cull 0.00162，平均预测 675.2 个实例。validation 平均 GT 为 109.0、平均候选为 4747.9，因此 precision 仍受大规模负候选影响，accuracy 和 balanced accuracy 必须与 precision 一起解释。

相对仅加入对比学习、采用同一 calibration 余量的安全基准，该配置将 accuracy 从 0.8506 提到 0.8687，precision 从 0.3746 提到 0.4365，balanced accuracy 从 0.8711 提到 0.8735，useful cull 从 0.7586 提到 0.7771，平均预测数从 766.1 降到 675.2；weighted recall 从 0.99384 小幅降至 0.99242，但仍满足本轮点估计安全要求。

这些结果只用于冻结长训候选参数，尚不是论文正式结论。快速扫描没有执行 bootstrap，也没有读取 684 个 test view-cell；正式安全资格仍需三种子长训后，在 calibration 计算 weighted recall 单侧置信下界，再以冻结阈值评价 validation，最后只对最终冻结模型读取一次 test。本轮没有修改默认 checkpoint、阈值或前端资产。
