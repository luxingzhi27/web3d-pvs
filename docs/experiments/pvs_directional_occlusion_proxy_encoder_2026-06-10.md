# pvs_directional_occlusion_proxy_encoder Implementation Record

日期：2026-06-10

> 历史实验记录说明：本文保留训练和调参过程，不替代当前架构、资源语义和可复现边界。当前实现细节见 `docs/current/neuralstreamweb3d_model_pipeline.md`；当前前端与数据集口径见 `docs/frontend/neuralstreamweb3d_runtime_implementation.md` 和 `docs/current/neuralstreamweb3d_dataset_protocol.md`。本文中的旧 epoch、旧导出目录和旧 benchmark 表格按当时实验语境理解，不能直接当作当前默认资产。

## 目标

本次实现的目标是替代上一轮运行时代价较高的动态遮挡池路线。新方案不再让前端为每个候选实例临时展开多个遮挡邻居，也不使用 dynamic-pool 作为 teacher。它把较重的点云和上下文建模放到离线训练阶段，前端只读取固定实例特征表，并用当前相机到实例中心的视线方向进行轻量查询。

该方案暂定实验名为 `pvs_directional_occlusion_proxy_encoder`。2026-06-10 已完成一轮正式训练到 epoch 14，并按用户确认的 `pose recall >= 0.95` 安全目标导出 `recall95` 工作点和完整 test split 评测。后续又完成 `RVL=0.42` 的 epoch 24 权重评测和前端导出，当前前端默认资产已切换到该 epoch 24 版本。

2026-06-10 追加修改：由于当前模型仍存在明显过预测，训练脚本已把排斥式可见性损失（Repulsive Visibility Loss，RVL）默认打开。RVL 的输入是每个 pose 的候选实例预测概率、真实可见标签、可见权重和方向遮挡证据；输出是一个约束项，用于同时保护可见实例召回并压低不可见候选上的预测概率质量。当前默认模式为证据加权 RVL，即对有遮挡证据的不可见候选施加更强排斥。

## 模型数据流

每个实例首先通过离线实例点云编码器得到自身几何表示。这里的实例点云来自 GLB 点云模板和实例包围盒缩放后的局部采样，用于描述该实例的局部形状、场景归一化位置和尺度。

然后模型使用由训练数据直接构建的“方向遮挡证据”。遮挡证据的含义是：在某个采样相机下，一个候选实例没有出现在可见集合中，但同一视角下有其他可见实例位于它前方，并且两者的屏幕投影存在重叠。这个证据不是 teacher 模型输出，而是从真实可见集合、当前投影矩阵、包围盒屏幕投影和前后深度关系中弱监督构造出来。

离线编码器有两个输出头：

- 上下文特征：描述一个实例在场景结构和常见遮挡关系中的上下文状态，供可见性判断和 GLB 下载优先级共享使用。
- 方向遮挡代理特征：按照相机视线方向和前后深度层级组织的压缩遮挡表示。运行时不再查询具体遮挡邻居，而是用当前视线方向从代理特征中软选择一份遮挡信息。

运行时输入是候选实例 ID、固定实例特征表和当前相机查询特征。运行时输出包括实例级可见性分数、视觉效用分数和 GLB 下载优先级分数。实例级输出用于 `componentModelList`，GLB 级输出用于下载、解码和缓存排序。

## 新增文件

| 文件 | 作用 |
|---|---|
| `neural_instance_culling/dataset/build_directional_occlusion_evidence.py` | 从 pose CSR 数据集、MVP 投影和真实可见集合构建方向遮挡证据 |
| `neural_instance_culling/model/directional_occlusion_proxy_encoder_model.py` | 定义离线重型点云编码器、上下文头、方向遮挡代理头和轻量相机查询头 |
| `neural_instance_culling/model/train_directional_occlusion_proxy_encoder.py` | 训练、阈值扫描、固定运行时特征导出和 test 评估入口 |

## 训练监督

实例可见性监督来自 pose-level 可见集合，仍以高召回为安全约束。当前训练损失包含集合级可见性损失、遮挡代理证据损失、可见实例保护项、预测预算约束、遮挡 hard negative 排序项、视觉效用损失、GLB 优先级损失和 RVL。

预测预算约束用于抑制“为了高召回把所有候选都预测为可见”的退化行为。它按每个 pose 的真实可见实例数量设置安全预算，超过预算的预测数量会被惩罚。这个约束不能替代 recall / weighted recall / bad cull 报告，只是训练阶段的行为约束。

GLB 优先级监督不简单等同于“GLB 内是否含有可见实例”。它结合实例可见标签、可见权重和 GLB 成本，训练模型在统一表征上输出下载排序分数。

当前默认 loss 组合参数如下。这组参数是依据高召回 PVS 任务经验和本轮小规模消融设定的启发式初值，还不是充分网格搜索后的最优参数；因此后续正式论文实验仍需要继续调 `RVL` 强度、预测预算强度和阈值选择策略。

| 损失项 | 默认权重 / 参数 | 作用 |
|---|---:|---|
| 集合二分类 BCE | 0.3 | 对每个候选实例提供基础可见 / 不可见监督 |
| 集合 Tversky | 1.4，FN 侧 beta = 8.0 | 在样本不平衡下保护真实可见实例召回 |
| 集合数量约束 | 0.12 | 约束预测集合规模，不把全部候选都抬成可见 |
| 集合排序项 | 0.5 | 拉开真实可见实例与不可见实例的可见性分数 |
| 遮挡代理证据 | 0.25 | 让遮挡代理特征学习训练数据构造出的方向遮挡证据 |
| 可见实例保护项 | 0.08 | 避免遮挡代理把真实可见实例过度压低 |
| 遮挡代理稀疏项 | 0.02 | 约束遮挡代理不要退化为全方向高响应 |
| 预测预算约束 | 0.25，安全倍数 2.5 | 抑制高召回目标下的过预测 |
| 遮挡 hard negative 排序 | 0.20，margin 0.30 | 对有遮挡证据的不可见候选提高排序压力 |
| 视觉效用损失 | 0.20 | 让输出分数保留可见权重较高实例的效用排序 |
| GLB 优先级损失 | 0.20 | 在共享表征上学习下载 / 解码排序 |
| RVL | 默认 `evidence`，总权重 0.08，FN 权重 0.25，FP 权重 1.0 | 直接惩罚不可见候选上的预测概率总质量，并对有遮挡证据的负样本加权 |

需要特别说明：原始 `pvs_directional_occlusion_proxy_encoder` 是联合训练版本，实例点云编码器、上下文特征头、遮挡代理特征头和运行时查询头一起接受可见性 / 调度目标的梯度。训练完成后导出固定实例特征表，前端运行时不执行点云编码器。为了让论文叙事更清楚，本次又加入固定特征微调路径：先读取已经导出的、与当前视角无关的实例特征表，再冻结离线点云和上下文编码器，只训练轻量视角查询头和输出头。这样可以更明确地区分“离线视角无关特征提取”和“运行时视角条件化查询”。

## Loss v2 分层设计

2026-06-10 进一步整理 loss 结构。此前代码把 RVL、预算约束、遮挡排序和调度目标都平铺在总 loss 中，虽然数学上可以训练，但概念层级不清晰。新的实现把 loss 分成四层：

```text
L_total =
  L_visibility
+ L_occlusion_proxy
+ L_runtime_scheduler
+ L_regularization
```

第一层 `L_visibility` 是实例可见性主目标，输入是候选实例的可见性 logit、真实可见标签、可见权重、pose 分段和方向遮挡证据，输出是主可见性监督。它内部包含 BCE、Tversky、集合数量、可见/不可见排序、预测预算约束、遮挡 hard negative 排序和 RVL。这样 RVL 不再是一个和可见性 loss 平级的外部项，而是可见性目标中专门抑制过预测的一部分；预测预算和遮挡 hard negative 排序也一起归入“可见性行为约束”。

第二层 `L_occlusion_proxy` 是遮挡代理表征约束，输入是模型内部的遮挡抑制量、方向遮挡证据和真实可见标签，输出是对遮挡代理特征的弱监督。它只负责让“遮挡代理”有明确语义：有遮挡证据的不可见实例应该产生较高抑制，真实可见实例不能被过度抑制，没有证据的位置不应到处高响应。

第三层 `L_runtime_scheduler` 是运行时调度目标，包含视觉效用分数和 GLB 下载优先级。它不应该反向污染实例级可见性判断，因此权重应保持中等，主要服务排序和下载预算。

第四层 `L_regularization` 是参数正则，只用于防止权重膨胀。

当前训练脚本新增 `--loss-profile`，用于避免每次实验手写一长串权重。第一组 profile 含义如下：

| profile | 核心目的 | 主要变化 |
|---|---|---|
| `legacy` | 保留原始 CLI 权重 | 不做 profile 覆盖，用于复现旧设置 |
| `balanced_v2` | 新默认平衡方案 | RVL 合入可见性主目标；略降低代理和预算权重，避免辅助目标压过主可见性 |
| `rvl_strong_v2` | 验证更强 RVL 是否能进一步压低过预测 | 相比 `balanced_v2` 把 RVL 总权重从 0.08 提到 0.12 |
| `budget_tight_v2` | 验证更强预测预算是否能降低 avg pred | 降低安全倍数到 2.15，提高预算约束权重到 0.30 |
| `proxy_light_v2` | 验证遮挡代理辅助监督是否过强 | 进一步降低遮挡代理 evidence / guard / sparsity 和代理排序权重 |

第一组调参不直接改点云编码器，只使用固定实例特征微调。原因是当前要先判断 loss 设计是否合理；如果同时重新训练离线点云编码器，很难分清收益来自 loss 还是来自表示重学。

## Loss v2 第一组调参结果

执行命令模板如下，四个 profile 分别在 GPU 0/1/2/3 并行运行：

```bash
CUDA_VISIBLE_DEVICES=<gpu> conda run --no-capture-output -n slm_pvs python -u neural_instance_culling/model/train_directional_occlusion_proxy_encoder.py \
  --dataset-dir neural_instance_culling/dataset/out/pose_csr_viewcell_back_camera_pvs_v1 \
  --evidence-dir neural_instance_culling/dataset/out/directional_occlusion_evidence_v1 \
  --runtime-meta hkust-v3/assets/runtimeVisibilityMeta.json \
  --glb-points neural_instance_culling/dataset/out/glb_points_v3.bin \
  --init-checkpoint neural_instance_culling/model/out/pvs_directional_occlusion_proxy_encoder/best_recall95.pt \
  --fixed-runtime-features neural_instance_culling/model/out/pvs_directional_occlusion_proxy_encoder/instance_runtime_features_fp16.bin \
  --freeze-offline-encoder \
  --loss-profile <balanced_v2|rvl_strong_v2|budget_tight_v2|proxy_light_v2> \
  --epochs 6 \
  --steps-per-epoch 600 \
  --eval-every 2 \
  --eval-pose-steps 256 \
  --pose-set-batch-size 2 \
  --target-recall 0.95 \
  --target-weighted-recall 0.99 \
  --device cuda \
  --amp \
  --lr 8e-5
```

统一指标评测命令：

```bash
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n slm_pvs python -u neural_instance_culling/benchmark/evaluate_unified_pvs_metrics.py \
  --models pvs_directional_occlusion_proxy_encoder_loss_v2_balanced_ft,pvs_directional_occlusion_proxy_encoder_loss_v2_rvl_strong_ft,pvs_directional_occlusion_proxy_encoder_loss_v2_budget_tight_ft,pvs_directional_occlusion_proxy_encoder_loss_v2_proxy_light_ft \
  --dataset-dir neural_instance_culling/dataset/out/pose_csr_viewcell_back_camera_pvs_v1 \
  --runtime-meta hkust-v3/assets/runtimeVisibilityMeta.json \
  --glb-index hkust-v3/assets/glbIndex.json \
  --glb-root hkust-v3/assets \
  --output-dir neural_instance_culling/benchmark/out/unified_pvs_metrics_directional_occlusion_proxy_loss_v2_ft \
  --split test \
  --target-recall 0.95 \
  --target-weighted-recall 0.99 \
  --target-utility-recall 0.98 \
  --device cuda
```

完整 test split 使用 `684` 个唯一 test viewcell。安全工作点均为 threshold `0.010`：

| profile | precision | recall | weighted recall | utility recall | useful cull | bad cull | avg pred | GLB byte reduction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `balanced_v2` | 0.680 | 0.955 | 0.996 | 0.972 | 0.868 | 0.00131 | 674.33 | 0.858 |
| `rvl_strong_v2` | 0.685 | 0.953 | 0.996 | 0.971 | 0.869 | 0.00133 | 666.27 | 0.858 |
| `budget_tight_v2` | 0.684 | 0.953 | 0.996 | 0.971 | 0.869 | 0.00132 | 669.51 | 0.858 |
| `proxy_light_v2` | 0.681 | 0.954 | 0.996 | 0.971 | 0.868 | 0.00131 | 672.75 | 0.858 |

与上一轮 `loss_full_evidence_ft` 的安全工作点相比，`rvl_strong_v2` 将 avg pred 从 `679.85` 降到 `666.27`，precision 从 `0.677` 提到 `0.685`，useful cull 从 `0.867` 提到 `0.869`；代价是 recall 从 `0.956` 降到 `0.953`，bad cull 从 `0.00129` 到 `0.00133`，仍满足当前普通召回 `>=0.95` 和 weighted recall `>=0.99` 安全目标。

这说明 loss v2 的方向是有效的：把 RVL 合入可见性主目标并稍微减轻遮挡代理辅助项后，模型的过预测有所下降。但四个 profile 的差异并不大，说明当前瓶颈已经不只是 loss 权重，而可能来自固定实例特征本身的可分性、遮挡证据噪声或阈值粒度。

当前建议保留 `rvl_strong_v2` 作为下一轮固定特征微调的主候选；如果要更保守地保护 recall，则选 `balanced_v2`。下一轮更有价值的调参应围绕 `rvl_strong_v2` 做局部搜索：RVL 权重 `0.10 / 0.12 / 0.14 / 0.16`，同时固定预算安全倍数 `2.4`，避免预算项和 RVL 同时变化导致归因不清。完成后必须补正式 image PER。

## RVL 权重快速筛选

2026-06-10 继续对 RVL 权重做固定特征快速筛选。此轮不是从头训练，而是从 `best_recall95.pt` 初始化，读取固定实例特征，冻结离线编码器，只训练运行时查询头和输出头。目的只是快速判断 RVL 权重的大致量级，不作为正式从头训练结论。

第一轮扫 `0.16 / 0.24 / 0.36 / 0.48`，第二轮扫 `0.60 / 0.80 / 1.00 / 1.20`。每组使用 3 epoch、每 epoch 500 step。由于默认阈值网格在低阈值区较粗，又对候选权重做了细阈值重评，输出保存在 `neural_instance_culling/benchmark/out/rvl_weight_fine_scan_directional_proxy/summary.json`。

细阈值扫描结果：

| RVL 权重 | 最佳安全阈值 | precision | recall | weighted recall | avg pred | candidate reduction |
|---:|---:|---:|---:|---:|---:|---:|
| 0.24 | 0.0100 | 0.687 | 0.951 | 0.995 | 653.2 | 0.896 |
| 0.36 | 0.0035 | 0.694 | 0.950 | 0.997 | 612.2 | 0.902 |
| 0.48 | 0.0030 | 0.690 | 0.951 | 0.997 | 613.5 | 0.902 |
| 0.60 | 0.0060 | 0.687 | 0.951 | 0.996 | 649.9 | 0.896 |
| 0.80 | 0.0045 | 0.688 | 0.950 | 0.996 | 648.8 | 0.897 |

结论：`0.10~0.16` 的 RVL 权重确实偏小；`0.36~0.48` 是当前更合适的区间。`0.36` 在满足普通 recall `0.950` 和 weighted recall `0.997` 时，把 avg pred 降到 `612.2`，比 `0.24` 少约 `41` 个预测实例；`0.48` 指标非常接近。继续增大到 `0.60` 以上没有收益，反而需要更多预测才能恢复安全召回，说明过强 RVL 已经开始压坏分数尺度。

下一步从头训练不应再扫很小权重。建议四 GPU 并行从头短训：

| 权重 | 目的 |
|---:|---|
| 0.30 | 检查比 0.36 略弱时是否更稳 |
| 0.36 | 当前固定特征快速筛选最佳 |
| 0.42 | 在 0.36 和 0.48 之间细化 |
| 0.48 | 当前固定特征快速筛选并列候选 |

如果从头短训仍显示 `0.36~0.48` 稳定，再选择 1 到 2 个权重做正式长训和 image PER。

## Epoch24 前端阈值切换到 bestF1

2026-06-10 追加前端阈值调整：用户要求前端使用 epoch 24 的 best-F1 阈值，而不是高召回工作点阈值。本次没有重新训练，也没有改动 `instance_pvs_assets.bin` 权重，只修改前端 `instance_model_meta.json` 中的 `visibilityThreshold`、`thresholdSelection` 和 `testWorkpoint`。

`eval_summary_epoch24.json` 中的 `workpoints.bestF1` 对应：

| 指标 | 数值 |
|---|---:|
| threshold | `0.6800000071525574` |
| pose precision | `0.8261313081262845` |
| pose recall | `0.8699753306230631` |
| pose F1 | `0.8251439218836464` |
| weighted recall | `0.9911287796379514` |
| avg pred | `267.8625730994152` |
| avg candidate | `6272.937134502924` |
| candidate reduction | `0.9572986995794848` |
| eval pose count | `684` |

该工作点的意义是提高 F1 和降低预测数量，但普通 pose recall 从高召回工作点的约 `0.951` 降到约 `0.870`。因此它适合前端调试或观察低预测数量效果，不能被写成满足当前 `pose recall >= 0.95` 安全约束的正式主工作点。后续如果继续保留 bestF1 作为前端默认，需要补 image PER、miss pixel rate 和同位姿前端画面风险检查。

## 正式训练与 95% 召回工作点

本轮全量训练原计划运行 24 epoch。训练到 epoch 14 后，用户确认普通召回率不必强行达到 97%，`95%` 也可以作为当前可接受的画面安全约束。因此本次停止继续训练，保留 epoch 14 作为 `recall95` 候选，而不继续沿用旧的 97% 偏好 checkpoint。

训练命令：

```bash
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n slm_pvs python -u neural_instance_culling/model/train_directional_occlusion_proxy_encoder.py \
  --dataset-dir neural_instance_culling/dataset/out/pose_csr_viewcell_back_camera_pvs_v1 \
  --evidence-dir neural_instance_culling/dataset/out/directional_occlusion_evidence_v1 \
  --runtime-meta hkust-v3/assets/runtimeVisibilityMeta.json \
  --glb-points neural_instance_culling/dataset/out/glb_points_v3.bin \
  --output-dir neural_instance_culling/model/out/pvs_directional_occlusion_proxy_encoder \
  --experiment-name pvs_directional_occlusion_proxy_encoder \
  --epochs 24 \
  --steps-per-epoch 900 \
  --eval-every 2 \
  --device cuda \
  --amp \
  > neural_instance_culling/model/out/pvs_directional_occlusion_proxy_encoder/train_stdout.log \
  2> neural_instance_culling/model/out/pvs_directional_occlusion_proxy_encoder/train_stderr.log
```

保留文件：

| 文件 | 含义 |
|---|---|
| `neural_instance_culling/model/out/pvs_directional_occlusion_proxy_encoder/best.pt` | 旧选择规则下的 97% 偏好 checkpoint，保留但不作为本次 95% 工作点 |
| `neural_instance_culling/model/out/pvs_directional_occlusion_proxy_encoder/best_recall95_candidate_epoch14.pt` | epoch 14 候选 |
| `neural_instance_culling/model/out/pvs_directional_occlusion_proxy_encoder/best_recall95.pt` | 本次正式 95% 工作点 checkpoint |
| `neural_instance_culling/model/out/pvs_directional_occlusion_proxy_encoder/eval_summary_recall95.json` | 使用 95% 普通召回和 99% 加权召回目标的完整 test 阈值扫描 |
| `neural_instance_culling/model/out/pvs_directional_occlusion_proxy_encoder/instance_runtime_features_fp16.bin` | 前端运行时可读取的固定实例特征表；运行时不执行点云编码器或图传播 |

导出和完整 test 阈值扫描命令：

```bash
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n slm_pvs python -u neural_instance_culling/model/train_directional_occlusion_proxy_encoder.py \
  --dataset-dir neural_instance_culling/dataset/out/pose_csr_viewcell_back_camera_pvs_v1 \
  --evidence-dir neural_instance_culling/dataset/out/directional_occlusion_evidence_v1 \
  --runtime-meta hkust-v3/assets/runtimeVisibilityMeta.json \
  --glb-points neural_instance_culling/dataset/out/glb_points_v3.bin \
  --output-dir neural_instance_culling/model/out/pvs_directional_occlusion_proxy_encoder \
  --experiment-name pvs_directional_occlusion_proxy_encoder \
  --device cuda \
  --amp \
  --target-recall 0.95 \
  --target-weighted-recall 0.99 \
  --export-eval-checkpoint neural_instance_culling/model/out/pvs_directional_occlusion_proxy_encoder/best_recall95_candidate_epoch14.pt \
  --checkpoint-alias best_recall95.pt \
  --eval-summary-name eval_summary_recall95.json \
  > neural_instance_culling/model/out/pvs_directional_occlusion_proxy_encoder/export_eval_recall95_stdout.log \
  2> neural_instance_culling/model/out/pvs_directional_occlusion_proxy_encoder/export_eval_recall95_stderr.log
```

完整 test split 使用 `684` 个唯一 test viewcell。`eval_summary_recall95.json` 中选择的工作点为 threshold `0.010`：

| 指标 | 数值 | 含义 |
|---|---:|---|
| pose precision | 0.679 | 每个 pose 先算 precision 再平均，表示预测为可见的实例中有多少真实可见 |
| pose recall | 0.953 | 每个 pose 的真实可见实例找回率，满足当前 `>=0.95` 安全目标 |
| weighted recall | 0.996 | 按 `visible_weights` 加权的 GT 找回率；权重是 rvcServer 重要性，不是真实像素面积 |
| avg candidate | 6272.94 | 每个 pose 后退扩大视锥候选实例数 |
| avg GT | 158.08 | 每个 pose 真实可见实例数 |
| avg pred | 688.99 | 每个 pose 预测保留实例数 |
| candidate reduction | 0.890 | 预测数量相对候选集合的减少比例；该指标混合了正确剔除和错误剔除，不能单独作为主结论 |

## 统一指标评测

统一指标命令：

```bash
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n slm_pvs python -u neural_instance_culling/benchmark/evaluate_unified_pvs_metrics.py \
  --models pvs_directional_occlusion_proxy_encoder_recall95 \
  --dataset-dir neural_instance_culling/dataset/out/pose_csr_viewcell_back_camera_pvs_v1 \
  --runtime-meta hkust-v3/assets/runtimeVisibilityMeta.json \
  --glb-index hkust-v3/assets/glbIndex.json \
  --glb-root hkust-v3/assets \
  --output-dir neural_instance_culling/benchmark/out/unified_pvs_metrics_pvs_directional_occlusion_proxy_encoder_recall95 \
  --split test \
  --target-recall 0.95 \
  --target-weighted-recall 0.99 \
  --target-utility-recall 0.98 \
  --device cuda
```

输出目录：`neural_instance_culling/benchmark/out/unified_pvs_metrics_pvs_directional_occlusion_proxy_encoder_recall95`。

主安全工作点同样为 threshold `0.010`：

| 指标 | 数值 | 含义 |
|---|---:|---|
| pose accuracy | 0.927 | 候选实例逐项分类正确率，受大量不可见候选影响，只作辅助 |
| balanced accuracy | 0.928 | recall 与 specificity 的平均，避免普通 accuracy 被负样本主导 |
| pose precision | 0.679 | 预测可见集合的纯度 |
| pose recall | 0.953 | 普通实例集合召回 |
| weighted recall | 0.996 | 重要性加权召回 |
| weak utility recall | 0.970 | 使用 `log1p(visible_weights)` 的弱视觉效用找回率；未达到本次统一脚本设定的 0.98 辅助目标 |
| useful cull | 0.864 | `TN / candidate`，表示正确剔除的不可见候选占候选比例 |
| bad cull | 0.00138 | `FN / candidate`，表示错误剔除真实可见实例占候选比例 |
| raw reduction | 0.890 | `1 - avg_pred / avg_candidate`，只作辅助 |
| avg pred / avg GT | 4.36 | 平均预测数量是真实可见数量的 4.36 倍 |
| GLB byte reduction | 0.854 | 预测 GLB 字节数相对候选 GLB 字节数的减少比例 |

如果把 weak utility recall `>=0.98` 也作为硬约束，统一评测会选择 threshold `0.001`：pose recall `0.974`、weighted recall更高、bad cull 更低，但 avg pred 上升到 `804.68`，useful cull 降到 `0.842`。当前应把 threshold `0.010` 作为“普通 95% 召回安全工作点”，把 threshold `0.001` 作为“效用更保守工作点”备选。

预算 GLB utility 结果：

| GLB 预算 | utility recall | required recall | 平均选中 GLB | 字节削减 |
|---:|---:|---:|---:|---:|
| 50 | 0.904 | 0.877 | 43.04 | 0.614 |
| 100 | 0.927 | 0.903 | 78.51 | 0.531 |
| 200 | 0.948 | 0.926 | 146.45 | 0.417 |
| 384 | 0.967 | 0.950 | 262.60 | 0.278 |

达到 0.98 weak utility 目标所需的字节前缀平均为 `215.05` 个 GLB，平均字节 `57,728,781`，相对候选字节削减 `0.794`，实际 utility recall `0.996`。

## 图像级 small PER

图像评测命令：

```bash
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n slm_pvs python -u neural_instance_culling/benchmark/evaluate_viewcell_image_per.py \
  --model-name pvs_directional_occlusion_proxy_encoder_recall95 \
  --runtime-meta hkust-v3/assets/runtimeVisibilityMeta.json \
  --viewcell-dataset neural_instance_culling/dataset/out/pose_csr_viewcell_back_camera_pvs_v1 \
  --pose-csr neural_instance_culling/dataset/out/pose_csr_viewcell_back_camera_pvs_v1 \
  --glb-root hkust-v3/assets \
  --glb-index hkust-v3/assets/glbIndex.json \
  --glb-points-meta neural_instance_culling/dataset/out/glb_points_v3_meta.json \
  --output-dir neural_instance_culling/benchmark/out/viewcell_image_per_pvs_directional_occlusion_proxy_encoder_recall95_small \
  --split test \
  --max-viewcells 8 \
  --subposes-per-viewcell 2 \
  --width 480 \
  --height 270 \
  --device cuda \
  --image-renderer true_glb \
  --chrome-exe /usr/bin/google-chrome \
  --threshold-policy runner \
  --skip-raw-subpose-gt
```

本机未找到历史 raw 子姿态目录 `neural_instance_culling/collector/out/hkust_rvc_viewcell_froxel_v1_1080p_rx9070_raw`，因此这次 small PER 使用 viewcell union fallback 作为每个 subpose 的参考集合。它可以证明 true GLB 渲染链路和模型输出没有明显崩溃，但不能替代带 raw 子姿态 GT 的正式图像评测。

输出目录：`neural_instance_culling/benchmark/out/viewcell_image_per_pvs_directional_occlusion_proxy_encoder_recall95_small`。渲染结果：16/16 subpose 成功，本地 GLB 缺失数为 0，自一致性 PER 为 0。

| 指标 | 数值 | 含义 |
|---|---:|---|
| component pose precision | 0.7077 | small PER 样本上的构件集合 precision |
| component pose recall | 0.9563 | small PER 样本上的构件集合 recall |
| weighted recall | 0.9984 | small PER 样本上的重要性加权召回 |
| image PER | 0.007982 | reference 有效像素中 test ID 不一致的比例 |
| miss pixel rate | 0.000926 | reference 有效像素中被预测漏掉成背景的比例 |
| wrong-ID pixel rate | 0.007057 | reference 有效像素中被其他 GLB 覆盖的比例 |
| extra pixel rate | 0.00000048 | reference 背景上多出的 test 像素比例 |

该 small PER 低于上一版 dynamic-pool 报告中的 small PER `0.008532`，但二者的样本和参考 GT 口径未必完全一致，不能直接作为胜负结论。后续必须补带 raw 子姿态 GT 的同口径 image PER。

## RVL 实现与消融结果

NeuralPVS 使用加权 Dice 损失处理可见区域稀疏问题，并引入 Repulsive Visibility Loss（RVL）抑制过预测。这里的 RVL 指“排斥式可见性损失”：它一方面继续惩罚漏掉真实可见区域，另一方面直接惩罚不可见区域上的预测概率总质量。换成本项目的实例级任务，RVL 的作用是避免模型为了追求高召回，把大量候选实例都抬成可见。

当前实现中的集合级 Tversky 损失已经属于 Dice / Tversky 家族，主要负责在 pose-level 集合上平衡 TP、FP 和 FN，并通过较高 FN 权重保护召回。但 Tversky 本身主要优化集合重叠，不够直接约束“不可见候选上的概率总质量”。因此训练脚本新增 RVL，并从 2026-06-10 起默认使用证据加权 RVL。复现实验如果需要关闭 RVL，必须显式传入 `--rvl-mode off --rvl-loss-weight 0.0`。

实例级 RVL 的定义如下。对一个 pose 的候选实例集合，令 `p_i` 表示模型预测第 `i` 个实例可见的概率，`y_i` 表示真实可见标签，`w_i` 表示可见权重。先定义：

```text
soft FN = sum((1 - p_i) * y_i * visible_weight_i)
soft FP = sum(p_i * (1 - y_i) * negative_weight_i)
visible mass = sum(y_i * visible_weight_i)
```

则 RVL 为：

```text
L_rvl =
  lambda_fn * soft FN / max(visible mass, eps)
+ lambda_fp * soft FP / max(number of visible instances, 1)
```

这个定义的预期作用是：真实可见实例的预测概率过低时，`soft FN` 会把它往可见方向拉；真实不可见实例的预测概率过高时，`soft FP` 会把它往不可见方向推。`soft FP` 使用真实可见实例数量归一化，而不是使用候选数量归一化，是为了避免在候选集合很大时误报惩罚被稀释。

负样本权重 `negative_weight_i` 不简单固定为 1。本次实现保留三种口径：

| 版本 | 负样本权重 | 目的 |
|---|---|---|
| `rvl_uniform` | 所有不可见候选权重为 1 | 验证 RVL 基础排斥项是否降低 avg pred |
| `rvl_evidence_weighted` | 使用方向遮挡证据强度提高 hard negative 权重 | 重点压低有明确遮挡证据的不可见实例 |
| `rvl_cost_aware` | 在遮挡证据基础上叠加 GLB 字节成本或资源成本 | 优先减少高成本误报，服务下载调度 |

加入 RVL 后的总损失应记录为：

```text
L_total =
  L_pose_set_visibility
+ L_proxy_evidence
+ L_visible_guard
+ L_prediction_budget
+ L_proxy_rank
+ L_visual_utility
+ L_glb_priority
+ lambda_rvl * L_rvl
+ L_regularization
```

RVL 不能替代高召回集合损失，也不能替代正式指标。它只是用来约束“高召回下过预测”的辅助项。调参时必须同时观察：

| 指标 | 期望变化 |
|---|---|
| pose recall | 不低于当前安全目标，当前可接受目标为 `>= 0.95` |
| weighted recall | 接近或超过 `0.99` |
| bad cull = FN / candidate | 不能明显升高 |
| useful cull = TN / candidate | 应提高 |
| avg pred / avg GT | 应降低 |
| GLB byte reduction | 应提高或不劣于当前模型 |
| image PER | 不能恶化 |

本次已完成一轮固定特征微调消融。四组实验都从 `best_recall95.pt` 初始化，读取同一份 `instance_runtime_features_fp16.bin`，冻结离线编码器，只训练运行时查询头和输出头。这样做的目的是先验证 loss 是否改善过预测，而不把点云编码器重新训练带来的变化混入对比。

四组实验使用四张 GPU 并行运行：

```bash
CUDA_VISIBLE_DEVICES=<0|1|2|3> conda run --no-capture-output -n slm_pvs python -u neural_instance_culling/model/train_directional_occlusion_proxy_encoder.py \
  --dataset-dir neural_instance_culling/dataset/out/pose_csr_viewcell_back_camera_pvs_v1 \
  --evidence-dir neural_instance_culling/dataset/out/directional_occlusion_evidence_v1 \
  --runtime-meta hkust-v3/assets/runtimeVisibilityMeta.json \
  --glb-points neural_instance_culling/dataset/out/glb_points_v3.bin \
  --init-checkpoint neural_instance_culling/model/out/pvs_directional_occlusion_proxy_encoder/best_recall95.pt \
  --fixed-runtime-features neural_instance_culling/model/out/pvs_directional_occlusion_proxy_encoder/instance_runtime_features_fp16.bin \
  --freeze-offline-encoder \
  --epochs 6 \
  --steps-per-epoch 600 \
  --eval-every 2 \
  --eval-pose-steps 256 \
  --pose-set-batch-size 2 \
  --target-recall 0.95 \
  --target-weighted-recall 0.99 \
  --device cuda \
  --amp \
  --lr 8e-5
```

对应实验：

| 实验名 | RVL 设置 | 说明 |
|---|---|---|
| `pvs_directional_occlusion_proxy_encoder_loss_full_off_ft` | `off`, 权重 0.0 | 固定特征微调的无 RVL 对照 |
| `pvs_directional_occlusion_proxy_encoder_loss_full_uniform_ft` | `uniform`, 权重 0.08 | 对所有不可见候选使用均匀排斥 |
| `pvs_directional_occlusion_proxy_encoder_loss_full_evidence_ft` | `evidence`, 权重 0.08 | 对有方向遮挡证据的不可见候选加权排斥；当前默认 |
| `pvs_directional_occlusion_proxy_encoder_loss_full_cost_ft` | `cost_aware`, 权重 0.06 | 在遮挡证据基础上叠加 GLB 成本权重 |

统一指标评测命令：

```bash
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n slm_pvs python -u neural_instance_culling/benchmark/evaluate_unified_pvs_metrics.py \
  --models pvs_directional_occlusion_proxy_encoder_loss_full_off_ft,pvs_directional_occlusion_proxy_encoder_loss_full_uniform_ft,pvs_directional_occlusion_proxy_encoder_loss_full_evidence_ft,pvs_directional_occlusion_proxy_encoder_loss_full_cost_ft \
  --dataset-dir neural_instance_culling/dataset/out/pose_csr_viewcell_back_camera_pvs_v1 \
  --runtime-meta hkust-v3/assets/runtimeVisibilityMeta.json \
  --glb-index hkust-v3/assets/glbIndex.json \
  --glb-root hkust-v3/assets \
  --output-dir neural_instance_culling/benchmark/out/unified_pvs_metrics_directional_occlusion_proxy_loss_full_ft \
  --split test \
  --target-recall 0.95 \
  --target-weighted-recall 0.99 \
  --target-utility-recall 0.98 \
  --device cuda
```

完整 test split 使用 `684` 个唯一 test viewcell。安全工作点结果如下：

| 模型 | threshold | precision | recall | weighted recall | utility recall | useful cull | bad cull | avg pred | GLB byte reduction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| no RVL fixed-feature | 0.020 | 0.6865 | 0.9510 | 0.9952 | 0.9689 | 0.8691 | 0.00150 | 660.70 | 0.8620 |
| uniform RVL fixed-feature | 0.010 | 0.6757 | 0.9564 | 0.9960 | 0.9728 | 0.8669 | 0.00129 | 681.01 | 0.8571 |
| evidence RVL fixed-feature | 0.010 | 0.6772 | 0.9563 | 0.9960 | 0.9727 | 0.8671 | 0.00129 | 679.85 | 0.8572 |
| cost-aware RVL fixed-feature | 0.010 | 0.6767 | 0.9564 | 0.9960 | 0.9728 | 0.8670 | 0.00129 | 680.79 | 0.8570 |

这个表不能简单解读为“RVL 一定优于 no RVL”。在自动选择 `pose recall >= 0.95`、`weighted recall >= 0.99` 的安全工作点时，无 RVL 版本可以选择更高的 threshold `0.020`，因此平均预测数量更低、useful cull 更高，但它的 ordinary recall、utility recall 和 bad cull 更差。换句话说，无 RVL 在这个阈值口径下更激进，RVL 版本更保守。

为了单独观察 RVL 是否压低过预测，应比较同一个 threshold `0.010`：

| 模型 | threshold | precision | recall | weighted recall | useful cull | bad cull | avg pred | GLB byte reduction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| no RVL fixed-feature | 0.010 | 0.6573 | 0.9597 | 0.9968 | 0.8630 | 0.00115 | 697.40 | 0.8549 |
| evidence RVL fixed-feature | 0.010 | 0.6772 | 0.9563 | 0.9960 | 0.8671 | 0.00129 | 679.85 | 0.8572 |

同阈值下，证据加权 RVL 把平均预测数量从 `697.40` 降到 `679.85`，precision 从 `0.6573` 提升到 `0.6772`，GLB 字节削减从 `0.8549` 提升到 `0.8572`；代价是 recall 从 `0.9597` 降到 `0.9563`，bad cull 从 `0.00115` 升到 `0.00129`，但仍满足当前 `>=0.95` 普通召回和 `>=0.99` 加权召回安全目标。因此，RVL 作为默认 loss 是合理的，但还不能说明当前权重就是最优。

预算 GLB utility 上，RVL 三个版本在 `0.98` utility 目标下所需平均 GLB 数约为 `195`，无 RVL 为 `203.68`；这说明 RVL 对下载排序也有轻微收益。但差距不大，后续需要结合图像 PER 和前端延迟再判断是否值得作为正式主线。

下一步建议不再回到无 RVL 默认，而是在证据加权 RVL 上继续做强度消融，例如 `0.04 / 0.08 / 0.12 / 0.16`，并同步校准 threshold、预算安全倍数和 image PER。若目标是论文叙事更清晰，应优先做固定特征微调的长训和全量图像评测；若目标是冲击最高指标，再做完整联合训练版本。

## RVL 0.42 从头长训中间 checkpoint 测试

2026-06-10 根据固定特征快速扫描和从头短训结果，选择 RVL 权重 `0.42` 开启 `40 epoch` 从头训练，实验目录为 `neural_instance_culling/model/out/pvs_directional_occlusion_proxy_encoder_rvl_w042_full40`。训练命令使用 `--loss-profile legacy`，显式设置 `--rvl-mode evidence --rvl-loss-weight 0.42`，并保留 `pose recall >= 0.95`、`weighted recall >= 0.99` 作为安全工作点选择目标。

用户要求先比较 epoch 24 和 epoch 14。实际检查发现，当前 `full40` 目录只保存了 `best.pt` 和 `last.pt`，训练过程没有逐 epoch 保存 checkpoint；后来复制出的 `epoch14_best_snapshot.pt` 与 `best.pt` 文件哈希相同，checkpoint 元数据中的最佳 epoch 也是 `24`，因此它不是 epoch 14 权重。当前只能使用训练日志中的 epoch 14 验证集指标，不能对当前 RVL 0.42 训练线的 epoch 14 做完整 test replay。为避免后续再次丢失中间权重，训练脚本已修改为每次验证时额外保存 `checkpoint_epoch_XXX.pt`。

为了提供参考，另外测试了旧实验目录中保留下来的 `best_recall95_candidate_epoch14.pt`。它是 `pvs_directional_occlusion_proxy_encoder` 原始训练线的 epoch 14 候选，不是当前 RVL 0.42 full40 的同一条训练曲线；因此只能作为旧候选对照，不能解释为当前 epoch14 到 epoch24 的严格演化。

完整 test split 仍使用 `684` 个唯一 test viewcell。两次测试的安全工作点如下：

| checkpoint | 来源 | threshold | precision | recall | weighted recall | avg pred | avg GT | avg candidate | candidate reduction | agg precision | agg recall |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `epoch24_snapshot.pt` | 当前 RVL 0.42 full40 训练线 | 0.010 | 0.7006 | 0.9507 | 0.9965 | 669.76 | 158.08 | 6272.94 | 0.8932 | 0.2324 | 0.9845 |
| `best_recall95_candidate_epoch14.pt` | 旧原始训练线 epoch 14 候选 | 0.010 | 0.6790 | 0.9533 | 0.9957 | 688.99 | 158.08 | 6272.94 | 0.8902 | 0.2257 | 0.9839 |

从完整 test 结果看，当前 RVL 0.42 的 epoch24 在满足普通召回 `0.9507` 和加权召回 `0.9965` 的前提下，比旧 epoch14 候选少预测约 `19.23` 个实例，precision 提高约 `2.16` 个百分点，候选削减率从 `0.8902` 提高到 `0.8932`。代价是普通 recall 从 `0.9533` 降到 `0.9507`，但仍满足当前 `>=0.95` 目标。

当前判断：如果只看完整 test 安全工作点，epoch24 更适合作为当前保留候选；但由于真正的当前 epoch14 权重不存在，还不能断言“当前 RVL 0.42 在 epoch14 到 epoch24 之间单调变好”。后续应等待 full40 训练结束后，用新保存的 eval checkpoint 比较后续 epoch，并补充 image PER、GLB 字节和前端延迟指标。

## RVL 0.42 full40 训练完成与前端导出

2026-06-10，`pvs_directional_occlusion_proxy_encoder_rvl_w042_full40` 已完成 40/40 epoch 训练。训练进程正常退出，保留文件包括：

| 文件 | 作用 |
|---|---|
| `best.pt` | 训练过程中按目标选择出的最佳 checkpoint |
| `last.pt` | 第 40 epoch 结束权重 |
| `train_stdout.log` / `train_stderr.log` | 训练日志 |
| `instance_runtime_features_fp16.bin` | 训练结束时导出的运行时固定实例特征 |
| `eval_summary.json` | 训练结束时生成的评估摘要 |

`best.pt` checkpoint 内部记录的训练验证工作点来自 epoch 38：

| 指标 | 数值 |
|---|---:|
| threshold | `0.0020000000949949026` |
| pose precision | `0.6669` |
| pose recall | `0.9503` |
| pose F1 | `0.7473` |
| weighted recall | `0.9993` |
| avg pred | `900.77` |
| eval pose count | `512` |

由于 `full40/eval_summary.json` 的时间戳与 `last.pt` 接近，不能直接假定它对应 `best.pt`。因此本次为 `best.pt` 单独建立评估导出目录：

```text
neural_instance_culling/model/out/pvs_directional_occlusion_proxy_encoder_rvl_w042_full40_best_eval
```

独立 export/eval 在完整 test split 的 `684` 个唯一 test viewcell 上得到：

| 工作点 | threshold | pose precision | pose recall | weighted recall | pose F1 | avg pred | avg GT | avg candidate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `bestF1` | `0.6400` | `0.8466` | `0.8744` | `0.9906` | `0.8382` | `318.63` | `158.08` | `6272.94` |
| safety target | `0.0050` | `0.7172` | `0.9546` | `0.9968` | `0.7852` | `684.68` | `158.08` | `6272.94` |

本次用户明确要求把前端阈值调整为 `bestF1`，因此导出的前端 meta 写入：

```text
visibilityThreshold = 0.6399999856948853
thresholdSelection = bestF1
runtimeModelName = pvs_directional_occlusion_proxy_encoder_rvl_w042_full40_best
```

需要注意：`bestF1` 工作点提高了 F1 并显著降低 avg pred，但普通 pose recall 只有 `0.8744`，不满足当前 `>=0.95` 的高召回安全目标。后续如果将它作为正式论文主线或默认部署工作点，必须补 image PER、miss pixel rate、bad cull、GLB 字节削减和前端同位姿 smoke，证明漏预测不会造成不可接受的画面损失。

前端导出目录：

```text
slm2viewer/public/assets/neural_instance_culling/pvs_directional_occlusion_proxy_encoder_rvl_w042_full40_best
```

前端修改：

- `slm2viewer/src/neuralCullingBackendMode.js` 默认资产目录切换到 full40 best。
- `slm2viewer/slm2/SLM2Loader.js` cache-busting 版本号切换到 `directional-proxy-rvl-w042-full40-best-20260610`。
- `slm2viewer/src/viewer.js` 调试面板资产估算改为 full40 best。
- `neural_instance_culling/model/export_directional_occlusion_proxy_frontend.py` 的 display name 不再硬编码 epoch24。

验证：

- 已停止并确认常见前端端口无监听进程。
- `python -m py_compile` passed。
- `node --check` passed。
- 资产 layout byte range 检查通过，`maxLayoutEnd = 13,844,856`，与 `instance_pvs_assets.bin` 大小一致。
- `npm run build` passed，并同步 `public/` 到 `dist/`。

## 已执行验证

语法检查：

```bash
python -m py_compile \
  neural_instance_culling/dataset/build_directional_occlusion_evidence.py \
  neural_instance_culling/model/directional_occlusion_proxy_encoder_model.py \
  neural_instance_culling/model/train_directional_occlusion_proxy_encoder.py
```

遮挡证据 smoke：

```bash
conda run -n slm_pvs python -u neural_instance_culling/dataset/build_directional_occlusion_evidence.py \
  --dataset-dir neural_instance_culling/dataset/out/pose_csr_viewcell_back_camera_pvs_v1 \
  --runtime-meta hkust-v3/assets/runtimeVisibilityMeta.json \
  --output-dir /tmp/slm_evidence_smoke \
  --splits train \
  --max-poses 2 \
  --max-targets-per-pose 256 \
  --max-sources-per-pose 128 \
  --target-chunk-size 128 \
  --direction-bins 8 \
  --depth-shells 3 \
  --source-k 8
```

smoke 结果：2 个 train pose 中产生 4202 个正遮挡证据对，602 个非零证据单元。该结果只验证证据生成链路，不代表全量数据统计。

训练 smoke：

```bash
conda run -n slm_pvs python -u neural_instance_culling/model/train_directional_occlusion_proxy_encoder.py \
  --dataset-dir neural_instance_culling/dataset/out/pose_csr_viewcell_back_camera_pvs_v1 \
  --evidence-dir /tmp/slm_evidence_smoke \
  --runtime-meta hkust-v3/assets/runtimeVisibilityMeta.json \
  --glb-points neural_instance_culling/dataset/out/glb_points_v3.bin \
  --output-dir /tmp/smoke_pvs_directional_occlusion_proxy_encoder \
  --experiment-name smoke_pvs_directional_occlusion_proxy_encoder \
  --epochs 1 \
  --steps-per-epoch 1 \
  --eval-every 1 \
  --eval-pose-steps 1 \
  --max-candidates-per-pose 256 \
  --feature-export-batch-size 512 \
  --device cuda \
  --amp
```

训练 smoke 能完成一次前向、反向、阈值扫描、checkpoint 保存和固定特征导出。由于只训练 1 step 且遮挡证据只来自 2 个 pose，所有指标只能作为链路可运行证明，不能作为模型效果结论。

## 已执行的全量运行命令

全量遮挡证据构建：

```bash
conda run -n slm_pvs python -u neural_instance_culling/dataset/build_directional_occlusion_evidence.py \
  --dataset-dir neural_instance_culling/dataset/out/pose_csr_viewcell_back_camera_pvs_v1 \
  --runtime-meta hkust-v3/assets/runtimeVisibilityMeta.json \
  --output-dir neural_instance_culling/dataset/out/directional_occlusion_evidence_v1 \
  --splits train \
  --direction-bins 8 \
  --depth-shells 3 \
  --source-k 8 \
  > neural_instance_culling/dataset/out/directional_occlusion_evidence_v1_build_stdout.log \
  2> neural_instance_culling/dataset/out/directional_occlusion_evidence_v1_build_stderr.log
```

正式训练命令见上文“正式训练与 95% 召回工作点”。保留在此处的旧计划命令不再重复，避免与本次实际采用的 95% 工作点选择规则混淆。

后续仍需补充：

- 带 raw 子姿态 GT 的同口径 image PER。
- 前端 Worker / WebGPU smoke 延迟。
- 如果要设为前端默认资产，还必须完成浏览器同位姿对比和 runtime schema 文档。

## 保留状态

当前状态：实现已完成，smoke、正式 epoch 14 训练、95% 工作点导出、完整 684 test 统一指标和 small PER fallback 评测已完成。它尚未设为主线，尚未接入前端默认资产。

后续风险：

- 方向遮挡证据是包围盒投影弱监督，不是真实像素级遮挡分解，可能对细碎构件和复杂穿插结构存在噪声。
- 预测预算约束需要和高召回目标共同校准，否则可能压低漏检风险较高 pose 的预测数量。
- 遮挡代理能否替代运行时邻居展开，需要通过全量 test、image PER 和前端延迟共同判断。
