# V4 综合可见性唯一主线与执行协议

日期：2026-08-21

## 目的

本实验固定论文模型的唯一训练主线，消除近期实验中架构、数据划分和损失口径交叉复用的问题。模型从头训练，保留以下三项核心机制：

1. 分层遮挡关系先验和逐实例校准生存场；
2. 视点区域矩包络频谱查询；
3. 面向高加权召回的综合可见性损失。

本阶段只研究实例可见性。视觉效用、下载优先级、GLB 字节预算和资源调度不进入训练目标，也不参与参数扫描排名。前端运行结构保持不变，仍只读取固定实例特征表并执行一次轻量视角查询。

## 固定数据协议

主实验使用 `pose_csr_hkust_v3_main_stratified_calibration_fov66_v1`：

| split | view-cell 数量 | 用途 |
|---|---:|---|
| train | 5926 | 参数学习和离线关系编码 |
| calibration | 659 | 每个 checkpoint 独立冻结安全阈值 |
| validation | 730 | 选择参数配置和 checkpoint |
| test | 684 | 全部配置冻结后的一次性正式测试 |

真实渲染 FOV 为 60 度，后退候选、采样和模型查询 FOV 为 66 度。候选、GT、可见权重、查询中心和后退相机字段保持原样。主实验禁止从旧 checkpoint 初始化，禁止在 test 上选择阈值。

原 V4 关系表按旧空间 split 的 2772 个 train view-cell 构建，不能直接配合恢复后的主实验 split。关系表因此按当前 train 标签重建：硬件三角形深度缓存中的非 train 行会被过滤，缓存没有覆盖的 train view-cell 不生成伪造遮挡边。该关系表仍是 train-only 离线输入。

## 综合可见性损失

### 设计原因

完整旧 RVL 已含高正类权重 BCE、Tversky、预测数量和困难负例排序；Pose Balanced Frontier 又含逐 pose 平衡 BCE 和困难尾部排序。把两者与对比损失直接相加，会重复强化正例和排序梯度，容易重新出现低阈值、过预测和训练越久分类分离越差的问题。

新目标给每一项分配唯一职责：

- **逐 pose 平衡分类：** 每个 view-cell 内分别归一化正例和负例，再对 view-cell 等权平均，学习普通可见/不可见判别，避免候选数量大的 pose 支配训练。
- **RVL 加权召回保护：** 只保留 RVL 对重要可见实例的保护作用。按可见权重计算软加权漏检率，仅在漏检率超过目标时施加单侧平滑约束。该项不再包含第二套 BCE、Tversky、数量预算或负例排序。
- **共享困难边界：** 每个 pose 只选择当前分数最低的重要正例和分数最高的负例。相同样本同时用于 logit 间隔和训练期表征对比，避免两套困难样本挖掘相互冲突。
- **生存场监督：** 三角形深度剥离的删失监督、关系一致性和逐实例校准正则保持独立，继续负责遮挡表示本身，不与最终分类损失重复。

整体形式为：

```text
综合可见性损失
  = 逐 pose 平衡分类
  + 召回保护权重 × 单侧 RVL 加权召回保护
  + 边界权重 × 边界渐入系数
    × [(1 - 对比混合比例) × logit 间隔
       + 对比混合比例 × 表征间隔]
```

RVL 保护项以 aggregate weighted recall 为主体，并加入较小的最差 pose 尾部项，防止安全余量全部由少数容易 view-cell 提供。正式安全门仍由 calibration 上的 weighted recall 点估计及其单侧置信下界决定，普通 pose recall 只作诊断。

表征对比使用最终可见性查询隐藏特征。训练期小投影头把困难正例拉向同 pose 的其余可见实例原型，并推离共享困难负例。只有违反当前间隔的样本产生梯度；间隔满足后停止施压。投影头不保存到运行时模型，也不增加前端特征表和前向算子。

## 参数扫描

单种子快速扫描固定为 8 组，每组从头训练 10 epoch、每 epoch 100 step，四张 GPU 并行。扫描只覆盖以下高影响参数：

- 学习率：`1e-4`、`2e-4`、`3e-4`；
- 召回保护权重：`0.15`、`0.20`、`0.30`、`0.45`；
- 共享边界权重：`0.10`、`0.15`、`0.20`、`0.30`；
- 对比混合比例：`0`、`0.15`、`0.25`、`0.35`、`0.50`；
- logit 间隔：`0.30`、`0.50`、`0.75`。

每个 checkpoint 使用自己的 calibration 阈值。选择顺序固定为：

1. calibration 通过 weighted recall 安全门，且 validation weighted recall 大于 0.99；
2. balanced accuracy；
3. precision；
4. instance accuracy；
5. useful cull；
6. 平均预测实例数更少。

即使 8 组都未达到安全门，也选择相对最优配置继续正式长训，不允许 pilot 门控取消长训。

## 80 epoch 正式矩阵

选定配置后，以下 5 个成员均使用 `20260801/02/03` 三个种子，从头训练 80 epoch：

| 成员 | 分层关系/生存场 | 区域矩频谱 | RVL 召回保护 | 表征对比 |
|---|---|---|---|---|
| 完整模型 | 开启 | 开启 | 开启 | 开启 |
| 去除分层关系先验 | 改为同容量几何控制，保留逐实例校准 | 开启 | 开启 | 开启 |
| 去除区域矩频谱 | 开启 | 改为中心点查询 | 开启 | 开启 |
| 去除 RVL 召回保护 | 开启 | 开启 | 关闭 | 开启 |
| 去除对比表征分离 | 开启 | 开启 | 开启 | 关闭 |

失败成员也必须完成计划训练，并如实标记为没有合格安全工作点。正式测试只在配置、checkpoint 和 calibration 阈值全部冻结后执行。

## 运行入口

唯一 runner：

```bash
conda run -n slm_pvs python \
  neural_instance_culling/benchmark/run_pvs_v4_integrated_visibility_mainline_v1.py \
  preflight --data-root /mnt/sda/rhyang/slm

conda run -n slm_pvs python \
  neural_instance_culling/benchmark/run_pvs_v4_integrated_visibility_mainline_v1.py \
  smoke --data-root /mnt/sda/rhyang/slm --gpu-ids 0

conda run -n slm_pvs python \
  neural_instance_culling/benchmark/run_pvs_v4_integrated_visibility_mainline_v1.py \
  scan --data-root /mnt/sda/rhyang/slm --gpu-ids 0 1 2 3

conda run -n slm_pvs python \
  neural_instance_culling/benchmark/run_pvs_v4_integrated_visibility_mainline_v1.py \
  formal80 --data-root /mnt/sda/rhyang/slm --gpu-ids 0 1 2 3
```

## 扫描前代码审计修正

正式扫描前完成了两轮独立代码复核，并修正以下会影响长训可信度的问题：

- 训练期对比投影头只接收可见性梯度，关系、生存和调度目标不拥有该参数。多目标梯度投影现在只在安全目标与辅助目标共同拥有的参数坐标上计算冲突和投影系数，避免投影头的独占梯度稀释修正量并触发整组关系梯度回退清零。完整关系 CUDA smoke 中 `relationGradientProjectionFallbackZeroed = 0`。
- 最佳安全 checkpoint 的阈值仍只来自该 epoch 的 calibration；通过 calibration 与 validation 双重 weighted recall 安全门后，epoch 按 validation 的 balanced accuracy、precision、accuracy、useful cull 和平均预测数量依次选择，calibration useful cull 只作最后平局项。
- 独立 validation 重放同时计算 weighted recall 点估计和单侧置信下界，runner 只有在两者都大于 `0.99` 时才把成员列为安全成员。
- runner 会核对成员的 seed、变体、训练轮数和综合损失参数。不完整或不匹配的本实验成员会从头重跑；完整成员保持不动。
- 已存在的 validation 结果必须与当前选定 checkpoint、epoch、seed、阈值、模型描述和 calibration 摘要一致，否则强制重评。`formal80` 还会验证八组扫描和对应评价全部完成，不能用部分或旧扫描摘要启动长训。

## 快速扫描结果

八组单种子 10 epoch 扫描均从头完成，并在各自 calibration 冻结阈值后重放完整 validation。八组均满足 calibration 与 validation 的 aggregate weighted recall 点估计及其单侧置信下界大于 `0.99`。

| 配置 | 阈值 | weighted recall | 下界 | balanced accuracy | accuracy | precision | useful cull | 平均预测数 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| s00 | 0.28 | 0.99549 | 0.99407 | 0.62775 | 0.58565 | 0.03654 | 0.57022 | 2004.8 |
| s01 | 0.32 | 0.99479 | 0.99329 | **0.73907** | 0.73600 | 0.06195 | 0.71896 | 1306.3 |
| s02 | 0.34 | 0.99425 | 0.99270 | 0.67646 | **0.84714** | **0.07480** | **0.83572** | **725.2** |
| s03 | 0.32 | 0.99700 | 0.99596 | 0.64622 | 0.46545 | 0.03489 | 0.44626 | 2611.2 |
| s04 | 0.30 | 0.99801 | 0.99710 | 0.68124 | 0.48387 | 0.03819 | 0.46347 | 2535.2 |
| s05 | 0.38 | 0.99661 | 0.99542 | 0.58526 | 0.31817 | 0.02844 | 0.29830 | 3316.9 |
| s06 | 0.34 | 0.99967 | 0.99953 | 0.50941 | 0.05971 | 0.02340 | 0.03719 | 4569.2 |
| s07 | 0.22 | 0.99809 | 0.99736 | 0.61655 | 0.33519 | 0.03061 | 0.31426 | 3246.2 |

按预登记的“安全成员中先比较 balanced accuracy，再比较 precision、accuracy、useful cull 和预测数量”规则，正式长训选择 `s01`：学习率 `2e-4`、召回保护权重 `0.30`、共享边界权重 `0.15`、对比混合比例 `0.15`、logit 间隔 `0.50`。其 pose 宏平均 precision、recall、weighted recall 和 balanced accuracy 分别为 `0.12919`、`0.95593`、`0.99590` 和 `0.74738`。

`s02` 在 accuracy、precision、useful cull 和平均预测数上更好，但 balanced accuracy 明显低于 `s01`，说明它通过漏掉更多普通可见实例换取了剔除；其 aggregate recall 仅为 `0.49755`。它因此不覆盖预登记排序成为正式配置。扫描轮数较短，以上结果只用于固定长训超参数，不能作为论文最终结果。

## 当前验证状态

- 正式主 split 关系 CSR 已从已登记的硬件三角形深度缓存完成重建。它以当前 `5926` 个 train view-cell 为规范训练集合，仅保留缓存中属于当前 train 的 `10145` 个渲染行；缓存覆盖 `2029` 个唯一 train view-cell，其余 train view-cell 不合成遮挡关系。
- 正式关系表包含 `376929` 条保留关系边和 `3038552` 条生存监督记录，其中 `2496487` 条为观测到首遮挡的事件。原生后退视锥候选审计通过，没有补入 GT。
- 主 split、固定架构、禁止 checkpoint 初始化和关闭非可见性目标的 preflight 已通过。
- 使用完整正式关系表完成单步 CUDA 前向、反向、校准和 checkpoint 写出；总损失为 `2.0564`，峰值 CUDA 显存约 `3.03 GiB`，未出现非有限数值。
- 综合损失、多目标梯度投影、关系构建、训练主入口、导出、数据契约、评价器和 runner 共 `152` 项相关 unittest 已通过。
- 八组快速扫描已经完成并选定 `s01`；下一步执行五变体、三种子、80 epoch 正式长训。正式长训仍不得修改当前默认 checkpoint、阈值和前端资产。
