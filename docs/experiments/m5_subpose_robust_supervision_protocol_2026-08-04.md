# M5 Dense Subpose 鲁棒监督实验协议

日期：2026-08-04  
状态：协议已冻结，三种子训练执行中；dense 图像评价等待训练完成  
对应阶段：投稿计划 M5

## 1. 动机

当前硬件 GPU dense 评价已经证明渲染链路和实例绑定正确，但冻结模型的 miss-pixel rate 为 `0.7488%`，view-cell 长尾 p95 也未达到 `<1%`。当前训练只使用 view-cell 内所有 subpose 的可见实例并集，以及这些实例的最大屏幕覆盖代理；它没有使用同一实例在多少个 subpose 中出现的信息。

这会把两类正例混在一起：在整个 view-cell 中稳定出现的实例，以及只在一个位置出现但该位置具有较高视觉贡献的实例。后者正是 dense 评价中的鲁棒性风险。本实验使用数据集中已经保存的 `visible_hit_counts.bin` 和 `subpose_offsets.bin` 补充监督，不引入 teacher、不改变候选集合、不补入 GT，也不使用前端白名单。

## 2. 新监督的定义

对一个 view-cell 中的候选实例 `i`，已有标签为：

- `y_i`：实例是否出现在该 view-cell 的 dense subpose 可见并集中；
- `w_i`：该实例在 dense subpose 中的最大屏幕覆盖代理，不宣称为真实像素覆盖率；
- `r_i = hit_count_i / successful_subpose_count`：该实例出现在多少比例的成功 subpose 中。

仅对 `y_i=1` 的正例构造鲁棒风险权重：

```text
visual_mass_i = log(1 + w_i)
frequency_i = r_i ^ 0.5
risk_i = visual_mass_i * (1 + 1.0 * (1 - frequency_i))
```

损失由正例漏检项和风险最高正例的分数边界组成：

```text
L_robust_mass = sum_i risk_i * (1 - sigmoid(logit_i)) / sum_i risk_i
L_robust_tail = mean_top8(risk_i) softplus(1.5 - logit_i)
L_subpose_robust = L_robust_mass + 0.5 * L_robust_tail
```

该损失没有对负例增加可见奖励，因此不能单独推动所有实例输出为可见。原有集合分类、RVL、预测预算、hard-negative 排序和下载/效用损失仍然保留，继续约束误报和资源膨胀。外层权重冻结为 `0.35`。

## 3. 正式变体和随机种子

正式变体名称为：

```text
pvs_m5_subpose_robust_v1_hkust_spatial_fov66_seed20260801_full40
pvs_m5_subpose_robust_v1_hkust_spatial_fov66_seed20260802_full40
pvs_m5_subpose_robust_v1_hkust_spatial_fov66_seed20260803_full40
```

三种子为 `20260801`、`20260802`、`20260803`，每个运行 `40` epoch、`900` steps/epoch。基础损失使用 `m5_subpose_robust_v1`，其余模型结构、方向遮挡代理、候选 CSR、FOV、训练/validation/calibration/test split 与当前正式 HKUST 数据完全相同。

当前主线和第一轮 M5 视觉修复输出仅作为对照，不能被覆盖。新模型只有在完整 validation/calibration dense 图像评价和集合安全指标都达到登记门槛后，才有资格进入 one-shot test 候选。

## 4. 固定数据与评价协议

- 数据集：`pose_csr_hkust_v3_spatial_raw_subpose_aabb_fov66_v1`；
- `7,999` 个 view-cell、`279,008` 个 dense subpose；
- 候选是成功 subpose 的后退相机 AABB 候选并集，禁止 GT 正例补入；
- 模型输入和后退相机 FOV 为 `66°`，真实图像评价 FOV 为 `60°`；
- 阈值只能从每个 checkpoint 自己的 calibration split 冻结；
- 安全阈值还必须满足普通 pose recall `>=0.95`；只有同时满足普通 recall、加权召回点估计和加权召回单侧置信下界的校准行才可进入正式图像评价；
- calibration weighted recall 点估计至少 `0.9925`，view-cell 聚类 bootstrap 单侧 95% 下界大于 `0.99`；
- test split 在训练、checkpoint 选择、阈值选择和 dense validation/calibration 评价中保持封存；
- 正式图像评价必须通过硬件 GPU 门，并保存 `gpuBackend`、`gpuGate`、Chrome 日志和 `nvidia-smi` 证据。

必须同时报告 pose/aggregate recall、weighted recall、precision、F1、accuracy、balanced accuracy、useful cull、bad cull、平均预测数、GLB 字节，以及 dense subpose 的 miss-pixel rate、wrong-ID pixel rate、view-cell 均值和 p95。不能用 useful cull 或 weighted recall 单独替代图像安全结论。

## 5. 质量门和失败处理

主质量门为 validation 和 calibration 的 aggregate mean miss-pixel rate `<0.5%`、view-cell miss-pixel p95 `<1%`，同时保持 pose recall `>=0.95`、weighted recall 安全约束和无系统性 pop-in。若鲁棒监督改善图像安全但增加误报，必须完整报告该 trade-off；若只提高预测数量而没有改善图像指标，则降级为失败实验。

如果本轮仍未通过，下一轮必须先根据逐 view-cell、出现频率和视觉贡献的联合诊断提出新的机制假设，不能只扫描阈值或继续增大正例权重。最多登记十轮有明确假设的尝试；每一轮都使用独立输出目录和独立报告。

## 6. 可复现入口

```bash
bash neural_instance_culling/benchmark/run_m5_subpose_robust_training.sh \
  > neural_instance_culling/benchmark/out/m5_subpose_robust_training_20260804.log 2>&1
```

入口使用 `conda run -n slm_pvs`、CUDA 和四卡中的三个固定 GPU 槽，并将每个进程的 stdout/stderr 写入对应模型目录。训练入口在启用该损失但缺少 `visible_hit_counts.bin` 或 `subpose_offsets.bin` 时直接失败，禁止静默退化为旧监督。

### 2026-08-04 当前执行记录

三种子已按登记配置启动，分别固定到物理 GPU `0`、`1`、`2`；当前机器为 4 张 NVIDIA RTX A6000，驱动 `535.183.01`。每个训练目录均保存 `train_stdout.log`、`train_stderr.log`、`train_metrics.jsonl` 和 checkpoint，训练过程不使用浏览器软件渲染，也不读取 test split。当前训练入口若 CUDA 或数据资源检查失败会直接退出，不会切换到 CPU 训练来伪造正式结果。

训练日志位置：

```text
neural_instance_culling/model/out/pvs_m5_subpose_robust_v1_hkust_spatial_fov66_seed20260801_full40/
neural_instance_culling/model/out/pvs_m5_subpose_robust_v1_hkust_spatial_fov66_seed20260802_full40/
neural_instance_culling/model/out/pvs_m5_subpose_robust_v1_hkust_spatial_fov66_seed20260803_full40/
```

本记录中的“硬件 GPU”指 CUDA 训练或 Chrome 硬件 Vulkan 光栅化，二者都必须有对应证据；Chrome 的 SwiftShader、llvmpipe、softpipe、swrast 等软件后端只允许显式的小规模语义调试，不能进入本实验的正式数据、图像质量或性能结论。统一规则见 `docs/current/hardware_gpu_execution_policy.md`。

## 7. 独立 dense 图像评价入口

三种子训练完成后，独立的图像评价入口会等待三个 `calibration_ready_summary.json`，然后只读取
validation/calibration，使用各 checkpoint 自己冻结的 calibration 阈值生成实例级 Color-ID manifest，并调用
硬件 GPU 浏览器批处理。它使用独立输出目录，不能覆盖上一轮 M5 图像结果：

```bash
bash neural_instance_culling/benchmark/run_m5_subpose_robust_image_evaluation.sh \
  > neural_instance_culling/benchmark/out/m5_subpose_robust_image_dense_hw_20260804_launcher.log 2>&1
```

该入口的目录模板和三种子名称与本协议第 3 节完全一致；`--subposes-per-viewcell 0` 表示每个 view-cell 的全部
dense subpose。浏览器默认必须通过硬件 GPU 门，输出应保存 `gpuBackend`、`gpuGate`、Chrome 日志和
`nvidia-smi` 证据。若检测到软件渲染器，评价直接失败；不能把软件渲染结果合并到硬件目录或质量报告。

## 8. 当前边界

这轮实验只解决 view-cell 内 subpose 鲁棒监督，不改变模型输入粒度、前端预测频率、GLB 调度规则或默认资产。它也不能证明方向代理的独立因果贡献；方向代理仍以 M4-v2 的 `route_b_system` 结论为准。
