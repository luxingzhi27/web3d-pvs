# M5 图像安全修复实验预注册协议

日期：2026-08-02
状态：预注册；等待 M4 核心消融矩阵释放训练资源后执行。
对应阶段：投稿计划 M5，实例级 60°图像安全门。

## 1. 修复动机

HKUST 空间隔离 validation 的完整实例级评价已经证明，当前模型的平均漏像素率接近门槛，但长尾明显超标：213 个 view-cell 的单视点漏像素率 p95 为 `3.1973%`，最大值为 `13.1422%`。根因回查显示，问题不是 GLB 缺失、实例绑定或候选集合漏正样本，而是部分高屏幕贡献实例在模型阈值下得到过低分数。

当前训练损失把 `visible_weights` 先做 `log1p` 并以 `1024` 截断。对于 Color-ID 数据中表示屏幕覆盖的 parts-per-million 权重，这会让 `1,024` 以上的构件几乎获得相同的正例权重，无法区分一个占据约 `0.2%` 屏幕的实例和一个占据约 `7%` 屏幕的实例。这个设计对普通集合召回有帮助，但与 M5 的真实漏像素目标不完全一致。

本实验不把权重解释成真实几何像素覆盖率：当前数据的准确语义是“三维 Color-ID 光栅化后、在 view-cell 的成功 subpose 上取最大值的屏幕覆盖代理”。修复目标是利用该代理构造可微的视觉安全训练信号，并在真实 60°实例 ID 渲染中验证它是否确实降低漏像素长尾。

## 2. 研究假设

主假设是：在保持候选集合、视场角和 false-positive 约束不变时，直接最小化每个 pose 的软漏视觉质量，能够比 `log1p` 饱和权重更好地保护高贡献实例。

对一个 pose 的候选实例集合 `C`，模型输出可见概率 `p_i`，屏幕覆盖代理为 `w_i`，定义归一化视觉质量损失：

```text
L_mass = sum_i [w_i * (1 - p_i)] / max(sum_i w_i, epsilon)
```

该项只对 GT 可见实例产生漏检梯度，不会单独压低不可见实例分数。因此它必须和 RVL 的误报项、预测预算项以及 hard-negative 排序共同使用，不能单独替换整个可见性损失。为保护少数最大构件，再加入按 `w_i` 排序的上尾正例 margin 项：

```text
L_tail = mean_{i in top-k(w)} softplus(margin - logit_i)
L_visual = L_mass + lambda_tail * L_tail
```

其中 `top-k` 只在 pose 内选择正例，`k` 和 `lambda_tail` 在 validation/calibration 预注册，不根据 test 图像结果调整。

## 3. 对照和实验变体

每个变体使用同一 HKUST 空间四路 split、同一候选 CSR、同一方向遮挡证据、同一模型容量、同一 40 epoch 训练预算和至少三个随机种子。实验目录使用稳定名称，不覆盖当前主线或 M4 输出。

| 变体 | 变化 | 目的 |
|---|---|---|
| `m5_visual_mass_linear` | `L_mass`，`lambda_tail=0` | 检查线性视觉质量监督是否足够 |
| `m5_visual_mass_tail` | `L_mass + L_tail`，固定 `k=8` | 保护高贡献长尾构件 |
| `m5_visual_mass_soft` | 对 `w_i` 使用平方根变换后归一化，再加 `L_tail` | 检查线性权重是否过度牺牲普通剔除效率 |
| `m5_control_log1p` | 保持当前损失和参数，仅换独立 seed | 估计训练随机性，避免把随机波动误判为修复收益 |

为避免在图像结果生成后反向调权重，第一轮的数值在启动训练前冻结如下：

| 变体 | `visual_safety_loss_weight` | `weight_power` | `tail_k` | `tail_margin` | `tail_weight` |
|---|---:|---:|---:|---:|---:|
| `m5_visual_mass_linear` | `0.20` | `1.0` | `0` | `1.0` | `0.0` |
| `m5_visual_mass_tail` | `0.20` | `1.0` | `8` | `1.0` | `0.5` |
| `m5_visual_mass_soft` | `0.20` | `0.5` | `8` | `1.0` | `0.5` |
| `m5_control_log1p` | `0.0` | `1.0` | `8` | `1.0` | `0.5` |

这里的外层权重 `0.20` 与现有下载效用辅助项同量级，避免视觉损失替代 RVL；`weight_power=1.0` 保留
屏幕覆盖代理的相对比例，`0.5` 用于检验软化长尾是否能减少过度预测。四个变体都使用
`rvl_strong_v2` 的其余固定参数，不在 validation 或 calibration 上继续搜索这些数值。

第一轮只比较四个预注册变体，不进行无边界权重搜索。若主假设不成立，下一轮必须针对诊断结果提出新的机制假设；每轮最多保留十次有明确假设的尝试，不能通过反复扫描阈值代替模型改进。

## 4. 安全与选择规则

1. 训练期间只使用 train；checkpoint 选择只看完整固定 validation。
2. 独立 calibration 选择阈值，要求 weighted recall 点估计至少 `0.9925`，并要求按 view-cell 聚类 bootstrap 的单侧 95% 下置信界大于 `0.99`。
3. 所有变体使用相同的 calibration 规则，不共享当前模型的阈值，也不读取 test 来调参。
4. 先满足 weighted recall 和图像安全约束，再比较 pose precision、useful cull、bad cull、平均预测数和 GLB 字节。
5. 变体只有在 validation/calibration 的 miss-pixel 均值和 p95 同时改善，并且普通 recall、bad cull 没有超过预注册退化限度时，才进入 one-shot test 候选。
6. test 只在模型、特征、阈值和后处理全部冻结后运行一次；test 失败时保留失败结果并重新命名下一轮实验。

## 5. 质量门

正式 M5 门为：完整 validation 和 calibration 的 mean miss-pixel rate `<0.5%`、view-cell p95 `<1%`，并且不出现系统性 pop-in。普通 pose recall 应尽量不低于 `0.95`；若低于该值，必须逐视点检查漏像素是否确实集中在低视觉贡献实例，不能仅凭 weighted recall 解释通过。

图像评价必须继续使用真实 `60°`渲染相机、实例级 `componentGlobalId`、完整本地 GLB 清单和同位姿 reference/prediction。M5 的质量门与 M10 的移动性能门分开判断；浏览器 SwiftShader smoke 不得替代图像质量或移动设备证据。

## 6. 需要记录的证据

每个变体必须保存：训练 stdout/stderr、完整 loss 分量和梯度范数、checkpoint 配置、特征/数据 manifest、calibration threshold rows、view-cell bootstrap、validation/calibration 图像汇总、逐 view-cell 长尾表和运行命令。报告至少解释以下问题：

- 高覆盖正例的分数分布是否整体上移；
- 漏像素是否从少数大构件转移为大量低贡献构件；
- `useful cull` 的变化是否来自有效剔除而非错误漏检；
- 额外预测和 GLB 字节是否在可接受范围；
- 图像安全改善是否在不同 view-cell 空间块上稳定，而不是只改善少数区域。

## 7. 当前准入状态

该协议本身不改变当前主线和 M5 的 No-Go 状态。M4 三种子消融完成后，先根据其独立效应决定保留完整方向代理路线还是转为简化路线，再执行本协议；这样可以避免在核心架构尚未确定时产生不可比较的修复模型。

## 8. 可复现实验入口

正式矩阵入口为：

```bash
bash neural_instance_culling/benchmark/run_m5_visual_safety_repair.sh \
  > neural_instance_culling/benchmark/out/m5_visual_safety_repair_queue.log 2>&1
```

脚本默认使用 `20260801/20260802/20260803` 三个 seed 和四张 GPU，并在启动前等待当前正式 M4/M11 会话
退出。每个变体写入独立的 `model/out/<experiment_name>/`，如果发现不完整目录会直接失败，不覆盖中间结果。
当前脚本已在 `m5_visual_repair` tmux 会话中排队；截至本记录生成时仍在等待 M4/M11 完成，尚未产生修复模型
或图像指标。
