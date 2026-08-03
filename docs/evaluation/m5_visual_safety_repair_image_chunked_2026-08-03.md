# M5 视觉安全修复分块图像评价

日期：2026-08-03  
场景：HKUST v3  
评价范围：validation/calibration；test split 未读取

## 目的与协议

上一轮 M5 图像评价因单个浏览器页面持续累积全场景 GLB 解码对象而未完成。本轮只修复评价器的资源生命周期，不改变模型、候选集合、真实可见集合、阈值或样本语义。

每个分块使用独立的 headless Chrome 页面，保留完整的 3,273 个 GLB inventory、实例绑定、候选和预测实例 ID；同一分块内复用相同相机的 reference 渲染。44 个分块完成后按像素计数合并，而不是平均各块百分比。渲染视场角为 60°，模型输入和后退相机口径为 66°。

执行命令：

```bash
conda run --no-capture-output -n slm_pvs python -u \
  neural_instance_culling/benchmark/run_m5_visual_safety_image_evaluation.py \
  --output-name m5_visual_safety_repair_image_chunked_20260803 \
  --chunk-samples 16 --render-timeout-sec 7200 --width 320 --height 180
```

正式输入仍来自四个视觉安全损失变体、三个 seed 的 validation/calibration manifest。阈值沿用各 checkpoint 在 calibration 冻结的阈值；本轮没有读取 test，也没有重新扫描阈值。

## 完整性结果

| 项目 | 结果 |
|---|---:|
| 源 batch | 24 |
| 分块 | 44/44 |
| 总样本 | 16,248 |
| validation 样本 | 7,968（12 batch × 664） |
| calibration 样本 | 8,280（12 batch × 690） |
| GLB inventory | 3,273，完整保留 |
| 渲染失败 | 0 |
| 缺失 GLB 样本 | 0 |
| self-consistency PER | 0 |
| extra-pixel | 0 |
| testRead | false |

## 图像结果

像素合并结果使用真实 reference 的有效实例像素作为 miss-pixel 分母；背景像素不计入 miss-pixel。`visible_weights` 仍是可见重要性代理，不被解释为精确像素覆盖率。

| 数据划分 | 样本 | 像素合并 miss-pixel | 逐 view-cell miss 均值 | 逐 view-cell miss p95 | wrong-ID 像素合并 |
|---|---:|---:|---:|---:|---:|
| validation | 7,968 | 0.8837% | 0.8958% | 6.0033% | 0.5930% |
| calibration | 8,280 | 0.6920% | 0.6487% | 5.0171% | 0.7757% |
| 全部 validation+calibration | 16,248 | 0.78596% | 0.76986% | 未作为主门槛 | 0.6861% |

预注册质量门为 validation/calibration 图像的 mean miss-pixel `<0.5%`、view-cell p95 `<1%`，并且不能出现系统性 pop-in。本轮 validation 两项均未达到门槛，尤其是逐 view-cell p95 为 6.0033%；因此 M5 仍为 `No-Go`。该结论不能通过降低阈值、补入 GT、缩小候选集合或前端白名单修复。

各修复变体的 validation 结果如下，完整的 calibration 表和逐 batch 数据保存在输出目录的 `summary.json`、`summary.md` 以及分块 `sample_image_metrics.json` 中：

| 变体/seed | miss-pixel 合并 | view-cell 均值 | view-cell p95 |
|---|---:|---:|---:|
| visual_mass_linear / 20260801 | 0.9545% | 0.9599% | 6.7302% |
| visual_mass_tail / 20260801 | 1.1474% | 1.1826% | 8.2798% |
| visual_mass_soft / 20260801 | 1.0105% | 1.0193% | 6.3114% |
| control_log1p / 20260801 | 0.9858% | 0.9845% | 6.7688% |
| visual_mass_linear / 20260802 | 1.0263% | 1.0374% | 7.5715% |
| visual_mass_tail / 20260802 | 0.7157% | 0.7400% | 4.3028% |
| visual_mass_soft / 20260802 | 0.9560% | 0.9839% | 7.2747% |
| control_log1p / 20260802 | 1.2198% | 1.2181% | 8.4681% |
| visual_mass_linear / 20260803 | 0.6275% | 0.6435% | 3.7054% |
| visual_mass_tail / 20260803 | 0.6474% | 0.6582% | 3.6991% |
| visual_mass_soft / 20260803 | 0.6018% | 0.6169% | 3.4898% |
| control_log1p / 20260803 | 0.7114% | 0.7049% | 4.2681% |

即使 validation 中最好的 `visual_mass_soft / 20260803` 仍高于 0.5%，且 p95 高于 1%，所以不能据此宣布视觉安全修复成功。self-consistency PER 为 0 表明 reference/prediction 的实例 ID 绑定和颜色编码在本轮没有引入自相矛盾；这不等于模型画面质量达到门槛。

## 资源与运行边界

分块评价的浏览器页面数为 44，GLB 加载调用为 143,580。该数字反映“每块完整 inventory 加载一次”的评价成本，不是前端线上下载成本；Chrome 使用 SwiftShader，不能用来宣称 NVIDIA GPU 或移动设备性能。M4-v2 的 GLB 字节、WebGPU 延迟和主线程延迟字段仍按其自身协议保持 `not_available`。

## 保留结论

本轮保留分块 renderer 和 AABB 缺失时的 fail-open 修复，因为它们修复了评价器正确性和浏览器资源生命周期问题。M5 的模型视觉安全门仍为 `No-Go`；下一轮若要提升画面安全，应基于漏像素构件分布设计新的独立机制实验，并重新在 calibration 冻结阈值，不能把本轮结果并入默认模型或前端资产。
