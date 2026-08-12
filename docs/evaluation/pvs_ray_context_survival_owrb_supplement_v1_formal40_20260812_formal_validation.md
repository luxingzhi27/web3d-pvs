# 视线关系场、生存场与 OWRB 补充矩阵评价

本报告只汇总 calibration 冻结阈值后的 validation 结果。补充矩阵是独立的五变体机制对照，不生成核心 2×2×2 因子主效应；所有成员使用同一 pose 顺序、后退相机候选集合、实例 GT 和候选哈希。
paired bootstrap 先按 seed 重采样，再在每个 seed 内按 pose 重采样。

- 变体：5；seed：3；pose：213。
- bootstrap：10000 次；test split 读取：否。
- weighted recall 是画面安全主指标；useful cull 必须与 bad cull、recall 和 weighted recall 一起解释。
- Color-ID 图像评价：已完成硬件 GPU 正式评价。
- WebGPU 数值 parity：未登记；不能据此报告浏览器 WebGPU 结果。

## 画面安全指标

| 变体 | seed | 阈值 | pose recall | aggregate recall | pose weighted recall | aggregate weighted recall | pose useful cull | aggregate useful cull | pose bad cull | aggregate bad cull | safety-adjusted useful cull |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `triangle_context32_direct9_monotone` | 20260801 | 0.600000 | 0.8251 | 0.7609 | 0.9903 | 0.9912 | 0.6587 | 0.9155 | 0.0243 | 0.0106 | 0.5721 |
| `triangle_context32_direct9_monotone` | 20260802 | 0.360000 | 0.8220 | 0.7602 | 0.9911 | 0.9920 | 0.6588 | 0.9158 | 0.0270 | 0.0106 | 0.5701 |
| `triangle_context32_direct9_monotone` | 20260803 | 0.600000 | 0.7851 | 0.6374 | 0.9885 | 0.9892 | 0.6774 | 0.9332 | 0.0355 | 0.0161 | 0.5590 |
| `triangle_context64_direct9_monotone` | 20260801 | 0.220000 | 0.8125 | 0.6515 | 0.9932 | 0.9936 | 0.6680 | 0.9289 | 0.0313 | 0.0154 | 0.5713 |
| `triangle_context64_direct9_monotone` | 20260802 | 0.520000 | 0.8074 | 0.7270 | 0.9895 | 0.9905 | 0.6664 | 0.9217 | 0.0299 | 0.0121 | 0.5661 |
| `triangle_context64_direct9_monotone` | 20260803 | 0.620000 | 0.8123 | 0.7313 | 0.9899 | 0.9908 | 0.6632 | 0.9207 | 0.0280 | 0.0119 | 0.5670 |
| `triangle_context32_fourier117_monotone` | 20260801 | 0.880000 | 0.8722 | 0.8348 | 0.9960 | 0.9965 | 0.7153 | 0.9170 | 0.0097 | 0.0073 | 0.6568 |
| `triangle_context32_fourier117_monotone` | 20260802 | 0.840000 | 0.8622 | 0.8189 | 0.9959 | 0.9963 | 0.7206 | 0.9197 | 0.0118 | 0.0080 | 0.6541 |
| `triangle_context32_fourier117_monotone` | 20260803 | 0.900000 | 0.8596 | 0.7881 | 0.9957 | 0.9961 | 0.7246 | 0.9259 | 0.0115 | 0.0094 | 0.6557 |
| `triangle_context32_direct9_unconstrained28` | 20260801 | 0.540000 | 0.8294 | 0.7748 | 0.9909 | 0.9917 | 0.6555 | 0.9125 | 0.0240 | 0.0100 | 0.5723 |
| `triangle_context32_direct9_unconstrained28` | 20260802 | 0.500000 | 0.8116 | 0.7341 | 0.9900 | 0.9909 | 0.6636 | 0.9206 | 0.0287 | 0.0118 | 0.5669 |
| `triangle_context32_direct9_unconstrained28` | 20260803 | 0.580000 | 0.7876 | 0.5653 | 0.9909 | 0.9911 | 0.6813 | 0.9379 | 0.0386 | 0.0192 | 0.5648 |
| `aabb_context32_direct9_monotone` | 20260801 | 0.500000 | 0.7874 | 0.5760 | 0.9900 | 0.9905 | 0.6787 | 0.9372 | 0.0370 | 0.0188 | 0.5626 |
| `aabb_context32_direct9_monotone` | 20260802 | 0.360000 | 0.8183 | 0.7307 | 0.9908 | 0.9916 | 0.6629 | 0.9210 | 0.0285 | 0.0119 | 0.5710 |
| `aabb_context32_direct9_monotone` | 20260803 | 0.560000 | 0.7885 | 0.5656 | 0.9911 | 0.9913 | 0.6794 | 0.9378 | 0.0369 | 0.0192 | 0.5639 |

## 分类诊断指标

| 变体 | seed | pose precision | aggregate precision | pose F1 | aggregate F1 | pose Jaccard | aggregate Jaccard | pose accuracy | aggregate accuracy | pose balanced acc. | aggregate balanced acc. | pose specificity | aggregate specificity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `triangle_context32_direct9_monotone` | 20260801 | 0.4246 | 0.4556 | 0.5046 | 0.5700 | 0.3602 | 0.3986 | 0.8118 | 0.9492 | 0.7825 | 0.8594 | 0.7399 | 0.9579 |
| `triangle_context32_direct9_monotone` | 20260802 | 0.4207 | 0.4573 | 0.4979 | 0.5710 | 0.3532 | 0.3996 | 0.8093 | 0.9494 | 0.7815 | 0.8592 | 0.7410 | 0.9582 |
| `triangle_context32_direct9_monotone` | 20260803 | 0.4470 | 0.5559 | 0.4999 | 0.5939 | 0.3580 | 0.4223 | 0.8193 | 0.9614 | 0.7776 | 0.8069 | 0.7700 | 0.9764 |
| `triangle_context64_direct9_monotone` | 20260801 | 0.4162 | 0.5180 | 0.4826 | 0.5772 | 0.3408 | 0.4056 | 0.8141 | 0.9577 | 0.7844 | 0.8117 | 0.7563 | 0.9719 |
| `triangle_context64_direct9_monotone` | 20260802 | 0.4295 | 0.4863 | 0.4978 | 0.5828 | 0.3543 | 0.4112 | 0.8139 | 0.9539 | 0.7802 | 0.8457 | 0.7530 | 0.9644 |
| `triangle_context64_direct9_monotone` | 20260803 | 0.4282 | 0.4801 | 0.4996 | 0.5796 | 0.3557 | 0.4081 | 0.8126 | 0.9530 | 0.7794 | 0.8473 | 0.7466 | 0.9633 |
| `triangle_context32_fourier117_monotone` | 20260801 | 0.6450 | 0.4880 | 0.7127 | 0.6160 | 0.5813 | 0.4450 | 0.8830 | 0.9539 | 0.8361 | 0.8971 | 0.7999 | 0.9594 |
| `triangle_context32_fourier117_monotone` | 20260802 | 0.6430 | 0.5016 | 0.7093 | 0.6221 | 0.5782 | 0.4515 | 0.8862 | 0.9560 | 0.8355 | 0.8906 | 0.8088 | 0.9623 |
| `triangle_context32_fourier117_monotone` | 20260803 | 0.6537 | 0.5389 | 0.7151 | 0.6401 | 0.5837 | 0.4707 | 0.8906 | 0.9608 | 0.8365 | 0.8784 | 0.8134 | 0.9688 |
| `triangle_context32_direct9_unconstrained28` | 20260801 | 0.4201 | 0.4423 | 0.5031 | 0.5631 | 0.3583 | 0.3919 | 0.8089 | 0.9468 | 0.7821 | 0.8648 | 0.7348 | 0.9548 |
| `triangle_context32_direct9_unconstrained28` | 20260802 | 0.4290 | 0.4807 | 0.5002 | 0.5810 | 0.3559 | 0.4094 | 0.8123 | 0.9531 | 0.7796 | 0.8487 | 0.7475 | 0.9633 |
| `triangle_context32_direct9_unconstrained28` | 20260803 | 0.4392 | 0.5844 | 0.4894 | 0.5747 | 0.3481 | 0.4032 | 0.8201 | 0.9630 | 0.7823 | 0.7734 | 0.7771 | 0.9814 |
| `aabb_context32_direct9_monotone` | 20260801 | 0.4347 | 0.5786 | 0.4866 | 0.5773 | 0.3461 | 0.4058 | 0.8192 | 0.9627 | 0.7799 | 0.7783 | 0.7725 | 0.9806 |
| `aabb_context32_direct9_monotone` | 20260802 | 0.4227 | 0.4824 | 0.4951 | 0.5811 | 0.3513 | 0.4096 | 0.8117 | 0.9534 | 0.7826 | 0.8472 | 0.7470 | 0.9637 |
| `aabb_context32_direct9_monotone` | 20260803 | 0.4340 | 0.5830 | 0.4856 | 0.5742 | 0.3449 | 0.4027 | 0.8199 | 0.9629 | 0.7814 | 0.7734 | 0.7743 | 0.9813 |

## 剔除、资源与运行成本

| 变体 | seed | 平均预测数 | pose 平均 FP | pose 平均 FN | pose 平均 TN | aggregate 平均 FP | aggregate 平均 FN | aggregate 平均 TN | 预测/候选 | 预测/GT | 候选 GLB 数 | 预测 GLB 数 | 候选 GLB 字节 | 预测 GLB 字节 | GLB 数量削减 | GLB 字节削减 | 下载效用召回 | 达到同等视觉效用所需字节 | 前向 p95 ms | 特征表字节 | 固定表维度 | 头输入维度 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `triangle_context32_direct9_monotone` | 20260801 | 382.5117 | 208.2207 | 54.7653 | 4737.1174 | 208.2207 | 54.7653 | 4737.1174 | 0.0739 | 1.6699 | 963.7746 | 119.2300 | 183619879.7371 | 43983312.1502 | 0.6466 | 0.5366 | 0.9914 | 15763425.9718 | 2.114 | 5875272 | 156 | 177 |
| `triangle_context32_direct9_monotone` | 20260802 | 380.8263 | 206.6901 | 54.9202 | 4738.6479 | 206.6901 | 54.9202 | 4738.6479 | 0.0736 | 1.6626 | 963.7746 | 117.7465 | 183619879.7371 | 43946696.4319 | 0.6505 | 0.5396 | 0.9921 | 15608831.7559 | 2.166 | 5875272 | 156 | 177 |
| `triangle_context32_direct9_monotone` | 20260803 | 262.6291 | 116.6338 | 83.0610 | 4828.7042 | 116.6338 | 83.0610 | 4828.7042 | 0.0508 | 1.1466 | 963.7746 | 84.4930 | 183619879.7371 | 36482070.0657 | 0.6830 | 0.5634 | 0.9894 | 11840252.4319 | 3.044 | 5875272 | 156 | 177 |
| `triangle_context64_direct9_monotone` | 20260801 | 288.0657 | 138.8357 | 79.8263 | 4806.5023 | 138.8357 | 79.8263 | 4806.5023 | 0.0557 | 1.2576 | 963.7746 | 96.8545 | 183619879.7371 | 41051270.5352 | 0.6650 | 0.5461 | 0.9939 | 13647409.2207 | 2.196 | 7080456 | 188 | 217 |
| `triangle_context64_direct9_monotone` | 20260802 | 342.4225 | 175.8920 | 62.5258 | 4769.4460 | 175.8920 | 62.5258 | 4769.4460 | 0.0662 | 1.4949 | 963.7746 | 104.1268 | 183619879.7371 | 40735482.0845 | 0.6639 | 0.5497 | 0.9906 | 13870380.8638 | 3.204 | 7080456 | 188 | 217 |
| `triangle_context64_direct9_monotone` | 20260803 | 348.9061 | 181.3991 | 61.5493 | 4763.9390 | 181.3991 | 61.5493 | 4763.9390 | 0.0674 | 1.5232 | 963.7746 | 107.4883 | 183619879.7371 | 41358792.9202 | 0.6581 | 0.5469 | 0.9909 | 14324981.3146 | 2.156 | 7080456 | 188 | 217 |
| `triangle_context32_fourier117_monotone` | 20260801 | 391.8451 | 200.6197 | 37.8310 | 4744.7183 | 200.6197 | 37.8310 | 4744.7183 | 0.0757 | 1.7107 | 963.7746 | 154.5775 | 183619879.7371 | 58008166.8357 | 0.6708 | 0.6249 | 0.9967 | 22563577.4648 | 2.532 | 5875272 | 156 | 285 |
| `triangle_context32_fourier117_monotone` | 20260802 | 373.9390 | 186.3709 | 41.4883 | 4758.9671 | 186.3709 | 41.4883 | 4758.9671 | 0.0723 | 1.6325 | 963.7746 | 149.5023 | 183619879.7371 | 57533230.9108 | 0.6795 | 0.6292 | 0.9966 | 21557168.0000 | 5.152 | 5875272 | 156 | 285 |
| `triangle_context32_fourier117_monotone` | 20260803 | 334.9953 | 154.4695 | 48.5305 | 4790.8685 | 154.4695 | 48.5305 | 4790.8685 | 0.0647 | 1.4625 | 963.7746 | 137.9577 | 183619879.7371 | 55597521.9531 | 0.6844 | 0.6292 | 0.9964 | 19887479.5681 | 2.659 | 5875272 | 156 | 285 |
| `triangle_context32_direct9_unconstrained28` | 20260801 | 401.2254 | 223.7606 | 51.5915 | 4721.5775 | 223.7606 | 51.5915 | 4721.5775 | 0.0775 | 1.7516 | 963.7746 | 124.9577 | 183619879.7371 | 45029683.7183 | 0.6419 | 0.5356 | 0.9918 | 16631038.8732 | 1.837 | 5875272 | 156 | 177 |
| `triangle_context32_direct9_unconstrained28` | 20260802 | 349.8028 | 181.6479 | 60.9014 | 4763.6901 | 181.6479 | 60.9014 | 4763.6901 | 0.0676 | 1.5271 | 963.7746 | 108.6009 | 183619879.7371 | 41667602.7793 | 0.6583 | 0.5471 | 0.9910 | 14378428.4507 | 1.828 | 5875272 | 156 | 177 |
| `triangle_context32_direct9_unconstrained28` | 20260803 | 221.5962 | 92.1033 | 99.5634 | 4853.2347 | 92.1033 | 99.5634 | 4853.2347 | 0.0428 | 0.9674 | 963.7746 | 77.1831 | 183619879.7371 | 35003771.6620 | 0.6907 | 0.5689 | 0.9913 | 10784149.3333 | 2.750 | 5875272 | 156 | 177 |
| `aabb_context32_direct9_monotone` | 20260801 | 228.0188 | 96.0798 | 97.1174 | 4849.2582 | 96.0798 | 97.1174 | 4849.2582 | 0.0441 | 0.9955 | 963.7746 | 77.6995 | 183619879.7371 | 34835971.7183 | 0.6874 | 0.5674 | 0.9907 | 10819324.9014 | 2.829 | 5875272 | 156 | 177 |
| `aabb_context32_direct9_monotone` | 20260802 | 346.9577 | 179.5915 | 61.6901 | 4765.7465 | 179.5915 | 61.6901 | 4765.7465 | 0.0671 | 1.5147 | 963.7746 | 107.0469 | 183619879.7371 | 41787104.9014 | 0.6589 | 0.5451 | 0.9918 | 14394873.7089 | 2.095 | 5875272 | 156 | 177 |
| `aabb_context32_direct9_monotone` | 20260803 | 222.1925 | 92.6432 | 99.5070 | 4852.6948 | 92.6432 | 99.5070 | 4852.6948 | 0.0429 | 0.9700 | 963.7746 | 78.8357 | 183619879.7371 | 35284656.3756 | 0.6864 | 0.5662 | 0.9916 | 10461322.7981 | 3.030 | 5875272 | 156 | 177 |

## 图像质量

| 变体 | seed | PER | mean miss-pixel | p95 miss-pixel | wrong-ID pixel | extra-pixel |
|---|---:|---:|---:|---:|---:|---:|
| `triangle_context32_direct9_monotone` | 20260801 | 0.0127 | 0.0097 | 0.0519 | 0.0035 | 0.0000 |
| `triangle_context32_direct9_monotone` | 20260802 | 0.0112 | 0.0068 | 0.0296 | 0.0048 | 0.0000 |
| `triangle_context32_direct9_monotone` | 20260803 | 0.0131 | 0.0065 | 0.0340 | 0.0068 | 0.0000 |
| `triangle_context64_direct9_monotone` | 20260801 | 0.0061 | 0.0010 | 0.0027 | 0.0053 | 0.0000 |
| `triangle_context64_direct9_monotone` | 20260802 | 0.0136 | 0.0102 | 0.0677 | 0.0040 | 0.0000 |
| `triangle_context64_direct9_monotone` | 20260803 | 0.0129 | 0.0096 | 0.0444 | 0.0037 | 0.0000 |
| `triangle_context32_fourier117_monotone` | 20260801 | 0.0051 | 0.0043 | 0.0128 | 0.0011 | 0.0000 |
| `triangle_context32_fourier117_monotone` | 20260802 | 0.0051 | 0.0039 | 0.0128 | 0.0015 | 0.0000 |
| `triangle_context32_fourier117_monotone` | 20260803 | 0.0053 | 0.0044 | 0.0128 | 0.0012 | 0.0000 |
| `triangle_context32_direct9_unconstrained28` | 20260801 | 0.0121 | 0.0090 | 0.0428 | 0.0034 | 0.0000 |
| `triangle_context32_direct9_unconstrained28` | 20260802 | 0.0127 | 0.0087 | 0.0556 | 0.0046 | 0.0000 |
| `triangle_context32_direct9_unconstrained28` | 20260803 | 0.0078 | 0.0012 | 0.0057 | 0.0067 | 0.0000 |
| `aabb_context32_direct9_monotone` | 20260801 | 0.0087 | 0.0022 | 0.0101 | 0.0070 | 0.0000 |
| `aabb_context32_direct9_monotone` | 20260802 | 0.0108 | 0.0056 | 0.0175 | 0.0054 | 0.0000 |
| `aabb_context32_direct9_monotone` | 20260803 | 0.0078 | 0.0012 | 0.0067 | 0.0068 | 0.0000 |

## 配对差值和因子效应

差值定义为左侧减右侧；补充矩阵只报告注册的单因素机制对照，不把它们包装成完整因子主效应。
所有区间均保持同一 seed/pose 配对重采样；区间跨零时不宣称稳定改善。

| 比较 | 指标 | 差值 | 95% 区间 | 方向 | 跨零 |
|---|---|---:|---|---|---|
| `context64_minus_context32` | `pose_precision` | -0.0062 | [-0.0188, 0.0083] | negative | 是 |
| `context64_minus_context32` | `aggregate_precision` | 0.0052 | [-0.0718, 0.0627] | positive | 是 |
| `context64_minus_context32` | `pose_recall` | 0.0000 | [-0.0177, 0.0265] | positive | 是 |
| `context64_minus_context32` | `aggregate_recall` | -0.0162 | [-0.1053, 0.0890] | negative | 是 |
| `context64_minus_context32` | `pose_weighted_recall` | 0.0009 | [-0.0015, 0.0034] | positive | 是 |
| `context64_minus_context32` | `aggregate_weighted_recall` | 0.0008 | [-0.0014, 0.0031] | positive | 是 |
| `context64_minus_context32` | `pose_accuracy` | 0.0000 | [-0.0066, 0.0054] | positive | 是 |
| `context64_minus_context32` | `aggregate_accuracy` | 0.0016 | [-0.0076, 0.0091] | positive | 是 |
| `context64_minus_context32` | `pose_balanced_accuracy` | 0.0008 | [-0.0022, 0.0042] | positive | 是 |
| `context64_minus_context32` | `aggregate_balanced_accuracy` | -0.0069 | [-0.0468, 0.0381] | negative | 是 |
| `context64_minus_context32` | `pose_f1` | -0.0075 | [-0.0210, 0.0022] | negative | 是 |
| `context64_minus_context32` | `aggregate_f1` | 0.0016 | [-0.0192, 0.0190] | positive | 是 |
| `context64_minus_context32` | `pose_useful_cull` | 0.0009 | [-0.0133, 0.0106] | positive | 是 |
| `context64_minus_context32` | `aggregate_useful_cull` | 0.0023 | [-0.0122, 0.0135] | positive | 是 |
| `context64_minus_context32` | `pose_bad_cull` | 0.0008 | [-0.0071, 0.0072] | positive | 是 |
| `context64_minus_context32` | `aggregate_bad_cull` | 0.0007 | [-0.0040, 0.0050] | positive | 是 |
| `context64_minus_context32` | `pose_avg_fp_count` | -11.8059 | [-70.9353, 58.5651] | negative | 是 |
| `context64_minus_context32` | `pose_avg_fn_count` | 3.7183 | [-19.9646, 25.3085] | positive | 是 |
| `context64_minus_context32` | `pose_avg_tn_count` | 11.8059 | [-59.7192, 71.2131] | positive | 是 |
| `context64_minus_context32` | `aggregate_avg_fp_count` | -11.8059 | [-70.2968, 59.5211] | negative | 是 |
| `context64_minus_context32` | `aggregate_avg_fn_count` | 3.7183 | [-20.6088, 25.3746] | positive | 是 |
| `context64_minus_context32` | `aggregate_avg_tn_count` | 11.8059 | [-61.3353, 69.3993] | positive | 是 |
| `context64_minus_context32` | `pose_avg_pred_count` | -15.5243 | [-94.8768, 79.8880] | negative | 是 |
| `context64_minus_context32` | `aggregate_avg_pred_count` | -15.5243 | [-95.0423, 78.0757] | negative | 是 |
| `context64_minus_context32` | `aggregate_predicted_glb_bytes` | -422177.7027 | [-3726679.4333, 4553724.4991] | negative | 是 |
| `context64_minus_context32` | `pose_download_utility_recall` | 0.0009 | [-0.0016, 0.0034] | positive | 是 |
| `context64_minus_context32` | `aggregate_download_utility_recall` | 0.0008 | [-0.0014, 0.0031] | positive | 是 |
| `context64_minus_context32` | `pose_glb_bytes_at_achieved_visual_utility` | -456579.5869 | [-2391889.3560, 2326257.9922] | negative | 是 |
| `context64_minus_context32` | `aggregate_glb_bytes_at_achieved_visual_utility` | -456579.5869 | [-2390918.8408, 2292430.4837] | negative | 是 |
| `context64_minus_context32` | `aggregate_glb_byte_reduction` | 0.0010 | [-0.0154, 0.0119] | positive | 是 |
| `context64_minus_context32` | `pose_image_per` | -0.0014 | [-0.0068, 0.0026] | negative | 是 |
| `context64_minus_context32` | `aggregate_image_per` | -0.0015 | [-0.0069, 0.0023] | negative | 是 |
| `context64_minus_context32` | `pose_image_miss_pixel_rate` | -0.0007 | [-0.0081, 0.0044] | negative | 是 |
| `context64_minus_context32` | `aggregate_image_miss_pixel_rate` | -0.0008 | [-0.0084, 0.0045] | negative | 是 |
| `context64_minus_context32` | `pose_image_wrong_id_pixel_rate` | -0.0007 | [-0.0032, 0.0019] | negative | 是 |
| `context64_minus_context32` | `aggregate_image_extra_pixel_rate` | 0.0000 | [0.0000, 0.0000] | zero | 是 |
| `fourier117_minus_direct9` | `pose_precision` | 0.2164 | [0.1961, 0.2372] | positive | 否 |
| `fourier117_minus_direct9` | `aggregate_precision` | 0.0199 | [-0.0144, 0.0476] | positive | 是 |
| `fourier117_minus_direct9` | `pose_recall` | 0.0539 | [0.0348, 0.0758] | positive | 否 |
| `fourier117_minus_direct9` | `aggregate_recall` | 0.0944 | [0.0467, 0.1536] | positive | 否 |
| `fourier117_minus_direct9` | `pose_weighted_recall` | 0.0059 | [0.0039, 0.0081] | positive | 否 |
| `fourier117_minus_direct9` | `aggregate_weighted_recall` | 0.0055 | [0.0035, 0.0078] | positive | 否 |
| `fourier117_minus_direct9` | `pose_accuracy` | 0.0731 | [0.0620, 0.0847] | positive | 否 |
| `fourier117_minus_direct9` | `aggregate_accuracy` | 0.0035 | [-0.0004, 0.0072] | positive | 是 |
| `fourier117_minus_direct9` | `pose_balanced_accuracy` | 0.0555 | [0.0468, 0.0646] | positive | 否 |
| `fourier117_minus_direct9` | `aggregate_balanced_accuracy` | 0.0469 | [0.0255, 0.0739] | positive | 否 |
| `fourier117_minus_direct9` | `pose_f1` | 0.2116 | [0.1965, 0.2269] | positive | 否 |
| `fourier117_minus_direct9` | `aggregate_f1` | 0.0478 | [0.0341, 0.0659] | positive | 否 |
| `fourier117_minus_direct9` | `pose_useful_cull` | 0.0552 | [0.0420, 0.0681] | positive | 否 |
| `fourier117_minus_direct9` | `aggregate_useful_cull` | -0.0006 | [-0.0072, 0.0048] | negative | 是 |
| `fourier117_minus_direct9` | `pose_bad_cull` | -0.0179 | [-0.0249, -0.0123] | negative | 否 |
| `fourier117_minus_direct9` | `aggregate_bad_cull` | -0.0042 | [-0.0067, -0.0020] | negative | 否 |
| `fourier117_minus_direct9` | `pose_avg_fp_count` | 3.3052 | [-23.7090, 35.1237] | positive | 是 |
| `fourier117_minus_direct9` | `pose_avg_fn_count` | -21.6322 | [-34.7419, -10.6822] | negative | 否 |
| `fourier117_minus_direct9` | `pose_avg_tn_count` | -3.3052 | [-35.7858, 24.2192] | negative | 是 |
| `fourier117_minus_direct9` | `aggregate_avg_fp_count` | 3.3052 | [-24.9313, 35.2697] | positive | 是 |
| `fourier117_minus_direct9` | `aggregate_avg_fn_count` | -21.6322 | [-35.0728, -10.7354] | negative | 否 |
| `fourier117_minus_direct9` | `aggregate_avg_tn_count` | -3.3052 | [-36.3824, 24.1334] | negative | 是 |
| `fourier117_minus_direct9` | `pose_avg_pred_count` | 24.9374 | [-13.0112, 69.3768] | positive | 是 |
| `fourier117_minus_direct9` | `aggregate_avg_pred_count` | 24.9374 | [-12.4735, 69.1058] | positive | 是 |
| `fourier117_minus_direct9` | `aggregate_predicted_glb_bytes` | 15575613.6839 | [10265438.7493, 21345805.5648] | positive | 否 |
| `fourier117_minus_direct9` | `pose_download_utility_recall` | 0.0060 | [0.0041, 0.0082] | positive | 否 |
| `fourier117_minus_direct9` | `aggregate_download_utility_recall` | 0.0056 | [0.0036, 0.0080] | positive | 否 |
| `fourier117_minus_direct9` | `pose_glb_bytes_at_achieved_visual_utility` | 6931904.9577 | [5058427.5729, 8973475.4216] | positive | 否 |
| `fourier117_minus_direct9` | `aggregate_glb_bytes_at_achieved_visual_utility` | 6931904.9577 | [5042705.4856, 9035256.7847] | positive | 否 |
| `fourier117_minus_direct9` | `aggregate_glb_byte_reduction` | 0.0812 | [0.0538, 0.1067] | positive | 否 |
| `fourier117_minus_direct9` | `pose_image_per` | -0.0073 | [-0.0110, -0.0042] | negative | 否 |
| `fourier117_minus_direct9` | `aggregate_image_per` | -0.0072 | [-0.0114, -0.0039] | negative | 否 |
| `fourier117_minus_direct9` | `pose_image_miss_pixel_rate` | -0.0035 | [-0.0070, -0.0009] | negative | 否 |
| `fourier117_minus_direct9` | `aggregate_image_miss_pixel_rate` | -0.0037 | [-0.0076, -0.0009] | negative | 否 |
| `fourier117_minus_direct9` | `pose_image_wrong_id_pixel_rate` | -0.0038 | [-0.0060, -0.0021] | negative | 否 |
| `fourier117_minus_direct9` | `aggregate_image_extra_pixel_rate` | 0.0000 | [0.0000, 0.0000] | zero | 是 |
| `unconstrained28_minus_monotone` | `pose_precision` | -0.0013 | [-0.0090, 0.0080] | negative | 是 |
| `unconstrained28_minus_monotone` | `aggregate_precision` | 0.0129 | [-0.0127, 0.0352] | positive | 是 |
| `unconstrained28_minus_monotone` | `pose_recall` | -0.0012 | [-0.0099, 0.0059] | negative | 是 |
| `unconstrained28_minus_monotone` | `aggregate_recall` | -0.0281 | [-0.0733, 0.0133] | negative | 是 |
| `unconstrained28_minus_monotone` | `pose_weighted_recall` | 0.0006 | [-0.0011, 0.0027] | positive | 是 |
| `unconstrained28_minus_monotone` | `aggregate_weighted_recall` | 0.0004 | [-0.0010, 0.0023] | positive | 是 |
| `unconstrained28_minus_monotone` | `pose_accuracy` | 0.0003 | [-0.0029, 0.0034] | positive | 是 |
| `unconstrained28_minus_monotone` | `aggregate_accuracy` | 0.0009 | [-0.0022, 0.0039] | positive | 是 |
| `unconstrained28_minus_monotone` | `pose_balanced_accuracy` | 0.0008 | [-0.0021, 0.0049] | positive | 是 |
| `unconstrained28_minus_monotone` | `aggregate_balanced_accuracy` | -0.0129 | [-0.0333, 0.0050] | negative | 是 |
| `unconstrained28_minus_monotone` | `pose_f1` | -0.0032 | [-0.0104, 0.0023] | negative | 是 |
| `unconstrained28_minus_monotone` | `aggregate_f1` | -0.0054 | [-0.0246, 0.0093] | negative | 是 |
| `unconstrained28_minus_monotone` | `pose_useful_cull` | 0.0018 | [-0.0031, 0.0055] | positive | 是 |
| `unconstrained28_minus_monotone` | `aggregate_useful_cull` | 0.0022 | [-0.0028, 0.0060] | positive | 是 |
| `unconstrained28_minus_monotone` | `pose_bad_cull` | 0.0015 | [-0.0002, 0.0037] | positive | 是 |
| `unconstrained28_minus_monotone` | `aggregate_bad_cull` | 0.0012 | [-0.0006, 0.0034] | positive | 是 |
| `unconstrained28_minus_monotone` | `pose_avg_fp_count` | -11.3443 | [-31.0564, 14.4734] | negative | 是 |
| `unconstrained28_minus_monotone` | `pose_avg_fn_count` | 6.4366 | [-3.0128, 17.6293] | positive | 是 |
| `unconstrained28_minus_monotone` | `pose_avg_tn_count` | 11.3443 | [-14.7717, 30.8795] | positive | 是 |
| `unconstrained28_minus_monotone` | `aggregate_avg_fp_count` | -11.3443 | [-30.8702, 13.9950] | negative | 是 |
| `unconstrained28_minus_monotone` | `aggregate_avg_fn_count` | 6.4366 | [-3.0000, 17.5244] | positive | 是 |
| `unconstrained28_minus_monotone` | `aggregate_avg_tn_count` | 11.3443 | [-14.3662, 30.7029] | positive | 是 |
| `unconstrained28_minus_monotone` | `pose_avg_pred_count` | -17.7809 | [-46.3209, 17.6687] | negative | 是 |
| `unconstrained28_minus_monotone` | `aggregate_avg_pred_count` | -17.7809 | [-46.1309, 17.4842] | negative | 是 |
| `unconstrained28_minus_monotone` | `aggregate_predicted_glb_bytes` | -903673.4961 | [-2456936.2254, 973395.8316] | negative | 是 |
| `unconstrained28_minus_monotone` | `pose_download_utility_recall` | 0.0006 | [-0.0011, 0.0027] | positive | 是 |
| `unconstrained28_minus_monotone` | `aggregate_download_utility_recall` | 0.0005 | [-0.0009, 0.0023] | positive | 是 |
| `unconstrained28_minus_monotone` | `pose_glb_bytes_at_achieved_visual_utility` | -472964.5008 | [-1456338.5734, 836959.4333] | negative | 是 |
| `unconstrained28_minus_monotone` | `aggregate_glb_bytes_at_achieved_visual_utility` | -472964.5008 | [-1466551.9070, 805009.7224] | negative | 是 |
| `unconstrained28_minus_monotone` | `aggregate_glb_byte_reduction` | 0.0040 | [-0.0008, 0.0083] | positive | 是 |
| `unconstrained28_minus_monotone` | `pose_image_per` | -0.0015 | [-0.0056, 0.0016] | negative | 是 |
| `unconstrained28_minus_monotone` | `aggregate_image_per` | -0.0015 | [-0.0057, 0.0014] | negative | 是 |
| `unconstrained28_minus_monotone` | `pose_image_miss_pixel_rate` | -0.0014 | [-0.0052, 0.0017] | negative | 是 |
| `unconstrained28_minus_monotone` | `aggregate_image_miss_pixel_rate` | -0.0014 | [-0.0055, 0.0017] | negative | 是 |
| `unconstrained28_minus_monotone` | `pose_image_wrong_id_pixel_rate` | -0.0001 | [-0.0008, 0.0005] | negative | 是 |
| `unconstrained28_minus_monotone` | `aggregate_image_extra_pixel_rate` | 0.0000 | [0.0000, 0.0000] | zero | 是 |
| `aabb_minus_triangle_relation` | `pose_precision` | -0.0003 | [-0.0122, 0.0110] | negative | 是 |
| `aabb_minus_triangle_relation` | `aggregate_precision` | 0.0584 | [0.0175, 0.1173] | positive | 否 |
| `aabb_minus_triangle_relation` | `pose_recall` | -0.0127 | [-0.0356, 0.0041] | negative | 是 |
| `aabb_minus_triangle_relation` | `aggregate_recall` | -0.0954 | [-0.1786, -0.0293] | negative | 否 |
| `aabb_minus_triangle_relation` | `pose_weighted_recall` | 0.0007 | [-0.0012, 0.0029] | positive | 是 |
| `aabb_minus_triangle_relation` | `aggregate_weighted_recall` | 0.0003 | [-0.0015, 0.0025] | positive | 是 |
| `aabb_minus_triangle_relation` | `pose_accuracy` | 0.0035 | [-0.0002, 0.0085] | positive | 是 |
| `aabb_minus_triangle_relation` | `aggregate_accuracy` | 0.0063 | [0.0014, 0.0135] | positive | 否 |
| `aabb_minus_triangle_relation` | `pose_balanced_accuracy` | 0.0008 | [-0.0038, 0.0046] | positive | 是 |
| `aabb_minus_triangle_relation` | `aggregate_balanced_accuracy` | -0.0422 | [-0.0799, -0.0121] | negative | 否 |
| `aabb_minus_triangle_relation` | `pose_f1` | -0.0117 | [-0.0206, -0.0030] | negative | 否 |
| `aabb_minus_triangle_relation` | `aggregate_f1` | -0.0008 | [-0.0276, 0.0219] | negative | 是 |
| `aabb_minus_triangle_relation` | `pose_useful_cull` | 0.0087 | [0.0014, 0.0196] | positive | 否 |
| `aabb_minus_triangle_relation` | `aggregate_useful_cull` | 0.0105 | [0.0037, 0.0213] | positive | 否 |
| `aabb_minus_triangle_relation` | `pose_bad_cull` | 0.0052 | [0.0006, 0.0120] | positive | 否 |
| `aabb_minus_triangle_relation` | `aggregate_bad_cull` | 0.0042 | [0.0013, 0.0084] | positive | 否 |
| `aabb_minus_triangle_relation` | `pose_avg_fp_count` | -54.4100 | [-111.3226, -18.9667] | negative | 否 |
| `aabb_minus_triangle_relation` | `pose_avg_fn_count` | 21.8560 | [6.6651, 43.2114] | positive | 否 |
| `aabb_minus_triangle_relation` | `pose_avg_tn_count` | 54.4100 | [18.9401, 112.6908] | positive | 否 |
| `aabb_minus_triangle_relation` | `aggregate_avg_fp_count` | -54.4100 | [-108.1729, -19.0046] | negative | 否 |
| `aabb_minus_triangle_relation` | `aggregate_avg_fn_count` | 21.8560 | [6.5005, 43.3041] | positive | 否 |
| `aabb_minus_triangle_relation` | `aggregate_avg_tn_count` | 54.4100 | [18.8805, 108.0646] | positive | 否 |
| `aabb_minus_triangle_relation` | `pose_avg_pred_count` | -76.2660 | [-151.1422, -27.6208] | negative | 否 |
| `aabb_minus_triangle_relation` | `aggregate_avg_pred_count` | -76.2660 | [-151.8587, -27.6133] | negative | 否 |
| `aabb_minus_triangle_relation` | `aggregate_predicted_glb_bytes` | -4168115.2175 | [-8883076.0474, -994742.0726] | negative | 否 |
| `aabb_minus_triangle_relation` | `pose_download_utility_recall` | 0.0007 | [-0.0012, 0.0029] | positive | 是 |
| `aabb_minus_triangle_relation` | `aggregate_download_utility_recall` | 0.0004 | [-0.0015, 0.0025] | positive | 是 |
| `aabb_minus_triangle_relation` | `pose_glb_bytes_at_achieved_visual_utility` | -2512329.5837 | [-4930207.6649, -941594.8225] | negative | 否 |
| `aabb_minus_triangle_relation` | `aggregate_glb_bytes_at_achieved_visual_utility` | -2512329.5837 | [-4883480.1734, -951172.6025] | negative | 否 |
| `aabb_minus_triangle_relation` | `aggregate_glb_byte_reduction` | 0.0130 | [0.0021, 0.0292] | positive | 否 |
| `aabb_minus_triangle_relation` | `pose_image_per` | -0.0033 | [-0.0067, -0.0005] | negative | 否 |
| `aabb_minus_triangle_relation` | `aggregate_image_per` | -0.0033 | [-0.0070, -0.0003] | negative | 否 |
| `aabb_minus_triangle_relation` | `pose_image_miss_pixel_rate` | -0.0047 | [-0.0087, -0.0013] | negative | 否 |
| `aabb_minus_triangle_relation` | `aggregate_image_miss_pixel_rate` | -0.0047 | [-0.0092, -0.0011] | negative | 否 |
| `aabb_minus_triangle_relation` | `pose_image_wrong_id_pixel_rate` | 0.0014 | [-0.0004, 0.0036] | positive | 是 |
| `aabb_minus_triangle_relation` | `aggregate_image_extra_pixel_rate` | 0.0000 | [0.0000, 0.0000] | zero | 是 |

## useful cull 的解释边界

例如某个 pose 有 1000 个候选实例、100 个真实可见实例，模型只预测 10 个且其中 10 个恰好正确，则 TP=10、FP=0、FN=90、TN=900。此时 useful cull=0.90，看起来很高，但 recall=0.10、bad cull=0.09；若漏掉的实例具有较大 visible weight，weighted recall 也会明显下降。这种结果是错误剔除风险，不能解释为模型性能更好。

方向代理、上下文和损失只有在 weighted recall 安全性不恶化，并且 precision、balanced accuracy、F1、有效剔除、图像质量或资源成本至少一项的配对置信区间稳定改善时，才可写成预测或系统贡献。
