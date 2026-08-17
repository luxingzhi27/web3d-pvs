# 视线关系场、生存场与 OWRB 矩阵评价

本报告只汇总 calibration 冻结阈值后的 validation 结果。所有成员使用同一 pose 顺序、后退相机候选集合和实例 GT；paired bootstrap 先按 seed 重采样，再在每个 seed 内按 pose 重采样。

- 变体：8；seed：3；pose：213。
- bootstrap：10000 次；test split 读取：否。
- weighted recall 是画面安全主指标；useful cull 必须与 bad cull、recall 和 weighted recall 一起解释。
- Color-ID 图像评价：已完成硬件 GPU 正式评价。
- WebGPU 数值 parity：已通过软件后端数值校验，最大绝对误差 0.00000286，最大相对误差 0.00000679；当前适配器未通过 NVIDIA WebGPU 硬件门，不能据此报告硬件 WebGPU 延迟。

## 画面安全指标

| 变体 | seed | 阈值 | pose recall | aggregate recall | pose weighted recall | aggregate weighted recall | pose useful cull | aggregate useful cull | pose bad cull | aggregate bad cull | safety-adjusted useful cull |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `context_off_survival_off_rvl` | 20260801 | 0.480000 | 0.8128 | 0.6654 | 0.9890 | 0.9897 | 0.5846 | 0.9024 | 0.0253 | 0.0148 | 0.4997 |
| `context_off_survival_off_rvl` | 20260802 | 0.400000 | 0.8093 | 0.6391 | 0.9892 | 0.9899 | 0.5881 | 0.9120 | 0.0258 | 0.0160 | 0.5006 |
| `context_off_survival_off_rvl` | 20260803 | 0.540000 | 0.8042 | 0.6529 | 0.9879 | 0.9887 | 0.5854 | 0.9069 | 0.0262 | 0.0154 | 0.4945 |
| `context_on_survival_off_rvl` | 20260801 | 0.480000 | 0.8110 | 0.6586 | 0.9889 | 0.9898 | 0.5863 | 0.9047 | 0.0260 | 0.0151 | 0.4999 |
| `context_on_survival_off_rvl` | 20260802 | 0.100000 | 0.8490 | 0.7577 | 0.9915 | 0.9925 | 0.5622 | 0.8862 | 0.0188 | 0.0107 | 0.5024 |
| `context_on_survival_off_rvl` | 20260803 | 0.320000 | 0.8415 | 0.7395 | 0.9917 | 0.9925 | 0.5714 | 0.8926 | 0.0195 | 0.0115 | 0.5062 |
| `context_off_survival_on_rvl` | 20260801 | 0.680000 | 0.8134 | 0.7450 | 0.9893 | 0.9903 | 0.6609 | 0.9182 | 0.0253 | 0.0113 | 0.5655 |
| `context_off_survival_on_rvl` | 20260802 | 0.580000 | 0.8102 | 0.7325 | 0.9893 | 0.9903 | 0.6634 | 0.9201 | 0.0283 | 0.0118 | 0.5654 |
| `context_off_survival_on_rvl` | 20260803 | 0.500000 | 0.7908 | 0.5600 | 0.9914 | 0.9916 | 0.6806 | 0.9383 | 0.0390 | 0.0195 | 0.5666 |
| `context_on_survival_on_rvl` | 20260801 | 0.540000 | 0.8176 | 0.7310 | 0.9913 | 0.9922 | 0.6615 | 0.9193 | 0.0289 | 0.0119 | 0.5693 |
| `context_on_survival_on_rvl` | 20260802 | 0.540000 | 0.8169 | 0.7163 | 0.9917 | 0.9926 | 0.6649 | 0.9226 | 0.0297 | 0.0126 | 0.5718 |
| `context_on_survival_on_rvl` | 20260803 | 0.800000 | 0.7867 | 0.6629 | 0.9881 | 0.9889 | 0.6754 | 0.9305 | 0.0334 | 0.0149 | 0.5582 |
| `context_off_survival_off_safety` | 20260801 | 0.075000 | 0.8679 | 0.6533 | 0.9941 | 0.9939 | 0.5533 | 0.9048 | 0.0194 | 0.0153 | 0.5054 |
| `context_off_survival_off_safety` | 20260802 | 0.200000 | 0.8502 | 0.5680 | 0.9935 | 0.9929 | 0.5546 | 0.9156 | 0.0239 | 0.0191 | 0.4963 |
| `context_off_survival_off_safety` | 20260803 | 0.100000 | 0.8905 | 0.6991 | 0.9961 | 0.9956 | 0.5249 | 0.8914 | 0.0170 | 0.0133 | 0.4920 |
| `context_on_survival_off_safety` | 20260801 | 0.050000 | 0.8850 | 0.6995 | 0.9954 | 0.9951 | 0.5341 | 0.8843 | 0.0176 | 0.0133 | 0.4976 |
| `context_on_survival_off_safety` | 20260802 | 0.075000 | 0.8902 | 0.7047 | 0.9961 | 0.9958 | 0.5226 | 0.8835 | 0.0170 | 0.0131 | 0.4897 |
| `context_on_survival_off_safety` | 20260803 | 0.075000 | 0.8945 | 0.6993 | 0.9964 | 0.9960 | 0.5180 | 0.8821 | 0.0163 | 0.0133 | 0.4878 |
| `context_off_survival_on_safety` | 20260801 | 0.000487 | 0.9497 | 0.9341 | 0.9990 | 0.9991 | 0.3765 | 0.5181 | 0.0082 | 0.0029 | 0.3764 |
| `context_off_survival_on_safety` | 20260802 | 0.020000 | 0.9111 | 0.8331 | 0.9972 | 0.9972 | 0.4596 | 0.5590 | 0.0154 | 0.0074 | 0.4408 |
| `context_off_survival_on_safety` | 20260803 | 0.000237 | 0.9363 | 0.9761 | 0.9974 | 0.9976 | 0.4272 | 0.6274 | 0.0097 | 0.0011 | 0.4210 |
| `context_on_survival_on_safety` | 20260801 | 0.001000 | 0.9402 | 0.9099 | 0.9986 | 0.9987 | 0.4114 | 0.5694 | 0.0100 | 0.0040 | 0.4072 |
| `context_on_survival_on_safety` | 20260802 | 0.010000 | 0.9278 | 0.8660 | 0.9979 | 0.9980 | 0.4278 | 0.5434 | 0.0115 | 0.0059 | 0.4178 |
| `context_on_survival_on_safety` | 20260803 | 0.000487 | 0.9300 | 0.9740 | 0.9970 | 0.9971 | 0.4736 | 0.6828 | 0.0108 | 0.0012 | 0.4637 |

## 分类诊断指标

| 变体 | seed | pose precision | aggregate precision | pose F1 | aggregate F1 | pose Jaccard | aggregate Jaccard | pose accuracy | aggregate accuracy | pose balanced acc. | aggregate balanced acc. | pose specificity | aggregate specificity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `context_off_survival_off_rvl` | 20260801 | 0.3466 | 0.3559 | 0.4151 | 0.4638 | 0.2831 | 0.3019 | 0.7367 | 0.9319 | 0.7259 | 0.8048 | 0.6390 | 0.9442 |
| `context_off_survival_off_rvl` | 20260802 | 0.3504 | 0.3929 | 0.4150 | 0.4866 | 0.2830 | 0.3216 | 0.7397 | 0.9403 | 0.7267 | 0.7967 | 0.6441 | 0.9543 |
| `context_off_survival_off_rvl` | 20260803 | 0.3489 | 0.3720 | 0.4172 | 0.4740 | 0.2844 | 0.3106 | 0.7366 | 0.9359 | 0.7213 | 0.8009 | 0.6384 | 0.9490 |
| `context_on_survival_off_rvl` | 20260801 | 0.3480 | 0.3634 | 0.4154 | 0.4684 | 0.2832 | 0.3058 | 0.7377 | 0.9338 | 0.7261 | 0.8026 | 0.6412 | 0.9466 |
| `context_on_survival_off_rvl` | 20260802 | 0.3252 | 0.3255 | 0.4090 | 0.4554 | 0.2782 | 0.2948 | 0.7208 | 0.9198 | 0.7274 | 0.8425 | 0.6059 | 0.9273 |
| `context_on_survival_off_rvl` | 20260803 | 0.3373 | 0.3414 | 0.4191 | 0.4671 | 0.2858 | 0.3048 | 0.7293 | 0.9253 | 0.7296 | 0.8367 | 0.6176 | 0.9339 |
| `context_off_survival_on_rvl` | 20260801 | 0.4305 | 0.4677 | 0.5052 | 0.5747 | 0.3609 | 0.4032 | 0.8131 | 0.9512 | 0.7776 | 0.8529 | 0.7419 | 0.9607 |
| `context_off_survival_on_rvl` | 20260802 | 0.4280 | 0.4766 | 0.4999 | 0.5775 | 0.3557 | 0.4060 | 0.8125 | 0.9525 | 0.7785 | 0.8476 | 0.7469 | 0.9627 |
| `context_off_survival_on_rvl` | 20260803 | 0.4368 | 0.5864 | 0.4864 | 0.5729 | 0.3454 | 0.4014 | 0.8190 | 0.9630 | 0.7839 | 0.7708 | 0.7771 | 0.9817 |
| `context_on_survival_on_rvl` | 20260801 | 0.4250 | 0.4703 | 0.4971 | 0.5724 | 0.3529 | 0.4009 | 0.8100 | 0.9516 | 0.7816 | 0.8464 | 0.7456 | 0.9619 |
| `context_on_survival_on_rvl` | 20260802 | 0.4239 | 0.4890 | 0.4947 | 0.5812 | 0.3505 | 0.4097 | 0.8126 | 0.9543 | 0.7841 | 0.8408 | 0.7514 | 0.9653 |
| `context_on_survival_on_rvl` | 20260803 | 0.4466 | 0.5376 | 0.5018 | 0.5937 | 0.3600 | 0.4222 | 0.8194 | 0.9598 | 0.7764 | 0.8182 | 0.7661 | 0.9736 |
| `context_off_survival_off_safety` | 20260801 | 0.2945 | 0.3623 | 0.3700 | 0.4661 | 0.2538 | 0.3038 | 0.7112 | 0.9337 | 0.7317 | 0.8000 | 0.5956 | 0.9467 |
| `context_off_survival_off_safety` | 20260802 | 0.3020 | 0.3851 | 0.3702 | 0.4590 | 0.2532 | 0.2979 | 0.7081 | 0.9407 | 0.7250 | 0.7630 | 0.5998 | 0.9580 |
| `context_off_survival_off_safety` | 20260803 | 0.2701 | 0.3248 | 0.3472 | 0.4435 | 0.2397 | 0.2849 | 0.6853 | 0.9223 | 0.7264 | 0.8159 | 0.5624 | 0.9327 |
| `context_on_survival_off_safety` | 20260801 | 0.2744 | 0.3023 | 0.3533 | 0.4221 | 0.2420 | 0.2675 | 0.6939 | 0.9152 | 0.7285 | 0.8124 | 0.5721 | 0.9252 |
| `context_on_survival_off_safety` | 20260802 | 0.2657 | 0.3018 | 0.3447 | 0.4226 | 0.2369 | 0.2679 | 0.6829 | 0.9147 | 0.7249 | 0.8146 | 0.5595 | 0.9245 |
| `context_on_survival_off_safety` | 20260803 | 0.2576 | 0.2960 | 0.3352 | 0.4159 | 0.2316 | 0.2626 | 0.6792 | 0.9131 | 0.7245 | 0.8111 | 0.5546 | 0.9230 |
| `context_off_survival_on_safety` | 20260801 | 0.2128 | 0.0863 | 0.2899 | 0.1580 | 0.2075 | 0.0858 | 0.5457 | 0.5594 | 0.6834 | 0.7381 | 0.4171 | 0.5420 |
| `context_off_survival_on_safety` | 20260802 | 0.2397 | 0.0850 | 0.3130 | 0.1543 | 0.2271 | 0.0836 | 0.6216 | 0.5959 | 0.7198 | 0.7090 | 0.5285 | 0.5849 |
| `context_off_survival_on_safety` | 20260803 | 0.2139 | 0.1163 | 0.2935 | 0.2079 | 0.2077 | 0.1160 | 0.5948 | 0.6707 | 0.7009 | 0.8163 | 0.4655 | 0.6565 |
| `context_on_survival_on_safety` | 20260801 | 0.2189 | 0.0944 | 0.2959 | 0.1711 | 0.2119 | 0.0935 | 0.5788 | 0.6097 | 0.6991 | 0.7528 | 0.4579 | 0.5958 |
| `context_on_survival_on_safety` | 20260802 | 0.2285 | 0.0851 | 0.3042 | 0.1549 | 0.2198 | 0.0840 | 0.5937 | 0.5818 | 0.7069 | 0.7173 | 0.4860 | 0.5686 |
| `context_on_survival_on_safety` | 20260803 | 0.2271 | 0.1364 | 0.3098 | 0.2393 | 0.2181 | 0.1359 | 0.6403 | 0.7259 | 0.7228 | 0.8442 | 0.5156 | 0.7144 |

## 剔除、资源与运行成本

| 变体 | seed | 平均预测数 | pose 平均 FP | pose 平均 FN | pose 平均 TN | aggregate 平均 FP | aggregate 平均 FN | aggregate 平均 TN | 预测/候选 | 预测/GT | 候选 GLB 数 | 预测 GLB 数 | 候选 GLB 字节 | 预测 GLB 字节 | GLB 数量削减 | GLB 字节削减 | 下载效用召回 | 达到同等视觉效用所需字节 | 前向 p95 ms | 特征表字节 | 固定表维度 | 头输入维度 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `context_off_survival_off_rvl` | 20260801 | 428.2300 | 275.8075 | 76.6338 | 4669.5305 | 275.8075 | 76.6338 | 4669.5305 | 0.0828 | 1.8695 | 963.7746 | 115.6854 | 183619879.7371 | 42793141.7465 | 0.5897 | 0.4650 | 0.9899 | 10885013.8404 | 1.600 | 5875272 | 156 | 177 |
| `context_off_survival_off_rvl` | 20260802 | 372.5869 | 226.1972 | 82.6667 | 4719.1408 | 226.1972 | 82.6667 | 4719.1408 | 0.0720 | 1.6266 | 963.7746 | 108.5634 | 183619879.7371 | 42266974.3850 | 0.5891 | 0.4658 | 0.9902 | 10690916.2066 | 1.597 | 5875272 | 156 | 177 |
| `context_off_survival_off_rvl` | 20260803 | 401.9718 | 252.4225 | 79.5070 | 4692.9155 | 252.4225 | 79.5070 | 4692.9155 | 0.0777 | 1.7549 | 963.7746 | 114.9014 | 183619879.7371 | 42474376.3380 | 0.5878 | 0.4669 | 0.9889 | 11195746.4413 | 1.620 | 5875272 | 156 | 177 |
| `context_on_survival_off_rvl` | 20260801 | 415.0939 | 264.2347 | 78.1972 | 4681.1033 | 264.2347 | 78.1972 | 4681.1033 | 0.0802 | 1.8122 | 963.7746 | 116.3756 | 183619879.7371 | 42936101.7465 | 0.5894 | 0.4653 | 0.9901 | 11026287.2676 | 2.112 | 5875272 | 156 | 177 |
| `context_on_survival_off_rvl` | 20260802 | 533.1362 | 359.5915 | 55.5117 | 4585.7465 | 359.5915 | 55.5117 | 4585.7465 | 0.1030 | 2.3275 | 963.7746 | 147.9343 | 183619879.7371 | 50664910.2535 | 0.5551 | 0.4430 | 0.9926 | 13953162.0657 | 1.584 | 5875272 | 156 | 177 |
| `context_on_survival_off_rvl` | 20260803 | 496.1362 | 326.7512 | 59.6714 | 4618.5869 | 326.7512 | 59.6714 | 4618.5869 | 0.0959 | 2.1660 | 963.7746 | 142.7653 | 183619879.7371 | 47575291.0610 | 0.5636 | 0.4529 | 0.9927 | 13377264.3192 | 1.591 | 5875272 | 156 | 177 |
| `context_off_survival_on_rvl` | 20260801 | 364.8685 | 194.2160 | 58.4038 | 4751.1221 | 194.2160 | 58.4038 | 4751.1221 | 0.0705 | 1.5929 | 963.7746 | 112.8310 | 183619879.7371 | 41698642.6103 | 0.6510 | 0.5437 | 0.9904 | 15288525.2958 | 4.043 | 5875272 | 156 | 177 |
| `context_off_survival_on_rvl` | 20260802 | 352.0610 | 184.2676 | 61.2629 | 4761.0704 | 184.2676 | 61.2629 | 4761.0704 | 0.0680 | 1.5370 | 963.7746 | 107.9765 | 183619879.7371 | 41088032.3380 | 0.6581 | 0.5472 | 0.9904 | 14297821.3521 | 2.617 | 5875272 | 156 | 177 |
| `context_off_survival_on_rvl` | 20260803 | 218.7230 | 90.4601 | 100.7934 | 4854.8779 | 90.4601 | 100.7934 | 4854.8779 | 0.0423 | 0.9549 | 963.7746 | 76.1174 | 183619879.7371 | 34903945.4460 | 0.6908 | 0.5694 | 0.9919 | 10713356.1315 | 2.137 | 5875272 | 156 | 177 |
| `context_on_survival_on_rvl` | 20260801 | 356.0094 | 188.5681 | 61.6150 | 4756.7700 | 188.5681 | 61.6150 | 4756.7700 | 0.0688 | 1.5542 | 963.7746 | 108.6948 | 183619879.7371 | 41254796.4883 | 0.6577 | 0.5476 | 0.9923 | 14369909.2019 | 2.148 | 5875272 | 156 | 177 |
| `context_on_survival_on_rvl` | 20260802 | 335.5540 | 171.4742 | 64.9765 | 4773.8638 | 171.4742 | 64.9765 | 4773.8638 | 0.0648 | 1.4649 | 963.7746 | 105.5164 | 183619879.7371 | 41697579.9812 | 0.6615 | 0.5464 | 0.9927 | 14240536.8826 | 5.534 | 5875272 | 156 | 177 |
| `context_on_survival_on_rvl` | 20260803 | 282.4225 | 130.5869 | 77.2207 | 4814.7512 | 130.5869 | 77.2207 | 4814.7512 | 0.0546 | 1.2330 | 963.7746 | 87.6197 | 183619879.7371 | 36409455.5117 | 0.6795 | 0.5621 | 0.9891 | 12251980.0188 | 3.993 | 5875272 | 156 | 177 |
| `context_off_survival_off_safety` | 20260801 | 413.0798 | 263.4413 | 79.4178 | 4681.8967 | 263.4413 | 79.4178 | 4681.8967 | 0.0798 | 1.8034 | 963.7746 | 163.5446 | 183619879.7371 | 68322845.0329 | 0.5225 | 0.3833 | 0.9942 | 12813620.6948 | 1.591 | 5875272 | 156 | 177 |
| `context_off_survival_off_safety` | 20260802 | 337.7793 | 207.6854 | 98.9624 | 4737.6526 | 207.6854 | 98.9624 | 4737.6526 | 0.0653 | 1.4747 | 963.7746 | 139.7465 | 183619879.7371 | 64315323.8873 | 0.5297 | 0.3843 | 0.9935 | 11881888.3944 | 1.577 | 5875272 | 156 | 177 |
| `context_off_survival_off_safety` | 20260803 | 493.0563 | 332.9296 | 68.9296 | 4612.4085 | 332.9296 | 68.9296 | 4612.4085 | 0.0953 | 2.1526 | 963.7746 | 215.4225 | 183619879.7371 | 88186226.2347 | 0.4729 | 0.3160 | 0.9959 | 14688512.1502 | 3.323 | 5875272 | 156 | 177 |
| `context_on_survival_off_safety` | 20260801 | 530.0282 | 369.8075 | 68.8357 | 4575.5305 | 369.8075 | 68.8357 | 4575.5305 | 0.1024 | 2.3140 | 963.7746 | 192.7277 | 183619879.7371 | 79882091.8122 | 0.4986 | 0.3458 | 0.9955 | 14059887.7183 | 1.831 | 5875272 | 156 | 177 |
| `context_on_survival_off_safety` | 20260802 | 534.9343 | 373.5070 | 67.6291 | 4571.8310 | 373.5070 | 67.6291 | 4571.8310 | 0.1034 | 2.3354 | 963.7746 | 205.9577 | 183619879.7371 | 88722958.0845 | 0.4791 | 0.3146 | 0.9962 | 14966187.5869 | 1.636 | 5875272 | 156 | 177 |
| `context_on_survival_off_safety` | 20260803 | 541.1315 | 380.9577 | 68.8826 | 4564.3803 | 380.9577 | 68.8826 | 4564.3803 | 0.1046 | 2.3624 | 963.7746 | 250.2394 | 183619879.7371 | 95092490.1033 | 0.4540 | 0.2972 | 0.9965 | 15640340.6948 | 1.595 | 5875272 | 156 | 177 |
| `context_off_survival_on_safety` | 20260801 | 2478.7042 | 2264.7324 | 15.0845 | 2680.6056 | 2264.7324 | 15.0845 | 2680.6056 | 0.4790 | 10.8214 | 963.7746 | 447.4272 | 183619879.7371 | 151266971.5305 | 0.3737 | 0.1588 | 0.9991 | 25790219.7746 | 2.135 | 5875272 | 156 | 177 |
| `context_off_survival_on_safety` | 20260802 | 2243.6009 | 2052.7840 | 38.2394 | 2892.5540 | 2052.7840 | 38.2394 | 2892.5540 | 0.4336 | 9.7950 | 963.7746 | 218.0563 | 183619879.7371 | 97017465.8216 | 0.5495 | 0.3517 | 0.9973 | 16160948.0188 | 2.178 | 5875272 | 156 | 177 |
| `context_off_survival_on_safety` | 20260803 | 1922.3005 | 1698.7136 | 5.4695 | 3246.6244 | 1698.7136 | 5.4695 | 3246.6244 | 0.3715 | 8.3923 | 963.7746 | 433.2347 | 183619879.7371 | 126104929.2770 | 0.3876 | 0.2072 | 0.9976 | 28929326.1033 | 2.124 | 5875272 | 156 | 177 |
| `context_on_survival_on_safety` | 20260801 | 2207.5446 | 1999.1268 | 20.6385 | 2946.2113 | 1999.1268 | 20.6385 | 2946.2113 | 0.4266 | 9.6376 | 963.7746 | 390.4789 | 183619879.7371 | 141858200.5822 | 0.4127 | 0.1922 | 0.9988 | 23253369.5775 | 2.124 | 5875272 | 156 | 177 |
| `context_on_survival_on_safety` | 20260802 | 2331.8169 | 2133.4648 | 30.7042 | 2811.8732 | 2133.4648 | 30.7042 | 2811.8732 | 0.4506 | 10.1801 | 963.7746 | 267.7324 | 183619879.7371 | 110947729.3333 | 0.4990 | 0.2880 | 0.9980 | 18578063.8122 | 2.138 | 5875272 | 156 | 177 |
| `context_on_survival_on_safety` | 20260803 | 1635.6009 | 1412.5070 | 5.9624 | 3532.8310 | 1412.5070 | 5.9624 | 3532.8310 | 0.3161 | 7.1406 | 963.7746 | 370.9624 | 183619879.7371 | 110099908.5822 | 0.4401 | 0.2822 | 0.9971 | 28854746.4413 | 2.127 | 5875272 | 156 | 177 |

## 图像质量

| 变体 | seed | PER | mean miss-pixel | p95 miss-pixel | wrong-ID pixel | extra-pixel |
|---|---:|---:|---:|---:|---:|---:|
| `context_off_survival_off_rvl` | 20260801 | 0.0126 | 0.0049 | 0.0285 | 0.0084 | 0.0000 |
| `context_off_survival_off_rvl` | 20260802 | 0.0123 | 0.0048 | 0.0279 | 0.0083 | 0.0000 |
| `context_off_survival_off_rvl` | 20260803 | 0.0147 | 0.0081 | 0.0443 | 0.0073 | 0.0000 |
| `context_on_survival_off_rvl` | 20260801 | 0.0123 | 0.0053 | 0.0302 | 0.0079 | 0.0000 |
| `context_on_survival_off_rvl` | 20260802 | 0.0103 | 0.0054 | 0.0350 | 0.0056 | 0.0000 |
| `context_on_survival_off_rvl` | 20260803 | 0.0098 | 0.0043 | 0.0212 | 0.0060 | 0.0000 |
| `context_off_survival_on_rvl` | 20260801 | 0.0153 | 0.0124 | 0.0677 | 0.0034 | 0.0000 |
| `context_off_survival_on_rvl` | 20260802 | 0.0144 | 0.0113 | 0.0667 | 0.0038 | 0.0000 |
| `context_off_survival_on_rvl` | 20260803 | 0.0060 | 0.0007 | 0.0043 | 0.0056 | 0.0000 |
| `context_on_survival_on_rvl` | 20260801 | 0.0104 | 0.0045 | 0.0186 | 0.0063 | 0.0000 |
| `context_on_survival_on_rvl` | 20260802 | 0.0093 | 0.0041 | 0.0108 | 0.0055 | 0.0000 |
| `context_on_survival_on_rvl` | 20260803 | 0.0145 | 0.0097 | 0.0428 | 0.0052 | 0.0000 |
| `context_off_survival_off_safety` | 20260801 | 0.0046 | 0.0017 | 0.0033 | 0.0029 | 0.0000 |
| `context_off_survival_off_safety` | 20260802 | 0.0052 | 0.0010 | 0.0022 | 0.0040 | 0.0000 |
| `context_off_survival_off_safety` | 20260803 | 0.0029 | 0.0005 | 0.0006 | 0.0021 | 0.0000 |
| `context_on_survival_off_safety` | 20260801 | 0.0033 | 0.0007 | 0.0010 | 0.0024 | 0.0000 |
| `context_on_survival_off_safety` | 20260802 | 0.0029 | 0.0005 | 0.0006 | 0.0021 | 0.0000 |
| `context_on_survival_off_safety` | 20260803 | 0.0026 | 0.0005 | 0.0006 | 0.0019 | 0.0000 |
| `context_off_survival_on_safety` | 20260801 | 0.0005 | 0.0000 | 0.0000 | 0.0004 | 0.0000 |
| `context_off_survival_on_safety` | 20260802 | 0.0018 | 0.0000 | 0.0001 | 0.0016 | 0.0000 |
| `context_off_survival_on_safety` | 20260803 | 0.0050 | 0.0015 | 0.0093 | 0.0032 | 0.0000 |
| `context_on_survival_on_safety` | 20260801 | 0.0007 | 0.0000 | 0.0000 | 0.0006 | 0.0000 |
| `context_on_survival_on_safety` | 20260802 | 0.0013 | 0.0000 | 0.0000 | 0.0011 | 0.0000 |
| `context_on_survival_on_safety` | 20260803 | 0.0058 | 0.0020 | 0.0155 | 0.0035 | 0.0000 |

## 配对差值和因子效应

差值定义为左侧减右侧；派生因子效应使用完整 2×2×2 对比并保持同一 seed/pose 重采样。区间跨零时不宣称稳定改善。

| 比较 | 指标 | 差值 | 95% 区间 | 方向 | 跨零 |
|---|---|---:|---|---|---|
| `context_on_minus_off_rvl_survival_off` | `pose_precision` | -0.0118 | [-0.0248, 0.0013] | negative | 是 |
| `context_on_minus_off_rvl_survival_off` | `aggregate_precision` | -0.0302 | [-0.0661, 0.0068] | negative | 是 |
| `context_on_minus_off_rvl_survival_off` | `pose_recall` | 0.0251 | [-0.0014, 0.0430] | positive | 是 |
| `context_on_minus_off_rvl_survival_off` | `aggregate_recall` | 0.0661 | [-0.0058, 0.1217] | positive | 是 |
| `context_on_minus_off_rvl_survival_off` | `pose_weighted_recall` | 0.0020 | [0.0000, 0.0041] | positive | 否 |
| `context_on_minus_off_rvl_survival_off` | `aggregate_weighted_recall` | 0.0022 | [0.0003, 0.0041] | positive | 否 |
| `context_on_minus_off_rvl_survival_off` | `pose_accuracy` | -0.0084 | [-0.0185, 0.0006] | negative | 是 |
| `context_on_minus_off_rvl_survival_off` | `aggregate_accuracy` | -0.0097 | [-0.0211, 0.0017] | negative | 是 |
| `context_on_minus_off_rvl_survival_off` | `pose_balanced_accuracy` | 0.0031 | [-0.0012, 0.0083] | positive | 是 |
| `context_on_minus_off_rvl_survival_off` | `aggregate_balanced_accuracy` | 0.0264 | [-0.0017, 0.0484] | positive | 是 |
| `context_on_minus_off_rvl_survival_off` | `pose_f1` | -0.0012 | [-0.0068, 0.0034] | negative | 是 |
| `context_on_minus_off_rvl_survival_off` | `aggregate_f1` | -0.0112 | [-0.0317, 0.0049] | negative | 是 |
| `context_on_minus_off_rvl_survival_off` | `pose_useful_cull` | -0.0128 | [-0.0258, 0.0014] | negative | 是 |
| `context_on_minus_off_rvl_survival_off` | `aggregate_useful_cull` | -0.0126 | [-0.0263, 0.0021] | negative | 是 |
| `context_on_minus_off_rvl_survival_off` | `pose_bad_cull` | -0.0044 | [-0.0082, 0.0005] | negative | 是 |
| `context_on_minus_off_rvl_survival_off` | `aggregate_bad_cull` | -0.0029 | [-0.0056, 0.0003] | negative | 是 |
| `context_on_minus_off_rvl_survival_off` | `pose_avg_fp_count` | 65.3834 | [-10.6062, 135.6288] | positive | 是 |
| `context_on_minus_off_rvl_survival_off` | `pose_avg_fn_count` | -15.1424 | [-29.1977, 1.3350] | negative | 是 |
| `context_on_minus_off_rvl_survival_off` | `pose_avg_tn_count` | -65.3834 | [-134.9880, 10.8952] | negative | 是 |
| `context_on_minus_off_rvl_survival_off` | `aggregate_avg_fp_count` | 65.3834 | [-10.3823, 136.8372] | positive | 是 |
| `context_on_minus_off_rvl_survival_off` | `aggregate_avg_fn_count` | -15.1424 | [-29.1412, 1.3349] | negative | 是 |
| `context_on_minus_off_rvl_survival_off` | `aggregate_avg_tn_count` | -65.3834 | [-134.5212, 10.4023] | negative | 是 |
| `context_on_minus_off_rvl_survival_off` | `pose_avg_pred_count` | 80.5258 | [-12.0566, 162.5231] | positive | 是 |
| `context_on_minus_off_rvl_survival_off` | `aggregate_avg_pred_count` | 80.5258 | [-12.1208, 162.3369] | positive | 是 |
| `context_on_minus_off_rvl_survival_off` | `aggregate_predicted_glb_bytes` | 4547270.1972 | [156456.1300, 8656323.4842] | positive | 否 |
| `context_on_minus_off_rvl_survival_off` | `pose_download_utility_recall` | 0.0020 | [0.0000, 0.0039] | positive | 否 |
| `context_on_minus_off_rvl_survival_off` | `aggregate_download_utility_recall` | 0.0021 | [0.0003, 0.0040] | positive | 否 |
| `context_on_minus_off_rvl_survival_off` | `pose_glb_bytes_at_achieved_visual_utility` | 1861679.0548 | [175565.5000, 3410234.3113] | positive | 否 |
| `context_on_minus_off_rvl_survival_off` | `aggregate_glb_bytes_at_achieved_visual_utility` | 1861679.0548 | [206592.5074, 3421567.0438] | positive | 否 |
| `context_on_minus_off_rvl_survival_off` | `aggregate_glb_byte_reduction` | -0.0121 | [-0.0228, 0.0001] | negative | 是 |
| `context_on_minus_off_rvl_survival_off` | `pose_image_per` | -0.0024 | [-0.0054, -0.0001] | negative | 否 |
| `context_on_minus_off_rvl_survival_off` | `aggregate_image_per` | -0.0024 | [-0.0053, -0.0003] | negative | 否 |
| `context_on_minus_off_rvl_survival_off` | `pose_image_miss_pixel_rate` | -0.0009 | [-0.0038, 0.0010] | negative | 是 |
| `context_on_minus_off_rvl_survival_off` | `aggregate_image_miss_pixel_rate` | -0.0009 | [-0.0040, 0.0009] | negative | 是 |
| `context_on_minus_off_rvl_survival_off` | `pose_image_wrong_id_pixel_rate` | -0.0015 | [-0.0028, -0.0004] | negative | 否 |
| `context_on_minus_off_rvl_survival_off` | `aggregate_image_extra_pixel_rate` | 0.0000 | [0.0000, 0.0000] | zero | 是 |
| `context_on_minus_off_rvl_survival_on` | `pose_precision` | 0.0001 | [-0.0067, 0.0097] | positive | 是 |
| `context_on_minus_off_rvl_survival_on` | `aggregate_precision` | -0.0113 | [-0.0487, 0.0135] | negative | 是 |
| `context_on_minus_off_rvl_survival_on` | `pose_recall` | 0.0023 | [-0.0063, 0.0087] | positive | 是 |
| `context_on_minus_off_rvl_survival_on` | `aggregate_recall` | 0.0242 | [-0.0209, 0.0973] | positive | 是 |
| `context_on_minus_off_rvl_survival_on` | `pose_weighted_recall` | 0.0003 | [-0.0034, 0.0030] | positive | 是 |
| `context_on_minus_off_rvl_survival_on` | `aggregate_weighted_recall` | 0.0005 | [-0.0027, 0.0029] | positive | 是 |
| `context_on_minus_off_rvl_survival_on` | `pose_accuracy` | -0.0008 | [-0.0038, 0.0022] | negative | 是 |
| `context_on_minus_off_rvl_survival_on` | `aggregate_accuracy` | -0.0003 | [-0.0034, 0.0021] | negative | 是 |
| `context_on_minus_off_rvl_survival_on` | `pose_balanced_accuracy` | 0.0007 | [-0.0073, 0.0065] | positive | 是 |
| `context_on_minus_off_rvl_survival_on` | `aggregate_balanced_accuracy` | 0.0114 | [-0.0094, 0.0448] | positive | 是 |
| `context_on_minus_off_rvl_survival_on` | `pose_f1` | 0.0007 | [-0.0091, 0.0145] | positive | 是 |
| `context_on_minus_off_rvl_survival_on` | `aggregate_f1` | 0.0074 | [-0.0049, 0.0300] | positive | 是 |
| `context_on_minus_off_rvl_survival_on` | `pose_useful_cull` | -0.0011 | [-0.0055, 0.0023] | negative | 是 |
| `context_on_minus_off_rvl_survival_on` | `aggregate_useful_cull` | -0.0014 | [-0.0077, 0.0027] | negative | 是 |
| `context_on_minus_off_rvl_survival_on` | `pose_bad_cull` | -0.0002 | [-0.0056, 0.0038] | negative | 是 |
| `context_on_minus_off_rvl_survival_on` | `aggregate_bad_cull` | -0.0011 | [-0.0046, 0.0010] | negative | 是 |
| `context_on_minus_off_rvl_survival_on` | `pose_avg_fp_count` | 7.2285 | [-14.0283, 37.8569] | positive | 是 |
| `context_on_minus_off_rvl_survival_on` | `pose_avg_fn_count` | -5.5493 | [-23.5889, 4.9265] | negative | 是 |
| `context_on_minus_off_rvl_survival_on` | `pose_avg_tn_count` | -7.2285 | [-38.0509, 14.3636] | negative | 是 |
| `context_on_minus_off_rvl_survival_on` | `aggregate_avg_fp_count` | 7.2285 | [-14.1331, 37.7348] | positive | 是 |
| `context_on_minus_off_rvl_survival_on` | `aggregate_avg_fn_count` | -5.5493 | [-23.7866, 5.0094] | negative | 是 |
| `context_on_minus_off_rvl_survival_on` | `aggregate_avg_tn_count` | -7.2285 | [-38.4572, 14.2304] | negative | 是 |
| `context_on_minus_off_rvl_survival_on` | `pose_avg_pred_count` | 12.7778 | [-18.2293, 61.0664] | positive | 是 |
| `context_on_minus_off_rvl_survival_on` | `aggregate_avg_pred_count` | 12.7778 | [-18.3256, 59.8968] | positive | 是 |
| `context_on_minus_off_rvl_survival_on` | `aggregate_predicted_glb_bytes` | 557070.5290 | [-556925.7153, 1794523.9994] | positive | 是 |
| `context_on_minus_off_rvl_survival_on` | `pose_download_utility_recall` | 0.0003 | [-0.0033, 0.0030] | positive | 是 |
| `context_on_minus_off_rvl_survival_on` | `aggregate_download_utility_recall` | 0.0005 | [-0.0028, 0.0030] | positive | 是 |
| `context_on_minus_off_rvl_survival_on` | `pose_glb_bytes_at_achieved_visual_utility` | 187574.4413 | [-1017403.9031, 1558589.7128] | positive | 是 |
| `context_on_minus_off_rvl_survival_on` | `aggregate_glb_bytes_at_achieved_visual_utility` | 187574.4413 | [-994946.0723, 1531582.8078] | positive | 是 |
| `context_on_minus_off_rvl_survival_on` | `aggregate_glb_byte_reduction` | -0.0014 | [-0.0075, 0.0041] | negative | 是 |
| `context_on_minus_off_rvl_survival_on` | `pose_image_per` | -0.0006 | [-0.0066, 0.0081] | negative | 是 |
| `context_on_minus_off_rvl_survival_on` | `aggregate_image_per` | -0.0005 | [-0.0064, 0.0080] | negative | 是 |
| `context_on_minus_off_rvl_survival_on` | `pose_image_miss_pixel_rate` | -0.0021 | [-0.0098, 0.0078] | negative | 是 |
| `context_on_minus_off_rvl_survival_on` | `aggregate_image_miss_pixel_rate` | -0.0020 | [-0.0101, 0.0084] | negative | 是 |
| `context_on_minus_off_rvl_survival_on` | `pose_image_wrong_id_pixel_rate` | 0.0014 | [-0.0004, 0.0035] | positive | 是 |
| `context_on_minus_off_rvl_survival_on` | `aggregate_image_extra_pixel_rate` | 0.0000 | [0.0000, 0.0000] | zero | 是 |
| `survival_on_minus_off_rvl_context_off` | `pose_precision` | 0.0831 | [0.0748, 0.0916] | positive | 否 |
| `survival_on_minus_off_rvl_context_off` | `aggregate_precision` | 0.1366 | [0.0799, 0.2103] | positive | 否 |
| `survival_on_minus_off_rvl_context_off` | `pose_recall` | -0.0039 | [-0.0163, 0.0075] | negative | 是 |
| `survival_on_minus_off_rvl_context_off` | `aggregate_recall` | 0.0267 | [-0.0854, 0.1026] | positive | 是 |
| `survival_on_minus_off_rvl_context_off` | `pose_weighted_recall` | 0.0013 | [-0.0010, 0.0040] | positive | 是 |
| `survival_on_minus_off_rvl_context_off` | `aggregate_weighted_recall` | 0.0013 | [-0.0009, 0.0035] | positive | 是 |
| `survival_on_minus_off_rvl_context_off` | `pose_accuracy` | 0.0772 | [0.0667, 0.0882] | positive | 否 |
| `survival_on_minus_off_rvl_context_off` | `aggregate_accuracy` | 0.0196 | [0.0112, 0.0295] | positive | 否 |
| `survival_on_minus_off_rvl_context_off` | `pose_balanced_accuracy` | 0.0554 | [0.0459, 0.0657] | positive | 否 |
| `survival_on_minus_off_rvl_context_off` | `aggregate_balanced_accuracy` | 0.0230 | [-0.0260, 0.0568] | positive | 是 |
| `survival_on_minus_off_rvl_context_off` | `pose_f1` | 0.0814 | [0.0682, 0.0929] | positive | 否 |
| `survival_on_minus_off_rvl_context_off` | `aggregate_f1` | 0.1002 | [0.0767, 0.1204] | positive | 否 |
| `survival_on_minus_off_rvl_context_off` | `pose_useful_cull` | 0.0823 | [0.0689, 0.0969] | positive | 否 |
| `survival_on_minus_off_rvl_context_off` | `aggregate_useful_cull` | 0.0184 | [0.0074, 0.0323] | positive | 否 |
| `survival_on_minus_off_rvl_context_off` | `pose_bad_cull` | 0.0051 | [-0.0013, 0.0130] | positive | 是 |
| `survival_on_minus_off_rvl_context_off` | `aggregate_bad_cull` | -0.0012 | [-0.0049, 0.0039] | negative | 是 |
| `survival_on_minus_off_rvl_context_off` | `pose_avg_fp_count` | -95.1612 | [-163.2490, -38.2982] | negative | 否 |
| `survival_on_minus_off_rvl_context_off` | `pose_avg_fn_count` | -6.1158 | [-25.0297, 19.6060] | negative | 是 |
| `survival_on_minus_off_rvl_context_off` | `pose_avg_tn_count` | 95.1612 | [37.7821, 163.5214] | positive | 否 |
| `survival_on_minus_off_rvl_context_off` | `aggregate_avg_fp_count` | -95.1612 | [-164.4134, -37.8696] | negative | 否 |
| `survival_on_minus_off_rvl_context_off` | `aggregate_avg_fn_count` | -6.1158 | [-25.1116, 19.9317] | negative | 是 |
| `survival_on_minus_off_rvl_context_off` | `aggregate_avg_tn_count` | 95.1612 | [36.7978, 165.0773] | positive | 否 |
| `survival_on_minus_off_rvl_context_off` | `pose_avg_pred_count` | -89.0454 | [-181.1068, -15.0655] | negative | 否 |
| `survival_on_minus_off_rvl_context_off` | `aggregate_avg_pred_count` | -89.0454 | [-178.4730, -16.1192] | negative | 否 |
| `survival_on_minus_off_rvl_context_off` | `aggregate_predicted_glb_bytes` | -3281290.6917 | [-7233053.3801, -297726.0554] | negative | 否 |
| `survival_on_minus_off_rvl_context_off` | `pose_download_utility_recall` | 0.0013 | [-0.0011, 0.0039] | positive | 是 |
| `survival_on_minus_off_rvl_context_off` | `aggregate_download_utility_recall` | 0.0012 | [-0.0010, 0.0035] | positive | 是 |
| `survival_on_minus_off_rvl_context_off` | `pose_glb_bytes_at_achieved_visual_utility` | 2509342.0970 | [-356839.9161, 4973174.6293] | positive | 是 |
| `survival_on_minus_off_rvl_context_off` | `aggregate_glb_bytes_at_achieved_visual_utility` | 2509342.0970 | [-300494.1889, 4953037.0025] | positive | 是 |
| `survival_on_minus_off_rvl_context_off` | `aggregate_glb_byte_reduction` | 0.0875 | [0.0739, 0.1026] | positive | 否 |
| `survival_on_minus_off_rvl_context_off` | `pose_image_per` | -0.0016 | [-0.0088, 0.0041] | negative | 是 |
| `survival_on_minus_off_rvl_context_off` | `aggregate_image_per` | -0.0013 | [-0.0086, 0.0046] | negative | 是 |
| `survival_on_minus_off_rvl_context_off` | `pose_image_miss_pixel_rate` | 0.0022 | [-0.0065, 0.0096] | positive | 是 |
| `survival_on_minus_off_rvl_context_off` | `aggregate_image_miss_pixel_rate` | 0.0025 | [-0.0064, 0.0103] | positive | 是 |
| `survival_on_minus_off_rvl_context_off` | `pose_image_wrong_id_pixel_rate` | -0.0038 | [-0.0057, -0.0016] | negative | 否 |
| `survival_on_minus_off_rvl_context_off` | `aggregate_image_extra_pixel_rate` | 0.0000 | [0.0000, 0.0000] | zero | 是 |
| `survival_on_minus_off_rvl_context_on` | `pose_precision` | 0.0950 | [0.0777, 0.1103] | positive | 否 |
| `survival_on_minus_off_rvl_context_on` | `aggregate_precision` | 0.1555 | [0.1089, 0.1953] | positive | 否 |
| `survival_on_minus_off_rvl_context_on` | `pose_recall` | -0.0268 | [-0.0547, 0.0039] | negative | 是 |
| `survival_on_minus_off_rvl_context_on` | `aggregate_recall` | -0.0152 | [-0.0770, 0.0642] | negative | 是 |
| `survival_on_minus_off_rvl_context_on` | `pose_weighted_recall` | -0.0004 | [-0.0036, 0.0022] | negative | 是 |
| `survival_on_minus_off_rvl_context_on` | `aggregate_weighted_recall` | -0.0004 | [-0.0036, 0.0022] | negative | 是 |
| `survival_on_minus_off_rvl_context_on` | `pose_accuracy` | 0.0848 | [0.0705, 0.0986] | positive | 否 |
| `survival_on_minus_off_rvl_context_on` | `aggregate_accuracy` | 0.0290 | [0.0177, 0.0410] | positive | 否 |
| `survival_on_minus_off_rvl_context_on` | `pose_balanced_accuracy` | 0.0530 | [0.0433, 0.0621] | positive | 否 |
| `survival_on_minus_off_rvl_context_on` | `aggregate_balanced_accuracy` | 0.0079 | [-0.0195, 0.0410] | positive | 是 |
| `survival_on_minus_off_rvl_context_on` | `pose_f1` | 0.0834 | [0.0756, 0.0913] | positive | 否 |
| `survival_on_minus_off_rvl_context_on` | `aggregate_f1` | 0.1188 | [0.0980, 0.1367] | positive | 否 |
| `survival_on_minus_off_rvl_context_on` | `pose_useful_cull` | 0.0940 | [0.0754, 0.1103] | positive | 否 |
| `survival_on_minus_off_rvl_context_on` | `aggregate_useful_cull` | 0.0296 | [0.0151, 0.0437] | positive | 否 |
| `survival_on_minus_off_rvl_context_on` | `pose_bad_cull` | 0.0092 | [0.0027, 0.0153] | positive | 否 |
| `survival_on_minus_off_rvl_context_on` | `aggregate_bad_cull` | 0.0007 | [-0.0029, 0.0034] | positive | 是 |
| `survival_on_minus_off_rvl_context_on` | `pose_avg_fp_count` | -153.3161 | [-224.1333, -77.3453] | negative | 否 |
| `survival_on_minus_off_rvl_context_on` | `pose_avg_fn_count` | 3.4773 | [-15.0770, 17.7998] | positive | 是 |
| `survival_on_minus_off_rvl_context_on` | `pose_avg_tn_count` | 153.3161 | [78.1428, 223.0450] | positive | 否 |
| `survival_on_minus_off_rvl_context_on` | `aggregate_avg_fp_count` | -153.3161 | [-224.5890, -79.2515] | negative | 否 |
| `survival_on_minus_off_rvl_context_on` | `aggregate_avg_fn_count` | 3.4773 | [-15.2088, 17.8044] | positive | 是 |
| `survival_on_minus_off_rvl_context_on` | `aggregate_avg_tn_count` | 153.3161 | [77.8509, 223.6966] | positive | 否 |
| `survival_on_minus_off_rvl_context_on` | `pose_avg_pred_count` | -156.7934 | [-239.7077, -62.8200] | negative | 否 |
| `survival_on_minus_off_rvl_context_on` | `aggregate_avg_pred_count` | -156.7934 | [-239.0468, -62.4750] | negative | 否 |
| `survival_on_minus_off_rvl_context_on` | `aggregate_predicted_glb_bytes` | -7271490.3599 | [-11798916.3189, -1813524.5479] | negative | 否 |
| `survival_on_minus_off_rvl_context_on` | `pose_download_utility_recall` | -0.0004 | [-0.0035, 0.0023] | negative | 是 |
| `survival_on_minus_off_rvl_context_on` | `aggregate_download_utility_recall` | -0.0004 | [-0.0035, 0.0022] | negative | 是 |
| `survival_on_minus_off_rvl_context_on` | `pose_glb_bytes_at_achieved_visual_utility` | 835237.4836 | [-1184183.7174, 3331035.8086] | positive | 是 |
| `survival_on_minus_off_rvl_context_on` | `aggregate_glb_bytes_at_achieved_visual_utility` | 835237.4836 | [-1174812.9302, 3268994.2258] | positive | 是 |
| `survival_on_minus_off_rvl_context_on` | `aggregate_glb_byte_reduction` | 0.0983 | [0.0820, 0.1121] | positive | 否 |
| `survival_on_minus_off_rvl_context_on` | `pose_image_per` | 0.0002 | [-0.0029, 0.0045] | positive | 是 |
| `survival_on_minus_off_rvl_context_on` | `aggregate_image_per` | 0.0006 | [-0.0024, 0.0049] | positive | 是 |
| `survival_on_minus_off_rvl_context_on` | `pose_image_miss_pixel_rate` | 0.0010 | [-0.0017, 0.0054] | positive | 是 |
| `survival_on_minus_off_rvl_context_on` | `aggregate_image_miss_pixel_rate` | 0.0014 | [-0.0014, 0.0063] | positive | 是 |
| `survival_on_minus_off_rvl_context_on` | `pose_image_wrong_id_pixel_rate` | -0.0008 | [-0.0020, 0.0003] | negative | 是 |
| `survival_on_minus_off_rvl_context_on` | `aggregate_image_extra_pixel_rate` | 0.0000 | [0.0000, 0.0000] | zero | 是 |
| `context_on_minus_off_safety_survival_off` | `pose_precision` | -0.0230 | [-0.0356, -0.0128] | negative | 否 |
| `context_on_minus_off_safety_survival_off` | `aggregate_precision` | -0.0574 | [-0.0840, -0.0305] | negative | 否 |
| `context_on_minus_off_safety_survival_off` | `pose_recall` | 0.0204 | [0.0043, 0.0386] | positive | 否 |
| `context_on_minus_off_safety_survival_off` | `aggregate_recall` | 0.0611 | [0.0029, 0.1328] | positive | 否 |
| `context_on_minus_off_safety_survival_off` | `pose_weighted_recall` | 0.0014 | [0.0003, 0.0026] | positive | 否 |
| `context_on_minus_off_safety_survival_off` | `aggregate_weighted_recall` | 0.0015 | [0.0004, 0.0027] | positive | 否 |
| `context_on_minus_off_safety_survival_off` | `pose_accuracy` | -0.0162 | [-0.0251, -0.0065] | negative | 否 |
| `context_on_minus_off_safety_survival_off` | `aggregate_accuracy` | -0.0179 | [-0.0271, -0.0096] | negative | 否 |
| `context_on_minus_off_safety_survival_off` | `pose_balanced_accuracy` | -0.0018 | [-0.0042, 0.0011] | negative | 是 |
| `context_on_minus_off_safety_survival_off` | `aggregate_balanced_accuracy` | 0.0197 | [-0.0038, 0.0499] | positive | 是 |
| `context_on_minus_off_safety_survival_off` | `pose_f1` | -0.0180 | [-0.0260, -0.0118] | negative | 否 |
| `context_on_minus_off_safety_survival_off` | `aggregate_f1` | -0.0360 | [-0.0496, -0.0231] | negative | 否 |
| `context_on_minus_off_safety_survival_off` | `pose_useful_cull` | -0.0193 | [-0.0314, -0.0073] | negative | 否 |
| `context_on_minus_off_safety_survival_off` | `aggregate_useful_cull` | -0.0206 | [-0.0324, -0.0094] | negative | 否 |
| `context_on_minus_off_safety_survival_off` | `pose_bad_cull` | -0.0031 | [-0.0066, -0.0007] | negative | 否 |
| `context_on_minus_off_safety_survival_off` | `aggregate_bad_cull` | -0.0027 | [-0.0060, -0.0001] | negative | 否 |
| `context_on_minus_off_safety_survival_off` | `pose_avg_fp_count` | 106.7387 | [50.1622, 168.9172] | positive | 否 |
| `context_on_minus_off_safety_survival_off` | `pose_avg_fn_count` | -13.9875 | [-30.7323, -0.5646] | negative | 否 |
| `context_on_minus_off_safety_survival_off` | `pose_avg_tn_count` | -106.7387 | [-168.5084, -49.7133] | negative | 否 |
| `context_on_minus_off_safety_survival_off` | `aggregate_avg_fp_count` | 106.7387 | [50.0344, 170.0869] | positive | 否 |
| `context_on_minus_off_safety_survival_off` | `aggregate_avg_fn_count` | -13.9875 | [-30.8890, -0.4709] | negative | 否 |
| `context_on_minus_off_safety_survival_off` | `aggregate_avg_tn_count` | -106.7387 | [-168.1286, -49.5516] | negative | 否 |
| `context_on_minus_off_safety_survival_off` | `pose_avg_pred_count` | 120.7261 | [51.2494, 197.0092] | positive | 否 |
| `context_on_minus_off_safety_survival_off` | `aggregate_avg_pred_count` | 120.7261 | [51.3224, 196.8980] | positive | 否 |
| `context_on_minus_off_safety_survival_off` | `aggregate_predicted_glb_bytes` | 14291048.2817 | [7119566.9770, 23854863.6225] | positive | 否 |
| `context_on_minus_off_safety_survival_off` | `pose_download_utility_recall` | 0.0014 | [0.0005, 0.0025] | positive | 否 |
| `context_on_minus_off_safety_survival_off` | `aggregate_download_utility_recall` | 0.0015 | [0.0006, 0.0026] | positive | 否 |
| `context_on_minus_off_safety_survival_off` | `pose_glb_bytes_at_achieved_visual_utility` | 1760798.2535 | [877319.7236, 3049137.7698] | positive | 否 |
| `context_on_minus_off_safety_survival_off` | `aggregate_glb_bytes_at_achieved_visual_utility` | 1760798.2535 | [870817.4651, 3080715.0097] | positive | 否 |
| `context_on_minus_off_safety_survival_off` | `aggregate_glb_byte_reduction` | -0.0420 | [-0.0679, -0.0196] | negative | 否 |
| `context_on_minus_off_safety_survival_off` | `pose_image_per` | -0.0013 | [-0.0023, -0.0002] | negative | 否 |
| `context_on_minus_off_safety_survival_off` | `aggregate_image_per` | -0.0013 | [-0.0023, -0.0003] | negative | 否 |
| `context_on_minus_off_safety_survival_off` | `pose_image_miss_pixel_rate` | -0.0005 | [-0.0012, -0.0000] | negative | 否 |
| `context_on_minus_off_safety_survival_off` | `aggregate_image_miss_pixel_rate` | -0.0004 | [-0.0012, -0.0000] | negative | 否 |
| `context_on_minus_off_safety_survival_off` | `pose_image_wrong_id_pixel_rate` | -0.0008 | [-0.0018, -0.0001] | negative | 否 |
| `context_on_minus_off_safety_survival_off` | `aggregate_image_extra_pixel_rate` | 0.0000 | [0.0000, 0.0000] | zero | 是 |
| `context_on_minus_off_safety_survival_on` | `pose_precision` | 0.0027 | [-0.0107, 0.0128] | positive | 是 |
| `context_on_minus_off_safety_survival_on` | `aggregate_precision` | 0.0094 | [0.0001, 0.0195] | positive | 否 |
| `context_on_minus_off_safety_survival_on` | `pose_recall` | 0.0003 | [-0.0100, 0.0160] | positive | 是 |
| `context_on_minus_off_safety_survival_on` | `aggregate_recall` | 0.0022 | [-0.0234, 0.0320] | positive | 是 |
| `context_on_minus_off_safety_survival_on` | `pose_weighted_recall` | -0.0000 | [-0.0006, 0.0007] | negative | 是 |
| `context_on_minus_off_safety_survival_on` | `aggregate_weighted_recall` | -0.0000 | [-0.0006, 0.0007] | negative | 是 |
| `context_on_minus_off_safety_survival_on` | `pose_accuracy` | 0.0169 | [-0.0271, 0.0450] | positive | 是 |
| `context_on_minus_off_safety_survival_on` | `aggregate_accuracy` | 0.0304 | [-0.0138, 0.0560] | positive | 是 |
| `context_on_minus_off_safety_survival_on` | `pose_balanced_accuracy` | 0.0082 | [-0.0123, 0.0218] | positive | 是 |
| `context_on_minus_off_safety_survival_on` | `aggregate_balanced_accuracy` | 0.0170 | [0.0083, 0.0273] | positive | 否 |
| `context_on_minus_off_safety_survival_on` | `pose_f1` | 0.0045 | [-0.0084, 0.0159] | positive | 是 |
| `context_on_minus_off_safety_survival_on` | `aggregate_f1` | 0.0150 | [0.0008, 0.0304] | positive | 否 |
| `context_on_minus_off_safety_survival_on` | `pose_useful_cull` | 0.0165 | [-0.0312, 0.0458] | positive | 是 |
| `context_on_minus_off_safety_survival_on` | `aggregate_useful_cull` | 0.0304 | [-0.0152, 0.0562] | positive | 是 |
| `context_on_minus_off_safety_survival_on` | `pose_bad_cull` | -0.0004 | [-0.0037, 0.0020] | negative | 是 |
| `context_on_minus_off_safety_survival_on` | `aggregate_bad_cull` | -0.0001 | [-0.0014, 0.0010] | negative | 是 |
| `context_on_minus_off_safety_survival_on` | `pose_avg_fp_count` | -157.0438 | [-303.1698, 77.9900] | negative | 是 |
| `context_on_minus_off_safety_survival_on` | `pose_avg_fn_count` | -0.4961 | [-7.2958, 5.3255] | negative | 是 |
| `context_on_minus_off_safety_survival_on` | `pose_avg_tn_count` | 157.0438 | [-77.7152, 301.7899] | positive | 是 |
| `context_on_minus_off_safety_survival_on` | `aggregate_avg_fp_count` | -157.0438 | [-301.1929, 78.7357] | negative | 是 |
| `context_on_minus_off_safety_survival_on` | `aggregate_avg_fn_count` | -0.4961 | [-7.2397, 5.3006] | negative | 是 |
| `context_on_minus_off_safety_survival_on` | `aggregate_avg_tn_count` | 157.0438 | [-76.9468, 302.9296] | positive | 是 |
| `context_on_minus_off_safety_survival_on` | `pose_avg_pred_count` | -156.5477 | [-303.8362, 85.3459] | negative | 是 |
| `context_on_minus_off_safety_survival_on` | `aggregate_avg_pred_count` | -156.5477 | [-304.4830, 84.7275] | negative | 是 |
| `context_on_minus_off_safety_survival_on` | `aggregate_predicted_glb_bytes` | -3827842.7105 | [-15891680.0291, 13547045.5829] | negative | 是 |
| `context_on_minus_off_safety_survival_on` | `pose_download_utility_recall` | -0.0000 | [-0.0006, 0.0007] | negative | 是 |
| `context_on_minus_off_safety_survival_on` | `aggregate_download_utility_recall` | 0.0000 | [-0.0006, 0.0007] | positive | 是 |
| `context_on_minus_off_safety_survival_on` | `pose_glb_bytes_at_achieved_visual_utility` | -64771.3552 | [-2469704.3235, 2263694.7042] | negative | 是 |
| `context_on_minus_off_safety_survival_on` | `aggregate_glb_bytes_at_achieved_visual_utility` | -64771.3552 | [-2385919.2798, 2303009.1531] | negative | 是 |
| `context_on_minus_off_safety_survival_on` | `aggregate_glb_byte_reduction` | 0.0149 | [-0.0620, 0.0726] | positive | 是 |
| `context_on_minus_off_safety_survival_on` | `pose_image_per` | 0.0002 | [-0.0004, 0.0009] | positive | 是 |
| `context_on_minus_off_safety_survival_on` | `aggregate_image_per` | 0.0002 | [-0.0005, 0.0010] | positive | 是 |
| `context_on_minus_off_safety_survival_on` | `pose_image_miss_pixel_rate` | 0.0002 | [-0.0000, 0.0006] | positive | 是 |
| `context_on_minus_off_safety_survival_on` | `aggregate_image_miss_pixel_rate` | 0.0002 | [-0.0000, 0.0008] | positive | 是 |
| `context_on_minus_off_safety_survival_on` | `pose_image_wrong_id_pixel_rate` | 0.0000 | [-0.0004, 0.0004] | positive | 是 |
| `context_on_minus_off_safety_survival_on` | `aggregate_image_extra_pixel_rate` | 0.0000 | [0.0000, 0.0000] | zero | 是 |
| `survival_on_minus_off_safety_context_off` | `pose_precision` | -0.0667 | [-0.0844, -0.0504] | negative | 否 |
| `survival_on_minus_off_safety_context_off` | `aggregate_precision` | -0.2615 | [-0.3075, -0.2104] | negative | 否 |
| `survival_on_minus_off_safety_context_off` | `pose_recall` | 0.0629 | [0.0412, 0.0848] | positive | 否 |
| `survival_on_minus_off_safety_context_off` | `aggregate_recall` | 0.2743 | [0.2355, 0.3170] | positive | 否 |
| `survival_on_minus_off_safety_context_off` | `pose_weighted_recall` | 0.0033 | [0.0014, 0.0052] | positive | 否 |
| `survival_on_minus_off_safety_context_off` | `aggregate_weighted_recall` | 0.0038 | [0.0019, 0.0056] | positive | 否 |
| `survival_on_minus_off_safety_context_off` | `pose_accuracy` | -0.1142 | [-0.1623, -0.0756] | negative | 否 |
| `survival_on_minus_off_safety_context_off` | `aggregate_accuracy` | -0.3236 | [-0.3730, -0.2561] | negative | 否 |
| `survival_on_minus_off_safety_context_off` | `pose_balanced_accuracy` | -0.0264 | [-0.0489, -0.0047] | negative | 否 |
| `survival_on_minus_off_safety_context_off` | `aggregate_balanced_accuracy` | -0.0385 | [-0.0712, 0.0004] | negative | 是 |
| `survival_on_minus_off_safety_context_off` | `pose_f1` | -0.0636 | [-0.0819, -0.0471] | negative | 否 |
| `survival_on_minus_off_safety_context_off` | `aggregate_f1` | -0.2828 | [-0.3237, -0.2347] | negative | 否 |
| `survival_on_minus_off_safety_context_off` | `pose_useful_cull` | -0.1231 | [-0.1749, -0.0827] | negative | 否 |
| `survival_on_minus_off_safety_context_off` | `aggregate_useful_cull` | -0.3358 | [-0.3857, -0.2655] | negative | 否 |
| `survival_on_minus_off_safety_context_off` | `pose_bad_cull` | -0.0090 | [-0.0127, -0.0053] | negative | 否 |
| `survival_on_minus_off_safety_context_off` | `aggregate_bad_cull` | -0.0121 | [-0.0152, -0.0095] | negative | 否 |
| `survival_on_minus_off_safety_context_off` | `pose_avg_fp_count` | 1737.3912 | [1363.7201, 2095.2756] | positive | 否 |
| `survival_on_minus_off_safety_context_off` | `pose_avg_fn_count` | -62.8388 | [-78.2290, -48.6588] | negative | 否 |
| `survival_on_minus_off_safety_context_off` | `pose_avg_tn_count` | -1737.3912 | [-2101.0799, -1359.0700] | negative | 否 |
| `survival_on_minus_off_safety_context_off` | `aggregate_avg_fp_count` | 1737.3912 | [1365.9456, 2088.2708] | positive | 否 |
| `survival_on_minus_off_safety_context_off` | `aggregate_avg_fn_count` | -62.8388 | [-78.7247, -48.8027] | negative | 否 |
| `survival_on_minus_off_safety_context_off` | `aggregate_avg_tn_count` | -1737.3912 | [-2095.3829, -1360.7314] | negative | 否 |
| `survival_on_minus_off_safety_context_off` | `pose_avg_pred_count` | 1800.2300 | [1429.3797, 2153.6504] | positive | 否 |
| `survival_on_minus_off_safety_context_off` | `aggregate_avg_pred_count` | 1800.2300 | [1423.7977, 2155.6737] | positive | 否 |
| `survival_on_minus_off_safety_context_off` | `aggregate_predicted_glb_bytes` | 51188323.8247 | [31399002.1153, 81271610.4510] | positive | 否 |
| `survival_on_minus_off_safety_context_off` | `pose_download_utility_recall` | 0.0031 | [0.0012, 0.0051] | positive | 否 |
| `survival_on_minus_off_safety_context_off` | `aggregate_download_utility_recall` | 0.0034 | [0.0017, 0.0053] | positive | 否 |
| `survival_on_minus_off_safety_context_off` | `pose_glb_bytes_at_achieved_visual_utility` | 10498824.2191 | [4538028.2668, 15663228.7214] | positive | 否 |
| `survival_on_minus_off_safety_context_off` | `aggregate_glb_bytes_at_achieved_visual_utility` | 10498824.2191 | [4402410.2825, 15651535.3831] | positive | 否 |
| `survival_on_minus_off_safety_context_off` | `aggregate_glb_byte_reduction` | -0.1220 | [-0.2190, -0.0367] | negative | 否 |
| `survival_on_minus_off_safety_context_off` | `pose_image_per` | -0.0018 | [-0.0046, 0.0020] | negative | 是 |
| `survival_on_minus_off_safety_context_off` | `aggregate_image_per` | -0.0018 | [-0.0047, 0.0020] | negative | 是 |
| `survival_on_minus_off_safety_context_off` | `pose_image_miss_pixel_rate` | -0.0005 | [-0.0020, 0.0009] | negative | 是 |
| `survival_on_minus_off_safety_context_off` | `aggregate_image_miss_pixel_rate` | -0.0006 | [-0.0021, 0.0009] | negative | 是 |
| `survival_on_minus_off_safety_context_off` | `pose_image_wrong_id_pixel_rate` | -0.0012 | [-0.0028, 0.0011] | negative | 是 |
| `survival_on_minus_off_safety_context_off` | `aggregate_image_extra_pixel_rate` | 0.0000 | [0.0000, 0.0000] | zero | 是 |
| `survival_on_minus_off_safety_context_on` | `pose_precision` | -0.0411 | [-0.0573, -0.0265] | negative | 否 |
| `survival_on_minus_off_safety_context_on` | `aggregate_precision` | -0.1947 | [-0.2283, -0.1586] | negative | 否 |
| `survival_on_minus_off_safety_context_on` | `pose_recall` | 0.0428 | [0.0271, 0.0588] | positive | 否 |
| `survival_on_minus_off_safety_context_on` | `aggregate_recall` | 0.2154 | [0.1539, 0.2811] | positive | 否 |
| `survival_on_minus_off_safety_context_on` | `pose_weighted_recall` | 0.0019 | [0.0005, 0.0034] | positive | 否 |
| `survival_on_minus_off_safety_context_on` | `aggregate_weighted_recall` | 0.0023 | [0.0009, 0.0038] | positive | 否 |
| `survival_on_minus_off_safety_context_on` | `pose_accuracy` | -0.0811 | [-0.1177, -0.0403] | negative | 否 |
| `survival_on_minus_off_safety_context_on` | `aggregate_accuracy` | -0.2753 | [-0.3337, -0.1918] | negative | 否 |
| `survival_on_minus_off_safety_context_on` | `pose_balanced_accuracy` | -0.0164 | [-0.0322, -0.0006] | negative | 否 |
| `survival_on_minus_off_safety_context_on` | `aggregate_balanced_accuracy` | -0.0413 | [-0.0983, 0.0276] | negative | 是 |
| `survival_on_minus_off_safety_context_on` | `pose_f1` | -0.0411 | [-0.0589, -0.0242] | negative | 否 |
| `survival_on_minus_off_safety_context_on` | `aggregate_f1` | -0.2318 | [-0.2777, -0.1783] | negative | 否 |
| `survival_on_minus_off_safety_context_on` | `pose_useful_cull` | -0.0873 | [-0.1254, -0.0455] | negative | 否 |
| `survival_on_minus_off_safety_context_on` | `aggregate_useful_cull` | -0.2848 | [-0.3407, -0.2014] | negative | 否 |
| `survival_on_minus_off_safety_context_on` | `pose_bad_cull` | -0.0062 | [-0.0094, -0.0032] | negative | 否 |
| `survival_on_minus_off_safety_context_on` | `aggregate_bad_cull` | -0.0095 | [-0.0132, -0.0065] | negative | 否 |
| `survival_on_minus_off_safety_context_on` | `pose_avg_fp_count` | 1473.6088 | [1053.8464, 1847.5221] | positive | 否 |
| `survival_on_minus_off_safety_context_on` | `pose_avg_fn_count` | -49.3474 | [-67.5814, -33.0887] | negative | 否 |
| `survival_on_minus_off_safety_context_on` | `pose_avg_tn_count` | -1473.6088 | [-1852.2261, -1052.3407] | negative | 否 |
| `survival_on_minus_off_safety_context_on` | `aggregate_avg_fp_count` | 1473.6088 | [1058.8872, 1848.4977] | positive | 否 |
| `survival_on_minus_off_safety_context_on` | `aggregate_avg_fn_count` | -49.3474 | [-68.1878, -33.4146] | negative | 否 |
| `survival_on_minus_off_safety_context_on` | `aggregate_avg_tn_count` | -1473.6088 | [-1854.3180, -1058.5717] | negative | 否 |
| `survival_on_minus_off_safety_context_on` | `pose_avg_pred_count` | 1522.9562 | [1119.1498, 1884.9036] | positive | 否 |
| `survival_on_minus_off_safety_context_on` | `aggregate_avg_pred_count` | 1522.9562 | [1118.0987, 1885.6374] | positive | 否 |
| `survival_on_minus_off_safety_context_on` | `aggregate_predicted_glb_bytes` | 33069432.8326 | [14711607.7728, 60537247.9124] | positive | 否 |
| `survival_on_minus_off_safety_context_on` | `pose_download_utility_recall` | 0.0017 | [0.0003, 0.0033] | positive | 否 |
| `survival_on_minus_off_safety_context_on` | `aggregate_download_utility_recall` | 0.0019 | [0.0005, 0.0035] | positive | 否 |
| `survival_on_minus_off_safety_context_on` | `pose_glb_bytes_at_achieved_visual_utility` | 8673254.6103 | [3812883.6808, 13656453.4516] | positive | 否 |
| `survival_on_minus_off_safety_context_on` | `aggregate_glb_bytes_at_achieved_visual_utility` | 8673254.6103 | [3689407.1142, 13572228.0997] | positive | 否 |
| `survival_on_minus_off_safety_context_on` | `aggregate_glb_byte_reduction` | -0.0651 | [-0.1481, -0.0102] | negative | 否 |
| `survival_on_minus_off_safety_context_on` | `pose_image_per` | -0.0003 | [-0.0028, 0.0031] | negative | 是 |
| `survival_on_minus_off_safety_context_on` | `aggregate_image_per` | -0.0004 | [-0.0030, 0.0032] | negative | 是 |
| `survival_on_minus_off_safety_context_on` | `pose_image_miss_pixel_rate` | 0.0001 | [-0.0009, 0.0014] | positive | 是 |
| `survival_on_minus_off_safety_context_on` | `aggregate_image_miss_pixel_rate` | 0.0000 | [-0.0011, 0.0014] | positive | 是 |
| `survival_on_minus_off_safety_context_on` | `pose_image_wrong_id_pixel_rate` | -0.0004 | [-0.0019, 0.0017] | negative | 是 |
| `survival_on_minus_off_safety_context_on` | `aggregate_image_extra_pixel_rate` | 0.0000 | [0.0000, 0.0000] | zero | 是 |
| `safety_minus_rvl_context_off_survival_off` | `pose_precision` | -0.0598 | [-0.0781, -0.0449] | negative | 否 |
| `safety_minus_rvl_context_off_survival_off` | `aggregate_precision` | -0.0162 | [-0.0534, 0.0127] | negative | 是 |
| `safety_minus_rvl_context_off_survival_off` | `pose_recall` | 0.0608 | [0.0395, 0.0862] | positive | 否 |
| `safety_minus_rvl_context_off_survival_off` | `aggregate_recall` | -0.0124 | [-0.0731, 0.0526] | negative | 是 |
| `safety_minus_rvl_context_off_survival_off` | `pose_weighted_recall` | 0.0058 | [0.0039, 0.0085] | positive | 否 |
| `safety_minus_rvl_context_off_survival_off` | `aggregate_weighted_recall` | 0.0048 | [0.0028, 0.0073] | positive | 否 |
| `safety_minus_rvl_context_off_survival_off` | `pose_accuracy` | -0.0361 | [-0.0512, -0.0236] | negative | 否 |
| `safety_minus_rvl_context_off_survival_off` | `aggregate_accuracy` | -0.0037 | [-0.0132, 0.0043] | negative | 是 |
| `safety_minus_rvl_context_off_survival_off` | `pose_balanced_accuracy` | 0.0031 | [-0.0044, 0.0107] | positive | 是 |
| `safety_minus_rvl_context_off_survival_off` | `aggregate_balanced_accuracy` | -0.0079 | [-0.0344, 0.0202] | negative | 是 |
| `safety_minus_rvl_context_off_survival_off` | `pose_f1` | -0.0533 | [-0.0716, -0.0381] | negative | 否 |
| `safety_minus_rvl_context_off_survival_off` | `aggregate_f1` | -0.0186 | [-0.0449, 0.0056] | negative | 是 |
| `safety_minus_rvl_context_off_survival_off` | `pose_useful_cull` | -0.0418 | [-0.0596, -0.0275] | negative | 否 |
| `safety_minus_rvl_context_off_survival_off` | `aggregate_useful_cull` | -0.0032 | [-0.0153, 0.0066] | negative | 是 |
| `safety_minus_rvl_context_off_survival_off` | `pose_bad_cull` | -0.0057 | [-0.0103, -0.0012] | negative | 否 |
| `safety_minus_rvl_context_off_survival_off` | `aggregate_bad_cull` | 0.0005 | [-0.0022, 0.0035] | positive | 是 |
| `safety_minus_rvl_context_off_survival_off` | `pose_avg_fp_count` | 16.5430 | [-33.3039, 77.7705] | positive | 是 |
| `safety_minus_rvl_context_off_survival_off` | `pose_avg_fn_count` | 2.8341 | [-11.6966, 17.8659] | positive | 是 |
| `safety_minus_rvl_context_off_survival_off` | `pose_avg_tn_count` | -16.5430 | [-79.7863, 33.1896] | negative | 是 |
| `safety_minus_rvl_context_off_survival_off` | `aggregate_avg_fp_count` | 16.5430 | [-33.6166, 79.2406] | positive | 是 |
| `safety_minus_rvl_context_off_survival_off` | `aggregate_avg_fn_count` | 2.8341 | [-11.4508, 17.8812] | positive | 是 |
| `safety_minus_rvl_context_off_survival_off` | `aggregate_avg_tn_count` | -16.5430 | [-79.1126, 32.8125] | negative | 是 |
| `safety_minus_rvl_context_off_survival_off` | `pose_avg_pred_count` | 13.7089 | [-48.1414, 87.8647] | positive | 是 |
| `safety_minus_rvl_context_off_survival_off` | `aggregate_avg_pred_count` | 13.7089 | [-47.5581, 87.6808] | positive | 是 |
| `safety_minus_rvl_context_off_survival_off` | `aggregate_predicted_glb_bytes` | 31096634.2285 | [20449224.1664, 45088471.4977] | positive | 否 |
| `safety_minus_rvl_context_off_survival_off` | `pose_download_utility_recall` | 0.0059 | [0.0040, 0.0085] | positive | 否 |
| `safety_minus_rvl_context_off_survival_off` | `aggregate_download_utility_recall` | 0.0049 | [0.0031, 0.0073] | positive | 否 |
| `safety_minus_rvl_context_off_survival_off` | `pose_glb_bytes_at_achieved_visual_utility` | 2204114.9171 | [1049637.3642, 3683817.3876] | positive | 否 |
| `safety_minus_rvl_context_off_survival_off` | `aggregate_glb_bytes_at_achieved_visual_utility` | 2204114.9171 | [1031569.0778, 3650292.7413] | positive | 否 |
| `safety_minus_rvl_context_off_survival_off` | `aggregate_glb_byte_reduction` | -0.1047 | [-0.1478, -0.0746] | negative | 否 |
| `safety_minus_rvl_context_off_survival_off` | `pose_image_per` | -0.0099 | [-0.0139, -0.0071] | negative | 否 |
| `safety_minus_rvl_context_off_survival_off` | `aggregate_image_per` | -0.0090 | [-0.0133, -0.0061] | negative | 否 |
| `safety_minus_rvl_context_off_survival_off` | `pose_image_miss_pixel_rate` | -0.0049 | [-0.0080, -0.0028] | negative | 否 |
| `safety_minus_rvl_context_off_survival_off` | `aggregate_image_miss_pixel_rate` | -0.0045 | [-0.0078, -0.0025] | negative | 否 |
| `safety_minus_rvl_context_off_survival_off` | `pose_image_wrong_id_pixel_rate` | -0.0050 | [-0.0067, -0.0035] | negative | 否 |
| `safety_minus_rvl_context_off_survival_off` | `aggregate_image_extra_pixel_rate` | 0.0000 | [0.0000, 0.0000] | zero | 是 |
| `safety_minus_rvl_context_off_survival_on` | `pose_precision` | -0.2096 | [-0.2307, -0.1858] | negative | 否 |
| `safety_minus_rvl_context_off_survival_on` | `aggregate_precision` | -0.4143 | [-0.4766, -0.3651] | negative | 否 |
| `safety_minus_rvl_context_off_survival_on` | `pose_recall` | 0.1276 | [0.1004, 0.1558] | positive | 否 |
| `safety_minus_rvl_context_off_survival_on` | `aggregate_recall` | 0.2353 | [0.0996, 0.4044] | positive | 否 |
| `safety_minus_rvl_context_off_survival_on` | `pose_weighted_recall` | 0.0078 | [0.0053, 0.0107] | positive | 否 |
| `safety_minus_rvl_context_off_survival_on` | `aggregate_weighted_recall` | 0.0072 | [0.0050, 0.0100] | positive | 否 |
| `safety_minus_rvl_context_off_survival_on` | `pose_accuracy` | -0.2275 | [-0.2656, -0.1902] | negative | 否 |
| `safety_minus_rvl_context_off_survival_on` | `aggregate_accuracy` | -0.3469 | [-0.3897, -0.2961] | negative | 否 |
| `safety_minus_rvl_context_off_survival_on` | `pose_balanced_accuracy` | -0.0787 | [-0.0970, -0.0595] | negative | 否 |
| `safety_minus_rvl_context_off_survival_on` | `aggregate_balanced_accuracy` | -0.0693 | [-0.1406, 0.0386] | negative | 是 |
| `safety_minus_rvl_context_off_survival_on` | `pose_f1` | -0.1984 | [-0.2216, -0.1766] | negative | 否 |
| `safety_minus_rvl_context_off_survival_on` | `aggregate_f1` | -0.4016 | [-0.4407, -0.3538] | negative | 否 |
| `safety_minus_rvl_context_off_survival_on` | `pose_useful_cull` | -0.2472 | [-0.2852, -0.2054] | negative | 否 |
| `safety_minus_rvl_context_off_survival_on` | `aggregate_useful_cull` | -0.3574 | [-0.3986, -0.3114] | negative | 否 |
| `safety_minus_rvl_context_off_survival_on` | `pose_bad_cull` | -0.0198 | [-0.0295, -0.0118] | negative | 否 |
| `safety_minus_rvl_context_off_survival_on` | `aggregate_bad_cull` | -0.0104 | [-0.0182, -0.0044] | negative | 否 |
| `safety_minus_rvl_context_off_survival_on` | `pose_avg_fp_count` | 1849.0955 | [1552.2993, 2161.9264] | positive | 否 |
| `safety_minus_rvl_context_off_survival_on` | `pose_avg_fn_count` | -53.8889 | [-94.1399, -22.8016] | negative | 否 |
| `safety_minus_rvl_context_off_survival_on` | `pose_avg_tn_count` | -1849.0955 | [-2163.6546, -1542.9279] | negative | 否 |
| `safety_minus_rvl_context_off_survival_on` | `aggregate_avg_fp_count` | 1849.0955 | [1553.8788, 2153.6880] | positive | 否 |
| `safety_minus_rvl_context_off_survival_on` | `aggregate_avg_fn_count` | -53.8889 | [-94.9543, -22.9156] | negative | 否 |
| `safety_minus_rvl_context_off_survival_on` | `aggregate_avg_tn_count` | -1849.0955 | [-2160.5368, -1548.2026] | negative | 否 |
| `safety_minus_rvl_context_off_survival_on` | `pose_avg_pred_count` | 1902.9844 | [1616.2534, 2202.4885] | positive | 否 |
| `safety_minus_rvl_context_off_survival_on` | `aggregate_avg_pred_count` | 1902.9844 | [1620.5912, 2203.8498] | positive | 否 |
| `safety_minus_rvl_context_off_survival_on` | `aggregate_predicted_glb_bytes` | 85566248.7449 | [57468793.1800, 111613538.1873] | positive | 否 |
| `safety_minus_rvl_context_off_survival_on` | `pose_download_utility_recall` | 0.0078 | [0.0052, 0.0107] | positive | 否 |
| `safety_minus_rvl_context_off_survival_on` | `aggregate_download_utility_recall` | 0.0071 | [0.0049, 0.0099] | positive | 否 |
| `safety_minus_rvl_context_off_survival_on` | `pose_glb_bytes_at_achieved_visual_utility` | 10193597.0391 | [2082112.5485, 18264397.0412] | positive | 否 |
| `safety_minus_rvl_context_off_survival_on` | `aggregate_glb_bytes_at_achieved_visual_utility` | 10193597.0391 | [2004680.1006, 18221676.0706] | positive | 否 |
| `safety_minus_rvl_context_off_survival_on` | `aggregate_glb_byte_reduction` | -0.3142 | [-0.3922, -0.1992] | negative | 否 |
| `safety_minus_rvl_context_off_survival_on` | `pose_image_per` | -0.0101 | [-0.0176, -0.0021] | negative | 否 |
| `safety_minus_rvl_context_off_survival_on` | `aggregate_image_per` | -0.0095 | [-0.0172, -0.0014] | negative | 否 |
| `safety_minus_rvl_context_off_survival_on` | `pose_image_miss_pixel_rate` | -0.0076 | [-0.0148, 0.0005] | negative | 是 |
| `safety_minus_rvl_context_off_survival_on` | `aggregate_image_miss_pixel_rate` | -0.0076 | [-0.0151, 0.0005] | negative | 是 |
| `safety_minus_rvl_context_off_survival_on` | `pose_image_wrong_id_pixel_rate` | -0.0025 | [-0.0037, -0.0012] | negative | 否 |
| `safety_minus_rvl_context_off_survival_on` | `aggregate_image_extra_pixel_rate` | 0.0000 | [0.0000, 0.0000] | zero | 是 |
| `safety_minus_rvl_context_on_survival_off` | `pose_precision` | -0.0710 | [-0.0833, -0.0583] | negative | 否 |
| `safety_minus_rvl_context_on_survival_off` | `aggregate_precision` | -0.0434 | [-0.0742, -0.0167] | negative | 否 |
| `safety_minus_rvl_context_on_survival_off` | `pose_recall` | 0.0560 | [0.0378, 0.0749] | positive | 否 |
| `safety_minus_rvl_context_on_survival_off` | `aggregate_recall` | -0.0174 | [-0.0774, 0.0456] | negative | 是 |
| `safety_minus_rvl_context_on_survival_off` | `pose_weighted_recall` | 0.0052 | [0.0038, 0.0068] | positive | 否 |
| `safety_minus_rvl_context_on_survival_off` | `aggregate_weighted_recall` | 0.0040 | [0.0027, 0.0056] | positive | 否 |
| `safety_minus_rvl_context_on_survival_off` | `pose_accuracy` | -0.0439 | [-0.0537, -0.0346] | negative | 否 |
| `safety_minus_rvl_context_on_survival_off` | `aggregate_accuracy` | -0.0120 | [-0.0207, -0.0023] | negative | 否 |
| `safety_minus_rvl_context_on_survival_off` | `pose_balanced_accuracy` | -0.0017 | [-0.0090, 0.0057] | negative | 是 |
| `safety_minus_rvl_context_on_survival_off` | `aggregate_balanced_accuracy` | -0.0146 | [-0.0404, 0.0123] | negative | 是 |
| `safety_minus_rvl_context_on_survival_off` | `pose_f1` | -0.0701 | [-0.0867, -0.0556] | negative | 否 |
| `safety_minus_rvl_context_on_survival_off` | `aggregate_f1` | -0.0434 | [-0.0690, -0.0212] | negative | 否 |
| `safety_minus_rvl_context_on_survival_off` | `pose_useful_cull` | -0.0484 | [-0.0587, -0.0372] | negative | 否 |
| `safety_minus_rvl_context_on_survival_off` | `aggregate_useful_cull` | -0.0112 | [-0.0218, 0.0001] | negative | 是 |
| `safety_minus_rvl_context_on_survival_off` | `pose_bad_cull` | -0.0045 | [-0.0089, -0.0003] | negative | 否 |
| `safety_minus_rvl_context_on_survival_off` | `aggregate_bad_cull` | 0.0008 | [-0.0019, 0.0037] | positive | 是 |
| `safety_minus_rvl_context_on_survival_off` | `pose_avg_fp_count` | 57.8983 | [-2.9217, 117.2163] | positive | 是 |
| `safety_minus_rvl_context_on_survival_off` | `pose_avg_fn_count` | 3.9890 | [-9.8122, 19.1182] | positive | 是 |
| `safety_minus_rvl_context_on_survival_off` | `pose_avg_tn_count` | -57.8983 | [-115.6653, 0.7200] | negative | 是 |
| `safety_minus_rvl_context_on_survival_off` | `aggregate_avg_fp_count` | 57.8983 | [-2.2177, 114.6384] | positive | 是 |
| `safety_minus_rvl_context_on_survival_off` | `aggregate_avg_fn_count` | 3.9890 | [-9.8842, 18.6156] | positive | 是 |
| `safety_minus_rvl_context_on_survival_off` | `aggregate_avg_tn_count` | -57.8983 | [-116.1308, 0.2077] | negative | 是 |
| `safety_minus_rvl_context_on_survival_off` | `pose_avg_pred_count` | 53.9092 | [-16.9776, 124.7955] | positive | 是 |
| `safety_minus_rvl_context_on_survival_off` | `aggregate_avg_pred_count` | 53.9092 | [-15.6929, 123.3450] | positive | 是 |
| `safety_minus_rvl_context_on_survival_off` | `aggregate_predicted_glb_bytes` | 40840412.3130 | [33088422.0457, 49815276.8227] | positive | 否 |
| `safety_minus_rvl_context_on_survival_off` | `pose_download_utility_recall` | 0.0054 | [0.0040, 0.0070] | positive | 否 |
| `safety_minus_rvl_context_on_survival_off` | `aggregate_download_utility_recall` | 0.0043 | [0.0030, 0.0058] | positive | 否 |
| `safety_minus_rvl_context_on_survival_off` | `pose_glb_bytes_at_achieved_visual_utility` | 2103234.1158 | [704440.1408, 3513485.5906] | positive | 否 |
| `safety_minus_rvl_context_on_survival_off` | `aggregate_glb_bytes_at_achieved_visual_utility` | 2103234.1158 | [683657.0624, 3488999.3914] | positive | 否 |
| `safety_minus_rvl_context_on_survival_off` | `aggregate_glb_byte_reduction` | -0.1345 | [-0.1581, -0.1146] | negative | 否 |
| `safety_minus_rvl_context_on_survival_off` | `pose_image_per` | -0.0088 | [-0.0112, -0.0067] | negative | 否 |
| `safety_minus_rvl_context_on_survival_off` | `aggregate_image_per` | -0.0078 | [-0.0102, -0.0058] | negative | 否 |
| `safety_minus_rvl_context_on_survival_off` | `pose_image_miss_pixel_rate` | -0.0045 | [-0.0059, -0.0032] | negative | 否 |
| `safety_minus_rvl_context_on_survival_off` | `aggregate_image_miss_pixel_rate` | -0.0040 | [-0.0053, -0.0028] | negative | 否 |
| `safety_minus_rvl_context_on_survival_off` | `pose_image_wrong_id_pixel_rate` | -0.0044 | [-0.0062, -0.0028] | negative | 否 |
| `safety_minus_rvl_context_on_survival_off` | `aggregate_image_extra_pixel_rate` | 0.0000 | [0.0000, 0.0000] | zero | 是 |
| `safety_minus_rvl_context_on_survival_on` | `pose_precision` | -0.2070 | [-0.2257, -0.1886] | negative | 否 |
| `safety_minus_rvl_context_on_survival_on` | `aggregate_precision` | -0.3937 | [-0.4298, -0.3608] | negative | 否 |
| `safety_minus_rvl_context_on_survival_on` | `pose_recall` | 0.1256 | [0.1052, 0.1494] | positive | 否 |
| `safety_minus_rvl_context_on_survival_on` | `aggregate_recall` | 0.2132 | [0.1357, 0.3115] | positive | 否 |
| `safety_minus_rvl_context_on_survival_on` | `pose_weighted_recall` | 0.0075 | [0.0054, 0.0098] | positive | 否 |
| `safety_minus_rvl_context_on_survival_on` | `aggregate_weighted_recall` | 0.0067 | [0.0047, 0.0090] | positive | 否 |
| `safety_minus_rvl_context_on_survival_on` | `pose_accuracy` | -0.2098 | [-0.2360, -0.1798] | negative | 否 |
| `safety_minus_rvl_context_on_survival_on` | `aggregate_accuracy` | -0.3162 | [-0.3715, -0.2390] | negative | 否 |
| `safety_minus_rvl_context_on_survival_on` | `pose_balanced_accuracy` | -0.0711 | [-0.0867, -0.0533] | negative | 否 |
| `safety_minus_rvl_context_on_survival_on` | `aggregate_balanced_accuracy` | -0.0637 | [-0.1250, 0.0205] | negative | 是 |
| `safety_minus_rvl_context_on_survival_on` | `pose_f1` | -0.1946 | [-0.2136, -0.1762] | negative | 否 |
| `safety_minus_rvl_context_on_survival_on` | `aggregate_f1` | -0.3940 | [-0.4355, -0.3489] | negative | 否 |
| `safety_minus_rvl_context_on_survival_on` | `pose_useful_cull` | -0.2297 | [-0.2539, -0.2021] | negative | 否 |
| `safety_minus_rvl_context_on_survival_on` | `aggregate_useful_cull` | -0.3256 | [-0.3777, -0.2500] | negative | 否 |
| `safety_minus_rvl_context_on_survival_on` | `pose_bad_cull` | -0.0199 | [-0.0251, -0.0151] | negative | 否 |
| `safety_minus_rvl_context_on_survival_on` | `aggregate_bad_cull` | -0.0094 | [-0.0139, -0.0060] | negative | 否 |
| `safety_minus_rvl_context_on_survival_on` | `pose_avg_fp_count` | 1684.8232 | [1290.3964, 2052.2616] | positive | 否 |
| `safety_minus_rvl_context_on_survival_on` | `pose_avg_fn_count` | -48.8357 | [-72.1360, -30.5790] | negative | 否 |
| `safety_minus_rvl_context_on_survival_on` | `pose_avg_tn_count` | -1684.8232 | [-2062.1699, -1284.7719] | negative | 否 |
| `safety_minus_rvl_context_on_survival_on` | `aggregate_avg_fp_count` | 1684.8232 | [1295.1757, 2049.8626] | positive | 否 |
| `safety_minus_rvl_context_on_survival_on` | `aggregate_avg_fn_count` | -48.8357 | [-72.4781, -30.5649] | negative | 否 |
| `safety_minus_rvl_context_on_survival_on` | `aggregate_avg_tn_count` | -1684.8232 | [-2060.4322, -1294.0849] | negative | 否 |
| `safety_minus_rvl_context_on_survival_on` | `pose_avg_pred_count` | 1733.6588 | [1355.1190, 2087.7388] | positive | 否 |
| `safety_minus_rvl_context_on_survival_on` | `aggregate_avg_pred_count` | 1733.6588 | [1354.4818, 2091.8426] | positive | 否 |
| `safety_minus_rvl_context_on_survival_on` | `aggregate_predicted_glb_bytes` | 81181335.5055 | [65109016.0953, 101254009.3244] | positive | 否 |
| `safety_minus_rvl_context_on_survival_on` | `pose_download_utility_recall` | 0.0075 | [0.0054, 0.0097] | positive | 否 |
| `safety_minus_rvl_context_on_survival_on` | `aggregate_download_utility_recall` | 0.0066 | [0.0046, 0.0089] | positive | 否 |
| `safety_minus_rvl_context_on_survival_on` | `pose_glb_bytes_at_achieved_visual_utility` | 9941251.2426 | [4394162.6552, 16557992.0243] | positive | 否 |
| `safety_minus_rvl_context_on_survival_on` | `aggregate_glb_bytes_at_achieved_visual_utility` | 9941251.2426 | [4348570.8059, 16514351.3485] | positive | 否 |
| `safety_minus_rvl_context_on_survival_on` | `aggregate_glb_byte_reduction` | -0.2979 | [-0.3521, -0.2536] | negative | 否 |
| `safety_minus_rvl_context_on_survival_on` | `pose_image_per` | -0.0093 | [-0.0122, -0.0068] | negative | 否 |
| `safety_minus_rvl_context_on_survival_on` | `aggregate_image_per` | -0.0088 | [-0.0121, -0.0061] | negative | 否 |
| `safety_minus_rvl_context_on_survival_on` | `pose_image_miss_pixel_rate` | -0.0054 | [-0.0087, -0.0031] | negative | 否 |
| `safety_minus_rvl_context_on_survival_on` | `aggregate_image_miss_pixel_rate` | -0.0053 | [-0.0092, -0.0029] | negative | 否 |
| `safety_minus_rvl_context_on_survival_on` | `pose_image_wrong_id_pixel_rate` | -0.0039 | [-0.0063, -0.0015] | negative | 否 |
| `safety_minus_rvl_context_on_survival_on` | `aggregate_image_extra_pixel_rate` | 0.0000 | [0.0000, 0.0000] | zero | 是 |
| `context_main_effect` | `pose_precision` | -0.0080 | [-0.0186, -0.0004] | negative | 否 |
| `context_main_effect` | `aggregate_precision` | -0.0224 | [-0.0348, -0.0103] | negative | 否 |
| `context_main_effect` | `pose_recall` | 0.0120 | [0.0025, 0.0250] | positive | 否 |
| `context_main_effect` | `aggregate_recall` | 0.0384 | [0.0007, 0.0680] | positive | 否 |
| `context_main_effect` | `pose_weighted_recall` | 0.0009 | [0.0001, 0.0020] | positive | 否 |
| `context_main_effect` | `aggregate_weighted_recall` | 0.0010 | [0.0002, 0.0021] | positive | 否 |
| `context_main_effect` | `pose_accuracy` | -0.0021 | [-0.0175, 0.0079] | negative | 是 |
| `context_main_effect` | `aggregate_accuracy` | 0.0006 | [-0.0141, 0.0093] | positive | 是 |
| `context_main_effect` | `pose_balanced_accuracy` | 0.0026 | [-0.0014, 0.0056] | positive | 是 |
| `context_main_effect` | `aggregate_balanced_accuracy` | 0.0186 | [0.0048, 0.0291] | positive | 否 |
| `context_main_effect` | `pose_f1` | -0.0035 | [-0.0111, 0.0048] | negative | 是 |
| `context_main_effect` | `aggregate_f1` | -0.0062 | [-0.0161, 0.0055] | negative | 是 |
| `context_main_effect` | `pose_useful_cull` | -0.0042 | [-0.0215, 0.0059] | negative | 是 |
| `context_main_effect` | `aggregate_useful_cull` | -0.0011 | [-0.0169, 0.0087] | negative | 是 |
| `context_main_effect` | `pose_bad_cull` | -0.0020 | [-0.0043, 0.0009] | negative | 是 |
| `context_main_effect` | `aggregate_bad_cull` | -0.0017 | [-0.0031, -0.0000] | negative | 否 |
| `context_main_effect` | `pose_avg_fp_count` | 5.5767 | [-45.9484, 86.9488] | positive | 是 |
| `context_main_effect` | `pose_avg_fn_count` | -8.7938 | [-16.2697, -0.1748] | negative | 否 |
| `context_main_effect` | `pose_avg_tn_count` | -5.5767 | [-86.9488, 45.9484] | negative | 是 |
| `context_main_effect` | `aggregate_avg_fp_count` | 5.5767 | [-45.9484, 86.9488] | positive | 是 |
| `context_main_effect` | `aggregate_avg_fn_count` | -8.7938 | [-16.2697, -0.1748] | negative | 否 |
| `context_main_effect` | `aggregate_avg_tn_count` | -5.5767 | [-86.9488, 45.9484] | negative | 是 |
| `context_main_effect` | `pose_avg_pred_count` | 14.3705 | [-44.3866, 101.7638] | positive | 是 |
| `context_main_effect` | `aggregate_avg_pred_count` | 14.3705 | [-44.3866, 101.7638] | positive | 是 |
| `context_main_effect` | `aggregate_predicted_glb_bytes` | 3891886.5743 | [-741026.7644, 11491946.8728] | positive | 是 |
| `context_main_effect` | `pose_download_utility_recall` | 0.0009 | [0.0001, 0.0019] | positive | 否 |
| `context_main_effect` | `aggregate_download_utility_recall` | 0.0010 | [0.0002, 0.0020] | positive | 否 |
| `context_main_effect` | `pose_glb_bytes_at_achieved_visual_utility` | 936320.0986 | [-427953.8619, 2174733.2720] | positive | 是 |
| `context_main_effect` | `aggregate_glb_bytes_at_achieved_visual_utility` | 936320.0986 | [-427953.8619, 2174733.2720] | positive | 是 |
| `context_main_effect` | `aggregate_glb_byte_reduction` | -0.0102 | [-0.0385, 0.0083] | negative | 是 |
| `context_main_effect` | `pose_image_per` | -0.0010 | [-0.0026, 0.0009] | negative | 是 |
| `context_main_effect` | `aggregate_image_per` | -0.0010 | [-0.0026, 0.0009] | negative | 是 |
| `context_main_effect` | `pose_image_miss_pixel_rate` | -0.0008 | [-0.0025, 0.0012] | negative | 是 |
| `context_main_effect` | `aggregate_image_miss_pixel_rate` | -0.0008 | [-0.0026, 0.0013] | negative | 是 |
| `context_main_effect` | `pose_image_wrong_id_pixel_rate` | -0.0002 | [-0.0009, 0.0005] | negative | 是 |
| `context_main_effect` | `aggregate_image_extra_pixel_rate` | 0.0000 | [0.0000, 0.0000] | zero | 是 |
| `survival_main_effect` | `pose_precision` | 0.0176 | [0.0053, 0.0290] | positive | 否 |
| `survival_main_effect` | `aggregate_precision` | -0.0410 | [-0.0777, 0.0087] | negative | 是 |
| `survival_main_effect` | `pose_recall` | 0.0187 | [0.0027, 0.0365] | positive | 否 |
| `survival_main_effect` | `aggregate_recall` | 0.1253 | [0.0912, 0.1663] | positive | 否 |
| `survival_main_effect` | `pose_weighted_recall` | 0.0016 | [0.0004, 0.0028] | positive | 否 |
| `survival_main_effect` | `aggregate_weighted_recall` | 0.0017 | [0.0005, 0.0031] | positive | 否 |
| `survival_main_effect` | `pose_accuracy` | -0.0083 | [-0.0330, 0.0138] | negative | 是 |
| `survival_main_effect` | `aggregate_accuracy` | -0.1376 | [-0.1648, -0.0959] | negative | 否 |
| `survival_main_effect` | `pose_balanced_accuracy` | 0.0164 | [0.0059, 0.0265] | positive | 否 |
| `survival_main_effect` | `aggregate_balanced_accuracy` | -0.0122 | [-0.0278, 0.0031] | negative | 是 |
| `survival_main_effect` | `pose_f1` | 0.0150 | [0.0069, 0.0230] | positive | 否 |
| `survival_main_effect` | `aggregate_f1` | -0.0739 | [-0.0984, -0.0459] | negative | 否 |
| `survival_main_effect` | `pose_useful_cull` | -0.0085 | [-0.0364, 0.0164] | negative | 是 |
| `survival_main_effect` | `aggregate_useful_cull` | -0.1431 | [-0.1706, -0.1002] | negative | 否 |
| `survival_main_effect` | `pose_bad_cull` | -0.0002 | [-0.0046, 0.0039] | negative | 是 |
| `survival_main_effect` | `aggregate_bad_cull` | -0.0055 | [-0.0078, -0.0038] | negative | 否 |
| `survival_main_effect` | `pose_avg_fp_count` | 740.6307 | [518.6636, 931.1737] | positive | 否 |
| `survival_main_effect` | `pose_avg_fn_count` | -28.7062 | [-40.3511, -19.6384] | negative | 否 |
| `survival_main_effect` | `pose_avg_tn_count` | -740.6307 | [-931.1737, -518.6636] | negative | 否 |
| `survival_main_effect` | `aggregate_avg_fp_count` | 740.6307 | [518.6636, 931.1737] | positive | 否 |
| `survival_main_effect` | `aggregate_avg_fn_count` | -28.7062 | [-40.3511, -19.6384] | negative | 否 |
| `survival_main_effect` | `aggregate_avg_tn_count` | -740.6307 | [-931.1737, -518.6636] | negative | 否 |
| `survival_main_effect` | `pose_avg_pred_count` | 769.3369 | [542.2986, 964.2370] | positive | 否 |
| `survival_main_effect` | `aggregate_avg_pred_count` | 769.3369 | [542.2986, 964.2370] | positive | 否 |
| `survival_main_effect` | `aggregate_predicted_glb_bytes` | 18426243.9014 | [8103002.5148, 34708707.4608] | positive | 否 |
| `survival_main_effect` | `pose_download_utility_recall` | 0.0014 | [0.0003, 0.0027] | positive | 否 |
| `survival_main_effect` | `aggregate_download_utility_recall` | 0.0015 | [0.0004, 0.0029] | positive | 否 |
| `survival_main_effect` | `pose_glb_bytes_at_achieved_visual_utility` | 5629164.6025 | [3006916.4852, 8178242.8966] | positive | 否 |
| `survival_main_effect` | `aggregate_glb_bytes_at_achieved_visual_utility` | 5629164.6025 | [3006916.4852, 8178242.8966] | positive | 否 |
| `survival_main_effect` | `aggregate_glb_byte_reduction` | -0.0003 | [-0.0518, 0.0341] | negative | 是 |
| `survival_main_effect` | `pose_image_per` | -0.0008 | [-0.0021, 0.0004] | negative | 是 |
| `survival_main_effect` | `aggregate_image_per` | -0.0007 | [-0.0021, 0.0006] | negative | 是 |
| `survival_main_effect` | `pose_image_miss_pixel_rate` | 0.0007 | [-0.0002, 0.0019] | positive | 是 |
| `survival_main_effect` | `aggregate_image_miss_pixel_rate` | 0.0009 | [-0.0002, 0.0022] | positive | 是 |
| `survival_main_effect` | `pose_image_wrong_id_pixel_rate` | -0.0015 | [-0.0028, 0.0000] | negative | 是 |
| `survival_main_effect` | `aggregate_image_extra_pixel_rate` | 0.0000 | [0.0000, 0.0000] | zero | 是 |
| `loss_main_effect` | `pose_precision` | -0.1368 | [-0.1528, -0.1206] | negative | 否 |
| `loss_main_effect` | `aggregate_precision` | -0.2169 | [-0.2504, -0.1890] | negative | 否 |
| `loss_main_effect` | `pose_recall` | 0.0925 | [0.0727, 0.1122] | positive | 否 |
| `loss_main_effect` | `aggregate_recall` | 0.1047 | [0.0315, 0.1859] | positive | 否 |
| `loss_main_effect` | `pose_weighted_recall` | 0.0066 | [0.0051, 0.0083] | positive | 否 |
| `loss_main_effect` | `aggregate_weighted_recall` | 0.0057 | [0.0042, 0.0073] | positive | 否 |
| `loss_main_effect` | `pose_accuracy` | -0.1293 | [-0.1429, -0.1169] | negative | 否 |
| `loss_main_effect` | `aggregate_accuracy` | -0.1697 | [-0.1902, -0.1402] | negative | 否 |
| `loss_main_effect` | `pose_balanced_accuracy` | -0.0371 | [-0.0447, -0.0295] | negative | 否 |
| `loss_main_effect` | `aggregate_balanced_accuracy` | -0.0389 | [-0.0805, 0.0127] | negative | 是 |
| `loss_main_effect` | `pose_f1` | -0.1291 | [-0.1435, -0.1147] | negative | 否 |
| `loss_main_effect` | `aggregate_f1` | -0.2144 | [-0.2388, -0.1903] | negative | 否 |
| `loss_main_effect` | `pose_useful_cull` | -0.1418 | [-0.1558, -0.1278] | negative | 否 |
| `loss_main_effect` | `aggregate_useful_cull` | -0.1743 | [-0.1942, -0.1480] | negative | 否 |
| `loss_main_effect` | `pose_bad_cull` | -0.0124 | [-0.0171, -0.0080] | negative | 否 |
| `loss_main_effect` | `aggregate_bad_cull` | -0.0046 | [-0.0082, -0.0013] | negative | 否 |
| `loss_main_effect` | `pose_avg_fp_count` | 902.0900 | [740.9668, 1059.0454] | positive | 否 |
| `loss_main_effect` | `pose_avg_fn_count` | -23.9754 | [-43.0887, -6.9349] | negative | 否 |
| `loss_main_effect` | `pose_avg_tn_count` | -902.0900 | [-1059.0454, -740.9668] | negative | 否 |
| `loss_main_effect` | `aggregate_avg_fp_count` | 902.0900 | [740.9668, 1059.0454] | positive | 否 |
| `loss_main_effect` | `aggregate_avg_fn_count` | -23.9754 | [-43.0887, -6.9349] | negative | 否 |
| `loss_main_effect` | `aggregate_avg_tn_count` | -902.0900 | [-1059.0454, -740.9668] | negative | 否 |
| `loss_main_effect` | `pose_avg_pred_count` | 926.0653 | [772.2835, 1079.1325] | positive | 否 |
| `loss_main_effect` | `aggregate_avg_pred_count` | 926.0653 | [772.2835, 1079.1325] | positive | 否 |
| `loss_main_effect` | `aggregate_predicted_glb_bytes` | 59671157.6980 | [46604649.5848, 72543431.8920] | positive | 否 |
| `loss_main_effect` | `pose_download_utility_recall` | 0.0066 | [0.0051, 0.0083] | positive | 否 |
| `loss_main_effect` | `aggregate_download_utility_recall` | 0.0057 | [0.0042, 0.0074] | positive | 否 |
| `loss_main_effect` | `pose_glb_bytes_at_achieved_visual_utility` | 6110549.3286 | [2310990.2348, 10353015.1780] | positive | 否 |
| `loss_main_effect` | `aggregate_glb_bytes_at_achieved_visual_utility` | 6110549.3286 | [2310990.2348, 10353015.1780] | positive | 否 |
| `loss_main_effect` | `aggregate_glb_byte_reduction` | -0.2128 | [-0.2474, -0.1700] | negative | 否 |
| `loss_main_effect` | `pose_image_per` | -0.0096 | [-0.0126, -0.0071] | negative | 否 |
| `loss_main_effect` | `aggregate_image_per` | -0.0088 | [-0.0120, -0.0062] | negative | 否 |
| `loss_main_effect` | `pose_image_miss_pixel_rate` | -0.0056 | [-0.0076, -0.0039] | negative | 否 |
| `loss_main_effect` | `aggregate_image_miss_pixel_rate` | -0.0053 | [-0.0074, -0.0036] | negative | 否 |
| `loss_main_effect` | `pose_image_wrong_id_pixel_rate` | -0.0040 | [-0.0054, -0.0027] | negative | 否 |
| `loss_main_effect` | `aggregate_image_extra_pixel_rate` | 0.0000 | [0.0000, 0.0000] | zero | 是 |
| `context_survival_interaction_rvl` | `pose_precision` | 0.0119 | [-0.0060, 0.0252] | positive | 是 |
| `context_survival_interaction_rvl` | `aggregate_precision` | 0.0189 | [-0.0218, 0.0758] | positive | 是 |
| `context_survival_interaction_rvl` | `pose_recall` | -0.0228 | [-0.0439, 0.0049] | negative | 是 |
| `context_survival_interaction_rvl` | `aggregate_recall` | -0.0419 | [-0.1280, 0.0231] | negative | 是 |
| `context_survival_interaction_rvl` | `pose_weighted_recall` | -0.0017 | [-0.0071, 0.0023] | negative | 是 |
| `context_survival_interaction_rvl` | `aggregate_weighted_recall` | -0.0017 | [-0.0067, 0.0019] | negative | 是 |
| `context_survival_interaction_rvl` | `pose_accuracy` | 0.0076 | [-0.0035, 0.0188] | positive | 是 |
| `context_survival_interaction_rvl` | `aggregate_accuracy` | 0.0094 | [-0.0012, 0.0220] | positive | 是 |
| `context_survival_interaction_rvl` | `pose_balanced_accuracy` | -0.0024 | [-0.0149, 0.0069] | negative | 是 |
| `context_survival_interaction_rvl` | `aggregate_balanced_accuracy` | -0.0151 | [-0.0500, 0.0140] | negative | 是 |
| `context_survival_interaction_rvl` | `pose_f1` | 0.0020 | [-0.0086, 0.0140] | positive | 是 |
| `context_survival_interaction_rvl` | `aggregate_f1` | 0.0186 | [-0.0067, 0.0435] | positive | 是 |
| `context_survival_interaction_rvl` | `pose_useful_cull` | 0.0117 | [-0.0006, 0.0263] | positive | 是 |
| `context_survival_interaction_rvl` | `aggregate_useful_cull` | 0.0112 | [-0.0009, 0.0275] | positive | 是 |
| `context_survival_interaction_rvl` | `pose_bad_cull` | 0.0041 | [0.0001, 0.0084] | positive | 否 |
| `context_survival_interaction_rvl` | `aggregate_bad_cull` | 0.0019 | [-0.0011, 0.0057] | positive | 是 |
| `context_survival_interaction_rvl` | `pose_avg_fp_count` | -58.1549 | [-141.5057, 4.4434] | negative | 是 |
| `context_survival_interaction_rvl` | `pose_avg_fn_count` | 9.5931 | [-5.6952, 29.7327] | positive | 是 |
| `context_survival_interaction_rvl` | `pose_avg_tn_count` | 58.1549 | [-4.4434, 141.5057] | positive | 是 |
| `context_survival_interaction_rvl` | `aggregate_avg_fp_count` | -58.1549 | [-141.5057, 4.4434] | negative | 是 |
| `context_survival_interaction_rvl` | `aggregate_avg_fn_count` | 9.5931 | [-5.6952, 29.7327] | positive | 是 |
| `context_survival_interaction_rvl` | `aggregate_avg_tn_count` | 58.1549 | [-4.4434, 141.5057] | positive | 是 |
| `context_survival_interaction_rvl` | `pose_avg_pred_count` | -67.7480 | [-170.0599, 2.7327] | negative | 是 |
| `context_survival_interaction_rvl` | `aggregate_avg_pred_count` | -67.7480 | [-170.0599, 2.7327] | negative | 是 |
| `context_survival_interaction_rvl` | `aggregate_predicted_glb_bytes` | -3990199.6682 | [-7836415.8023, -655520.7224] | negative | 否 |
| `context_survival_interaction_rvl` | `pose_download_utility_recall` | -0.0017 | [-0.0071, 0.0023] | negative | 是 |
| `context_survival_interaction_rvl` | `aggregate_download_utility_recall` | -0.0016 | [-0.0066, 0.0019] | negative | 是 |
| `context_survival_interaction_rvl` | `pose_glb_bytes_at_achieved_visual_utility` | -1674104.6135 | [-3322830.0565, -382173.5986] | negative | 否 |
| `context_survival_interaction_rvl` | `aggregate_glb_bytes_at_achieved_visual_utility` | -1674104.6135 | [-3322830.0565, -382173.5986] | negative | 否 |
| `context_survival_interaction_rvl` | `aggregate_glb_byte_reduction` | 0.0107 | [0.0024, 0.0212] | positive | 否 |
| `context_survival_interaction_rvl` | `pose_image_per` | 0.0018 | [-0.0059, 0.0130] | positive | 是 |
| `context_survival_interaction_rvl` | `aggregate_image_per` | 0.0019 | [-0.0056, 0.0130] | positive | 是 |
| `context_survival_interaction_rvl` | `pose_image_miss_pixel_rate` | -0.0012 | [-0.0106, 0.0119] | negative | 是 |
| `context_survival_interaction_rvl` | `aggregate_image_miss_pixel_rate` | -0.0011 | [-0.0108, 0.0120] | negative | 是 |
| `context_survival_interaction_rvl` | `pose_image_wrong_id_pixel_rate` | 0.0029 | [0.0009, 0.0052] | positive | 否 |
| `context_survival_interaction_rvl` | `aggregate_image_extra_pixel_rate` | 0.0000 | [0.0000, 0.0000] | zero | 是 |
| `context_survival_interaction_safety` | `pose_precision` | 0.0256 | [0.0214, 0.0300] | positive | 否 |
| `context_survival_interaction_safety` | `aggregate_precision` | 0.0668 | [0.0490, 0.0857] | positive | 否 |
| `context_survival_interaction_safety` | `pose_recall` | -0.0201 | [-0.0287, -0.0109] | negative | 否 |
| `context_survival_interaction_safety` | `aggregate_recall` | -0.0589 | [-0.1061, -0.0066] | negative | 否 |
| `context_survival_interaction_safety` | `pose_weighted_recall` | -0.0014 | [-0.0021, -0.0007] | negative | 否 |
| `context_survival_interaction_safety` | `aggregate_weighted_recall` | -0.0015 | [-0.0021, -0.0008] | negative | 否 |
| `context_survival_interaction_safety` | `pose_accuracy` | 0.0331 | [-0.0020, 0.0540] | positive | 是 |
| `context_survival_interaction_safety` | `aggregate_accuracy` | 0.0484 | [0.0126, 0.0705] | positive | 否 |
| `context_survival_interaction_safety` | `pose_balanced_accuracy` | 0.0100 | [-0.0122, 0.0242] | positive | 是 |
| `context_survival_interaction_safety` | `aggregate_balanced_accuracy` | -0.0028 | [-0.0414, 0.0307] | negative | 是 |
| `context_survival_interaction_safety` | `pose_f1` | 0.0226 | [0.0153, 0.0291] | positive | 否 |
| `context_survival_interaction_safety` | `aggregate_f1` | 0.0510 | [0.0326, 0.0645] | positive | 否 |
| `context_survival_interaction_safety` | `pose_useful_cull` | 0.0358 | [0.0009, 0.0567] | positive | 否 |
| `context_survival_interaction_safety` | `aggregate_useful_cull` | 0.0510 | [0.0173, 0.0730] | positive | 否 |
| `context_survival_interaction_safety` | `pose_bad_cull` | 0.0028 | [0.0012, 0.0044] | positive | 否 |
| `context_survival_interaction_safety` | `aggregate_bad_cull` | 0.0026 | [0.0003, 0.0049] | positive | 否 |
| `context_survival_interaction_safety` | `pose_avg_fp_count` | -263.7825 | [-389.6218, -89.6648] | negative | 否 |
| `context_survival_interaction_safety` | `pose_avg_fn_count` | 13.4914 | [1.3849, 25.0395] | positive | 否 |
| `context_survival_interaction_safety` | `pose_avg_tn_count` | 263.7825 | [89.6648, 389.6218] | positive | 否 |
| `context_survival_interaction_safety` | `aggregate_avg_fp_count` | -263.7825 | [-389.6218, -89.6648] | negative | 否 |
| `context_survival_interaction_safety` | `aggregate_avg_fn_count` | 13.4914 | [1.3849, 25.0395] | positive | 否 |
| `context_survival_interaction_safety` | `aggregate_avg_tn_count` | 263.7825 | [89.6648, 389.6218] | positive | 否 |
| `context_survival_interaction_safety` | `pose_avg_pred_count` | -277.2739 | [-401.8743, -114.3978] | negative | 否 |
| `context_survival_interaction_safety` | `aggregate_avg_pred_count` | -277.2739 | [-401.8743, -114.3978] | negative | 否 |
| `context_survival_interaction_safety` | `aggregate_predicted_glb_bytes` | -18118890.9922 | [-24347424.2014, -10770257.6017] | negative | 否 |
| `context_survival_interaction_safety` | `pose_download_utility_recall` | -0.0014 | [-0.0021, -0.0008] | negative | 否 |
| `context_survival_interaction_safety` | `aggregate_download_utility_recall` | -0.0015 | [-0.0021, -0.0009] | negative | 否 |
| `context_survival_interaction_safety` | `pose_glb_bytes_at_achieved_visual_utility` | -1825569.6088 | [-3751970.1653, -455221.2900] | negative | 否 |
| `context_survival_interaction_safety` | `aggregate_glb_bytes_at_achieved_visual_utility` | -1825569.6088 | [-3751970.1653, -455221.2900] | negative | 否 |
| `context_survival_interaction_safety` | `aggregate_glb_byte_reduction` | 0.0569 | [0.0075, 0.0941] | positive | 否 |
| `context_survival_interaction_safety` | `pose_image_per` | 0.0015 | [0.0009, 0.0021] | positive | 否 |
| `context_survival_interaction_safety` | `aggregate_image_per` | 0.0014 | [0.0009, 0.0020] | positive | 否 |
| `context_survival_interaction_safety` | `pose_image_miss_pixel_rate` | 0.0006 | [0.0002, 0.0013] | positive | 否 |
| `context_survival_interaction_safety` | `aggregate_image_miss_pixel_rate` | 0.0006 | [0.0002, 0.0013] | positive | 否 |
| `context_survival_interaction_safety` | `pose_image_wrong_id_pixel_rate` | 0.0009 | [0.0003, 0.0015] | positive | 否 |
| `context_survival_interaction_safety` | `aggregate_image_extra_pixel_rate` | 0.0000 | [0.0000, 0.0000] | zero | 是 |
| `context_loss_interaction_off` | `pose_precision` | -0.0112 | [-0.0216, -0.0012] | negative | 否 |
| `context_loss_interaction_off` | `aggregate_precision` | -0.0272 | [-0.0647, 0.0027] | negative | 是 |
| `context_loss_interaction_off` | `pose_recall` | -0.0047 | [-0.0319, 0.0185] | negative | 是 |
| `context_loss_interaction_off` | `aggregate_recall` | -0.0050 | [-0.0824, 0.0554] | negative | 是 |
| `context_loss_interaction_off` | `pose_weighted_recall` | -0.0006 | [-0.0034, 0.0015] | negative | 是 |
| `context_loss_interaction_off` | `aggregate_weighted_recall` | -0.0007 | [-0.0034, 0.0011] | negative | 是 |
| `context_loss_interaction_off` | `pose_accuracy` | -0.0078 | [-0.0179, 0.0016] | negative | 是 |
| `context_loss_interaction_off` | `aggregate_accuracy` | -0.0082 | [-0.0200, 0.0018] | negative | 是 |
| `context_loss_interaction_off` | `pose_balanced_accuracy` | -0.0048 | [-0.0104, 0.0003] | negative | 是 |
| `context_loss_interaction_off` | `aggregate_balanced_accuracy` | -0.0067 | [-0.0387, 0.0176] | negative | 是 |
| `context_loss_interaction_off` | `pose_f1` | -0.0168 | [-0.0222, -0.0121] | negative | 否 |
| `context_loss_interaction_off` | `aggregate_f1` | -0.0248 | [-0.0483, -0.0022] | negative | 否 |
| `context_loss_interaction_off` | `pose_useful_cull` | -0.0065 | [-0.0202, 0.0068] | negative | 是 |
| `context_loss_interaction_off` | `aggregate_useful_cull` | -0.0080 | [-0.0223, 0.0052] | negative | 是 |
| `context_loss_interaction_off` | `pose_bad_cull` | 0.0012 | [-0.0026, 0.0057] | positive | 是 |
| `context_loss_interaction_off` | `aggregate_bad_cull` | 0.0002 | [-0.0025, 0.0036] | positive | 是 |
| `context_loss_interaction_off` | `pose_avg_fp_count` | 41.3552 | [-25.9754, 116.4888] | positive | 是 |
| `context_loss_interaction_off` | `pose_avg_fn_count` | 1.1549 | [-12.9267, 18.7405] | positive | 是 |
| `context_loss_interaction_off` | `pose_avg_tn_count` | -41.3552 | [-116.4888, 25.9754] | negative | 是 |
| `context_loss_interaction_off` | `aggregate_avg_fp_count` | 41.3552 | [-25.9754, 116.4888] | positive | 是 |
| `context_loss_interaction_off` | `aggregate_avg_fn_count` | 1.1549 | [-12.9267, 18.7405] | positive | 是 |
| `context_loss_interaction_off` | `aggregate_avg_tn_count` | -41.3552 | [-116.4888, 25.9754] | negative | 是 |
| `context_loss_interaction_off` | `pose_avg_pred_count` | 40.2003 | [-43.9442, 128.6585] | positive | 是 |
| `context_loss_interaction_off` | `aggregate_avg_pred_count` | 40.2003 | [-43.9442, 128.6585] | positive | 是 |
| `context_loss_interaction_off` | `aggregate_predicted_glb_bytes` | 9743778.0845 | [2263066.4067, 16207371.5814] | positive | 否 |
| `context_loss_interaction_off` | `pose_download_utility_recall` | -0.0005 | [-0.0032, 0.0015] | negative | 是 |
| `context_loss_interaction_off` | `aggregate_download_utility_recall` | -0.0006 | [-0.0031, 0.0012] | negative | 是 |
| `context_loss_interaction_off` | `pose_glb_bytes_at_achieved_visual_utility` | -100880.8013 | [-1308620.9585, 1140082.8604] | negative | 是 |
| `context_loss_interaction_off` | `aggregate_glb_bytes_at_achieved_visual_utility` | -100880.8013 | [-1308620.9585, 1140082.8604] | negative | 是 |
| `context_loss_interaction_off` | `aggregate_glb_byte_reduction` | -0.0299 | [-0.0478, -0.0058] | negative | 否 |
| `context_loss_interaction_off` | `pose_image_per` | 0.0011 | [-0.0016, 0.0048] | positive | 是 |
| `context_loss_interaction_off` | `aggregate_image_per` | 0.0012 | [-0.0012, 0.0048] | positive | 是 |
| `context_loss_interaction_off` | `pose_image_miss_pixel_rate` | 0.0004 | [-0.0018, 0.0038] | positive | 是 |
| `context_loss_interaction_off` | `aggregate_image_miss_pixel_rate` | 0.0005 | [-0.0017, 0.0039] | positive | 是 |
| `context_loss_interaction_off` | `pose_image_wrong_id_pixel_rate` | 0.0007 | [-0.0002, 0.0014] | positive | 是 |
| `context_loss_interaction_off` | `aggregate_image_extra_pixel_rate` | 0.0000 | [0.0000, 0.0000] | zero | 是 |
| `context_loss_interaction_on` | `pose_precision` | 0.0026 | [-0.0073, 0.0118] | positive | 是 |
| `context_loss_interaction_on` | `aggregate_precision` | 0.0207 | [-0.0123, 0.0663] | positive | 是 |
| `context_loss_interaction_on` | `pose_recall` | -0.0020 | [-0.0145, 0.0103] | negative | 是 |
| `context_loss_interaction_on` | `aggregate_recall` | -0.0221 | [-0.1000, 0.0472] | negative | 是 |
| `context_loss_interaction_on` | `pose_weighted_recall` | -0.0003 | [-0.0029, 0.0029] | negative | 是 |
| `context_loss_interaction_on` | `aggregate_weighted_recall` | -0.0005 | [-0.0027, 0.0024] | negative | 是 |
| `context_loss_interaction_on` | `pose_accuracy` | 0.0177 | [-0.0272, 0.0454] | positive | 是 |
| `context_loss_interaction_on` | `aggregate_accuracy` | 0.0308 | [-0.0155, 0.0585] | positive | 是 |
| `context_loss_interaction_on` | `pose_balanced_accuracy` | 0.0075 | [-0.0178, 0.0285] | positive | 是 |
| `context_loss_interaction_on` | `aggregate_balanced_accuracy` | 0.0056 | [-0.0200, 0.0218] | positive | 是 |
| `context_loss_interaction_on` | `pose_f1` | 0.0038 | [-0.0054, 0.0138] | positive | 是 |
| `context_loss_interaction_on` | `aggregate_f1` | 0.0076 | [-0.0105, 0.0240] | positive | 是 |
| `context_loss_interaction_on` | `pose_useful_cull` | 0.0176 | [-0.0326, 0.0510] | positive | 是 |
| `context_loss_interaction_on` | `aggregate_useful_cull` | 0.0317 | [-0.0177, 0.0625] | positive | 是 |
| `context_loss_interaction_on` | `pose_bad_cull` | -0.0001 | [-0.0054, 0.0063] | negative | 是 |
| `context_loss_interaction_on` | `aggregate_bad_cull` | 0.0010 | [-0.0021, 0.0047] | positive | 是 |
| `context_loss_interaction_on` | `pose_avg_fp_count` | -164.2723 | [-329.7095, 90.1599] | negative | 是 |
| `context_loss_interaction_on` | `pose_avg_fn_count` | 5.0532 | [-10.5306, 24.0095] | positive | 是 |
| `context_loss_interaction_on` | `pose_avg_tn_count` | 164.2723 | [-90.1599, 329.7095] | positive | 是 |
| `context_loss_interaction_on` | `aggregate_avg_fp_count` | -164.2723 | [-329.7095, 90.1599] | negative | 是 |
| `context_loss_interaction_on` | `aggregate_avg_fn_count` | 5.0532 | [-10.5306, 24.0095] | positive | 是 |
| `context_loss_interaction_on` | `aggregate_avg_tn_count` | 164.2723 | [-90.1599, 329.7095] | positive | 是 |
| `context_loss_interaction_on` | `pose_avg_pred_count` | -169.3255 | [-350.5453, 100.8930] | negative | 是 |
| `context_loss_interaction_on` | `aggregate_avg_pred_count` | -169.3255 | [-350.5453, 100.8930] | negative | 是 |
| `context_loss_interaction_on` | `aggregate_predicted_glb_bytes` | -4384913.2394 | [-17269716.8078, 12971012.3491] | negative | 是 |
| `context_loss_interaction_on` | `pose_download_utility_recall` | -0.0003 | [-0.0028, 0.0030] | negative | 是 |
| `context_loss_interaction_on` | `aggregate_download_utility_recall` | -0.0005 | [-0.0027, 0.0025] | negative | 是 |
| `context_loss_interaction_on` | `pose_glb_bytes_at_achieved_visual_utility` | -252345.7966 | [-2176973.3052, 2352104.6845] | negative | 是 |
| `context_loss_interaction_on` | `aggregate_glb_bytes_at_achieved_visual_utility` | -252345.7966 | [-2176973.3052, 2352104.6845] | negative | 是 |
| `context_loss_interaction_on` | `aggregate_glb_byte_reduction` | 0.0163 | [-0.0615, 0.0798] | positive | 是 |
| `context_loss_interaction_on` | `pose_image_per` | 0.0008 | [-0.0073, 0.0065] | positive | 是 |
| `context_loss_interaction_on` | `aggregate_image_per` | 0.0007 | [-0.0073, 0.0063] | positive | 是 |
| `context_loss_interaction_on` | `pose_image_miss_pixel_rate` | 0.0022 | [-0.0078, 0.0098] | positive | 是 |
| `context_loss_interaction_on` | `aggregate_image_miss_pixel_rate` | 0.0022 | [-0.0078, 0.0101] | positive | 是 |
| `context_loss_interaction_on` | `pose_image_wrong_id_pixel_rate` | -0.0014 | [-0.0035, 0.0005] | negative | 是 |
| `context_loss_interaction_on` | `aggregate_image_extra_pixel_rate` | 0.0000 | [0.0000, 0.0000] | zero | 是 |
| `survival_loss_interaction_off` | `pose_precision` | -0.1498 | [-0.1691, -0.1317] | negative | 否 |
| `survival_loss_interaction_off` | `aggregate_precision` | -0.3981 | [-0.4360, -0.3628] | negative | 否 |
| `survival_loss_interaction_off` | `pose_recall` | 0.0668 | [0.0474, 0.0864] | positive | 否 |
| `survival_loss_interaction_off` | `aggregate_recall` | 0.2476 | [0.1623, 0.3676] | positive | 否 |
| `survival_loss_interaction_off` | `pose_weighted_recall` | 0.0020 | [-0.0022, 0.0055] | positive | 是 |
| `survival_loss_interaction_off` | `aggregate_weighted_recall` | 0.0025 | [-0.0012, 0.0057] | positive | 是 |
| `survival_loss_interaction_off` | `pose_accuracy` | -0.1913 | [-0.2406, -0.1545] | negative | 否 |
| `survival_loss_interaction_off` | `aggregate_accuracy` | -0.3432 | [-0.3929, -0.2814] | negative | 否 |
| `survival_loss_interaction_off` | `pose_balanced_accuracy` | -0.0818 | [-0.1027, -0.0585] | negative | 否 |
| `survival_loss_interaction_off` | `aggregate_balanced_accuracy` | -0.0615 | [-0.1178, 0.0255] | negative | 是 |
| `survival_loss_interaction_off` | `pose_f1` | -0.1450 | [-0.1717, -0.1199] | negative | 否 |
| `survival_loss_interaction_off` | `aggregate_f1` | -0.3830 | [-0.4265, -0.3254] | negative | 否 |
| `survival_loss_interaction_off` | `pose_useful_cull` | -0.2054 | [-0.2519, -0.1679] | negative | 否 |
| `survival_loss_interaction_off` | `aggregate_useful_cull` | -0.3542 | [-0.4018, -0.2979] | negative | 否 |
| `survival_loss_interaction_off` | `pose_bad_cull` | -0.0141 | [-0.0215, -0.0085] | negative | 否 |
| `survival_loss_interaction_off` | `aggregate_bad_cull` | -0.0110 | [-0.0171, -0.0069] | negative | 否 |
| `survival_loss_interaction_off` | `pose_avg_fp_count` | 1832.5524 | [1499.6543, 2170.7758] | positive | 否 |
| `survival_loss_interaction_off` | `pose_avg_fn_count` | -56.7230 | [-88.5232, -35.7772] | negative | 否 |
| `survival_loss_interaction_off` | `pose_avg_tn_count` | -1832.5524 | [-2170.7758, -1499.6543] | negative | 否 |
| `survival_loss_interaction_off` | `aggregate_avg_fp_count` | 1832.5524 | [1499.6543, 2170.7758] | positive | 否 |
| `survival_loss_interaction_off` | `aggregate_avg_fn_count` | -56.7230 | [-88.5232, -35.7772] | negative | 否 |
| `survival_loss_interaction_off` | `aggregate_avg_tn_count` | -1832.5524 | [-2170.7758, -1499.6543] | negative | 否 |
| `survival_loss_interaction_off` | `pose_avg_pred_count` | 1889.2754 | [1571.8630, 2218.5656] | positive | 否 |
| `survival_loss_interaction_off` | `aggregate_avg_pred_count` | 1889.2754 | [1571.8630, 2218.5656] | positive | 否 |
| `survival_loss_interaction_off` | `aggregate_predicted_glb_bytes` | 54469614.5164 | [33746629.2393, 82520208.0772] | positive | 否 |
| `survival_loss_interaction_off` | `pose_download_utility_recall` | 0.0018 | [-0.0024, 0.0054] | positive | 是 |
| `survival_loss_interaction_off` | `aggregate_download_utility_recall` | 0.0022 | [-0.0015, 0.0055] | positive | 是 |
| `survival_loss_interaction_off` | `pose_glb_bytes_at_achieved_visual_utility` | 7989482.1221 | [830044.1354, 14944545.7175] | positive | 否 |
| `survival_loss_interaction_off` | `aggregate_glb_bytes_at_achieved_visual_utility` | 7989482.1221 | [830044.1354, 14944545.7175] | positive | 否 |
| `survival_loss_interaction_off` | `aggregate_glb_byte_reduction` | -0.2095 | [-0.3000, -0.1158] | negative | 否 |
| `survival_loss_interaction_off` | `pose_image_per` | -0.0002 | [-0.0081, 0.0108] | negative | 是 |
| `survival_loss_interaction_off` | `aggregate_image_per` | -0.0005 | [-0.0086, 0.0105] | negative | 是 |
| `survival_loss_interaction_off` | `pose_image_miss_pixel_rate` | -0.0027 | [-0.0111, 0.0079] | negative | 是 |
| `survival_loss_interaction_off` | `aggregate_image_miss_pixel_rate` | -0.0031 | [-0.0117, 0.0074] | negative | 是 |
| `survival_loss_interaction_off` | `pose_image_wrong_id_pixel_rate` | 0.0025 | [0.0009, 0.0046] | positive | 否 |
| `survival_loss_interaction_off` | `aggregate_image_extra_pixel_rate` | 0.0000 | [0.0000, 0.0000] | zero | 是 |
| `survival_loss_interaction_on` | `pose_precision` | -0.1361 | [-0.1482, -0.1242] | negative | 否 |
| `survival_loss_interaction_on` | `aggregate_precision` | -0.3503 | [-0.3892, -0.3097] | negative | 否 |
| `survival_loss_interaction_on` | `pose_recall` | 0.0696 | [0.0469, 0.0951] | positive | 否 |
| `survival_loss_interaction_on` | `aggregate_recall` | 0.2306 | [0.1361, 0.3475] | positive | 否 |
| `survival_loss_interaction_on` | `pose_weighted_recall` | 0.0023 | [0.0005, 0.0044] | positive | 否 |
| `survival_loss_interaction_on` | `aggregate_weighted_recall` | 0.0027 | [0.0008, 0.0049] | positive | 否 |
| `survival_loss_interaction_on` | `pose_accuracy` | -0.1659 | [-0.1941, -0.1304] | negative | 否 |
| `survival_loss_interaction_on` | `aggregate_accuracy` | -0.3042 | [-0.3658, -0.2260] | negative | 否 |
| `survival_loss_interaction_on` | `pose_balanced_accuracy` | -0.0694 | [-0.0871, -0.0489] | negative | 否 |
| `survival_loss_interaction_on` | `aggregate_balanced_accuracy` | -0.0492 | [-0.1115, 0.0469] | negative | 是 |
| `survival_loss_interaction_on` | `pose_f1` | -0.1245 | [-0.1428, -0.1058] | negative | 否 |
| `survival_loss_interaction_on` | `aggregate_f1` | -0.3506 | [-0.3980, -0.2981] | negative | 否 |
| `survival_loss_interaction_on` | `pose_useful_cull` | -0.1813 | [-0.2069, -0.1495] | negative | 否 |
| `survival_loss_interaction_on` | `aggregate_useful_cull` | -0.3144 | [-0.3750, -0.2420] | negative | 否 |
| `survival_loss_interaction_on` | `pose_bad_cull` | -0.0154 | [-0.0217, -0.0095] | negative | 否 |
| `survival_loss_interaction_on` | `aggregate_bad_cull` | -0.0102 | [-0.0159, -0.0058] | negative | 否 |
| `survival_loss_interaction_on` | `pose_avg_fp_count` | 1626.9249 | [1230.3591, 2003.2838] | positive | 否 |
| `survival_loss_interaction_on` | `pose_avg_fn_count` | -52.8247 | [-82.2381, -30.1061] | negative | 否 |
| `survival_loss_interaction_on` | `pose_avg_tn_count` | -1626.9249 | [-2003.2838, -1230.3591] | negative | 否 |
| `survival_loss_interaction_on` | `aggregate_avg_fp_count` | 1626.9249 | [1230.3591, 2003.2838] | positive | 否 |
| `survival_loss_interaction_on` | `aggregate_avg_fn_count` | -52.8247 | [-82.2381, -30.1061] | negative | 否 |
| `survival_loss_interaction_on` | `aggregate_avg_tn_count` | -1626.9249 | [-2003.2838, -1230.3591] | negative | 否 |
| `survival_loss_interaction_on` | `pose_avg_pred_count` | 1679.7496 | [1302.4522, 2047.9574] | positive | 否 |
| `survival_loss_interaction_on` | `aggregate_avg_pred_count` | 1679.7496 | [1302.4522, 2047.9574] | positive | 否 |
| `survival_loss_interaction_on` | `aggregate_predicted_glb_bytes` | 40340923.1925 | [24783434.8462, 61458859.1124] | positive | 否 |
| `survival_loss_interaction_on` | `pose_download_utility_recall` | 0.0021 | [0.0003, 0.0041] | positive | 否 |
| `survival_loss_interaction_on` | `aggregate_download_utility_recall` | 0.0024 | [0.0006, 0.0044] | positive | 否 |
| `survival_loss_interaction_on` | `pose_glb_bytes_at_achieved_visual_utility` | 7838017.1268 | [3258425.9601, 14126888.7304] | positive | 否 |
| `survival_loss_interaction_on` | `aggregate_glb_bytes_at_achieved_visual_utility` | 7838017.1268 | [3258425.9601, 14126888.7304] | positive | 否 |
| `survival_loss_interaction_on` | `aggregate_glb_byte_reduction` | -0.1633 | [-0.2314, -0.1178] | negative | 否 |
| `survival_loss_interaction_on` | `pose_image_per` | -0.0005 | [-0.0023, 0.0012] | negative | 是 |
| `survival_loss_interaction_on` | `aggregate_image_per` | -0.0010 | [-0.0028, 0.0007] | negative | 是 |
| `survival_loss_interaction_on` | `pose_image_miss_pixel_rate` | -0.0009 | [-0.0044, 0.0012] | negative | 是 |
| `survival_loss_interaction_on` | `aggregate_image_miss_pixel_rate` | -0.0014 | [-0.0052, 0.0009] | negative | 是 |
| `survival_loss_interaction_on` | `pose_image_wrong_id_pixel_rate` | 0.0004 | [-0.0012, 0.0030] | positive | 是 |
| `survival_loss_interaction_on` | `aggregate_image_extra_pixel_rate` | 0.0000 | [0.0000, 0.0000] | zero | 是 |
| `context_survival_loss_three_way_interaction` | `pose_precision` | 0.0138 | [-0.0009, 0.0316] | positive | 是 |
| `context_survival_loss_three_way_interaction` | `aggregate_precision` | 0.0479 | [0.0060, 0.0813] | positive | 否 |
| `context_survival_loss_three_way_interaction` | `pose_recall` | 0.0028 | [-0.0304, 0.0313] | positive | 是 |
| `context_survival_loss_three_way_interaction` | `aggregate_recall` | -0.0170 | [-0.0625, 0.0377] | negative | 是 |
| `context_survival_loss_three_way_interaction` | `pose_weighted_recall` | 0.0003 | [-0.0041, 0.0064] | positive | 是 |
| `context_survival_loss_three_way_interaction` | `aggregate_weighted_recall` | 0.0002 | [-0.0036, 0.0058] | positive | 是 |
| `context_survival_loss_three_way_interaction` | `pose_accuracy` | 0.0255 | [-0.0200, 0.0547] | positive | 是 |
| `context_survival_loss_three_way_interaction` | `aggregate_accuracy` | 0.0390 | [-0.0088, 0.0702] | positive | 是 |
| `context_survival_loss_three_way_interaction` | `pose_balanced_accuracy` | 0.0124 | [-0.0161, 0.0386] | positive | 是 |
| `context_survival_loss_three_way_interaction` | `aggregate_balanced_accuracy` | 0.0123 | [-0.0018, 0.0283] | positive | 是 |
| `context_survival_loss_three_way_interaction` | `pose_f1` | 0.0206 | [0.0099, 0.0311] | positive | 否 |
| `context_survival_loss_three_way_interaction` | `aggregate_f1` | 0.0324 | [-0.0014, 0.0633] | positive | 是 |
| `context_survival_loss_three_way_interaction` | `pose_useful_cull` | 0.0241 | [-0.0254, 0.0552] | positive | 是 |
| `context_survival_loss_three_way_interaction` | `aggregate_useful_cull` | 0.0397 | [-0.0100, 0.0729] | positive | 是 |
| `context_survival_loss_three_way_interaction` | `pose_bad_cull` | -0.0014 | [-0.0059, 0.0023] | negative | 是 |
| `context_survival_loss_three_way_interaction` | `aggregate_bad_cull` | 0.0008 | [-0.0017, 0.0029] | positive | 是 |
| `context_survival_loss_three_way_interaction` | `pose_avg_fp_count` | -205.6275 | [-382.4548, 51.3908] | negative | 是 |
| `context_survival_loss_three_way_interaction` | `pose_avg_fn_count` | 3.8983 | [-8.6045, 14.7341] | positive | 是 |
| `context_survival_loss_three_way_interaction` | `pose_avg_tn_count` | 205.6275 | [-51.3908, 382.4548] | positive | 是 |
| `context_survival_loss_three_way_interaction` | `aggregate_avg_fp_count` | -205.6275 | [-382.4548, 51.3908] | negative | 是 |
| `context_survival_loss_three_way_interaction` | `aggregate_avg_fn_count` | 3.8983 | [-8.6045, 14.7341] | positive | 是 |
| `context_survival_loss_three_way_interaction` | `aggregate_avg_tn_count` | 205.6275 | [-51.3908, 382.4548] | positive | 是 |
| `context_survival_loss_three_way_interaction` | `pose_avg_pred_count` | -209.5258 | [-395.0067, 57.0694] | negative | 是 |
| `context_survival_loss_three_way_interaction` | `aggregate_avg_pred_count` | -209.5258 | [-395.0067, 57.0694] | negative | 是 |
| `context_survival_loss_three_way_interaction` | `aggregate_predicted_glb_bytes` | -14128691.3239 | [-22194202.8557, -3528393.0534] | negative | 否 |
| `context_survival_loss_three_way_interaction` | `pose_download_utility_recall` | 0.0002 | [-0.0041, 0.0062] | positive | 是 |
| `context_survival_loss_three_way_interaction` | `aggregate_download_utility_recall` | 0.0001 | [-0.0036, 0.0056] | positive | 是 |
| `context_survival_loss_three_way_interaction` | `pose_glb_bytes_at_achieved_visual_utility` | -151464.9953 | [-2653686.2022, 2494139.4191] | negative | 是 |
| `context_survival_loss_three_way_interaction` | `aggregate_glb_bytes_at_achieved_visual_utility` | -151464.9953 | [-2653686.2022, 2494139.4191] | negative | 是 |
| `context_survival_loss_three_way_interaction` | `aggregate_glb_byte_reduction` | 0.0462 | [-0.0131, 0.0885] | positive | 是 |
| `context_survival_loss_three_way_interaction` | `pose_image_per` | -0.0003 | [-0.0123, 0.0076] | negative | 是 |
| `context_survival_loss_three_way_interaction` | `aggregate_image_per` | -0.0005 | [-0.0123, 0.0071] | negative | 是 |
| `context_survival_loss_three_way_interaction` | `pose_image_miss_pixel_rate` | 0.0018 | [-0.0117, 0.0113] | positive | 是 |
| `context_survival_loss_three_way_interaction` | `aggregate_image_miss_pixel_rate` | 0.0017 | [-0.0118, 0.0113] | positive | 是 |
| `context_survival_loss_three_way_interaction` | `pose_image_wrong_id_pixel_rate` | -0.0021 | [-0.0040, -0.0004] | negative | 否 |
| `context_survival_loss_three_way_interaction` | `aggregate_image_extra_pixel_rate` | 0.0000 | [0.0000, 0.0000] | zero | 是 |

## useful cull 的解释边界

例如某个 pose 有 1000 个候选实例、100 个真实可见实例，模型只预测 10 个且其中 10 个恰好正确，则 TP=10、FP=0、FN=90、TN=900。此时 useful cull=0.90，看起来很高，但 recall=0.10、bad cull=0.09；若漏掉的实例具有较大 visible weight，weighted recall 也会明显下降。这种结果是错误剔除风险，不能解释为模型性能更好。

方向代理、上下文和损失只有在 weighted recall 安全性不恶化，并且 precision、balanced accuracy、F1、有效剔除、图像质量或资源成本至少一项的配对置信区间稳定改善时，才可写成预测或系统贡献。
