# M4-v2 完整因子消融评价

日期：2026-08-03  
评价阶段：validation-only；test split 未读取

## 摘要

本轮将方向遮挡代理和显式遮挡抑制头作为两个独立因子，形成 A/B/C/D 四个正式变体，并保留 AABB 与离线几何基线作为逐级参考。新的路线判定为 `route_b_system`。判定顺序是安全性、分类/剔除效果、系统效果，useful cull 不再单独决定路线。

方向代理的独立作用通过 `B-A` 和 `D-C` 同时观察；显式抑制通过 `C-A` 和 `D-B` 观察；交互项为 `D-B-C+A`。所有成员使用相同的 validation pose、后退相机候选、GT 和候选哈希。

## 数据与阈值

场景为 HKUST v3，validation pose 数为 `664`，候选身份摘要为 `8bd3e6a840c7624e2de459ef8057b24380c91936383c29f2368d93801f4c17bf`。模型与后退候选相机使用 66°，真实渲染约定为 60°。每个 checkpoint 的阈值只从自己的 calibration split 冻结；安全工作点要求 pose recall >= 0.95、weighted recall > 0.99 且 weighted recall 单侧 95% 下界 > 0.99。

## 指标口径

对每个 pose，候选集合记为 C，真实可见集合记为 G，预测集合记为 P。TP=P∩G，FP=P-G，FN=G-P，TN=C-(P∪G)。pose 宏平均先逐 pose 计算指标再平均；aggregate 先合并全部 pose 的计数再计算比例。

- 画面安全：recall、weighted recall、bad cull。weighted recall 使用 `visible_weights`，它是可见重要性证据，不是真实像素覆盖率。
- 分类诊断：precision、F1、Jaccard、accuracy、balanced accuracy、specificity。
- 剔除效率：useful cull=TN/candidate，只统计正确剔除的不可见候选；bad cull=FN/candidate，统计错误剔除的真实可见候选。
- 资源诊断：平均预测数、预测/候选、预测/GT。GLB 字节、像素和浏览器延迟必须有同位姿的额外证据，不能由 useful cull 推断。

## 成员结果：pose 宏平均

| 变体 | seed | 阈值 | pose recall | weighted recall | weighted LCB | pose precision | pose F1 | pose accuracy | balanced accuracy | useful cull | bad cull | 平均预测数 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `aabb_ray` | 20260801 | 0.00048697 | 0.971787 | 0.999930 | 0.999907 | 0.094881 | 0.156736 | 0.643125 | 0.780422 | 0.578005 | 0.000553 | 607.30 |
| `aabb_ray` | 20260802 | 0.00100000 | 0.949454 | 0.999515 | 0.999087 | 0.108598 | 0.177531 | 0.691642 | 0.796694 | 0.627423 | 0.001454 | 534.91 |
| `aabb_ray` | 20260803 | 0.00048697 | 0.968873 | 0.999907 | 0.999868 | 0.099399 | 0.163789 | 0.663201 | 0.790087 | 0.598404 | 0.000876 | 570.98 |
| `geometry_ray` | 20260801 | 0.00200000 | 0.941339 | 0.993582 | 0.989894 | 0.491421 | 0.593864 | 0.932780 | 0.931979 | 0.874887 | 0.007779 | 272.12 |
| `geometry_ray` | 20260802 | 0.00500000 | 0.938973 | 0.996104 | 0.994882 | 0.403320 | 0.512582 | 0.921221 | 0.924164 | 0.863258 | 0.007708 | 236.58 |
| `geometry_ray` | 20260803 | 0.00200000 | 0.940374 | 0.993493 | 0.989861 | 0.427551 | 0.528033 | 0.925023 | 0.927334 | 0.867315 | 0.007965 | 316.97 |
| `geometry_context_ray_no_inhibition` | 20260801 | 0.00200000 | 0.921743 | 0.990372 | 0.986331 | 0.493267 | 0.577525 | 0.935723 | 0.925106 | 0.880640 | 0.010589 | 244.13 |
| `geometry_context_ray_no_inhibition` | 20260802 | 0.00500000 | 0.943872 | 0.993889 | 0.991787 | 0.387201 | 0.490661 | 0.923046 | 0.927258 | 0.864610 | 0.007236 | 252.93 |
| `geometry_context_ray_no_inhibition` | 20260803 | 0.00500000 | 0.953178 | 0.993677 | 0.991122 | 0.393305 | 0.500948 | 0.924405 | 0.932156 | 0.864802 | 0.006069 | 218.76 |
| `geometry_context_proxy_ray_no_inhibition` | 20260801 | 0.00500000 | 0.939903 | 0.992762 | 0.988705 | 0.453218 | 0.551580 | 0.938253 | 0.934616 | 0.880646 | 0.008065 | 171.13 |
| `geometry_context_proxy_ray_no_inhibition` | 20260802 | 0.02000000 | 0.935929 | 0.992716 | 0.990304 | 0.399737 | 0.494578 | 0.927472 | 0.926407 | 0.869953 | 0.008153 | 237.04 |
| `geometry_context_proxy_ray_no_inhibition` | 20260803 | 0.01000000 | 0.934312 | 0.995121 | 0.992995 | 0.406661 | 0.514066 | 0.927053 | 0.924272 | 0.867769 | 0.006388 | 198.03 |
| `geometry_context_ray` | 20260801 | 0.00200000 | 0.949879 | 0.993283 | 0.989447 | 0.438649 | 0.540360 | 0.935000 | 0.937013 | 0.876223 | 0.006895 | 220.50 |
| `geometry_context_ray` | 20260802 | 0.00500000 | 0.944842 | 0.992251 | 0.989650 | 0.401849 | 0.503770 | 0.924595 | 0.929008 | 0.866625 | 0.007702 | 236.87 |
| `geometry_context_ray` | 20260803 | 0.00500000 | 0.953889 | 0.994849 | 0.992617 | 0.382282 | 0.492658 | 0.923781 | 0.932446 | 0.864778 | 0.006670 | 213.32 |
| `full` | 20260801 | 0.01000000 | 0.936150 | 0.994174 | 0.991829 | 0.371630 | 0.469690 | 0.916590 | 0.920536 | 0.858824 | 0.007906 | 257.14 |
| `full` | 20260802 | 0.02000000 | 0.928117 | 0.991119 | 0.987471 | 0.409741 | 0.505107 | 0.928809 | 0.923690 | 0.872262 | 0.009126 | 233.02 |
| `full` | 20260803 | 0.01000000 | 0.950622 | 0.994568 | 0.992374 | 0.389858 | 0.499646 | 0.924169 | 0.931202 | 0.865435 | 0.006938 | 200.50 |

## 成员结果：aggregate

aggregate 先合并全部 validation pose 的 TP、FP、FN、TN 和可见重要性权重，再计算比例；它不是 pose 宏平均的替代口径。

| 变体 | seed | 阈值 | aggregate recall | aggregate weighted recall | aggregate weighted LCB | aggregate precision | aggregate F1 | aggregate accuracy | aggregate balanced accuracy | aggregate useful cull | aggregate bad cull | 平均预测数 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `aabb_ray` | 20260801 | 0.00048697 | 0.984518 | 0.999930 | 0.999907 | 0.044786 | 0.085675 | 0.900223 | 0.942170 | 0.895549 | 0.000074 | 607.30 |
| `aabb_ray` | 20260802 | 0.00100000 | 0.971816 | 0.999522 | 0.999087 | 0.050191 | 0.095453 | 0.912545 | 0.942039 | 0.907931 | 0.000134 | 534.91 |
| `aabb_ray` | 20260803 | 0.00048697 | 0.978358 | 0.999901 | 0.999868 | 0.047337 | 0.090305 | 0.906408 | 0.942211 | 0.901762 | 0.000103 | 570.98 |
| `geometry_ray` | 20260801 | 0.00200000 | 0.949357 | 0.993301 | 0.989894 | 0.096382 | 0.174998 | 0.957498 | 0.953447 | 0.952990 | 0.000240 | 272.12 |
| `geometry_ray` | 20260802 | 0.00500000 | 0.913596 | 0.996155 | 0.994882 | 0.106687 | 0.191062 | 0.963267 | 0.938550 | 0.958929 | 0.000410 | 236.58 |
| `geometry_ray` | 20260803 | 0.00200000 | 0.946086 | 0.993104 | 0.989861 | 0.082460 | 0.151699 | 0.949759 | 0.947931 | 0.945267 | 0.000256 | 316.97 |
| `geometry_context_ray_no_inhibition` | 20260801 | 0.00200000 | 0.917030 | 0.990089 | 0.986331 | 0.103774 | 0.186449 | 0.962002 | 0.939623 | 0.957647 | 0.000394 | 244.13 |
| `geometry_context_ray_no_inhibition` | 20260802 | 0.00500000 | 0.884594 | 0.993670 | 0.991787 | 0.096620 | 0.174212 | 0.960181 | 0.922568 | 0.955981 | 0.000548 | 252.93 |
| `geometry_context_ray_no_inhibition` | 20260803 | 0.00500000 | 0.879579 | 0.993461 | 0.991122 | 0.111081 | 0.197252 | 0.966007 | 0.922999 | 0.961830 | 0.000572 | 218.76 |
| `geometry_context_proxy_ray_no_inhibition` | 20260801 | 0.00500000 | 0.897950 | 0.992228 | 0.988705 | 0.144960 | 0.249623 | 0.974367 | 0.936341 | 0.970103 | 0.000485 | 171.13 |
| `geometry_context_proxy_ray_no_inhibition` | 20260802 | 0.02000000 | 0.874509 | 0.992564 | 0.990304 | 0.101924 | 0.182569 | 0.962817 | 0.918874 | 0.958665 | 0.000596 | 237.04 |
| `geometry_context_proxy_ray_no_inhibition` | 20260803 | 0.01000000 | 0.848997 | 0.994785 | 0.992995 | 0.118439 | 0.207878 | 0.969278 | 0.909424 | 0.965247 | 0.000717 | 198.03 |
| `geometry_context_ray` | 20260801 | 0.00200000 | 0.941289 | 0.992925 | 0.989447 | 0.117937 | 0.209611 | 0.966294 | 0.953851 | 0.961825 | 0.000279 | 220.50 |
| `geometry_context_ray` | 20260802 | 0.00500000 | 0.880451 | 0.992000 | 0.989650 | 0.102689 | 0.183926 | 0.962902 | 0.921873 | 0.958722 | 0.000568 | 236.87 |
| `geometry_context_ray` | 20260803 | 0.00500000 | 0.875491 | 0.994519 | 0.992617 | 0.113382 | 0.200764 | 0.966902 | 0.921414 | 0.962745 | 0.000591 | 213.32 |
| `full` | 20260801 | 0.01000000 | 0.903238 | 0.993870 | 0.991829 | 0.097041 | 0.175253 | 0.959634 | 0.931571 | 0.955345 | 0.000459 | 257.14 |
| `full` | 20260802 | 0.02000000 | 0.872492 | 0.990631 | 0.987471 | 0.103441 | 0.184954 | 0.963488 | 0.918207 | 0.959345 | 0.000605 | 233.02 |
| `full` | 20260803 | 0.01000000 | 0.873910 | 0.994304 | 0.992374 | 0.120413 | 0.211662 | 0.969090 | 0.921727 | 0.964941 | 0.000599 | 200.50 |

## 诊断工作点

每个 checkpoint 的 best-F1、最高 precision 和原始冻结阈值都来自该 checkpoint 自己的 calibration 记录。它们只用于诊断阈值敏感性，不参与安全主工作点排名，也没有读取 validation/test 重新选阈值。

| 变体 | seed | 安全工作点状态 | 安全阈值 | best-F1 阈值 | best-F1 precision | best-F1 recall | best-F1 weighted recall | best-F1 F1 | 最高 precision 阈值 | 最高 precision | 对应 recall | 对应 weighted recall | 原始冻结阈值 |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `aabb_ray` | 20260801 | safe | 0.00048697 | 0.03000000 | 0.588030 | 0.617678 | 0.732027 | 0.524247 | 0.89999998 | 0.968640 | 0.111155 | 0.121452 | 0.00200000 |
| `aabb_ray` | 20260802 | safe | 0.00100000 | 0.05000000 | 0.577543 | 0.632172 | 0.749415 | 0.525389 | 0.89999998 | 0.929659 | 0.187870 | 0.233642 | 0.00200000 |
| `aabb_ray` | 20260803 | safe | 0.00048697 | 0.03000000 | 0.536855 | 0.643839 | 0.767227 | 0.510495 | 0.89999998 | 0.961320 | 0.094827 | 0.110837 | 0.00200000 |
| `geometry_ray` | 20260801 | safe | 0.00200000 | 0.89999998 | 0.796661 | 0.833272 | 0.983184 | 0.780364 | 0.89999998 | 0.796661 | 0.833272 | 0.983184 | 0.01000000 |
| `geometry_ray` | 20260802 | safe | 0.00500000 | 0.89999998 | 0.781632 | 0.822423 | 0.985491 | 0.764197 | 0.89999998 | 0.781632 | 0.822423 | 0.985491 | 0.02000000 |
| `geometry_ray` | 20260803 | safe | 0.00200000 | 0.89999998 | 0.764393 | 0.820811 | 0.982016 | 0.753027 | 0.89999998 | 0.764393 | 0.820811 | 0.982016 | 0.00500000 |
| `geometry_context_ray_no_inhibition` | 20260801 | safe | 0.00200000 | 0.89999998 | 0.756116 | 0.841643 | 0.981512 | 0.756476 | 0.89999998 | 0.756116 | 0.841643 | 0.981512 | 0.00200000 |
| `geometry_context_ray_no_inhibition` | 20260802 | safe | 0.00500000 | 0.89999998 | 0.731447 | 0.811109 | 0.965805 | 0.724170 | 0.89999998 | 0.731447 | 0.811109 | 0.965805 | 0.01000000 |
| `geometry_context_ray_no_inhibition` | 20260803 | safe | 0.00500000 | 0.89999998 | 0.752128 | 0.819998 | 0.976666 | 0.743703 | 0.89999998 | 0.752128 | 0.819998 | 0.976666 | 0.01000000 |
| `geometry_context_proxy_ray_no_inhibition` | 20260801 | safe | 0.00500000 | 0.89999998 | 0.715807 | 0.868918 | 0.984217 | 0.739877 | 0.89999998 | 0.715807 | 0.868918 | 0.984217 | 0.01000000 |
| `geometry_context_proxy_ray_no_inhibition` | 20260802 | safe | 0.02000000 | 0.89999998 | 0.688330 | 0.843116 | 0.968387 | 0.707885 | 0.89999998 | 0.688330 | 0.843116 | 0.968387 | 0.03000000 |
| `geometry_context_proxy_ray_no_inhibition` | 20260803 | safe | 0.01000000 | 0.89999998 | 0.744824 | 0.831843 | 0.978024 | 0.744066 | 0.89999998 | 0.744824 | 0.831843 | 0.978024 | 0.03000000 |
| `geometry_context_ray` | 20260801 | safe | 0.00200000 | 0.89999998 | 0.772318 | 0.845169 | 0.983790 | 0.771252 | 0.89999998 | 0.772318 | 0.845169 | 0.983790 | 0.01000000 |
| `geometry_context_ray` | 20260802 | safe | 0.00500000 | 0.89999998 | 0.751429 | 0.800444 | 0.949136 | 0.734005 | 0.89999998 | 0.751429 | 0.800444 | 0.949136 | 0.00500000 |
| `geometry_context_ray` | 20260803 | safe | 0.00500000 | 0.89999998 | 0.722806 | 0.830406 | 0.974936 | 0.727524 | 0.89999998 | 0.722806 | 0.830406 | 0.974936 | 0.01000000 |
| `full` | 20260801 | safe | 0.01000000 | 0.89999998 | 0.750478 | 0.834180 | 0.975525 | 0.747731 | 0.89999998 | 0.750478 | 0.834180 | 0.975525 | 0.02000000 |
| `full` | 20260802 | safe | 0.02000000 | 0.89999998 | 0.732672 | 0.817776 | 0.962595 | 0.726823 | 0.89999998 | 0.732672 | 0.817776 | 0.962595 | 0.02000000 |
| `full` | 20260803 | safe | 0.01000000 | 0.88000000 | 0.687423 | 0.845288 | 0.961744 | 0.706789 | 0.89999998 | 0.690019 | 0.839434 | 0.956994 | 0.03000000 |

每个成员的平均 TP、FP、FN、TN、预测/候选和预测/GT 已同时保存在汇总 JSON 的 `poseMacro` 与 `aggregate` 对象中；GLB、像素和浏览器成本若为 `not_available`，不参与路线排名。

## 完整成员诊断

下表把每个成员的两种统计口径和完整分类/剔除指标直接列出。`visual utility recall` 没有统一视觉效用监督，因此对所有成员明确记为 `not_available`，不能用 weighted recall 替代。

| 口径 | 变体 | seed | precision | recall | weighted recall | visual utility recall | F1 | Jaccard | accuracy | balanced accuracy | specificity | useful cull | bad cull | avg TP | avg FP | avg FN | avg TN | avg pred | pred/candidate | pred/GT |
|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pose_macro | `aabb_ray` | 20260801 | 0.094881 | 0.971787 | 0.999930 | not_available | 0.156736 | 0.094669 | 0.643125 | 0.780422 | 0.589058 | 0.578005 | 0.000553 | 27.20 | 580.11 | 0.43 | 5210.60 | 607.30 | 0.104378 | 21.982665 |
| aggregate | `aabb_ray` | 20260801 | 0.044786 | 0.984518 | 0.999930 | not_available | 0.085675 | 0.044755 | 0.900223 | 0.942170 | 0.899821 | 0.895549 | 0.000074 | 27.20 | 580.11 | 0.43 | 5210.60 | 607.30 | 0.104378 | 21.982665 |
| pose_macro | `aabb_ray` | 20260802 | 0.108598 | 0.949454 | 0.999515 | not_available | 0.177531 | 0.107894 | 0.691642 | 0.796694 | 0.643934 | 0.627423 | 0.001454 | 26.85 | 508.06 | 0.78 | 5282.64 | 534.91 | 0.091935 | 19.362189 |
| aggregate | `aabb_ray` | 20260802 | 0.050191 | 0.971816 | 0.999522 | not_available | 0.095453 | 0.050118 | 0.912545 | 0.942039 | 0.912262 | 0.907931 | 0.000134 | 26.85 | 508.06 | 0.78 | 5282.64 | 534.91 | 0.091935 | 19.362189 |
| pose_macro | `aabb_ray` | 20260803 | 0.099399 | 0.968873 | 0.999907 | not_available | 0.163789 | 0.099064 | 0.663201 | 0.790087 | 0.611300 | 0.598404 | 0.000876 | 27.03 | 543.95 | 0.60 | 5246.75 | 570.98 | 0.098135 | 20.667848 |
| aggregate | `aabb_ray` | 20260803 | 0.047337 | 0.978358 | 0.999901 | not_available | 0.090305 | 0.047288 | 0.906408 | 0.942211 | 0.906065 | 0.901762 | 0.000103 | 27.03 | 543.95 | 0.60 | 5246.75 | 570.98 | 0.098135 | 20.667848 |
| pose_macro | `geometry_ray` | 20260801 | 0.491421 | 0.941339 | 0.993582 | not_available | 0.593864 | 0.466761 | 0.932780 | 0.931979 | 0.922619 | 0.874887 | 0.007779 | 26.23 | 245.89 | 1.40 | 5544.81 | 272.12 | 0.046769 | 9.849924 |
| aggregate | `geometry_ray` | 20260801 | 0.096382 | 0.949357 | 0.993301 | not_available | 0.174998 | 0.095889 | 0.957498 | 0.953447 | 0.957537 | 0.952990 | 0.000240 | 26.23 | 245.89 | 1.40 | 5544.81 | 272.12 | 0.046769 | 9.849924 |
| pose_macro | `geometry_ray` | 20260802 | 0.403320 | 0.938973 | 0.996104 | not_available | 0.512582 | 0.380342 | 0.921221 | 0.924164 | 0.909354 | 0.863258 | 0.007708 | 25.24 | 211.34 | 2.39 | 5579.37 | 236.58 | 0.040660 | 8.563345 |
| aggregate | `geometry_ray` | 20260802 | 0.106687 | 0.913596 | 0.996155 | not_available | 0.191062 | 0.105621 | 0.963267 | 0.938550 | 0.963504 | 0.958929 | 0.000410 | 25.24 | 211.34 | 2.39 | 5579.37 | 236.58 | 0.040660 | 8.563345 |
| pose_macro | `geometry_ray` | 20260803 | 0.427551 | 0.940374 | 0.993493 | not_available | 0.528033 | 0.404752 | 0.925023 | 0.927334 | 0.914295 | 0.867315 | 0.007965 | 26.14 | 290.83 | 1.49 | 5499.87 | 316.97 | 0.054477 | 11.473234 |
| aggregate | `geometry_ray` | 20260803 | 0.082460 | 0.946086 | 0.993104 | not_available | 0.151699 | 0.082075 | 0.949759 | 0.947931 | 0.949777 | 0.945267 | 0.000256 | 26.14 | 290.83 | 1.49 | 5499.87 | 316.97 | 0.054477 | 11.473234 |
| pose_macro | `geometry_context_ray_no_inhibition` | 20260801 | 0.493267 | 0.921743 | 0.990372 | not_available | 0.577525 | 0.453182 | 0.935723 | 0.925106 | 0.928469 | 0.880640 | 0.010589 | 25.33 | 218.80 | 2.29 | 5571.91 | 244.13 | 0.041959 | 8.836786 |
| aggregate | `geometry_context_ray_no_inhibition` | 20260801 | 0.103774 | 0.917030 | 0.990089 | not_available | 0.186449 | 0.102809 | 0.962002 | 0.939623 | 0.962216 | 0.957647 | 0.000394 | 25.33 | 218.80 | 2.29 | 5571.91 | 244.13 | 0.041959 | 8.836786 |
| pose_macro | `geometry_context_ray_no_inhibition` | 20260802 | 0.387201 | 0.943872 | 0.993889 | not_available | 0.490661 | 0.371579 | 0.923046 | 0.927258 | 0.910643 | 0.864610 | 0.007236 | 24.44 | 228.49 | 3.19 | 5562.21 | 252.93 | 0.043471 | 9.155364 |
| aggregate | `geometry_context_ray_no_inhibition` | 20260802 | 0.096620 | 0.884594 | 0.993670 | not_available | 0.174212 | 0.095418 | 0.960181 | 0.922568 | 0.960541 | 0.955981 | 0.000548 | 24.44 | 228.49 | 3.19 | 5562.21 | 252.93 | 0.043471 | 9.155364 |
| pose_macro | `geometry_context_ray_no_inhibition` | 20260803 | 0.393305 | 0.953178 | 0.993677 | not_available | 0.500948 | 0.380342 | 0.924405 | 0.932156 | 0.911134 | 0.864802 | 0.006069 | 24.30 | 194.46 | 3.33 | 5596.25 | 218.76 | 0.037598 | 7.918338 |
| aggregate | `geometry_context_ray_no_inhibition` | 20260803 | 0.111081 | 0.879579 | 0.993461 | not_available | 0.197252 | 0.109417 | 0.966007 | 0.922999 | 0.966419 | 0.961830 | 0.000572 | 24.30 | 194.46 | 3.33 | 5596.25 | 218.76 | 0.037598 | 7.918338 |
| pose_macro | `geometry_context_proxy_ray_no_inhibition` | 20260801 | 0.453218 | 0.939903 | 0.992762 | not_available | 0.551580 | 0.428559 | 0.938253 | 0.934616 | 0.929330 | 0.880646 | 0.008065 | 24.81 | 146.32 | 2.82 | 5644.38 | 171.13 | 0.029412 | 6.194451 |
| aggregate | `geometry_context_proxy_ray_no_inhibition` | 20260801 | 0.144960 | 0.897950 | 0.992228 | not_available | 0.249623 | 0.142611 | 0.974367 | 0.936341 | 0.974731 | 0.970103 | 0.000485 | 24.81 | 146.32 | 2.82 | 5644.38 | 171.13 | 0.029412 | 6.194451 |
| pose_macro | `geometry_context_proxy_ray_no_inhibition` | 20260802 | 0.399737 | 0.935929 | 0.992716 | not_available | 0.494578 | 0.380868 | 0.927472 | 0.926407 | 0.916885 | 0.869953 | 0.008153 | 24.16 | 212.88 | 3.47 | 5577.83 | 237.04 | 0.040740 | 8.580026 |
| aggregate | `geometry_context_proxy_ray_no_inhibition` | 20260802 | 0.101924 | 0.874509 | 0.992564 | not_available | 0.182569 | 0.100455 | 0.962817 | 0.918874 | 0.963238 | 0.958665 | 0.000596 | 24.16 | 212.88 | 3.47 | 5577.83 | 237.04 | 0.040740 | 8.580026 |
| pose_macro | `geometry_context_proxy_ray_no_inhibition` | 20260803 | 0.406661 | 0.934312 | 0.995121 | not_available | 0.514066 | 0.390684 | 0.927053 | 0.924272 | 0.914232 | 0.867769 | 0.006388 | 23.45 | 174.58 | 4.17 | 5616.12 | 198.03 | 0.034036 | 7.168229 |
| aggregate | `geometry_context_proxy_ray_no_inhibition` | 20260803 | 0.118439 | 0.848997 | 0.994785 | not_available | 0.207878 | 0.115995 | 0.969278 | 0.909424 | 0.969852 | 0.965247 | 0.000717 | 23.45 | 174.58 | 4.17 | 5616.12 | 198.03 | 0.034036 | 7.168229 |
| pose_macro | `geometry_context_ray` | 20260801 | 0.438649 | 0.949879 | 0.993283 | not_available | 0.540360 | 0.421519 | 0.935000 | 0.937013 | 0.924148 | 0.876223 | 0.006895 | 26.00 | 194.49 | 1.62 | 5596.21 | 220.50 | 0.037897 | 7.981302 |
| aggregate | `geometry_context_ray` | 20260801 | 0.117937 | 0.941289 | 0.992925 | not_available | 0.209611 | 0.117076 | 0.966294 | 0.953851 | 0.966413 | 0.961825 | 0.000279 | 26.00 | 194.49 | 1.62 | 5596.21 | 220.50 | 0.037897 | 7.981302 |
| pose_macro | `geometry_context_ray` | 20260802 | 0.401849 | 0.944842 | 0.992251 | not_available | 0.503770 | 0.385115 | 0.924595 | 0.929008 | 0.913174 | 0.866625 | 0.007702 | 24.32 | 212.55 | 3.30 | 5578.16 | 236.87 | 0.040711 | 8.573975 |
| aggregate | `geometry_context_ray` | 20260802 | 0.102689 | 0.880451 | 0.992000 | not_available | 0.183926 | 0.101277 | 0.962902 | 0.921873 | 0.963295 | 0.958722 | 0.000568 | 24.32 | 212.55 | 3.30 | 5578.16 | 236.87 | 0.040711 | 8.573975 |
| pose_macro | `geometry_context_ray` | 20260803 | 0.382282 | 0.953889 | 0.994849 | not_available | 0.492658 | 0.369484 | 0.923781 | 0.932446 | 0.911004 | 0.864778 | 0.006670 | 24.19 | 189.13 | 3.44 | 5601.57 | 213.32 | 0.036664 | 7.721598 |
| aggregate | `geometry_context_ray` | 20260803 | 0.113382 | 0.875491 | 0.994519 | not_available | 0.200764 | 0.111583 | 0.966902 | 0.921414 | 0.967338 | 0.962745 | 0.000591 | 24.19 | 189.13 | 3.44 | 5601.57 | 213.32 | 0.036664 | 7.721598 |
| pose_macro | `full` | 20260801 | 0.371630 | 0.936150 | 0.994174 | not_available | 0.469690 | 0.354136 | 0.916590 | 0.920536 | 0.904923 | 0.858824 | 0.007906 | 24.95 | 232.19 | 2.67 | 5558.51 | 257.14 | 0.044195 | 9.307839 |
| aggregate | `full` | 20260801 | 0.097041 | 0.903238 | 0.993870 | not_available | 0.175253 | 0.096042 | 0.959634 | 0.931571 | 0.959903 | 0.955345 | 0.000459 | 24.95 | 232.19 | 2.67 | 5558.51 | 257.14 | 0.044195 | 9.307839 |
| pose_macro | `full` | 20260802 | 0.409741 | 0.928117 | 0.991119 | not_available | 0.505107 | 0.385086 | 0.928809 | 0.923690 | 0.919263 | 0.872262 | 0.009126 | 24.10 | 208.92 | 3.52 | 5581.78 | 233.02 | 0.040049 | 8.434693 |
| aggregate | `full` | 20260802 | 0.103441 | 0.872492 | 0.990631 | not_available | 0.184954 | 0.101900 | 0.963488 | 0.918207 | 0.963922 | 0.959345 | 0.000605 | 24.10 | 208.92 | 3.52 | 5581.78 | 233.02 | 0.040049 | 8.434693 |
| pose_macro | `full` | 20260803 | 0.389858 | 0.950622 | 0.994568 | not_available | 0.499646 | 0.376652 | 0.924169 | 0.931202 | 0.911782 | 0.865435 | 0.006938 | 24.14 | 176.36 | 3.48 | 5614.34 | 200.50 | 0.034460 | 7.257577 |
| aggregate | `full` | 20260803 | 0.120413 | 0.873910 | 0.994304 | not_available | 0.211662 | 0.118357 | 0.969090 | 0.921727 | 0.969545 | 0.964941 | 0.000599 | 24.14 | 176.36 | 3.48 | 5614.34 | 200.50 | 0.034460 | 7.257577 |

## 运行与资源成本

固定特征表字节数和 checkpoint/运行时特征文件来源保存在每个成员的 JSON 记录中；当前 intervention 没有保存逐 pose 前向计时、GLB 集合/字节曲线、同位姿图像或浏览器采样，因此下列系统指标对所有成员均为不可用，不能推断为零收益：

| 指标 | 状态 | 原因 |
|---|---|---|
| `glb_count_reduction` | not_available | not_available: intervention rows do not persist predicted/candidate GLB sets |
| `glb_byte_reduction` | not_available | not_available: intervention rows do not persist predicted/candidate GLB sets |
| `equal_visual_utility_glb_bytes` | not_available | not_available: no paired visual-utility/GLB budget curve attached |
| `forward_latency_ms` | not_available | not_available: stored interventions contain no forward timing |
| `webgpu_latency_ms` | not_available | not_available: no browser timing attached to this matrix |
| `main_thread_ms` | not_available | not_available: no browser timing attached to this matrix |
| `runtime_memory_bytes` | not_available | not_available: no browser memory sample attached to this matrix |
| `miss_pixel_rate` | not_available | not_available: no same-pose 60-degree instance-ID render attached to this intervention |
| `wrong_id_pixel_rate` | not_available | not_available: no same-pose 60-degree instance-ID render attached to this intervention |
| `extra_pixel_rate` | not_available | not_available: no same-pose 60-degree instance-ID render attached to this intervention |

固定特征表大小仍按成员分别记录；它是离线资产体积，不等同于前向延迟或移动设备内存峰值。

## 因子差值与置信区间

以下每项均为右侧因子组合减去左侧因子组合；区间由按 seed 聚类、seed 内按 pose 重采样的 10,000 次 paired bootstrap 得到。

### `B_minus_A`

| 统计口径 | 指标 | 差值 | 95% CI | 方向 | 是否跨零 |
|---|---|---:|---|---|---|
| pose_macro | `precision` | 0.013356 | [-0.038863, 0.015679] | positive | 是 |
| pose_macro | `recall` | -0.018866 | [-0.018372, 0.017548] | negative | 是 |
| pose_macro | `weighted_recall` | 0.001444 | [-0.001114, 0.002678] | positive | 是 |
| pose_macro | `f1` | 0.013117 | [-0.024841, 0.012924] | positive | 是 |
| pose_macro | `jaccard` | 0.010342 | [-0.023358, 0.012525] | positive | 是 |
| pose_macro | `accuracy` | 0.002648 | [0.001078, 0.005133] | positive | 否 |
| pose_macro | `balanced_accuracy` | -0.007884 | [-0.007560, 0.009248] | negative | 是 |
| pose_macro | `specificity` | 0.003098 | [-0.000002, 0.006380] | positive | 是 |
| pose_macro | `useful_cull` | 0.002967 | [-0.000525, 0.005474] | positive | 是 |
| pose_macro | `bad_cull` | 0.000319 | [-0.002403, 0.000929] | positive | 是 |
| pose_macro | `avg_tp_count` | -0.844880 | [-0.964859, -0.200803] | negative | 否 |
| pose_macro | `avg_fp_count` | -19.878012 | [-74.872377, -13.043085] | negative | 否 |
| pose_macro | `avg_fn_count` | 0.844880 | [0.200803, 0.964859] | positive | 否 |
| pose_macro | `avg_tn_count` | 19.878012 | [13.043085, 74.872377] | positive | 否 |
| pose_macro | `avg_pred_count` | -20.722892 | [-75.495896, -13.459438] | negative | 否 |
| pose_macro | `avg_candidate_count` | 0.000000 | [0.000000, 0.000000] | zero | 是 |
| pose_macro | `avg_gt_count` | 0.000000 | [0.000000, 0.000000] | zero | 是 |
| pose_macro | `pred_over_candidate` | -0.003562 | [-0.012887, -0.002329] | negative | 否 |
| pose_macro | `pred_over_gt` | -0.750109 | [-2.526622, -0.516212] | negative | 否 |
| aggregate | `precision` | 0.007358 | [0.004797, 0.042104] | positive | 否 |
| aggregate | `recall` | -0.030582 | [-0.034099, -0.007174] | negative | 否 |
| aggregate | `weighted_recall` | 0.001324 | [-0.001044, 0.002462] | positive | 是 |
| aggregate | `f1` | 0.010626 | [0.007244, 0.062749] | positive | 否 |
| aggregate | `jaccard` | 0.006578 | [0.004421, 0.040345] | positive | 否 |
| aggregate | `accuracy` | 0.003271 | [0.002167, 0.012429] | positive | 否 |
| aggregate | `balanced_accuracy` | -0.013575 | [-0.014324, -0.000345] | negative | 否 |
| aggregate | `specificity` | 0.003433 | [0.002231, 0.012604] | positive | 否 |
| aggregate | `useful_cull` | 0.003416 | [0.002221, 0.012538] | positive | 否 |
| aggregate | `bad_cull` | 0.000145 | [0.000034, 0.000165] | positive | 否 |
| aggregate | `avg_tp_count` | -0.844880 | [-0.957844, -0.195269] | negative | 否 |
| aggregate | `avg_fp_count` | -19.878012 | [-73.448883, -12.954606] | negative | 否 |
| aggregate | `avg_fn_count` | 0.844880 | [0.195269, 0.957844] | positive | 否 |
| aggregate | `avg_tn_count` | 19.878012 | [12.954606, 73.448883] | positive | 否 |
| aggregate | `avg_pred_count` | -20.722892 | [-74.084061, -13.344930] | negative | 否 |
| aggregate | `avg_candidate_count` | 0.000000 | [0.000000, 0.000000] | zero | 是 |
| aggregate | `avg_gt_count` | 0.000000 | [0.000000, 0.000000] | zero | 是 |
| aggregate | `pred_over_candidate` | -0.003562 | [-0.012682, -0.002295] | negative | 否 |
| aggregate | `pred_over_gt` | -0.750109 | [-2.504362, -0.508493] | negative | 否 |

### `C_minus_A`

| 统计口径 | 指标 | 差值 | 95% CI | 方向 | 是否跨零 |
|---|---|---:|---|---|---|
| pose_macro | `precision` | -0.011023 | [-0.053547, 0.013910] | negative | 是 |
| pose_macro | `recall` | 0.000711 | [-0.000357, 0.027410] | positive | 是 |
| pose_macro | `weighted_recall` | 0.001172 | [-0.001561, 0.003057] | positive | 是 |
| pose_macro | `f1` | -0.008290 | [-0.036128, 0.012579] | negative | 是 |
| pose_macro | `jaccard` | -0.010858 | [-0.030796, 0.012839] | negative | 是 |
| pose_macro | `accuracy` | -0.000624 | [-0.001958, 0.001973] | negative | 是 |
| pose_macro | `balanced_accuracy` | 0.000291 | [0.000056, 0.011548] | positive | 否 |
| pose_macro | `specificity` | -0.000130 | [-0.004550, 0.002638] | negative | 是 |
| pose_macro | `useful_cull` | -0.000024 | [-0.004493, 0.002162] | negative | 是 |
| pose_macro | `bad_cull` | 0.000600 | [-0.003545, 0.000770] | positive | 是 |
| pose_macro | `avg_tp_count` | -0.112952 | [-0.175715, 0.648092] | negative | 是 |
| pose_macro | `avg_fp_count` | -5.322289 | [-31.944352, -3.677347] | negative | 否 |
| pose_macro | `avg_fn_count` | 0.112952 | [-0.648092, 0.175715] | positive | 是 |
| pose_macro | `avg_tn_count` | 5.322289 | [3.677347, 31.944352] | positive | 否 |
| pose_macro | `avg_pred_count` | -5.435241 | [-31.546247, -3.630020] | negative | 否 |
| pose_macro | `avg_candidate_count` | 0.000000 | [0.000000, 0.000000] | zero | 是 |
| pose_macro | `avg_gt_count` | 0.000000 | [0.000000, 0.000000] | zero | 是 |
| pose_macro | `pred_over_candidate` | -0.000934 | [-0.005374, -0.000620] | negative | 否 |
| pose_macro | `pred_over_gt` | -0.196740 | [-1.057105, -0.131762] | negative | 否 |
| aggregate | `precision` | 0.002301 | [0.001665, 0.015957] | positive | 否 |
| aggregate | `recall` | -0.004089 | [-0.006241, 0.024462] | negative | 是 |
| aggregate | `weighted_recall` | 0.001058 | [-0.001579, 0.003001] | positive | 是 |
| aggregate | `f1` | 0.003512 | [0.002658, 0.025733] | positive | 否 |
| aggregate | `jaccard` | 0.002166 | [0.001610, 0.015935] | positive | 否 |
| aggregate | `accuracy` | 0.000895 | [0.000615, 0.005504] | positive | 否 |
| aggregate | `balanced_accuracy` | -0.001585 | [-0.002272, 0.013970] | negative | 是 |
| aggregate | `specificity` | 0.000919 | [0.000597, 0.005469] | positive | 否 |
| aggregate | `useful_cull` | 0.000915 | [0.000594, 0.005438] | positive | 否 |
| aggregate | `bad_cull` | 0.000019 | [-0.000110, 0.000030] | positive | 是 |
| aggregate | `avg_tp_count` | -0.112952 | [-0.174699, 0.636559] | negative | 是 |
| aggregate | `avg_fp_count` | -5.322289 | [-31.815361, -3.525653] | negative | 否 |
| aggregate | `avg_fn_count` | 0.112952 | [-0.636559, 0.174699] | positive | 是 |
| aggregate | `avg_tn_count` | 5.322289 | [3.525653, 31.815361] | positive | 否 |
| aggregate | `avg_pred_count` | -5.435241 | [-31.473230, -3.415336] | negative | 否 |
| aggregate | `avg_candidate_count` | 0.000000 | [0.000000, 0.000000] | zero | 是 |
| aggregate | `avg_gt_count` | 0.000000 | [0.000000, 0.000000] | zero | 是 |
| aggregate | `pred_over_candidate` | -0.000934 | [-0.005384, -0.000588] | negative | 否 |
| aggregate | `pred_over_gt` | -0.196740 | [-1.059928, -0.118373] | negative | 否 |

### `D_minus_C`

| 统计口径 | 指标 | 差值 | 95% CI | 方向 | 是否跨零 |
|---|---|---:|---|---|---|
| pose_macro | `precision` | 0.007575 | [-0.065944, 0.010487] | positive | 是 |
| pose_macro | `recall` | -0.003266 | [-0.017700, -0.003493] | negative | 否 |
| pose_macro | `weighted_recall` | -0.000281 | [-0.001618, 0.001582] | negative | 是 |
| pose_macro | `f1` | 0.006987 | [-0.069782, 0.007308] | positive | 是 |
| pose_macro | `jaccard` | 0.007168 | [-0.066421, 0.007425] | positive | 是 |
| pose_macro | `accuracy` | 0.000389 | [-0.018075, 0.004119] | positive | 是 |
| pose_macro | `balanced_accuracy` | -0.001244 | [-0.016150, -0.001381] | negative | 否 |
| pose_macro | `specificity` | 0.000778 | [-0.018882, 0.005865] | positive | 是 |
| pose_macro | `useful_cull` | 0.000657 | [-0.017096, 0.005406] | positive | 是 |
| pose_macro | `bad_cull` | 0.000269 | [0.000235, 0.001602] | positive | 否 |
| pose_macro | `avg_tp_count` | -0.043675 | [-1.074347, -0.033622] | negative | 否 |
| pose_macro | `avg_fp_count` | -12.775602 | [-12.962550, 37.173519] | negative | 是 |
| pose_macro | `avg_fn_count` | 0.043675 | [0.033622, 1.074347] | positive | 否 |
| pose_macro | `avg_tn_count` | 12.775602 | [-37.173519, 12.962550] | positive | 是 |
| pose_macro | `avg_pred_count` | -12.819277 | [-13.023130, 36.118938] | negative | 是 |
| pose_macro | `avg_candidate_count` | 0.000000 | [0.000000, 0.000000] | zero | 是 |
| pose_macro | `avg_gt_count` | 0.000000 | [0.000000, 0.000000] | zero | 是 |
| pose_macro | `pred_over_candidate` | -0.002203 | [-0.002229, 0.006182] | negative | 是 |
| pose_macro | `pred_over_gt` | -0.464021 | [-0.456168, 1.281706] | negative | 是 |
| aggregate | `precision` | 0.007031 | [-0.021179, 0.007047] | positive | 是 |
| aggregate | `recall` | -0.001581 | [-0.038721, -0.001008] | negative | 否 |
| aggregate | `weighted_recall` | -0.000215 | [-0.001938, 0.001818] | negative | 是 |
| aggregate | `f1` | 0.010899 | [-0.034564, 0.010791] | positive | 是 |
| aggregate | `jaccard` | 0.006774 | [-0.021294, 0.006766] | positive | 是 |
| aggregate | `accuracy` | 0.002188 | [-0.006255, 0.002192] | positive | 是 |
| aggregate | `balanced_accuracy` | 0.000313 | [-0.022063, 0.000383] | positive | 是 |
| aggregate | `specificity` | 0.002206 | [-0.006139, 0.002220] | positive | 是 |
| aggregate | `useful_cull` | 0.002196 | [-0.006108, 0.002208] | positive | 是 |
| aggregate | `bad_cull` | 0.000008 | [0.000005, 0.000183] | positive | 否 |
| aggregate | `avg_tp_count` | -0.043675 | [-1.072791, -0.028112] | negative | 否 |
| aggregate | `avg_fp_count` | -12.775602 | [-12.863077, 35.505710] | negative | 是 |
| aggregate | `avg_fn_count` | 0.043675 | [0.028112, 1.072791] | positive | 否 |
| aggregate | `avg_tn_count` | 12.775602 | [-35.505710, 12.863077] | positive | 是 |
| aggregate | `avg_pred_count` | -12.819277 | [-12.968951, 34.476707] | negative | 是 |
| aggregate | `avg_candidate_count` | 0.000000 | [0.000000, 0.000000] | zero | 是 |
| aggregate | `avg_gt_count` | 0.000000 | [0.000000, 0.000000] | zero | 是 |
| aggregate | `pred_over_candidate` | -0.002203 | [-0.002222, 0.005947] | negative | 是 |
| aggregate | `pred_over_gt` | -0.464021 | [-0.455056, 1.234481] | negative | 是 |

### `D_minus_B`

| 统计口径 | 指标 | 差值 | 95% CI | 方向 | 是否跨零 |
|---|---|---:|---|---|---|
| pose_macro | `precision` | -0.016803 | [-0.080726, 0.008877] | negative | 是 |
| pose_macro | `recall` | 0.016311 | [-0.008289, 0.015795] | positive | 是 |
| pose_macro | `weighted_recall` | -0.000553 | [-0.001929, 0.001876] | negative | 是 |
| pose_macro | `f1` | -0.014420 | [-0.081027, 0.009671] | negative | 是 |
| pose_macro | `jaccard` | -0.014032 | [-0.073445, 0.003293] | negative | 是 |
| pose_macro | `accuracy` | -0.002884 | [-0.021299, 0.001190] | negative | 是 |
| pose_macro | `balanced_accuracy` | 0.006930 | [-0.013792, 0.006593] | positive | 是 |
| pose_macro | `specificity` | -0.002450 | [-0.023988, 0.002155] | negative | 是 |
| pose_macro | `useful_cull` | -0.002334 | [-0.021468, 0.002098] | negative | 是 |
| pose_macro | `bad_cull` | 0.000550 | [-0.000282, 0.001081] | positive | 是 |
| pose_macro | `avg_tp_count` | 0.688253 | [-0.073293, 0.687764] | positive | 是 |
| pose_macro | `avg_fp_count` | 1.780120 | [-5.125025, 84.324912] | positive | 是 |
| pose_macro | `avg_fn_count` | -0.688253 | [-0.687764, 0.073293] | negative | 是 |
| pose_macro | `avg_tn_count` | -1.780120 | [-84.324912, 5.125025] | negative | 是 |
| pose_macro | `avg_pred_count` | 2.468373 | [-5.009551, 84.594039] | positive | 是 |
| pose_macro | `avg_candidate_count` | 0.000000 | [0.000000, 0.000000] | zero | 是 |
| pose_macro | `avg_gt_count` | 0.000000 | [0.000000, 0.000000] | zero | 是 |
| pose_macro | `pred_over_candidate` | 0.000424 | [-0.000867, 0.014534] | positive | 是 |
| pose_macro | `pred_over_gt` | 0.089348 | [-0.184071, 3.010377] | positive | 是 |
| aggregate | `precision` | 0.001975 | [-0.048940, 0.003765] | positive | 是 |
| aggregate | `recall` | 0.024913 | [-0.002893, 0.025436] | positive | 是 |
| aggregate | `weighted_recall` | -0.000481 | [-0.002321, 0.002161] | negative | 是 |
| aggregate | `f1` | 0.003785 | [-0.074489, 0.006193] | positive | 是 |
| aggregate | `jaccard` | 0.002362 | [-0.047353, 0.003902] | positive | 是 |
| aggregate | `accuracy` | -0.000188 | [-0.013823, 0.000894] | negative | 是 |
| aggregate | `balanced_accuracy` | 0.012303 | [-0.005259, 0.011899] | positive | 是 |
| aggregate | `specificity` | -0.000307 | [-0.013920, 0.000879] | negative | 是 |
| aggregate | `useful_cull` | -0.000306 | [-0.013853, 0.000875] | negative | 是 |
| aggregate | `bad_cull` | -0.000118 | [-0.000119, 0.000012] | negative | 是 |
| aggregate | `avg_tp_count` | 0.688253 | [-0.071787, 0.697327] | positive | 是 |
| aggregate | `avg_fp_count` | 1.780120 | [-5.054543, 80.079832] | positive | 是 |
| aggregate | `avg_fn_count` | -0.688253 | [-0.697327, 0.071787] | negative | 是 |
| aggregate | `avg_tn_count` | -1.780120 | [-80.079832, 5.054543] | negative | 是 |
| aggregate | `avg_pred_count` | 2.468373 | [-4.983911, 80.281928] | positive | 是 |
| aggregate | `avg_candidate_count` | 0.000000 | [0.000000, 0.000000] | zero | 是 |
| aggregate | `avg_gt_count` | 0.000000 | [0.000000, 0.000000] | zero | 是 |
| aggregate | `pred_over_candidate` | 0.000424 | [-0.000859, 0.013914] | positive | 是 |
| aggregate | `pred_over_gt` | 0.089348 | [-0.187476, 2.967588] | positive | 是 |

### `interaction_D_minus_B_minus_C_plus_A`

| 统计口径 | 指标 | 差值 | 95% CI | 方向 | 是否跨零 |
|---|---|---:|---|---|---|
| pose_macro | `precision` | -0.005780 | [-0.027513, -0.001688] | negative | 否 |
| pose_macro | `recall` | 0.015599 | [-0.030915, 0.015075] | positive | 是 |
| pose_macro | `weighted_recall` | -0.001725 | [-0.002920, 0.000924] | negative | 是 |
| pose_macro | `f1` | -0.006130 | [-0.043562, -0.001108] | negative | 否 |
| pose_macro | `jaccard` | -0.003174 | [-0.041801, -0.001955] | negative | 否 |
| pose_macro | `accuracy` | -0.002260 | [-0.020590, 0.000129] | negative | 是 |
| pose_macro | `balanced_accuracy` | 0.006640 | [-0.025392, 0.006264] | positive | 是 |
| pose_macro | `specificity` | -0.002320 | [-0.019657, 0.000231] | negative | 是 |
| pose_macro | `useful_cull` | -0.002310 | [-0.017159, 0.000494] | negative | 是 |
| pose_macro | `bad_cull` | -0.000050 | [-0.000117, 0.003441] | negative | 是 |
| pose_macro | `avg_tp_count` | 0.801205 | [-0.528640, 0.789157] | positive | 是 |
| pose_macro | `avg_fp_count` | 7.102410 | [5.956727, 110.468763] | positive | 否 |
| pose_macro | `avg_fn_count` | -0.801205 | [-0.789157, 0.528640] | negative | 是 |
| pose_macro | `avg_tn_count` | -7.102410 | [-110.468763, -5.956727] | negative | 否 |
| pose_macro | `avg_pred_count` | 7.903614 | [6.392558, 110.320921] | positive | 否 |
| pose_macro | `avg_candidate_count` | 0.000000 | [0.000000, 0.000000] | zero | 是 |
| pose_macro | `avg_gt_count` | 0.000000 | [0.000000, 0.000000] | zero | 是 |
| pose_macro | `pred_over_candidate` | 0.001358 | [0.001107, 0.018886] | positive | 否 |
| pose_macro | `pred_over_gt` | 0.286088 | [0.239730, 3.811658] | positive | 否 |
| aggregate | `precision` | -0.000326 | [-0.062691, -0.000455] | negative | 否 |
| aggregate | `recall` | 0.029001 | [-0.020904, 0.028877] | positive | 是 |
| aggregate | `weighted_recall` | -0.001539 | [-0.002918, 0.001016] | negative | 是 |
| aggregate | `f1` | 0.000273 | [-0.096837, -0.000004] | positive | 否 |
| aggregate | `jaccard` | 0.000196 | [-0.061236, 0.000047] | positive | 是 |
| aggregate | `accuracy` | -0.001083 | [-0.018427, -0.000931] | negative | 否 |
| aggregate | `balanced_accuracy` | 0.013887 | [-0.018462, 0.013629] | positive | 是 |
| aggregate | `specificity` | -0.001227 | [-0.018454, -0.001013] | negative | 否 |
| aggregate | `useful_cull` | -0.001221 | [-0.018369, -0.001009] | negative | 否 |
| aggregate | `bad_cull` | -0.000138 | [-0.000136, 0.000092] | negative | 是 |
| aggregate | `avg_tp_count` | 0.801205 | [-0.538680, 0.793675] | positive | 是 |
| aggregate | `avg_fp_count` | 7.102410 | [5.884990, 107.277560] | positive | 否 |
| aggregate | `avg_fn_count` | -0.801205 | [-0.793675, 0.538680] | negative | 是 |
| aggregate | `avg_tn_count` | -7.102410 | [-107.277560, -5.884990] | negative | 否 |
| aggregate | `avg_pred_count` | 7.903614 | [6.381451, 107.008747] | positive | 否 |
| aggregate | `avg_candidate_count` | 0.000000 | [0.000000, 0.000000] | zero | 是 |
| aggregate | `avg_gt_count` | 0.000000 | [0.000000, 0.000000] | zero | 是 |
| aggregate | `pred_over_candidate` | 0.001358 | [0.001094, 0.018315] | positive | 否 |
| aggregate | `pred_over_gt` | 0.286088 | [0.236802, 3.713792] | positive | 否 |

## 逐级输入表征增益

以下比较只用于展示从 AABB 到离线几何、再到上下文表征的逐级变化，不参与方向代理的三层路线判定。它们仍使用相同的 validation pose、候选集合和 paired bootstrap。

### `geometry_minus_aabb`

| 统计口径 | 指标 | 差值 | 95% CI | 方向 | 是否跨零 |
|---|---|---:|---|---|---|
| pose_macro | `precision` | 0.328152 | [0.296086, 0.395006] | positive | 否 |
| pose_macro | `recall` | -0.028499 | [-0.034059, -0.010409] | negative | 否 |
| pose_macro | `weighted_recall` | -0.006414 | [-0.007657, -0.003386] | negative | 否 |
| pose_macro | `f1` | 0.364244 | [0.336103, 0.435863] | positive | 否 |
| pose_macro | `jaccard` | 0.305688 | [0.273533, 0.370789] | positive | 否 |
| pose_macro | `accuracy` | 0.261822 | [0.230469, 0.290146] | positive | 否 |
| pose_macro | `balanced_accuracy` | 0.137248 | [0.126355, 0.152349] | positive | 否 |
| pose_macro | `specificity` | 0.302995 | [0.266354, 0.334589] | positive | 否 |
| pose_macro | `useful_cull` | 0.268911 | [0.236751, 0.297428] | positive | 否 |
| pose_macro | `bad_cull` | 0.007089 | [0.005303, 0.008493] | positive | 否 |
| pose_macro | `avg_tp_count` | -0.891566 | [-1.812262, -0.767056] | negative | 否 |
| pose_macro | `avg_fp_count` | -253.123494 | [-354.316039, -242.588830] | negative | 否 |
| pose_macro | `avg_fn_count` | 0.891566 | [0.767056, 1.812262] | positive | 否 |
| pose_macro | `avg_tn_count` | 253.123494 | [242.588830, 354.316039] | positive | 否 |
| pose_macro | `avg_pred_count` | -254.015060 | [-355.691642, -243.518587] | negative | 否 |
| pose_macro | `avg_candidate_count` | 0.000000 | [0.000000, 0.000000] | zero | 是 |
| pose_macro | `avg_gt_count` | 0.000000 | [0.000000, 0.000000] | zero | 是 |
| pose_macro | `pred_over_candidate` | -0.043658 | [-0.060783, -0.042218] | negative | 否 |
| pose_macro | `pred_over_gt` | -9.194614 | [-13.543119, -8.662491] | negative | 否 |
| aggregate | `precision` | 0.035123 | [0.035059, 0.067858] | positive | 否 |
| aggregate | `recall` | -0.032272 | [-0.064848, -0.027908] | negative | 否 |
| aggregate | `weighted_recall` | -0.006796 | [-0.008206, -0.003303] | negative | 否 |
| aggregate | `f1` | 0.061394 | [0.061200, 0.113989] | positive | 否 |
| aggregate | `jaccard` | 0.034787 | [0.034743, 0.066711] | positive | 否 |
| aggregate | `accuracy` | 0.043351 | [0.041794, 0.060241] | positive | 否 |
| aggregate | `balanced_accuracy` | 0.005720 | [-0.006611, 0.012136] | positive | 是 |
| aggregate | `specificity` | 0.043712 | [0.042155, 0.060778] | positive | 否 |
| aggregate | `useful_cull` | 0.043505 | [0.041962, 0.060485] | positive | 否 |
| aggregate | `bad_cull` | 0.000153 | [0.000131, 0.000311] | positive | 否 |
| aggregate | `avg_tp_count` | -0.891566 | [-1.803778, -0.762048] | negative | 否 |
| aggregate | `avg_fp_count` | -253.123494 | [-353.425452, -243.589608] | negative | 否 |
| aggregate | `avg_fn_count` | 0.891566 | [0.762048, 1.803778] | positive | 否 |
| aggregate | `avg_tn_count` | 253.123494 | [243.589608, 353.425452] | positive | 否 |
| aggregate | `avg_pred_count` | -254.015060 | [-354.748331, -244.664270] | negative | 否 |
| aggregate | `avg_candidate_count` | 0.000000 | [0.000000, 0.000000] | zero | 是 |
| aggregate | `avg_gt_count` | 0.000000 | [0.000000, 0.000000] | zero | 是 |
| aggregate | `pred_over_candidate` | -0.043658 | [-0.060699, -0.042133] | negative | 否 |
| aggregate | `pred_over_gt` | -9.194614 | [-13.523264, -8.647968] | negative | 否 |

### `context_minus_geometry`

| 统计口径 | 指标 | 差值 | 95% CI | 方向 | 是否跨零 |
|---|---|---:|---|---|---|
| pose_macro | `precision` | -0.045269 | [-0.053499, -0.002824] | negative | 否 |
| pose_macro | `recall` | 0.013515 | [0.004801, 0.014315] | positive | 否 |
| pose_macro | `weighted_recall` | 0.001356 | [-0.003711, 0.001595] | positive | 是 |
| pose_macro | `f1` | -0.035375 | [-0.052878, -0.009736] | negative | 否 |
| pose_macro | `jaccard` | -0.035268 | [-0.045497, 0.003512] | negative | 是 |
| pose_macro | `accuracy` | -0.001242 | [-0.002005, 0.004451] | negative | 是 |
| pose_macro | `balanced_accuracy` | 0.005112 | [0.003225, 0.006791] | positive | 否 |
| pose_macro | `specificity` | -0.003291 | [-0.003787, 0.004699] | negative | 是 |
| pose_macro | `useful_cull` | -0.002537 | [-0.003116, 0.004119] | negative | 是 |
| pose_macro | `bad_cull` | -0.001295 | [-0.001547, 0.000085] | negative | 是 |
| pose_macro | `avg_tp_count` | -1.950301 | [-2.129518, -0.225891] | negative | 否 |
| pose_macro | `avg_fp_count` | -101.694277 | [-106.192746, -0.383484] | negative | 否 |
| pose_macro | `avg_fn_count` | 1.950301 | [0.225891, 2.129518] | positive | 否 |
| pose_macro | `avg_tn_count` | 101.694277 | [0.383484, 106.192746] | positive | 否 |
| pose_macro | `avg_pred_count` | -103.644578 | [-108.092959, -1.307216] | negative | 否 |
| pose_macro | `avg_candidate_count` | 0.000000 | [0.000000, 0.000000] | zero | 是 |
| pose_macro | `avg_gt_count` | 0.000000 | [0.000000, 0.000000] | zero | 是 |
| pose_macro | `pred_over_candidate` | -0.017813 | [-0.018556, -0.000221] | negative | 否 |
| pose_macro | `pred_over_gt` | -3.751635 | [-3.750010, -0.042126] | negative | 否 |
| aggregate | `precision` | 0.030922 | [-0.003379, 0.033957] | positive | 是 |
| aggregate | `recall` | -0.070595 | [-0.073211, -0.008287] | negative | 否 |
| aggregate | `weighted_recall` | 0.001415 | [-0.003976, 0.001815] | positive | 是 |
| aggregate | `f1` | 0.049065 | [-0.006130, 0.053085] | positive | 是 |
| aggregate | `jaccard` | 0.029508 | [-0.003724, 0.032497] | positive | 是 |
| aggregate | `accuracy` | 0.017143 | [-0.000109, 0.017825] | positive | 是 |
| aggregate | `balanced_accuracy` | -0.026517 | [-0.029405, 0.000128] | negative | 是 |
| aggregate | `specificity` | 0.017562 | [0.000041, 0.018240] | positive | 否 |
| aggregate | `useful_cull` | 0.017478 | [0.000040, 0.018143] | positive | 否 |
| aggregate | `bad_cull` | 0.000335 | [0.000039, 0.000363] | positive | 否 |
| aggregate | `avg_tp_count` | -1.950301 | [-2.124561, -0.226393] | negative | 否 |
| aggregate | `avg_fp_count` | -101.694277 | [-106.512048, -0.228640] | negative | 否 |
| aggregate | `avg_fn_count` | 1.950301 | [0.226393, 2.124561] | positive | 否 |
| aggregate | `avg_tn_count` | 101.694277 | [0.228640, 106.512048] | positive | 否 |
| aggregate | `avg_pred_count` | -103.644578 | [-108.466968, -1.158484] | negative | 否 |
| aggregate | `avg_candidate_count` | 0.000000 | [0.000000, 0.000000] | zero | 是 |
| aggregate | `avg_gt_count` | 0.000000 | [0.000000, 0.000000] | zero | 是 |
| aggregate | `pred_over_candidate` | -0.017813 | [-0.018461, -0.000196] | negative | 否 |
| aggregate | `pred_over_gt` | -3.751635 | [-3.753592, -0.036469] | negative | 否 |

## 路线三层判定

安全层：`未通过`。B-A 和 D-C 都要求 recall、weighted recall 没有超过预登记容忍范围的下降，bad cull 区间上界不超过预登记安全增量，并且所有因子成员都存在合格安全工作点。
分类/剔除层：`通过`。只有 precision、balanced accuracy、F1、useful cull 的区间下界为正，或平均预测数量的区间上界为负，才算有预测贡献。
系统层：`未实现/未通过`。当前 intervention 记录没有同位姿图像或 GLB 资源曲线，因此不能将资源收益写成已证实结论。

## 方向代理的五个独立问题

路线结论把机制使用、分类效果、安全剔除、画面质量和资源成本分开记录。一个输入改变了 logits，只能说明模型读到了它，不能自动说明它改善了最终系统。

| 问题 | 本轮判定 | 证据边界 |
|---|---|---|
| 方向代理是否被模型使用 | 是，B/D 启用、A/C 关闭；B-A 与 D-C 是对应的配对比较 | 这是因子结构和 intervention 诊断的机制证据，不等同于效果提升 |
| 是否改善分类准确性 | 是 | 只有 precision、balanced accuracy、F1、useful cull 或平均预测数的 paired bootstrap 区间满足预注册方向，才算通过 |
| 是否改善安全约束下的剔除效率 | 未证明 | 必须先通过 B-A 和 D-C 的 recall、weighted recall、bad cull 安全层，再看分类/剔除层 |
| 是否改善图像质量 | 未实现或未证明 | 当前矩阵没有同位姿 miss-pixel、wrong-ID 或 p95 图像结果 |
| 是否改善下载和运行时成本 | 未实现或未证明 | 当前 intervention 没有 GLB 字节、首屏时间、WebGPU 或主线程配对记录 |

## 反例解释

例如 `geometry_context_proxy_ray_no_inhibition` seed `20260801` 在其记录中 pose 宏平均 useful cull 为 0.880646，但 pose recall 为 0.939903、weighted recall 为 0.992762，bad cull 为 0.008065。合并所有 pose 后，aggregate useful cull 为 0.970103、bad cull 为 0.000485，这说明大量 TN 可能掩盖 FN；useful cull 的升高不能解释为更好的模型，必须同时满足召回和画面安全约束。

## 系统指标边界

- `visual_utility_recall`：not_available: interventions.json has visible_weights but no unified visual-utility target
- `miss_pixel_rate`：not_available: no same-pose 60-degree instance-ID render attached to this intervention
- `wrong_id_pixel_rate`：not_available: no same-pose 60-degree instance-ID render attached to this intervention
- `extra_pixel_rate`：not_available: no same-pose 60-degree instance-ID render attached to this intervention
- `glb_count_reduction`：not_available: intervention rows do not persist predicted/candidate GLB sets
- `glb_byte_reduction`：not_available: intervention rows do not persist predicted/candidate GLB sets
- `equal_visual_utility_glb_bytes`：not_available: no paired visual-utility/GLB budget curve attached
- `forward_latency_ms`：not_available: stored interventions contain no forward timing
- `webgpu_latency_ms`：not_available: no browser timing attached to this matrix
- `main_thread_ms`：not_available: no browser timing attached to this matrix
- `runtime_memory_bytes`：not_available: no browser memory sample attached to this matrix

## 可复现产物

- 汇总：`neuralstreamweb3d-formal-m4-matrix-summary-v2`，文件为 benchmark 输出目录中的 `summary.json`。
- 路线 JSON 和 Markdown 使用独立的 `m4_formal_route_decision_v2` 名称。
- 旧 M4 summary、旧 route JSON 和旧报告未修改；本报告不改变默认模型、默认前端资产或旧 Route B 结论。

