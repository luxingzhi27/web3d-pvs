# 面向泛化的紧凑遮挡场 PVS：新模型架构、损失函数与消融实验完整设计

**工作版本：2026-09-16**  
**适用项目：`luxingzhi27/web3d-pvs` 后续主线重构**  
**工作名（仅用于内部讨论）：GCOF-PVS，Generalizable Compact Occlusion Field for PVS**

---

## 0. 执行摘要

这次重构的目标不是在当前 V4 上继续增加模块，而是把整篇论文收敛到一条可解释、可泛化、可消融验证的核心链路：

> **服务器端利用粗几何构造与可见标签无关的潜在遮挡关系，将其编译为每个 renderable unit 的紧凑方向性 survival field；浏览器仅用少量固定区域支持点解析查询该 field，即可在详细几何尚未下载之前预测一个 view-cell 的 PVS。**

新的方法应满足四条硬约束：

1. **模型结构从一开始支持跨场景泛化。** 最终模型不得包含 scene ID、instance embedding、per-instance residual、train-observed relation graph 或依赖目标场景标签的归一化统计。
2. **每个保留模块必须有明确物理含义。** 模块分别回答“对象是什么”“哪些对象可能遮挡它”“遮挡沿方向和深度如何变化”“区域内是否存在可见位置”。
3. **训练目标只负责学习概率排序与遮挡场，不负责硬编码 99% safety。** 99% weighted recall 与 bootstrap LCB 完全由独立 calibration 决定。
4. **消融不再是‘把创新点逐个删掉’，而是 hypothesis-driven controls。** 每一组实验回答一个明确问题，并尽量控制输入信息、runtime 维度、共享参数量或监督信号。

推荐的新主线模型可以压缩成：

```text
                OFFLINE / SERVER

Renderable Unit Geometry
        │
        ├─ Frozen 96D geometry feature
        │          │
        │      Shared projector
        │          ▼
        │        32D z_i ─────────────────────────────┐
        │                                            │
        └─ AABB ── Directional proxy-overlap graph   │
                         │                            │
                         ▼                            │
                 Shared edge compiler                │
                         │                            │
                  12 anchor responses                │
                         │                            │
             fixed SH(l<=1) projection               │
                         ▼                            │
                 4 x 7 survival field C_i            │
                                                      │
========================== transfer ==================│
                                                      │
                ONLINE / WEB CLIENT                   │
                                                      │
View-cell ── 9 fixed support points                   │
                  │                                   │
                  └── analytic field queries ─────────┘
                              │
                 S_center / S_max / S_mean / S_min
                              │
                     + scale-free query geometry
                              │
                         tiny shared MLP
                              │
                         visibility score
```

推荐最终训练目标只有两项：

\[
\boxed{\mathcal L = \mathcal L_{vis} + \mathcal L_{surv}}
\]

其中 `L_vis` 是无额外超参数的 pose-balanced logistic loss，`L_surv` 是 survival event/right-censor likelihood。两者都是归一化的 Bernoulli 型负对数似然，因此默认 1:1，不再引入 recall guard、CVaR、tail mining、margin、temperature、relation auxiliary、复杂 loss curriculum。

论文的核心创新不应再表述为“relation + survival + moment + calibration + loss 的若干模块”，而应表述为：

- **Pre-geometry from-region visibility formulation**：详细几何尚未到达客户端时，先预测整个观察区域的潜在可见集合。
- **Generalizable compact occlusion-field compiler**：从纯几何潜在遮挡关系直接生成固定大小、可连续方向查询的紧凑 survival field，且无 per-instance 可学习参数。
- **Analytic region-support query for Web runtime**：用固定少量区域支持点查询同一遮挡场，以极低运行成本近似 from-region PVS，并把 safety 与概率学习解耦到独立 calibration。

如果实验表明某个复杂组件没有明显改善 CNOR、Useful Cull、AP、跨 seed 稳定性或跨场景泛化，则应主动删除，而不是为了保持“创新点数量”继续保留。

---

# 1. 为什么当前 V4 需要结构性重构

## 1.1 当前模型最大的风险不是参数量，而是“因果链条不清楚”

当前 V4 同时包含：

- 96D fixed geometry feature；
- train-observed relation CSR；
- direction/depth shell；
- instance/local/structural 多层 attention；
- 4x7 survival coefficients；
- learned relation condition；
- learned direction basis；
- spectral moment envelope；
- boundary summary；
- low-rank summary；
- shared trunk；
- per-instance residual calibration；
- visibility/utility/download heads；
- pose-balanced BCE、recall guard、CVaR、hard-tail separation、survival NLL、relation consistency、regularization、calibration regularization 等训练项。

从工程角度，这套系统是完整的；从论文角度，问题是 reviewer 很难区分：

1. 哪个模块是真正不可替代的 inductive bias；
2. 哪些模块只是历史迭代留下的性能修补；
3. 哪些提升来自额外的 scene-specific memory；
4. 哪些提升来自复杂 loss 对安全阈值附近 score distribution 的人为塑形。

旧 HKUST 消融已经显示这种风险：`No Survival` 相比 Full 降得较多，但同容量的 Generic-28 已经追回了绝大部分差距；`No Moment` 和 `No Tail Margin` 的差距较小。因此仅靠“逐模块删除后都有一点下降”并不足以形成强方法故事。

## 1.2 当前泛化性存在结构性障碍

即使删除 per-instance residual，当前系统仍不能自然宣称 zero-shot，因为：

- relation graph 来源于 **train-observed visibility relations**；
- depth normalization 使用目标场景 train split 统计；
- residual calibration 是显式 scene-specific memory；
- Generic-28 等对照本质上也允许实例记忆。

因此新的架构必须把“泛化”作为输入契约，而不是在论文最后追加一个 cross-scene 实验。

## 1.3 当前 loss 的主要问题是职责重叠

当前 task loss 同时通过：

- PBCE 学基本分类；
- Recall Guard 逼近 99% recall；
- Pose-CVaR 保护困难 view-cell；
- Tail margin 拉开困难正负样本；
- calibration 最后又重新选择 99% safety operating point。

这形成了明显的职责重叠：训练和 calibration 都在负责 safety。新的设计应把二者彻底分开。

---

# 2. 新架构的设计原则

## 2.1 每个模块只解决一个问题

| 模块 | 唯一职责 | 不允许承担的职责 |
|---|---|---|
| Geometry Descriptor | 描述目标/遮挡物自身几何 | 不记忆 scene 或 instance ID |
| Proxy Relation Graph | 枚举“可能成为遮挡物”的几何关系 | 不使用 visibility GT |
| Occlusion Field Compiler | 把一组潜在遮挡关系压成紧凑方向场 | 不做 runtime 图传播 |
| Survival Field | 表示遮挡随方向和深度变化 | 不直接记忆最终 PVS 标签 |
| Region Support Query | 近似整个 view-cell 的存在性可见 | 不展开大量 subposes |
| Visibility Head | 对解析遮挡信息做小幅任务映射 | 不重新学习整个场景遮挡结构 |
| Calibration | 选择满足安全门的 operating point | 不更新模型参数 |

## 2.2 泛化的硬约束

最终 Full 模型必须满足：

- 不输入 scene ID；
- 不输入 instance ID embedding；
- 不允许可学习 per-instance parameter table；
- relation graph 只能来自 AABB/粗几何；
- 所有距离、尺度采用 target-relative / scale-free 表达；
- depth coordinate 不依赖 scene train quantile；
- model weights 在所有场景之间共享；
- 新场景只允许做确定性 server-side preprocessing；
- strict zero-shot 模式下连 target calibration label 都不能用；
- calibration-only 模式允许只用 target calibration split 选阈值，但模型权重保持冻结。

---

# 3. 新模型：Generalizable Compact Occlusion Field

## 3.1 输入单位与已有资产复用

不改变现有 renderable-unit 协议。每个 unit 保留：

- world-space AABB；
- resource / GLB mapping；
- 现有 frozen geometry encoder 输出的 96D feature；
- candidate / view-cell GT 用于训练与评价。

这意味着无需重新构建最昂贵的场景划分、Color-ID GT、streaming 管线和 HZB baseline。

---

## 3.2 Shared Geometry Projector：96D -> 32D

### 目标

把当前 96D 固定几何特征压缩成跨场景共享的 runtime descriptor：

\[
z_i = E_g(g_i), \qquad z_i \in \mathbb{R}^{32}.
\]

推荐结构：

```text
Input: 96D frozen geometry feature
Linear(96, 64)
SiLU
Linear(64, 32)
Output: z_i in R^32
```

约束：

- `E_g` 是共享网络；
- 不允许 target-scene-specific projection；
- 训练完成后 server 预计算 z_i，浏览器不运行 `E_g`；
- AABB 绝对尺寸不直接拼入 z_i，尺度关系通过后续相对几何特征显式提供。

为什么选 32D：

- 相比当前 96D 可直接降低 startup asset；
- 32D 足以作为 edge compiler 和 final head 的 compact object descriptor；
- 16/32/64D 作为容量/资产 Pareto 消融，而不是把 32D 当成未经验证的最佳值。

---

# 4. Geometry-derived Directional Proxy Relation Graph

这是新架构最关键的泛化改动：**关系图不再从 visibility GT 提取，而从 AABB 纯几何确定性构造。**

## 4.1 12 个固定方向 anchor

沿用 12 个固定方向 anchor，以覆盖水平方向和上下斜向。记为：

\[
a_k \in \mathbb{R}^{3},\quad k=1,\dots,12.
\]

每个 anchor 都预先构造两个正交屏幕轴 `u_k, v_k`，满足：

\[
u_k \perp a_k,\qquad v_k \perp a_k,\qquad u_k\perp v_k.
\]

这里的 `a_k` 定义为“从 target 指向潜在 camera 的方向”。

## 4.2 AABB 正交投影关系

对每个 instance i 的 AABB 8 个角点，分别投影到：

- 屏幕横轴 `u_k`；
- 屏幕纵轴 `v_k`；
- 深度轴 `a_k`。

得到三个区间：

\[
U_i^k=[u_i^-,u_i^+],\quad
V_i^k=[v_i^-,v_i^+],\quad
D_i^k=[d_i^-,d_i^+].
\]

对 source j 与 target i，如果：

1. `U_j^k` 与 `U_i^k` 相交；
2. `V_j^k` 与 `V_i^k` 相交；
3. source 在 anchor 方向上位于 target 前方；

则 j 是 i 在 anchor k 下的 **potential occluder**。

推荐前后判定先使用 AABB center projection：

\[
(c_j-c_i)\cdot a_k > 0.
\]

不使用可见标签，不声称这是精确遮挡，只把它当做宽松 potential relation。

## 4.3 Top-K bounded relation

对每个 `(target i, anchor k)`，按以下顺序筛选最多 K 个 source：

1. 更小的前向 depth gap 优先；
2. projected overlap ratio 更大优先；
3. source projected area 更大优先；
4. source ID 作为确定性 tie-break。

主配置：

```text
K = 8 potential occluders per target per anchor
12 anchors
maximum 96 offline edges / target
```

K=8 与当前项目常用 relation K 数量级一致，且 runtime 完全不保留这些边。K=4/8/16 只做附录敏感性。

## 4.4 推荐 edge feature（8D）

每条边只使用相对、尺度无关的几何量：

```text
1-3: unit relative direction (c_j - c_i) / ||c_j - c_i||
4:   log1p(||c_j-c_i|| / (r_i + eps))
5:   log((r_j + eps) / (r_i + eps))
6:   overlap_area / target_projected_area
7:   overlap_area / source_projected_area
8:   normalized front depth gap
```

其中 `r_i` 为 target AABB bounding-sphere radius。

禁止加入：

- world absolute XYZ；
- scene extent；
- scene ID；
- visibility frequency；
- train observed confidence；
- target-scene-specific statistics。

这样新场景 relation preprocessing 只依赖粗几何。

## 4.5 构图实现建议

不要 O(N^2) 枚举。对每个 anchor：

1. 把所有 projected rectangle 按 depth 排序；
2. 使用 interval tree / R-tree / sweep-line 维护可能与当前 target 的 U/V rectangle 重叠的前方 source；
3. 对 overlap candidate 维护固定大小 top-K heap；
4. 输出 deterministic CSR。

建议新 schema：

```text
pvs-geometry-proxy-relation-csr-v1
```

metadata 必须写：

```text
source = geometry_only
usesVisibilityLabels = false
anchors = 12
K = 8
projection = orthographic_aabb_overlap
```

这项 provenance 必须写死，防止以后实验不小心重新混入 train-observed edges。

---

# 5. Shared Occlusion Field Compiler

## 5.1 Edge message

对每条 potential relation `j -> i`：

\[
m_{ji}=\phi([z_i,z_j,e_{ji}]),
\]

输入维度：

```text
z_i: 32
z_j: 32
edge: 8
Total: 72
```

推荐：

```text
Linear(72, 64)
SiLU
Linear(64, 32)
SiLU
```

得到 32D edge message。

## 5.2 单层 anchor aggregation

不再使用 instance -> local -> structural 三层 hierarchy。

对每个 `(i,k)`：

```text
attention_logit = Linear(32,1)(m_ji)
alpha_ji = softmax over sources j within same target-anchor cell
h_i,k = sum_j alpha_ji * m_ji
```

如果该 anchor 没有 candidate source，则 `h_i,k = 0`。

这里只保留一层 learned attention，理由明确：同一个 anchor 下多个 potential occluder 对 target 的重要度并不相同。

不再需要：

- local group pooling；
- structural group pooling；
- graph propagation；
- relation confidence ranking head；
- edge classification head；
- depth-order auxiliary head。

## 5.3 Geometry base + relation delta

每个 target 自身几何先产生一个共享 base：

\[
q_i^{base}=H_b(z_i)\in\mathbb R^7.
\]

推荐：

```text
Linear(32,32) -> SiLU -> Linear(32,7)
```

每个 anchor token 产生 relation delta：

\[
\Delta q_{ik}=H_r(h_{ik})\in\mathbb R^7.
\]

推荐：

```text
Linear(32,32) -> SiLU -> Linear(32,7)
```

最终 anchor response：

\[
q_{ik}=q_i^{base}+\Delta q_{ik}.
\]

这比当前多层 base/gate/delta 结构更简单：没有额外 gate；没有 source-specific residual；没有 scene-specific memory。

---

# 6. 固定一阶球谐方向压缩：12x7 -> 4x7

## 6.1 为什么不用 learned direction basis

当前方向 basis 由 MLP 学习，会使 28D 表示的含义随训练改变。新架构使用固定的一阶实球谐/等价低阶方向基：

\[
b(d)=[1,d_x,d_y,d_z].
\]

它对应：

- 1 个方向无关项；
- 3 个一阶方向变化项。

这样 rank=4 有明确解释，不再只是经验选择。

## 6.2 从 12 anchor response 投影到 4x7 field

定义固定矩阵：

\[
B_{k,:}=b(a_k),\qquad B\in\mathbb R^{12\times4}.
\]

预计算：

\[
P=B^+\in\mathbb R^{4\times12},
\]

其中 `B+` 是 Moore-Penrose pseudoinverse。

对：

\[
Q_i\in\mathbb R^{12\times7},
\]

得到：

\[
\boxed{C_i=P Q_i\in\mathbb R^{4\times7}}.
\]

`P` 是固定常量，不训练。

这 28 个 FP16 数就是最终每个 unit 的 **directional survival field**。

---

# 7. 解析 Survival Field

对任意查询方向 d：

\[
\theta_i(d)=b(d)^T C_i\in\mathbb R^7.
\]

七个 raw parameter 转换为：

\[
p_{no}=\sigma(\theta_0),
\]

\[
(\pi_1,\pi_2)=softmax(\theta_{1:3}),
\]

\[
(\mu_1,\mu_2)=0.05+0.90\cdot\sigma(\theta_{3:5}),
\]

\[
(\sigma_1,\sigma_2)=0.03+softplus(\theta_{5:7}).
\]

按 μ 排序，并同步重排 π、σ。

定义 blocker CDF：

\[
F_i(d,\rho)=
(1-p_{no})
\sum_{m=1}^{2}
\pi_m\,
\sigma\left(\frac{\rho-\mu_m}{\sigma_m}\right).
\]

survival：

\[
\boxed{S_i(d,\rho)=1-F_i(d,\rho)}.
\]

物理解释：

> 从 target 朝查询 camera 方向行进，在相对深度 ρ 之前仍未遭遇足以遮挡 target 的 blocker 的概率。

## 7.1 新的 scale-free depth coordinate

为了从结构上支持新场景，不再使用目标场景 train split 的 q01/q99 depth quantile。

定义：

\[
\boxed{
\rho_i(x)=\frac{\|x-c_i\|}{\|x-c_i\|+r_i+\epsilon}
}
\]

它天然位于 [0,1)，并对场景绝对尺度不敏感。

因此新场景不需要训练标签或 train-depth statistics 即可查询 field。

---

# 8. From-region Query：9 个固定支持点，不再使用 learned spectral moment

## 8.1 支持点定义

view-cell 是水平圆盘，中心为 c，半径为 R。

使用：

- 1 个 center；
- 8 个圆周支持点，角度 `0,45,...,315 deg`。

\[
x_0=c,
\]

\[
x_{m+1}=c+R(\cos\theta_m\,r+\sin\theta_m\,f),
\quad \theta_m=m\pi/4.
\]

这里 r/f 是相机的水平 right/forward basis。

每个 support point 只做：

1. target -> support direction；
2. scale-free depth；
3. 4x7 field 的 analytic survival evaluation。

不为每个点运行独立 MLP。

## 8.2 区域统计量

得到 9 个 survival：

\[
S_{i0},\dots,S_{i8}.
\]

固定压缩成：

\[
S_{center}=S_{i0},
\]

\[
S_{max}=\max_k S_{ik},
\]

\[
S_{mean}=\frac19\sum_k S_{ik},
\]

\[
S_{min}=\min_k S_{ik}.
\]

对应含义：

- `S_max`：区域内是否存在较可见的位置，最贴近 PVS 的 existential semantics；
- `S_mean`：区域整体典型遮挡程度；
- `S_min`：区域最不利位置；
- `S_center`：中心点本身。

## 8.3 Scale-free query geometry

再计算三个解析特征：

```text
rho_center = d_center / (d_center + r_i)
angular_scale = r_i / (d_center + r_i)
cell_scale = R / (d_center + r_i + R)
```

这三个量均不依赖 scene absolute scale。

---

# 9. Tiny Visibility Decoder

最终输入：

```text
32D z_i
4D survival region statistics
3D scale-free query geometry
----------------------------
39D total
```

推荐：

```text
Linear(39,32)
SiLU
Linear(32,1)
```

输出 visibility logit。

删除：

- relation_condition_head；
- direction_basis_head；
- spectral moment query；
- boundary_summary_head；
- low-rank summary；
- utility head；
- download head；
- per-instance calibration residual。

下载调度继续直接使用 visibility score 聚合到 resource/GLB，不宣称模型学习独立 download utility。

---

# 10. Runtime Asset 与 Web 实现

主配置每 instance：

```text
32D geometry descriptor FP16 = 64 B
4x7 survival field FP16     = 56 B
AABB FP32                    = 24 B
resource ID uint32            = 4 B
-----------------------------------
approx.                      = 148 B / instance
```

不含共享 MLP 权重和对齐开销。

HKUST 18,831 instances 的粗略 per-instance table：

\[
18831\times148 \approx 2.79\text{ MB}.
\]

这个数字只是设计预算，正式论文必须由最终 exporter 实测，不可提前当结果使用。

WebGPU runtime 每 candidate：

1. 后退 66° candidate AABB test；
2. 读取 32D + 28D；
3. 生成 9 个 support direction/depth；
4. 每点执行 4x7 dot + 两分量 logistic mixture；
5. 做 max/mean/min；
6. 执行一次 39->32->1 MLP；
7. 阈值判断；
8. 聚合到 GLB；
9. 真实 60° 视锥 re-filter 维持原有逻辑。

重要：运行时仍然不传 relation graph，因此跨场景泛化不会增加客户端图结构成本。

---

# 11. 新损失函数：Dual-Likelihood Objective

## 11.1 原则

训练只做两件事：

1. 最终 PVS 分类正确；
2. 中间 survival field 符合遮挡事件语义。

99% safety 不进入损失。

---

## 11.2 Pose-balanced Visibility Logistic Loss

对 pose p：

- candidate 集合 `C_p`；
- visible 集合 `V_p`；
- occluded/negative 集合 `O_p=C_p\V_p`；
- logit `z_pi`；
- visible contribution `w_pi`。

### 正样本固定重要度

为避免大屏幕实例完全支配 loss，同时保留视觉贡献信息：

\[
\bar w_{pi}=\frac{w_{pi}}{\max_{j\in V_p}w_{pj}+\epsilon},
\]

\[
\boxed{a_{pi}=1+\bar w_{pi}}.
\]

因此 `a` 始终在 [1,2]，没有需要调节的 floor/power 超参数。

正样本 loss：

\[
L_p^+=
\frac{
\sum_{i\in V_p} a_{pi}\,softplus(-z_{pi})
}{
\sum_{i\in V_p}a_{pi}
}.
\]

负样本 loss：

\[
L_p^-=
\frac1{|O_p|}
\sum_{i\in O_p} softplus(z_{pi}).
\]

如果同时有两类：

\[
L_p=\frac12(L_p^++L_p^-).
\]

如果只有一类，则只使用存在的那一项。

最后：

\[
\boxed{
\mathcal L_{vis}=\frac1P\sum_p L_p
}.
\]

### 为什么这样设计

- pose-balanced：大 candidate pose 不会压倒小 pose；
- class-balanced：大量 occluded candidate 不会淹没 visible class；
- 正样本视觉权重只有固定 1~2 倍调制；
- 无 recall target；
- 无 hard mining；
- 无 pairwise margin；
- 无 temperature；
- 无 curriculum。

---

## 11.3 Survival Event / Right-censor Likelihood

对于 survival observation j：

- `e_j=1`：目标深度之前发生 blocker event；
- `e_j=0`：right-censored，到观察深度前未发生 blocker；
- `S_j`：field 的 survival 概率；
- `omega_j`：由 stratified sampler 提供的概率校正权重。

\[
\boxed{
\mathcal L_{surv}
=-
\frac{
\sum_j\omega_j
[e_j\log(1-S_j)+(1-e_j)\log S_j]
}{
\sum_j\omega_j
}
}.
\]

这不是“为了提升指标的辅助 trick”，而是 survival representation 的定义性监督。

---

## 11.4 总损失

主配置：

\[
\boxed{
\mathcal L=\mathcal L_{vis}+\mathcal L_{surv}
}.
\]

默认 1:1 的理由：两项均为按自身样本归一化的 Bernoulli 型 NLL，量纲一致。不要先引入 `lambda_surv=0.25`；如果 1:1 出现明显数值失衡，再只做 0.5/1/2 的附录敏感性，不按 validation CNOR 搜最优权重。

优化器：

```text
AdamW
weight_decay = 1e-5
```

不再显式叠加额外 global L2 loss。

---

# 12. Safety 与 Calibration 的职责分离

模型训练后输出连续 score。

### In-scene 模式

在 calibration split 上选择满足：

\[
WR > 0.99
\]

且：

\[
LCB_{95\%}(WR)>0.99
\]

的最大安全阈值。

### Strict zero-shot 模式

目标场景不使用任何标签。阈值由 source-scene calibration sets 冻结：

推荐选择满足 **每个 source scene** 都通过 WR/LCB gate 的最高共同阈值，而不是简单把 source 样本混池后让大场景支配。

### Calibration-only transfer

模型权重完全冻结，只用目标场景 calibration split 重新选一个阈值。

因此论文可以明确区分：

```text
zero-shot model + zero-shot threshold
zero-shot model + target threshold calibration
scene-specific retraining  (不作为新主线)
```

---

# 13. 训练协议：从一开始按泛化方式写代码

## 13.1 Scene-balanced sampling

多场景训练时，不按 observation 总数混池。

推荐每 step：

1. uniform sample 一个 source scene；
2. 从该 scene 的 train poses uniform sample P 个 pose；
3. `include_empty=True`，允许候选存在但 visible=0 的 pose；
4. 独立从该 scene 的 survival observations 采样。

这样 Big City / IFCBench 不会因为样本更多压倒小场景。

## 13.2 不使用 ambiguity-balanced hard sampling 作为主方法

主方法先使用 uniform pose sampling。原因：

- 新 loss 已经按 pose 和 class 平衡；
- hard sampling 会成为另一个 training trick；
- 如果后续 hard sampling 真有明显收益，放到 supplementary sensitivity，而不是核心方法。

## 13.3 HKUST 重跑控制

第一阶段 HKUST 重跑必须复用当前：

```text
train        5926
calibration   659
validation    730
test          684
```

正式 architecture/loss 选择只看 train/calibration/validation；new V5 完全冻结后才允许读取 test。

---

# 14. 新消融实验总原则

新消融不再围绕“我们的每个创新模块都删一下”，而围绕四个科学问题：

1. **训练目标是否需要复杂 safety engineering？**
2. **遮挡信息到底来自 free memory、自身几何还是 scene relation？**
3. **方向性 compact field 是否必要，低阶表示是否足够？**
4. **from-region 查询究竟需要多少空间支持，tiny decoder 是否真的必要？**

所有主消融采用同样：

```text
HKUST split
40 x 900 updates
3 seeds: 20260801/02/03
AdamW
same calibration protocol
same bootstrap 10,000
same candidate CSR
same GT
```

除被研究变量外不改变其他配置。

---

# 15. Group A：Loss / Objective 消融

架构全部固定为新 Full：

```text
geometry_dim = 32
relation_graph = geometry_proxy_aabb_12dir_top8
relation_compiler = single_anchor_attention
field_basis = SH l<=1, 4x7
region_support = 9
visibility_head = 39->32->1
instance residual = disabled
relation auxiliary = disabled
```

## A0. Visibility-only

```text
L = L_vis
survival_loss_weight = 0
```

目的：最终 PVS 标签本身能否让 28D field 自行退化成有效隐变量？

## A1. Proposed Dual-Likelihood（新 Full）

```text
L = L_vis + L_surv
```

目的：显式物理 survival supervision 是否提高性能、稳定性和泛化。

## A2. NeuralPVS-style candidate objective + survival

仅替换 task loss。

对每个 pose 适配 NeuralPVS 的 weighted Dice：

\[
D=\frac{2TP}{2TP+\alpha FP+(1-\alpha)FN},\quad \alpha=0.001.
\]

再使用 candidate 版本 NoGuess/RVL：

\[
L_{NG}
=\frac{\sum_{y=0}p}{|C_p|}
+1-rac{\sum_{y=1}p}{|C_p|}.
\]

task：

\[
L_{task}=0.99(1-D)+0.01L_{NG}.
\]

总损失：

\[
L=L_{task}+L_{surv}.
\]

目的：证明新 loss 的收益不是简单复制 NeuralPVS 的强 FN-biased objective。

注意论文措辞必须是 **NeuralPVS-style loss adapted to candidate units**，不能称为 NeuralPVS method baseline。

## A3. Legacy V4 task loss + survival

保持新架构，只替换 task loss 为旧 HKUST 正式配置：

```text
pose_balanced_rvl_contrastive
recall_guard_weight = 0.30
recall_target = 0.99
recall_temperature = 0.05
pose_cvar_fraction = 0.25
pose_cvar_weight = 0.25
separation_weight = 0.20
positive_mass_fraction = 0.005
positive_count_cap = 64
negative_fraction = 0.01
negative_count_cap = 256
margin = 0.50
logit_temperature = 0.25
positive_importance_floor = 0.5
positive_importance_power = 0.5
legacy guard/tail curriculum retained exactly
```

但：

```text
relation auxiliary = 0
instance residual = disabled
explicit regularization = 0
```

目的：只比较 task objective 的复杂度与稳定性，不让旧 V4 其他机制造成混淆。

### Group A 判定

如果 A1 与 A3 的 CNOR/Useful Cull 接近，但 A1：

- seed std 更低；
- score Spearman 更高；
- calibration threshold std 更低；
- cross-scene 更稳；

则论文应采用 A1，即使 A3 单 seed 稍高。

---

# 16. Group B：Occlusion Representation 消融

统一使用 A1 Dual-Likelihood 和 9-point region query。

## B0. Geometry-only

```text
z_i = 32D
no C_i field
no relation graph
no survival loss
visibility decoder input = z_i + 3 query geometry features
```

为减少代码分叉，也可在统一 39D layout 中把 4 个 survival stats 置为固定常数，但必须在 metadata 中声明。

回答：没有任何显式遮挡状态时能做到多少。

## B1. Free Per-instance Field（scene memory control）

```text
C_i is nn.Parameter[N,4,7]
initialized N(0,0.02)
no relation compiler
no geometry->field network
same analytic survival query
L = L_vis + L_surv
```

这组运行时维度与 Full 完全相同，但有显式 per-instance memory。

回答：同样 28D，如果允许背答案能做到多少？

它不具备 zero-shot 能力，因此 generalization 表中标 `N/A`，不能把它当可部署泛化方法。

## B2. Geometry-conditioned Field

```text
C_i = MLP(z_i)
MLP output = 28
no relation graph
no per-instance parameter
same survival query
L = L_vis + L_surv
```

为了公平，可用 `_capacity_matched_widths` 类似机制，让该 MLP 的共享参数量与 Full relation compiler 在 ±1% 内匹配。

回答：仅自身几何是否能预测其遮挡环境。

## B3. Geometry-derived Relation Field（新 Full）

```text
geometry proxy graph
shared edge compiler
12 anchor responses
fixed SH1 projection
no per-instance residual
```

回答：scene context / potential occluders 是否提供必要信息。

## B4. Full + Legacy Per-instance Residual

在 B3 上加入旧 residual：

```text
residual shape = [N,4,7]
max_abs = 4.0
sparse_instance_penalty = 3.0
regularization weight = 0.02
warmup = 10%
ramp = 20%
```

其他配置不变。

回答：如果允许 scene-specific memory，实际能提高多少。

**保留门槛应很高**：如果收益 <1 CNOR point、Bytes/seed stability 无明确改善，最终模型删除 residual。

## B5. Train-observed Relation Graph Upper Bound（附录）

使用同一个 new compiler，但把 geometry proxy edges 换成旧 train-observed relation edges。

```text
compiler/network identical
only edge source differs
```

这不是可泛化主方法，只作为 upper-bound diagnostic：

> geometry-only proxy graph 相比使用标签观测到的 relation 损失了多少？

如果差距很小，新的 generalization story 会明显更强。

---

# 17. Group C：Directional Field 消融

统一使用 geometry-derived relation compiler 和 9-point query。

## C0. Isotropic Field / SH0

```text
basis = [1]
field shape = 1x7
runtime occlusion payload = 7 FP16
```

所有方向共享同一遮挡深度分布。

回答：方向性是否真的必要。

## C1. First-order SH / SH1（新 Full）

```text
basis = [1, dx, dy, dz]
field shape = 4x7
```

回答：低阶方向变化是否足以表达主要遮挡结构。

## C2. Second-order SH / SH2

使用标准固定二阶实 SH，9 个 basis：

```text
field shape = 9x7
```

参数量与 startup bytes 明显增加。

回答：Full 是否只是因为方向容量不够，还是 SH1 已经处于合理 Pareto 点。

论文主结论不是要求 SH1 一定最高，而是希望得到：

```text
SH0 clearly worse
SH1 near-best
SH2 marginal gain / larger asset
```

如果 SH2 显著更好，则不能强行保留 SH1，应根据 Pareto 重新选择。

---

# 18. Group D：From-region Query 消融

模型/field 完全相同，只改变 support query。

## D0. Center-only

```text
support_count = 1
points = center
region stats = [S_center,S_center,S_center,S_center]
```

回答：单点预测是否足以近似区域 PVS。

## D1. 5-point Cross

```text
center
+R right
-R right
+R forward
-R forward
```

统计 `center/max/mean/min`。

## D2. 9-point Ring（新 Full）

```text
center + 8 ring points at 45-degree intervals
```

回答：对角方向 coverage 是否带来有效收益。

## D3. Field-only Max（decoder ablation）

仍使用 9 点，但完全删除 39->32->1 head：

\[
p_i=S_{max}.
\]

训练时 `L_vis` 直接作用于 `S_max`，同时保留 `L_surv`。

回答：tiny decoder 是否真的必要，还是 analytic field 已经足够完成 PVS。

如果 D3 与 D2 很接近，应优先删除 decoder，论文会更漂亮。

## D4. 5x Point Runtime Reference（不重训）

使用 center-only model，在 center/+X/-X/+Z/-Z 五个位置分别独立 point query，结果 union。

比较：

```text
5 independent point queries
vs
1 region query with 5/9 analytic field evaluations + one shared head
```

重点报告质量和 runtime，而不仅是 CNOR。

---

# 19. Group E：资产/复杂度敏感性（附录或 Pareto 图）

## E1. Geometry descriptor dimension

```text
16D / 32D / 64D
```

其他不变。

报告：

- startup bytes；
- CNOR；
- Useful Cull；
- WebGPU p50/p95。

## E2. Proxy graph K

```text
K=4 / 8 / 16 per anchor
```

这是离线复杂度与 field quality 的敏感性，不影响 runtime asset。

## E3. Support count

已经由 1/5/9 主消融覆盖，不再额外扫更多点。

---

# 20. 新的稳定性指标：必须成为正式结果的一部分

新架构的目标之一就是避免“某个 seed 刚好落在更好的 safety operating point”。因此三 seed 不能只报告 mean CNOR。

## 20.1 主指标 mean ± sample std

至少：

- weighted recall；
- WR 95% LCB；
- CNOR；
- Useful Cull；
- Bad Cull；
- pose-macro AP / AP lift；
- GLB byte reduction；
- calibration threshold。

## 20.2 Seed score correlation

对 validation 中完全相同 `(pose, instance)` observations，保存 float32 score。

报告三对 seed 的：

- Spearman correlation；
- Pearson correlation。

解释：

- ranking correlation 高、阈值后指标波动大：主要是 calibration/margin 问题；
- ranking correlation 本身低：representation 训练不稳定。

## 20.3 Safety frontier gap

统计：

```text
positive score p05
negative score p95
p05_positive - p95_negative
```

也可以画全 validation 的 PR/WR-UsefulCull frontier。

不要把这个 gap 作为 loss，只把它作为稳定性诊断。

---

# 21. 低支持实例分析：用于证明结构化表示不是 28D 记忆

按 train split 中每个 instance 的 candidate observation count 分桶：

```text
Q1: lowest 25%
Q2: 25-50%
Q3: 50-75%
Q4: highest 25%
```

或额外报告 zero/very-low support。

比较：

```text
B1 Free Field
B2 Geometry Field
B3 Relation Field
```

每个 bucket 报：

- AP；
- weighted recall；
- CNOR / Useful Cull（有足够负样本时）；
- score calibration。

希望验证的假设：

> Free per-instance field 在低支持单位上更容易失效，而 shared geometry/relation compiler 依赖结构性输入，因此对低训练覆盖单位更稳。

这比单独比较 Full 0.904 vs Generic28 0.894 更能说明结构化表示价值。

---

# 22. Cross-scene Generalization：从模型设计阶段固定协议

## 22.1 不要一开始就宣称 universal zero-shot

目前可用场景数量有限，建议论文用词：

> cross-scene transfer / unseen-scene generalization across heterogeneous scenes

而不是 universal generalization。

## 22.2 Leave-one-scene-out（LOSO）

场景候选：

```text
HKUST
IFCBench / Metropolis
Sponza
Viking Village
Big City
```

架构、loss、K、descriptor dim、support count 在 LOSO 前全部冻结。

每个 fold：

```text
train: 4 scenes train splits
source calibration: 4 scenes calibration splits
held-out: 1 scene
```

## 22.3 两个 transfer operating points

### Zero-shot-global

模型权重冻结；阈值只由 4 个 source calibration scenes 决定。

建议要求每个 source scene 独立满足 safety gate，选最高共同安全阈值。

目标场景完全不使用 calibration label。

报告：

- held-out WR/LCB；
- CNOR；
- AP；
- Useful Cull；
- threshold shift 不适用。

### Target-calibrated

模型权重仍冻结，只用 held-out scene calibration split 重新选阈值。

报告：

- 同样的 test metrics；
- target threshold 与 global threshold 的差值。

这可以把：

```text
representation generalization
```

和：

```text
probability/threshold transfer
```

明确分开。

## 22.4 泛化对照

跨场景表至少放：

```text
Geometry Field (B2)
Relation Field Full (B3)
Free Field: N/A for zero-shot
```

如果 B3 相比 B2 在 held-out scene 上更稳定，才真正支持“relation compiler 学到可迁移遮挡先验”。

---

# 23. Baseline 重新组织

主论文 baseline 建议分成三类，不要把所有东西混在一张表里。

## Classical / Geometry-resident

- Keep-All；
- AABB frustum / AABB+Ray MLP；
- Geometry-shell HZB lossless；
- Geometry-shell HZB equal-asset。

## Learned / Existing Project

- Current V4 Legacy（冻结旧结果或按新 split 重放，不再作为新消融的一部分）；
- New GCOF-PVS Full。

## NeuralPVS

如果能可靠把 NeuralPVS voxel/froxel output 映射到相同 renderable-unit protocol，可以做方法 baseline；如果映射本身会改变任务，不强行数值比较。

无论是否运行完整 NeuralPVS，都应保留 A2 的 **NeuralPVS-style loss control**，但明确它只是 loss 对照，不是 NeuralPVS 方法对照。

---

# 24. 评价指标层次

## Safety（第一优先级）

- Aggregate Weighted Recall；
- one-sided 95% bootstrap LCB；
- Bad Cull；
- image miss / wrong-ID。

## Occlusion Efficiency

- CNOR；
- Useful Cull；
- Occlusion Recall（作为辅助，不单独主导结论）；
- predicted count。

## Threshold-free Quality

- pose-macro AP；
- positive prevalence；
- AP lift；
- PR curve。

## Streaming

- GLB byte reduction；
- Bytes@95 / Bytes@99；
- fixed-bandwidth time-to-correct-frame；
- real scheduler replay。

## System

- startup asset bytes；
- bytes / instance；
- WebGPU p50/p95；
- WASM p50/p95；
- A6000 / M2 / phone；
- candidate-count scaling。

## Generalization / Stability

- LOSO zero-shot metrics；
- target-calibrated metrics；
- seed std；
- seed score correlation；
- calibration threshold std；
- low-support bucket metrics。

---

# 25. 代码重构方案

建议**不要直接把现有 V4 文件改成不可复现状态**。新建 V5 路径，旧 V4 保持冻结。

## 25.1 新文件

```text
neural_instance_culling/model/v5/
    geometry_projector.py
    proxy_relation_graph.py
    occlusion_field_compiler.py
    survival_field.py
    region_support_query.py
    visibility_loss.py
    pvs_model_v5.py
    train_pvs_v5.py
```

benchmark：

```text
neural_instance_culling/benchmark/
    run_hkust_v5_ablation.py
    summarize_hkust_v5_ablation.py
    run_v5_cross_scene_loso.py
```

## 25.2 `geometry_projector.py`

```python
class GeometryProjector(nn.Module):
    def __init__(self, in_dim=96, out_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.SiLU(),
            nn.Linear(64, out_dim),
        )

    def forward(self, x):
        return self.net(x.float())
```

## 25.3 `proxy_relation_graph.py`

核心输出：

```python
@dataclass
class ProxyRelationCSR:
    source_ids: Tensor
    target_ids: Tensor
    anchor_ids: Tensor
    edge_features: Tensor  # [E,8]
```

必须有：

```python
assert metadata["usesVisibilityLabels"] is False
```

训练入口发现 schema 不是 geometry-only relation 时应拒绝作为 V5 Full 输入。

## 25.4 `occlusion_field_compiler.py`

```python
class OcclusionFieldCompiler(nn.Module):
    # z_i/z_j/edge -> edge message
    # attention per target-anchor
    # base + anchor delta
    # fixed SH projection -> [N,4,7]
```

SH projection matrix注册为 non-persistent 或 persistent buffer，但不能是 Parameter。

## 25.5 `survival_field.py`

只包含：

```text
fixed SH basis evaluation
7-param transform
mixture CDF
survival
```

必须做到 CPU/PyTorch/WebGPU/WASM 数值 parity。

## 25.6 `region_support_query.py`

统一提供：

```python
build_support_points(center, right, forward, radius, count=9)
query_region_stats(field, target_center, target_radius, support_points)
```

`count` 只允许 registered values：1/5/9，防止实验中随意改。

## 25.7 `visibility_loss.py`

只实现两个正式函数：

```python
pose_balanced_visibility_logistic_loss(...)
survival_censoring_nll(...)
```

NeuralPVS-style 和 legacy objective 放到 `benchmark/controls/`，不要污染主模型模块。

## 25.8 `train_pvs_v5.py`

主线总 loss 代码应尽量直接：

```python
visibility_loss = pose_balanced_visibility_logistic_loss(...)
survival_loss = survival_censoring_nll(...)
loss = visibility_loss + survival_loss
```

没有 schedule branch。

只有 ablation runner 显式请求 control loss 时才走 benchmark control implementation。

---

# 26. 推荐配置对象

```python
FULL_V5 = {
    "geometry_dim": 32,
    "relation_graph": "aabb_orthographic_overlap_v1",
    "relation_anchors": 12,
    "relation_topk": 8,
    "edge_dim": 8,
    "edge_hidden": 64,
    "edge_message_dim": 32,
    "direction_basis": "real_sh_l1",
    "field_shape": [4, 7],
    "support_points": 9,
    "visibility_hidden": 32,
    "instance_residual": False,
    "loss": "pose_balanced_logistic_plus_survival_nll",
    "pose_sampling": "uniform_include_empty",
    "optimizer": "AdamW",
    "weight_decay": 1e-5,
}
```

不要在 Full config 中出现：

```text
recall target
CVaR fraction
margin
hard-tail fraction
temperature
loss ramp schedule
per-instance residual blend
relation auxiliary weight
scene depth quantile
```

---

# 27. 实验执行顺序与止损条件

## Phase 0：单元测试与数值 parity

必须先完成：

1. proxy relation graph 不读取任何 label 文件；
2. 12 anchor projection deterministic；
3. PyTorch survival query 与 WebGPU/WASM 单值误差在容差内；
4. 1/5/9 support point 定义固定；
5. all-negative pose loss finite；
6. no-positive / no-negative batch 不产生 NaN；
7. test split 在训练/选择阶段不可访问。

## Phase 1：HKUST viability，单 seed 短训

推荐：

```text
seed 20260801
10-20 epochs
same steps/epoch
validation only
```

先跑：

```text
B2 Geometry Field
B3 Relation Field
D0 Center
D2 9-point
A0 Vis-only
A1 Dual-Likelihood
```

止损：如果新的 relation field 在 threshold-free AP 和 safety frontier 上显著落后当前 V4，先修 architecture，不立刻扩展完整消融。

## Phase 2：单 seed 完整 screening

40x900，先跑所有主要 candidate variant。

不要读 test。

## Phase 3：选定约 10 个 unique configuration，三 seed正式确认

优先：

```text
A1 Full
A0 Vis-only
A2 NeuralPVS-style
A3 Legacy task
B0 Geometry-only
B1 Free Field
B2 Geometry Field
B3 Relation Field
B4 +Residual
C0 SH0
C2 SH2
D0 center
D1 5-point
D3 field-only max
```

其中多项共享同一 Full checkpoint，实际 unique config 可控制在约 10-12 个。

## Phase 4：架构冻结后跨场景 LOSO

此时不能再根据 held-out scene 结果修改结构或 loss。

## Phase 5：最后一次 frozen test + streaming + runtime

只有 architecture、loss、threshold protocol、variant selection 全冻结后才读取 test。

---

# 28. 论文贡献与创新点应该怎么写

## Contribution 1：Pre-geometry From-region Visibility

建议表述：

> We study pre-geometry from-region visibility for progressive Web3D, where the client must identify potentially visible resources over a camera region before detailed scene geometry has been streamed.

它强调论文解决的问题不是普通 runtime occlusion culling，而是“可见性计算发生在 geometry download 之前”。

## Contribution 2：Generalizable Geometry-to-Occlusion Field Compiler

建议表述：

> We introduce a scene-agnostic occlusion-field compiler that converts purely geometric, directional proxy-overlap relations into a compact per-unit survival field. The compiler contains no scene IDs, train-observed relation graph, or per-instance learned memory, enabling the same network to be applied to unseen scenes.

真正创新不在单独的 GNN、survival analysis 或 SH，而在：

\[
\boxed{
\text{geometry-only potential occlusion relations}
\rightarrow
\text{shared learned compiler}
\rightarrow
\text{compact continuous directional survival asset}
}
\]

这个离线编译过程直接针对 pre-geometry PVS 的 Web 资产约束。

## Contribution 3：Compact Analytic Region Query

建议表述：

> We query the same compact field analytically at a fixed set of support positions and aggregate physically interpretable survival statistics, providing a single lightweight from-region prediction without runtime graph propagation or dense subpose expansion.

这里强调：

- continuous directional field；
- fixed 1/5/9 support；
- analytic survival；
- no learned spectral basis；
- no online graph。

## Contribution 4：System / Evidence（如果版面允许）

> We implement the method in WebGPU/WASM and evaluate startup asset size, mobile/desktop runtime, conservative visibility quality, and progressive resource delivery against geometry-resident occlusion baselines.

这一项是 system validation，不要把 scheduler 本身写成新算法。

---

# 29. Loss 在论文中的定位

新的 loss **不建议作为独立 headline contribution**。

更好的写法：

> To avoid coupling representation learning with a hand-designed safety operating point, we use a simple dual-likelihood objective: a pose-balanced visibility likelihood and a censored survival likelihood. The formal high-recall safety point is selected only after training by calibration.

它的价值是：

- 方法更可解释；
- 模型和 safety protocol 解耦；
- 降低 reviewer 对 training tricks 的怀疑；
- 更容易跨场景复用。

与 NeuralPVS 的差异不要包装成“我们发明了更好的 loss”，而应写成任务结构不同：

```text
NeuralPVS: sparse voxel/froxel segmentation + weighted Dice/RVL
Ours: candidate-unit ranking/classification + survival-field likelihood + external calibration
```

---

# 30. 新架构的创新边界：哪些话不要说

在完整文献检索前，不建议声称：

- first learned occlusion field；
- first use of survival analysis for visibility；
- first graph-based visibility model；
- first SH visibility representation；
- mathematically conservative PVS guarantee。

更稳妥的创新主张是组合层面的、任务特定的：

> **A compact, generalizable pre-geometry visibility representation that compiles geometry-derived potential occlusion relations into a continuous directional survival field and supports low-cost region queries in a browser.**

另外 9-point support 是近似，不是连续区域的严格几何证明。论文应使用：

```text
from-region prediction
region-aware query
empirically conservative operating point
```

而不是：

```text
guaranteed visibility for every continuous point in the disk
```

除非以后增加真正的连续界证明。

---

# 31. 预期论文故事重构

旧故事容易被读成：

```text
geometry
+ relation
+ hierarchy
+ survival
+ moment
+ boundary
+ residual
+ recall guard
+ tail margin
+ calibration
```

新故事应该只有：

```text
Problem:
geometry is unavailable when streaming decisions must be made.

Idea:
compile coarse server-side occlusion structure into a tiny transferable field.

Offline:
geometry -> potential occluder relations -> directional survival field.

Online:
view-cell -> a few analytic field queries -> PVS score.

Training:
visibility likelihood + survival likelihood.

Safety:
independent calibration.
```

一句方法总结建议：

> **We compile coarse geometric occlusion relations into a compact first-order directional survival field for each renderable unit, then query this field at a small fixed set of camera-region support points to predict PVS before detailed geometry is streamed.**

这句话应该能够覆盖整篇方法，而不是需要再补五个模块名。

---

# 32. 主要风险与备选方案

## 风险 1：geometry proxy relation 太弱

现象：B3 不明显优于 B2 Geometry Field。

处理顺序：

1. 先检查 proxy graph recall：train-observed true relation 中有多少被 geometry graph top-K 覆盖；
2. K 4/8/16 敏感性；
3. 改进 overlap ranking；
4. 不先重新增加 hierarchy/GNN 深度。

如果 proxy graph 对真实 blocker recall 很低，再考虑更好的 geometry proxy，而不是加 loss。

## 风险 2：SH1 方向容量不足

现象：SH2 明显优于 SH1。

处理：根据 asset-quality Pareto 选 SH2；不要为了“28D 好看”强保 SH1。

## 风险 3：9 points 对区域边界不够

先比较 1/5/9 和 5x point reference。

如果 9-point 仍明显落后 dense subpose union，可考虑：

- 13 点；
- ring max + analytic upper envelope；

但不要立即恢复 learned spectral moment。

## 风险 4：survival NLL 和 final label 冲突

检查：

- 两项 gradient norm；
- A0 vs A1；
- survival calibration curve。

只有确认明显数值冲突后再做 0.5/1/2 权重敏感性，不按一个场景过拟合权重。

## 风险 5：跨场景训练场景数仍然太少

论文只做“cross-scene transfer on five heterogeneous scenes”，并把更大规模数据集列为 limitation/future work。

---

# 33. 最终建议：什么结果出现时应该真正采用这条新主线

我建议 V5 成为论文正式模型至少满足以下大部分条件：

1. HKUST Full 三 seed 的 CNOR / Useful Cull 不显著差于当前 V4；
2. pose-macro AP 至少不退化；
3. seed score Spearman 明显更高，或 CNOR std 更低；
4. B3 Relation Field 稳定优于 B2 Geometry Field；
5. A1 Dual-Likelihood 不弱于或更稳于 Legacy task loss；
6. B4 per-instance residual 没有足够收益，从而可以删除；
7. 9-point region query 明显优于 center，并接近多点 reference；
8. startup asset / runtime 至少不比 V4 更差；
9. 在 LOSO 中 Relation Field 比 Geometry Field 表现出可重复的 unseen-scene 收益。

如果这些条件不成立，不应为了论文叙事强行使用 V5，而应根据消融结果收缩到实验真正支持的最简单模型。

---

# 34. 推荐的最终 Full 配置清单

```text
Representation
  frozen source geometry feature : 96D
  shared exported geometry       : 32D
  relation source                 : geometry-only AABB proxy overlap
  direction anchors               : 12
  top-K / target / anchor         : 8
  edge feature                    : 8D relative/scale-free
  edge MLP                        : 72 -> 64 -> 32
  pooling                         : one attention pool / target-anchor
  anchor field head               : base(32->32->7) + delta(32->32->7)
  directional basis               : fixed real SH l<=1
  runtime field                   : 4 x 7 FP16
  instance residual               : OFF

Region Query
  support points                  : 9
  depth coordinate                : d / (d + target_radius)
  region stats                    : center / max / mean / min survival
  query geometry                  : rho_center / angular_scale / cell_scale
  decoder                         : 39 -> 32 -> 1

Training
  pose sampling                   : uniform, include empty poses
  task loss                       : pose-balanced logistic
  survival loss                   : event/right-censor NLL
  total loss                      : L_vis + L_surv
  recall guard                    : OFF
  CVaR                            : OFF
  tail margin                     : OFF
  relation auxiliary              : OFF
  explicit global regularizer     : OFF
  AdamW weight decay              : 1e-5

Safety
  training target recall          : NONE
  calibration gate                : WR > .99 and one-sided 95% LCB > .99

Generalization
  scene ID                        : NONE
  instance ID embedding           : NONE
  per-instance learned table      : NONE
  target-scene relation labels    : NONE
  target-scene depth statistics   : NONE
```

---

# 35. 与当前仓库的对应关系

本设计基于当前仓库主线的实际结构进行重构，主要参考文件：

```text
neural_instance_culling/model/pvs_model.py
neural_instance_culling/model/train_pvs.py
neural_instance_culling/model/common/visibility_loss.py
neural_instance_culling/model/common/survival_loss.py
neural_instance_culling/model/common/relation_encoder.py
neural_instance_culling/benchmark/run_pvs.py
```

建议旧 V4 结果、checkpoint 和 runner 全部保留，只新增 V5 namespace。这样论文可以把 V4 作为“legacy complex model”基线，直接证明新架构是否在更少模块、更少 loss engineering 的条件下保持或改善效果。

---

# 36. 最短实施路径

如果不希望一次重写太多，按下面顺序推进：

**Step 1：先换 loss，不改架构。**  
在当前 V4 上实现 `pose_balanced_visibility_logistic`，关闭 guard/tail/relation aux/residual，比较 3 seed 稳定性。

**Step 2：先简化 runtime query。**  
继续使用现有 4x7 field，但删除 moment/boundary/direction-basis，改成固定 SH query + 1/5/9 support stats。验证核心 region idea。

**Step 3：重写 geometry-derived relation compiler。**  
让 field 不再依赖 train-observed graph。

**Step 4：32D geometry export + cross-scene shared training。**

这样每一步都能单独判断“是否值得继续”，不会再次形成一次性的大规模重构风险。

