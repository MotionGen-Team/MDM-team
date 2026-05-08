# 方案一 Pipeline 整理版

## 三层结构图

![三层结构图](三层结构图.png)

## Pipeline 流程图

```text
输入: x_t + timestep / text condition
    ->
统一条件构造
    ->
c [1, B, D]
    ->
Shared Embedding
    ->
两路共享表示
    -> h_struct 结构侧表示
    -> h_global 时间侧表示
    ->
总装层统一注入
    -> frame positional encoding
    -> unified condition c

h_struct
    -> Local Branch
    -> 按 body groups 切分
    -> 每组 temporal modeling
    -> group attention
    -> local feature L_t

h_struct
    -> Group Summary Builder
    -> group summary S_t

h_global + S_t
    -> Summary Fusion
    -> h_global + gate * delta
    -> h_global_enh

h_global_enh
    -> Global Transformer Blocks
    -> global feature G_t

L_t + G_t
    -> Fusion: concat + projection
    -> F_t
    -> Base Prediction Head
    -> y_base_raw

F_t + y_base_latent
    -> Residual TCN
    -> gate * delta_raw

y_raw = y_base_raw + delta_raw
    -> restore shape
    -> y_pred
    -> Reverse Update: x_t -> x_{t-1}
```

## 1. 方法概述

方案一的目标不是改写扩散框架，而是在每个 reverse step 内部把单步去噪器拆成更清晰的三层结构：

- 共享层
  先把输入变成可同时服务局部与全局建模的双路共享表示
- 局部/全局建模层
  一路做 body-group 局部动态建模，一路做结构增强的时间建模
- 输出修正层
  先给基础预测，再做轻量残差修正

当前版本与早期实现相比，最重要的变化有四个：

- 条件真正进入主干
- 帧级位置编码真正进入主干
- foot contact 左右脚映射修正
- 只在两处加入 gate：
  - `SummaryFusion`
  - `ResidualTCN`

## 2. 当前主链的核心思想

### 2.1 共享层不再只产出单一路径

方案一当前明确要求共享层同时产生：

- `h_struct`
  给局部建模和 group summary 使用
- `h_global`
  给全局时间建模使用

这意味着结构信息不会在一开始就完全压平到单一路径里。

### 2.2 局部分支和全局分支共享同一结构来源

方案一不是简单把原来的单路 transformer 拆成两个平行分支，而是进一步要求：

- `Local Branch` 从 `h_struct` 出发
- `Group Summary Builder` 也从 `h_struct` 出发
- `Global Branch` 通过 `s_t` 接收结构增强

这样 `Global Branch` 的结构来源是明确的，不是从压平后的全局 token 里临时猜出来的。

### 2.3 gate 只是注入强度控制，不是新架构

当前这版的 gate 只承担“保守注入”的作用：

- `SummaryFusion`
  控制 structure summary 对 `h_global` 的注入幅度
- `ResidualTCN`
  控制 refine 修正量对 `y_base_raw` 的扰动幅度

它们不改变主链顺序，也不改变模块职责。

## 3. 共享表示阶段

当前共享表示阶段的职责是：

- 把 `hml_vec` 通过 `StructureAdapter` 还原成结构侧 token
- 做轻量 temporal mixing
- 生成：
  - `h_struct [T, B, 22, D_s]`
  - `h_global [T, B, D]`

之后由总装层统一注入：

- frame positional encoding
- unified condition `c`

这一点很重要。当前版本不再把条件注入分散在多个子模块里，而是统一在总装层做。

## 4. Local Branch

Local Branch 当前仍保持原来的总体思路：

- 按 `left leg / right leg / torso` 分组
- 每组先聚合，再做局部 temporal modeling
- 通过轻量 group attention 建立有限的组间交互
- 最终回到统一的 `L_t [T, B, D]`

这版没有给 Local Branch 加 gate，也不打算在文档里把它写成新结构。

## 5. Global Branch

Global Branch 当前链路是：

```text
h_struct
-> GroupSummaryBuilder
-> s_t

h_global, s_t
-> SummaryFusion
-> h_global_enh

h_global_enh
-> GlobalTransformerBlocks
-> g_t
```

### 5.1 SummaryFusion 的当前写法

当前只在这里加入 gate：

```text
delta = CrossAttention(h_global, s_t)
gate = sigmoid(gate_logit)
h_global_enh = LayerNorm(h_global + gate * delta)
```

这意味着：

- `SummaryFusion` 仍然是 cross-attention 注入
- 没有改成新的分支结构
- 只是给原本的 summary 注入增加了一个可学习强度系数

### 5.2 为什么这里要加 gate

因为在训练早期，如果 summary 注入过强，可能直接拉坏时间主线分布。

加 gate 后：

- 初始注入很小
- 如果 summary 确实有用，训练会逐步把 gate 学大

## 6. Refine 层

Refine 层当前仍保持：

```text
Fusion
-> Base Prediction Head
-> y_base_raw
-> base latent projector
-> Residual TCN
-> delta_raw
-> y_raw = y_base_raw + delta_raw
```

### 6.1 当前 Residual TCN 的 gate

当前只在残差输出端加 gate：

```text
delta_raw = sigmoid(gate_logit) * out_proj(x2)
```

这意味着：

- refine 主链没改
- 还是两层 residual TCN block
- 只是防止 `delta_raw` 在训练早期过强地扰动 `y_base_raw`

## 7. 当前版本的关键改动总结

### 7.1 已修正的问题

- `timestep / text condition` 之前没有真正进入主干，现已修正
- 帧级 positional encoding 之前没有接入主干，现已修正
- foot contact 左右脚映射之前有误，现已修正

### 7.2 新增但尽量保守的改动

- `SummaryFusion` 增加可学习标量 gate
- `ResidualTCN` 增加可学习标量 gate

### 7.3 明确没有改的内容

- Shared / Local / Global / Refine 四段式总体结构
- Local Branch 的主链
- Fusion 的 `concat + projection`
- Base Prediction Head 的职责
- Residual TCN 的两层 block 骨架

## 8. 当前训练观察口径

当前训练时除了原有 loss 记录外，还会额外记录：

```text
output/plan_one_gate.csv
```

字段：

```text
step,summary_gate,residual_gate
```

记录频率：

```text
每 1000 step 一次
```

这个文件的作用是帮助判断：

- gate 是否一直维持很小
- gate 是否会随训练逐步变大
- 当前 loss 落后是否与分支注入强度相关

## 9. 当前建议

当前最合理的使用方式不是继续大改架构，而是：

1. 保持当前主结构。
2. 跑 gate 版短程训练。
3. 同时看 loss 和 `plan_one_gate.csv`。
4. 再决定是否调学习率或初始化。

也就是说，当前方案一已经不再处于“继续拆新模块”的阶段，而是进入“验证现有结构、控制注入强度、稳定训练表现”的阶段。
