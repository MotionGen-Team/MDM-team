# 方案一 Pipeline 整理版

## 三层结构图

![三层结构图](三层结构图.png)

## Pipeline 流程图

```text
输入: x_t + timestep embedding + condition embedding
    ->
共享嵌入阶段
    ->
分成两路共享表示:
    -> H_struct  (结构化共享表示)
    -> H_global  (时序共享表示)

H_struct
    -> Local Branch
    -> 按 body groups 切分: left leg / right leg / torso
    -> 每个 group 做 multi-scale temporal conv
    -> 组内特征汇聚
    -> 轻量 group-attention / gating
    -> 得到 local feature L_t

H_struct
    -> Group Summary Builder
    -> 按 body groups 切分
    -> 每组 pooling / projection
    -> 得到 group summary S_t

H_global
    -> 构造 temporal tokens

temporal tokens + group summary S_t
    -> Global Branch
    -> 结构增强的时间建模
    -> 轻量 attention: 优先 GQA / 备选 chunked
    -> 得到 global feature G_t

L_t + G_t
    -> Fusion: concat + projection
    -> 融合特征 F_t
    -> Base Prediction Head
    -> 基础预测 y_base

F_t + proj(y_base)
    -> TCN Coordination Residual Module
    -> 2 个 residual TCN blocks
    -> 输出协调残差 Δy

y = y_base + Δy
    -> Reverse Update: x_t -> x_{t-1}
```

## 1. 方法概述

方案一面向 MDM 在人体动作生成任务中存在的协调性不足与自然性欠佳问题，目标是在保留原始扩散生成框架的前提下，对单步去噪器进行结构化增强。当前 baseline 在生成行走、摆臂或躯干参与较强的动作时，容易出现左右肢体节奏不一致、局部补偿动作不自然、整体轨迹后程偏移等现象。这说明仅沿时间维度做统一建模，虽然可以恢复动作的大体轮廓，但对局部身体结构与整体运动趋势之间的协同刻画仍然不够充分。

基于这一观察，方案一将单步去噪过程拆成三层互补结构：

- `Local Branch` 负责局部身体组动态建模。
- `Global Branch` 负责整体时序趋势建模。
- `TCN Coordination Residual Module` 负责输出端的协调残差修正。

整个设计保持 MDM 的扩散采样流程不变，只重构每个 reverse step 内部的去噪器结构。核心思想不是替换原有主干，而是在单步预测时显式区分“局部动作是否协调”“全局趋势是否稳定”以及“输出结果是否还需要细粒度修正”这三类问题。

## 2. 整体生成流程

方案一仍遵循标准的扩散生成过程。文本条件或动作类别条件首先经过 `Text Encoder / Action Embedding` 得到条件嵌入；当前带噪动作表示 `x_t` 与 timestep embedding、condition embedding 一起进入单步去噪器，经过共享嵌入、局部分支、组摘要构建、全局分支、融合、基础预测与残差修正，得到当前步输出 `y`，再用于 reverse update 得到 `x_{t-1}`。

与旧版流程图不同，当前版本明确将共享阶段拆成两路：

- `H_struct` 用于保留身体结构相关信息，供 `Local Branch` 和 `Group Summary Builder` 共同使用。
- `H_global` 用于生成 `temporal tokens`，供 `Global Branch` 继续执行时间建模。

因此，方案一的关键变化不只是把单路 transformer 拆成 `local/global` 双分支，而是进一步规定：`group summary` 不能从已经被完全压平的全局 token 中临时提取，而必须从共享阶段保留下来的结构化表示中单独生成。

## 3. 单步去噪器设计

### 3.1 共享输入表示

在每个 reverse step 中，模型接收当前状态 `x_t`、timestep embedding 和 condition embedding，并通过共享嵌入阶段得到两类共享表示：

- `H_struct`：结构化共享表示，用于保留 body-group 相关信息。
- `H_global`：时序共享表示，用于构造主时间序列 token。

这一改动相对于原始 MDM 非常重要。原始流程中，输入经过统一线性映射后很快被压成单一路径的时序 token，结构信息会在早期被混入统一 latent 空间。方案一为了让 `Local Branch` 真正按身体组建模，也为了让 `Global Branch` 使用有来源的 `group summary`，必须在共享阶段显式保留一条结构侧表示，而不是只留下单一的 `[T, B, D]` 主干特征。

因此，当前共享阶段的职责不再只是“统一嵌入”，而是“同时生成可供局部支路和全局支路复用的两类共享表示”。

### 3.2 Local Branch

`Local Branch` 的目标是增强模型对局部身体结构动态的刻画能力。根据当前结构图，局部分支的设计已经明确为一条“先按 body groups 切分，再做组内时序建模和组间轻量交互”的路径。

当前已确定的处理流程为：

- 从 `H_struct` 出发。
- 按 body groups 将人体简化划分为 `left leg`、`right leg`、`torso` 三组。
- 对每个 group 单独做 `multi-scale temporal conv`。
- 在组内得到时序特征后进行组内特征汇聚。
- 再通过轻量 `group-attention / gating` 建立有限的组间信息交换。
- 输出局部分支特征 `L_t`。

这一设计的含义很明确。方案一不是希望用复杂骨架图卷积直接覆盖所有结构关系，而是先把最关键、最容易产生协调问题的身体区域单独建模，再用轻量机制让这些区域发生交互。这样做有三个直接好处：

- 左右腿与躯干的局部动态被显式拆开，便于捕捉步态不齐、支撑相错位等问题。
- 多尺度时间卷积可以同时覆盖短时节奏变化和稍长时间范围内的局部补偿动作。
- 组间交互控制在 `group-attention / gating` 这一轻量层面，复杂度可控，不会把局部分支重新做成一个重型全局建模器。

因此，`Local Branch` 的输出 `L_t` 应被理解为“带有明确身体组语义的局部动态表示”，而不是普通的局部卷积特征。

### 3.3 Group Summary Builder

`Group Summary Builder` 是当前流程图中新增且必须单列的模块。它的职责是从 `H_struct` 中提取供 `Global Branch` 使用的结构摘要 `S_t`。

这里需要明确一点：`group summary` 不再被视为 `Global Branch` 内部顺手提取的附属量，而是来自共享结构表示的独立中间结果。这样设计的原因在于，如果 `Global Branch` 只接收已经压平的时序 token，那么其中并没有稳定、显式的 body-group 来源，最终得到的所谓“group summary”只能是隐式学习到的伪结构信息，解释性和可控性都不够强。

当前 `Group Summary Builder` 的处理逻辑可规定为：

- 从 `H_struct` 出发。
- 按 `left leg / right leg / torso` 三组切分。
- 对每组特征做 pooling 或 projection。
- 形成粗粒度结构摘要 `S_t`。

在这个定义下，`S_t` 是结构侧输入对全局建模的补充信号，其作用不是替代时间主干，而是为 `Global Branch` 提供稳定的身体组上下文。

### 3.4 Global Branch

`Global Branch` 负责从整体时序角度建模动作趋势与长程依赖。根据当前流程图，`Global Branch` 的输入已经明确为双输入结构：

- 一路来自 `H_global` 构造得到的 `temporal tokens`
- 一路来自 `Group Summary Builder` 输出的 `group summary S_t`

当前已确定的处理路径为：

- 基于 `H_global` 构造主时序序列 `temporal tokens`。
- 将 `temporal tokens` 与 `group summary S_t` 一起送入结构增强的时间建模模块。
- 该模块优先采用轻量 attention，首选 `GQA`，备选 `chunked attention`。
- 最终输出全局分支特征 `G_t`。

这里最关键的收敛点有两个。

第一，主轴仍然是时间建模。也就是说，方案一没有偏离 MDM 原本以时间序列为核心的生成逻辑，`Global Branch` 仍以 temporal tokens 为主体。

第二，`group summary` 由 `H_struct` 单独生成，而不是从已经压平的全局输入中现提。这意味着 `Global Branch` 不再是完全“结构无感”的时间主干，而是在维持时序建模能力的同时，显式接收结构增强信号。

在实现优先级上，当前文档应明确写死：

- 首选 `GQA`，因为它在保持较强注意力建模能力的同时更节省显存和计算。
- 若实现复杂度或显存预算受限，则退而使用 `chunked attention`。

### 3.5 Fusion

局部分支输出 `L_t`，全局分支输出 `G_t` 后，模型通过融合模块得到统一特征 `F_t`。根据结构图，这里的融合方式已经确定为：

```text
Fusion: concat + projection
```

因此，文档中不再把融合写成开放问题，而应明确表述为先拼接、后投影的固定方案：

```text
F_t = Proj([L_t ; G_t])
```

这样的设计理由是直接且充分的：

- `concat` 能保留局部与全局两类特征的原始语义，不会过早做强耦合。
- `projection` 用于把拼接后的特征重新映射回统一维度，便于后续预测头和残差修正模块复用。

所以 `Fusion` 的角色是“特征对齐与统一表达”，不是再做一层复杂建模。

### 3.6 Base Prediction Head

基于融合特征 `F_t`，模型首先通过 `Base Prediction Head` 输出基础预测 `y_base`。它表示当前 reverse step 下、已经结合局部与全局上下文后的主预测结果。

这里建议文档中明确它的职责边界：

- `Base Prediction Head` 负责主去噪预测。
- 它应输出一个整体上合理、具备基本动作结构和时间趋势的结果。
- 它不负责专门解决输出端的细粒度协调问题，这部分留给后续残差模块。

这种职责拆分很重要。否则如果把所有修正压力都压到主预测头上，优化目标会变得混乱，既不利于训练，也不利于解释各模块贡献。

### 3.7 TCN Coordination Residual Module

在得到基础预测 `y_base` 后，方案一通过 `TCN Coordination Residual Module` 输出协调残差 `Δy`，并与基础预测相加得到最终输出。

根据结构图，这一模块的实现细节当前应直接固定为：

```text
input = concat(F_t, proj(y_base))
Δy = TCN(input)
y = y_base + Δy
```

其中有三个已确定细节需要明确写入文档。

第一，输入并不是简单的 `[F_t ; y_base]`，而是 `concat(F_t, proj(y_base))`。这意味着 `y_base` 在与融合特征拼接前要先经过一个投影层，以对齐维度与表示空间。

第二，TCN 的主体结构当前定为 `2 个 residual TCN blocks`。这说明该模块被控制在一个轻量、后端、修正型的容量范围内，而不是扩展成新的主生成器。

第三，模块输出的是协调残差 `Δy`，不是完整预测值。最终结果由：

```text
y = y_base + Δy
```

得到。

因此，这个模块的定位应被明确写成：它专门用于修正基础预测中仍残留的局部协调失配、节奏错位和细粒度时序不顺滑问题，尤其关注左右肢体协同、躯干与下肢配合、步态连续性等高频出现的失真模式。

## 4. 三层结构的设计动机

方案一采用三层结构，不是为了增加模块数量本身，而是因为动作生成中的“不自然”通常同时来自三个层面：

- 局部身体组内部的动态模式没有建好。
- 整体时间趋势与长程依赖没有维持好。
- 最终输出虽然大体正确，但仍有细粒度协调误差需要修补。

对应地：

- `Local Branch` 解决“局部身体组是否各自动得合理”。
- `Global Branch` 解决“整体动作是否沿时间方向保持稳定趋势”。
- `TCN Coordination Residual Module` 解决“最终输出是否足够协调、顺滑、自然”。

与之前版本相比，当前设计进一步强调：局部建模和结构摘要共享同一条结构侧输入来源，即 `H_struct`。这使得局部分支与全局分支之间不再只是并列关系，而是通过 `Group Summary Builder` 建立起明确的信息桥梁。这样做的直接收益是，`Global Branch` 的结构增强来源更加稳定，方法解释性也更强。

## 5. 当前已明确内容与未决内容

从当前结构图出发，方案一已经明确的内容包括：

- 扩散框架保持不变，改动集中在单步去噪器内部。
- 共享输入不再只产生单一表示，而是显式拆成 `H_struct` 和 `H_global` 两路。
- `Local Branch` 采用 `left leg / right leg / torso` 三组划分。
- 每个局部组先做 `multi-scale temporal conv`，再做组内汇聚与轻量 `group-attention / gating`。
- `Group Summary Builder` 从 `H_struct` 出发生成 `group summary S_t`。
- `Global Branch` 采用 `temporal tokens + group summary` 的结构增强式时间建模。
- 全局 attention 优先 `GQA`，备选 `chunked attention`。
- `Fusion` 固定为 `concat + projection`。
- `Base Prediction Head` 先给出基础预测 `y_base`。
- `TCN Coordination Residual Module` 的输入为 `concat(F_t, proj(y_base))`。
- 残差修正主干固定为 `2 个 residual TCN blocks`。
- 最终输出形式固定为 `y = y_base + Δy`。

相较之下，仍可保留为实现细节再行确定的部分主要包括：

- `H_struct` 与 `H_global` 在共享阶段的具体构造方式。
- `multi-scale temporal conv` 的具体卷积核组合与 dilation 设置。
- `group-attention / gating` 采用何种最轻量实现更合适。
- `Group Summary Builder` 中每组摘要采用平均池化、注意力池化还是小型 MLP 汇聚。
- `proj(y_base)` 的维度设置与是否共享投影参数。
- 2 个 residual TCN blocks 的通道数、卷积核大小与归一化方式。

也就是说，方案一的总体结构和大部分模块边界已经收敛，后续需要实验决定的更多是参数化实现，而不是方法逻辑本身。

## 6. 小结

方案一的核心不是改写 MDM 的扩散流程，而是在每个 reverse step 内构造一个清晰的三层去噪结构：局部分支关注身体组动态，全局分支关注结构增强的时间建模，后端 TCN 残差模块负责细粒度协调修正。与早期版本相比，当前方案已经进一步明确了 `group summary` 的来源，即它来自共享阶段保留下来的结构化表示 `H_struct`，而不是从压平后的全局输入中临时提取。

这使得方案一不仅有明确的研究动机，也具备了较清晰的实现落点。尤其是双路共享表示、组摘要构建、融合方式、全局 attention 优先级以及残差模块输入输出形式都已基本确定，适合作为后续代码改造、模块实验与消融分析的统一方案说明。
