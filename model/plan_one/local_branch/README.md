# 多尺度时间卷积模块 (Multi-Scale Temporal Conv)

## 文件说明

`multi_scale_temporal.py` - 实现了多尺度时间卷积模块，支持不同的卷积核大小和 dilation 组合。

## 三种变体配置

### 1. Baseline: (3, 5, 3-dil2)
- 分支1: kernel=3, dilation=1
- 分支2: kernel=5, dilation=1
- 分支3: kernel=3, dilation=2

### 2. 变体 1: (3, 7, 3-dil2)
- 分支1: kernel=3, dilation=1
- 分支2: kernel=7, dilation=1
- 分支3: kernel=3, dilation=2

### 3. 变体 2: (3, 5, 5-dil2)
- 分支1: kernel=3, dilation=1
- 分支2: kernel=5, dilation=1
- 分支3: kernel=5, dilation=2

## 使用方法

### 方式1: 直接在模型中使用

在创建 MDM 模型时，添加以下参数：

```python
model = MDM(
    # ... 其他参数 ...
    use_temporal_tcn=True,           # 启用时间 TCN
    ms_tcn_variant='baseline'        # 选择变体: 'baseline', '3-7-3-dil2', '3-5-5-dil2'
)
```

### 方式2: 在训练脚本中配置

修改训练脚本中的模型参数：

```python
args.use_temporal_tcn = True
args.ms_tcn_variant = '3-7-3-dil2'  # 或 'baseline', '3-5-5-dil2'
```

### 方式3: 直接实例化模块

```python
from model.plan_one.local_branch.multi_scale_temporal import get_multi_scale_tcn

# 创建指定变体的模块
tcn = get_multi_scale_tcn('baseline', latent_dim=256, dropout=0.1)

# 前向传播
# x: [seqlen, batch_size, latent_dim]
output = tcn(x)
```

## 实验对比

| 变体名称 | kernel_sizes | dilations | 感受野大小 |
|---------|-------------|-----------|-----------|
| baseline | (3, 5, 3) | (1, 1, 2) | 3, 5, 5 |
| 3-7-3-dil2 | (3, 7, 3) | (1, 1, 2) | 3, 7, 5 |
| 3-5-5-dil2 | (3, 5, 5) | (1, 1, 2) | 3, 5, 9 |

## 注意事项

1. 使用多尺度 TCN 时，必须同时设置 `use_temporal_tcn=True`
2. 如果不设置 `ms_tcn_variant`，则使用原始的 `LightweightTemporalTCN`
3. 三种变体都保持残差连接和输出层归一化
