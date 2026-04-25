# 多尺度时间卷积实验指南

## 文件说明

本实验实现了三种多尺度时间卷积变体，用于对比不同卷积核大小和 dilation 组合的效果。

### 核心文件

| 文件 | 说明 |
|------|------|
| `model/plan_one/local_branch/multi_scale_temporal.py` | 多尺度TCN模块实现 |
| `model/plan_one/local_branch/README.md` | 模块使用文档 |

### 实验脚本

| 文件 | 说明 |
|------|------|
| `demo_multi_scale_tcn.py` | 模块演示和测试 |
| `train_variants.py` | 训练三种变体的模型 |
| `generate_videos.py` | 使用训练好的模型生成视频 |
| `run_experiment.py` | 完整的实验流程（训练+生成） |
| `quick_start.py` | 快速启动向导 |

## 三种变体配置

### 1. Baseline: (3, 5, 3-dil2)
```python
kernel_sizes = (3, 5, 3)
dilations = (1, 1, 2)
```
- 分支1: kernel=3, dilation=1, 感受野=3
- 分支2: kernel=5, dilation=1, 感受野=5
- 分支3: kernel=3, dilation=2, 感受野=5

### 2. 变体 1: (3, 7, 3-dil2)
```python
kernel_sizes = (3, 7, 3)
dilations = (1, 1, 2)
```
- 分支1: kernel=3, dilation=1, 感受野=3
- 分支2: kernel=7, dilation=1, 感受野=7
- 分支3: kernel=3, dilation=2, 感受野=5

### 3. 变体 2: (3, 5, 5-dil2)
```python
kernel_sizes = (3, 5, 5)
dilations = (1, 1, 2)
```
- 分支1: kernel=3, dilation=1, 感受野=3
- 分支2: kernel=5, dilation=1, 感受野=5
- 分支3: kernel=5, dilation=2, 感受野=9

## 快速开始

### 1. 环境检查

```bash
python quick_start.py --step check
```

### 2. 模块测试

```bash
python demo_multi_scale_tcn.py
```

### 3. 训练模型

训练所有变体：
```bash
python train_variants.py --variant all --num_steps 200000
```

训练单个变体：
```bash
python train_variants.py --variant baseline --num_steps 200000
python train_variants.py --variant 3-7-3-dil2 --num_steps 200000
python train_variants.py --variant 3-5-5-dil2 --num_steps 200000
```

### 4. 生成视频

生成所有变体的视频：
```bash
python generate_videos.py --variant all --text_prompt "the person walked forward and is picking up his toolbox."
```

生成单个变体的视频：
```bash
python generate_videos.py --variant baseline
```

### 5. 完整流程

```bash
python run_experiment.py --mode both --variant all
```

## 手动训练（不使用脚本）

### Baseline
```bash
python -m train.train_mdm \
    --save_dir save/ms_tcn_baseline \
    --dataset humanml \
    --use_temporal_tcn \
    --ms_tcn_variant baseline \
    --eval_during_training \
    --gen_during_training \
    --use_ema \
    --mask_frames
```

### 变体 1: (3, 7, 3-dil2)
```bash
python -m train.train_mdm \
    --save_dir save/ms_tcn_3-7-3-dil2 \
    --dataset humanml \
    --use_temporal_tcn \
    --ms_tcn_variant 3-7-3-dil2 \
    --eval_during_training \
    --gen_during_training \
    --use_ema \
    --mask_frames
```

### 变体 2: (3, 5, 5-dil2)
```bash
python -m train.train_mdm \
    --save_dir save/ms_tcn_3-5-5-dil2 \
    --dataset humanml \
    --use_temporal_tcn \
    --ms_tcn_variant 3-5-5-dil2 \
    --eval_during_training \
    --gen_during_training \
    --use_ema \
    --mask_frames
```

## 手动生成视频

```bash
python -m sample.generate \
    --model_path save/ms_tcn_baseline/model000200000.pt \
    --text_prompt "the person walked forward and is picking up his toolbox." \
    --num_samples 3 \
    --num_repetitions 1 \
    --output_dir videos/baseline
```

## 输出目录结构

```
videos/
├── baseline/
│   ├── sample00_rep00.mp4
│   ├── sample01_rep00.mp4
│   ├── sample02_rep00.mp4
│   └── samples_00_to_02.mp4
├── 3-7-3-dil2/
│   ├── sample00_rep00.mp4
│   ├── sample01_rep00.mp4
│   ├── sample02_rep00.mp4
│   └── samples_00_to_02.mp4
└── 3-5-5-dil2/
    ├── sample00_rep00.mp4
    ├── sample01_rep00.mp4
    ├── sample02_rep00.mp4
    └── samples_00_to_02.mp4
```

## 对比分析

三种变体的主要区别：

| 特性 | baseline | 3-7-3-dil2 | 3-5-5-dil2 |
|------|----------|------------|------------|
| 最大感受野 | 5 | 7 | 9 |
| 长距离依赖 | 中等 | 较好 | 最好 |
| 计算复杂度 | 低 | 中 | 中 |
| 适合任务 | 一般性 | 中等范围模式 | 长范围依赖 |

## 注意事项

1. **数据集**: 确保已经下载并放置了 HumanML3D 数据集到 `./dataset/humanml/`

2. **预训练模型**: 如果没有预训练模型，需要先训练模型才能生成视频

3. **GPU 内存**: 训练需要较大的 GPU 内存，如果内存不足可以减小 batch size

4. **训练时间**: 完整训练 (200k steps) 可能需要数小时到数天，取决于 GPU

5. **演示模式**: 可以使用 `--num_steps 1000` 进行快速演示，但模型质量会很差

## 故障排除

### 模型找不到
```
错误: 模型不存在: save/ms_tcn_baseline/model000200000.pt
```
**解决**: 先运行训练脚本，或检查模型路径是否正确

### 数据集找不到
```
错误: 数据集未找到
```
**解决**: 按照 README.md 中的说明下载 HumanML3D 数据集

### CUDA 内存不足
```
RuntimeError: CUDA out of memory
```
**解决**: 减小 batch size 或使用更小的模型

## 实验记录

建议在实验过程中记录以下信息：

1. 训练损失曲线
2. 验证集性能
3. 生成视频的质量对比
4. 训练时间和推理速度

可以使用 `--train_platform_type WandBPlatform` 启用 WandB 记录。
