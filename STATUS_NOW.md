# 当前运行状态

## 正在执行的操作

脚本 `start_training_and_generate.py` 正在运行，执行以下流程：

### 阶段 1: 训练模型
- **baseline**: 训练 (3, 5, 3-dil2) 配置
- **3-7-3-dil2**: 训练 (3, 7, 3-dil2) 配置  
- **3-5-5-dil2**: 训练 (3, 5, 5-dil2) 配置
- 训练步数: 1000 步（演示用）

### 阶段 2: 生成视频
- 使用训练好的模型生成动作视频
- 文本提示: "the person walked forward and is picking up his toolbox."
- 每个变体生成 3 个样本
- 视频保存到 `videos/` 目录

## 预计完成时间

- 每个变体训练: 5-30 分钟
- 每个变体生成: 1-5 分钟
- **总计**: 约 20-100 分钟

## 如何查看进度

### 方法 1: 查看目录
```bash
# 查看是否生成了模型
ls save/ms_tcn_*/model*.pt

# 查看是否生成了视频
ls videos/*/*.mp4
```

### 方法 2: 查看文件大小
```bash
# 查看videos目录大小
du -sh videos/
```

### 方法 3: 等待脚本完成
脚本会在完成后自动显示结果总结。

## 预期输出结构

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

## 如果当前脚本失败

可以使用以下备用方案：

### 方案 1: 批处理脚本
```bash
run_experiment.bat
```

### 方案 2: 手动执行
```bash
# 训练
python train_variants.py --variant all --num_steps 1000

# 生成
python generate_videos.py --variant all
```

### 方案 3: 单个变体
```bash
# 只训练 baseline
python -m train.train_mdm --save_dir save/ms_tcn_baseline --dataset humanml --num_steps 1000 --use_temporal_tcn --ms_tcn_variant baseline --save_interval 500

# 只生成 baseline 视频  
python -m sample.generate --model_path save/ms_tcn_baseline/model000001000.pt --text_prompt "the person walked forward and is picking up his toolbox." --num_samples 3 --output_dir videos/baseline
```

## 注意事项

1. **不要关闭终端** - 训练正在进行中
2. **确保有足够的磁盘空间** - 模型和视频需要存储空间
3. **GPU 内存** - 如果遇到 CUDA 内存错误，需要减小 batch size

## 完成后

脚本完成后会显示：
- 每个变体的训练状态
- 每个变体的视频生成状态
- 视频文件的保存位置

---

**当前时间**: 脚本正在运行中...
**状态**: 训练阶段
