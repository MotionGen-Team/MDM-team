# 运行状态

## 当前操作

正在运行 `run_and_save_videos.py` 脚本，自动完成以下流程：

1. **训练模型** - 三种变体各训练 1000 步（演示用）
   - baseline: (3, 5, 3-dil2)
   - 3-7-3-dil2: (3, 7, 3-dil2)
   - 3-5-5-dil2: (3, 5, 5-dil2)

2. **生成视频** - 使用训练好的模型生成动作视频
   - 文本提示: "the person walked forward and is picking up his toolbox."
   - 样本数: 3
   - 保存位置: `videos/` 目录

## 预期输出

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

## 预计时间

- 训练（1000步）: 每变体约 5-30 分钟（取决于 GPU）
- 生成视频: 每变体约 1-5 分钟
- **总计**: 约 20-100 分钟

## 如何检查进度

### 查看训练日志
训练过程中会打印损失值和进度信息。

### 查看生成的文件
```bash
# 查看模型
ls save/ms_tcn_*/model*.pt

# 查看视频
ls videos/*/*.mp4
```

### 实时监控
```bash
# 查看videos目录大小（视频正在生成）
du -sh videos/

# 查看最新修改的文件
ls -lt videos/*/
```

## 如果训练中断

可以单独运行某个变体：

```bash
# 只训练 baseline
python -m train.train_mdm --save_dir save/ms_tcn_baseline --dataset humanml --num_steps 1000 --use_temporal_tcn --ms_tcn_variant baseline --save_interval 500

# 只生成 baseline 视频
python -m sample.generate --model_path save/ms_tcn_baseline/model000001000.pt --text_prompt "the person walked forward and is picking up his toolbox." --num_samples 3 --output_dir videos/baseline
```

## 注意事项

1. **数据集**: 确保 `./dataset/humanml/` 目录包含 HumanML3D 数据集
2. **GPU 内存**: 如果 CUDA 内存不足，可能需要减小 batch size
3. **训练步数**: 1000 步仅用于演示，实际应用建议 200000 步以上

## 手动运行命令

如果自动脚本遇到问题，可以手动执行：

### 1. 训练所有变体
```bash
python train_variants.py --variant all --num_steps 1000
```

### 2. 生成所有视频
```bash
python generate_videos.py --variant all
```

### 3. 完整流程
```bash
python run_experiment.py --mode both --variant all --num_steps 1000
```
