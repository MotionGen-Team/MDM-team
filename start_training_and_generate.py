#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
启动训练并生成视频
"""
import os
import sys
import time
import subprocess


def print_banner(text):
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70)


def run_training(variant, steps=1000):
    """运行训练"""
    save_dir = f"save/ms_tcn_{variant}"
    
    print_banner(f"开始训练: {variant}")
    print(f"训练步数: {steps}")
    print(f"保存目录: {save_dir}\n")
    
    cmd = [
        sys.executable, "-m", "train.train_mdm",
        "--save_dir", save_dir,
        "--dataset", "humanml",
        "--num_steps", str(steps),
        "--use_temporal_tcn",
        "--ms_tcn_variant", variant,
        "--save_interval", "500"
    ]
    
    print(f"命令: {' '.join(cmd)}\n")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"训练失败: {e}")
        return False
    except Exception as e:
        print(f"错误: {e}")
        return False


def find_latest_model(save_dir):
    """查找最新的模型文件"""
    import glob
    
    if not os.path.exists(save_dir):
        return None
    
    model_files = glob.glob(os.path.join(save_dir, "model*.pt"))
    if not model_files:
        return None
    
    # 按修改时间排序，返回最新的
    model_files.sort(key=os.path.getmtime)
    return model_files[-1]


def generate_video(variant, model_path):
    """生成视频"""
    output_dir = f"videos/{variant}"
    text_prompt = "the person walked forward and is picking up his toolbox."
    
    print_banner(f"生成视频: {variant}")
    print(f"模型: {model_path}")
    print(f"输出: {output_dir}")
    print(f"文本: {text_prompt}\n")
    
    cmd = [
        sys.executable, "-m", "sample.generate",
        "--model_path", model_path,
        "--text_prompt", text_prompt,
        "--num_samples", "3",
        "--num_repetitions", "1",
        "--output_dir", output_dir
    ]
    
    print(f"命令: {' '.join(cmd)}\n")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"生成失败: {e}")
        return False
    except Exception as e:
        print(f"错误: {e}")
        return False


def main():
    variants = ["baseline", "3-7-3-dil2", "3-5-5-dil2"]
    num_steps = 1000  # 演示用，实际建议 200000
    
    print_banner("多尺度TCN实验")
    print("\n变体列表:")
    for v in variants:
        print(f"  - {v}")
    print(f"\n训练步数: {num_steps}")
    print("预计时间: 20-100 分钟（取决于GPU）\n")
    
    # 创建videos目录
    os.makedirs("videos", exist_ok=True)
    
    results = {}
    
    for variant in variants:
        save_dir = f"save/ms_tcn_{variant}"
        
        # 1. 训练
        success = run_training(variant, num_steps)
        if not success:
            print(f"\n训练 {variant} 失败，跳过生成")
            results[variant] = {"train": False, "generate": False}
            continue
        
        results[variant] = {"train": True}
        
        # 2. 查找模型
        model_path = find_latest_model(save_dir)
        if not model_path:
            print(f"\n未找到 {variant} 的模型文件")
            results[variant]["generate"] = False
            continue
        
        print(f"\n找到模型: {model_path}")
        
        # 3. 生成视频
        success = generate_video(variant, model_path)
        results[variant]["generate"] = success
    
    # 总结
    print_banner("实验结果总结")
    
    for variant, result in results.items():
        train_status = "✓" if result.get("train") else "✗"
        gen_status = "✓" if result.get("generate") else "✗"
        print(f"\n{variant}:")
        print(f"  训练: {train_status}")
        print(f"  生成: {gen_status}")
    
    print("\n视频保存位置:")
    for variant in variants:
        video_dir = f"videos/{variant}"
        if os.path.exists(video_dir):
            mp4_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
            print(f"  {variant:12s}: {video_dir}/ ({len(mp4_files)} 个视频)")
        else:
            print(f"  {variant:12s}: 未生成")
    
    print("\n" + "="*70)
    print("完成!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
