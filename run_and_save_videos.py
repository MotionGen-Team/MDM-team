"""
运行实验并保存视频到 videos 文件夹
"""
import os
import sys
import glob
import subprocess


def run_command(cmd, description):
    """运行命令并打印输出"""
    print(f"\n{'='*70}")
    print(f"{description}")
    print(f"{'='*70}")
    print(f"命令: {cmd}\n")
    
    result = os.system(cmd)
    return result == 0


def find_model(save_dir):
    """查找模型文件"""
    if not os.path.exists(save_dir):
        return None
    
    model_files = glob.glob(os.path.join(save_dir, "model*.pt"))
    if model_files:
        # 返回最新的模型
        model_files.sort()
        return model_files[-1]
    return None


def main():
    # 配置
    variants = ['baseline', '3-7-3-dil2', '3-5-5-dil2']
    num_train_steps = 1000  # 演示用，实际建议 200000
    num_samples = 3
    text_prompt = "the person walked forward and is picking up his toolbox."
    
    print("\n" + "="*70)
    print("  多尺度TCN实验 - 自动生成视频")
    print("="*70)
    
    # 创建输出目录
    os.makedirs("videos", exist_ok=True)
    
    # 为每个变体训练和生成
    for variant in variants:
        print(f"\n\n{'#'*70}")
        print(f"# 处理变体: {variant}")
        print(f"{'#'*70}")
        
        save_dir = f"save/ms_tcn_{variant}"
        model_path = find_model(save_dir)
        
        # 1. 训练模型（如果没有）
        if model_path is None:
            print(f"\n[1/2] 训练 {variant} 模型...")
            train_cmd = f"python -m train.train_mdm --save_dir {save_dir} --dataset humanml --num_steps {num_train_steps} --use_temporal_tcn --ms_tcn_variant {variant} --save_interval 500"
            
            success = run_command(train_cmd, f"训练 {variant}")
            if not success:
                print(f"训练 {variant} 失败，跳过")
                continue
            
            # 重新查找模型
            model_path = find_model(save_dir)
            if model_path is None:
                print(f"无法找到训练好的模型，跳过 {variant}")
                continue
        else:
            print(f"\n[1/2] 找到已有模型: {model_path}")
        
        # 2. 生成视频
        print(f"\n[2/2] 生成 {variant} 视频...")
        output_dir = f"videos/{variant}"
        
        gen_cmd = f'python -m sample.generate --model_path {model_path} --text_prompt "{text_prompt}" --num_samples {num_samples} --num_repetitions 1 --output_dir {output_dir}'
        
        success = run_command(gen_cmd, f"生成 {variant} 视频")
        if success:
            print(f"✓ {variant} 视频生成成功，保存在: {output_dir}")
        else:
            print(f"✗ {variant} 视频生成失败")
    
    # 总结
    print("\n\n" + "="*70)
    print("实验完成!")
    print("="*70)
    print("\n生成的视频位置:")
    for variant in variants:
        video_dir = f"videos/{variant}"
        if os.path.exists(video_dir):
            mp4_files = glob.glob(os.path.join(video_dir, "*.mp4"))
            print(f"  {variant:12s}: {video_dir}/ ({len(mp4_files)} 个视频)")
        else:
            print(f"  {variant:12s}: 未生成")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
