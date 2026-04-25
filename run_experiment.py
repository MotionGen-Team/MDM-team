"""
多尺度时间卷积实验脚本
支持三种变体：baseline, 3-7-3-dil2, 3-5-5-dil2
"""
import os
import sys
import argparse


def train_model(variant, save_dir, dataset='humanml', **kwargs):
    """
    训练指定变体的模型
    
    Args:
        variant: 变体名称 ('baseline', '3-7-3-dil2', '3-5-5-dil2')
        save_dir: 模型保存目录
        dataset: 数据集名称
        **kwargs: 其他训练参数
    """
    # 构建训练命令
    cmd = f"python -m train.train_mdm --save_dir {save_dir} --dataset {dataset}"
    
    # 添加多尺度 TCN 参数
    if variant:
        cmd += f" --use_temporal_tcn --ms_tcn_variant {variant}"
    
    # 添加其他参数
    for key, value in kwargs.items():
        if isinstance(value, bool):
            if value:
                cmd += f" --{key}"
        else:
            cmd += f" --{key} {value}"
    
    print(f"\n{'='*60}")
    print(f"开始训练变体: {variant}")
    print(f"保存目录: {save_dir}")
    print(f"命令: {cmd}")
    print(f"{'='*60}\n")
    
    os.system(cmd)


def generate_videos(model_path, output_dir, text_prompt=None, num_samples=3, num_repetitions=1):
    """
    使用训练好的模型生成视频
    
    Args:
        model_path: 模型路径
        output_dir: 输出目录
        text_prompt: 文本提示（可选）
        num_samples: 样本数量
        num_repetitions: 重复次数
    """
    # 构建生成命令
    cmd = f"python -m sample.generate --model_path {model_path} --num_samples {num_samples} --num_repetitions {num_repetitions} --output_dir {output_dir}"
    
    if text_prompt:
        cmd += f' --text_prompt "{text_prompt}"'
    
    print(f"\n{'='*60}")
    print(f"开始生成视频")
    print(f"模型: {model_path}")
    print(f"输出: {output_dir}")
    print(f"命令: {cmd}")
    print(f"{'='*60}\n")
    
    os.system(cmd)


def main():
    parser = argparse.ArgumentParser(description='多尺度时间卷积实验')
    parser.add_argument('--mode', type=str, choices=['train', 'generate', 'both'], default='both',
                        help='运行模式: train(仅训练), generate(仅生成), both(训练+生成)')
    parser.add_argument('--variant', type=str, choices=['baseline', '3-7-3-dil2', '3-5-5-dil2', 'all'], default='all',
                        help='要运行的变体')
    parser.add_argument('--dataset', type=str, default='humanml', help='数据集名称')
    parser.add_argument('--text_prompt', type=str, default='the person walked forward and is picking up his toolbox.',
                        help='生成视频时的文本提示')
    parser.add_argument('--num_samples', type=int, default=3, help='生成样本数量')
    parser.add_argument('--num_repetitions', type=int, default=1, help='每个样本的重复次数')
    parser.add_argument('--training_steps', type=int, default=1000, help='训练步数（演示用，实际建议更多）')
    
    args = parser.parse_args()
    
    # 确定要运行的变体
    variants = []
    if args.variant == 'all':
        variants = ['baseline', '3-7-3-dil2', '3-5-5-dil2']
    else:
        variants = [args.variant]
    
    # 创建输出目录
    os.makedirs('videos', exist_ok=True)
    
    # 运行实验
    for variant in variants:
        save_dir = f"save/ms_tcn_{variant}"
        model_path = f"{save_dir}/model{args.training_steps:09d}.pt"
        output_dir = f"videos/{variant}"
        
        # 训练
        if args.mode in ['train', 'both']:
            train_model(
                variant=variant,
                save_dir=save_dir,
                dataset=args.dataset,
                # 训练参数
                eval_during_training=True,
                gen_during_training=True,
                save_interval=1000,
                num_steps=args.training_steps,
                # 多尺度 TCN 参数
                use_temporal_tcn=True,
                ms_tcn_variant=variant
            )
        
        # 生成
        if args.mode in ['generate', 'both']:
            # 检查模型是否存在
            if not os.path.exists(model_path):
                print(f"警告: 模型不存在: {model_path}")
                # 尝试查找其他模型文件
                import glob
                model_files = glob.glob(f"{save_dir}/model*.pt")
                if model_files:
                    model_path = model_files[-1]  # 使用最新的模型
                    print(f"使用找到的最新模型: {model_path}")
                else:
                    print(f"跳过变体 {variant}，未找到模型文件")
                    continue
            
            generate_videos(
                model_path=model_path,
                output_dir=output_dir,
                text_prompt=args.text_prompt,
                num_samples=args.num_samples,
                num_repetitions=args.num_repetitions
            )
    
    print(f"\n{'='*60}")
    print("实验完成!")
    print(f"视频保存在: videos/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
