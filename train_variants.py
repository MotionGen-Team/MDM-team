"""
训练多尺度TCN三种变体的脚本
"""
import os
import sys
import argparse


def train_variant(variant, save_dir, dataset='humanml', num_steps=200000, **kwargs):
    """
    训练指定变体的模型
    
    Args:
        variant: 变体名称 ('baseline', '3-7-3-dil2', '3-5-5-dil2')
        save_dir: 模型保存目录
        dataset: 数据集名称
        num_steps: 训练步数
        **kwargs: 其他训练参数
    """
    # 构建基础命令
    cmd_parts = [
        "python", "-m", "train.train_mdm",
        "--save_dir", save_dir,
        "--dataset", dataset,
        "--num_steps", str(num_steps),
        "--use_temporal_tcn"  # 启用时间TCN
    ]
    
    # 添加变体参数
    if variant:
        cmd_parts.extend(["--ms_tcn_variant", variant])
    
    # 添加推荐参数
    cmd_parts.extend([
        "--eval_during_training",
        "--gen_during_training",
        "--use_ema",
        "--mask_frames",
        "--save_interval", "10000"
    ])
    
    # 添加其他参数
    for key, value in kwargs.items():
        if isinstance(value, bool):
            if value:
                cmd_parts.append(f"--{key}")
        else:
            cmd_parts.extend([f"--{key}", str(value)])
    
    cmd = ' '.join(cmd_parts)
    
    print(f"\n{'='*70}")
    print(f"开始训练变体: {variant}")
    print(f"保存目录: {save_dir}")
    print(f"{'='*70}")
    print(f"命令:\n{cmd}")
    print()
    
    # 执行训练
    result = os.system(cmd)
    
    return result == 0


def main():
    parser = argparse.ArgumentParser(description='训练多尺度TCN变体')
    parser.add_argument('--variant', type=str,
                        choices=['baseline', '3-7-3-dil2', '3-5-5-dil2', 'all'],
                        default='all',
                        help='要训练的变体')
    parser.add_argument('--dataset', type=str, default='humanml',
                        help='数据集名称')
    parser.add_argument('--num_steps', type=int, default=200000,
                        help='训练步数')
    parser.add_argument('--save_dir_prefix', type=str, default='save',
                        help='保存目录前缀')
    
    args = parser.parse_args()
    
    # 确定要训练的变体
    if args.variant == 'all':
        variants = ['baseline', '3-7-3-dil2', '3-5-5-dil2']
    else:
        variants = [args.variant]
    
    # 训练每个变体
    results = {}
    for variant in variants:
        save_dir = os.path.join(args.save_dir_prefix, f"ms_tcn_{variant}")
        
        success = train_variant(
            variant=variant,
            save_dir=save_dir,
            dataset=args.dataset,
            num_steps=args.num_steps
        )
        
        results[variant] = success
        
        if not success:
            print(f"\n警告: 变体 {variant} 训练可能出错")
    
    # 打印总结
    print(f"\n{'='*70}")
    print("训练结果总结")
    print(f"{'='*70}")
    for variant, success in results.items():
        status = "✓ 完成" if success else "✗ 失败"
        print(f"{variant:15s}: {status}")
    
    print(f"\n模型保存在: {args.save_dir_prefix}/")
    print("="*70)


if __name__ == "__main__":
    main()
