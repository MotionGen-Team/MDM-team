"""
使用多尺度TCN变体生成动作视频
"""
import os
import sys
import argparse


def generate_with_variant(variant, model_path, output_dir, text_prompt, num_samples=3):
    """使用指定变体生成视频"""
    
    if not os.path.exists(model_path):
        print(f"错误: 模型不存在: {model_path}")
        print("请先训练模型或下载预训练模型")
        return False
    
    # 构建生成命令
    cmd = [
        "python", "-m", "sample.generate",
        "--model_path", model_path,
        "--num_samples", str(num_samples),
        "--num_repetitions", "1",
        "--output_dir", output_dir,
        "--text_prompt", text_prompt
    ]
    
    print(f"\n{'='*60}")
    print(f"生成视频 - 变体: {variant}")
    print(f"{'='*60}")
    print(f"模型: {model_path}")
    print(f"输出: {output_dir}")
    print(f"文本: {text_prompt}")
    print(f"\n命令: {' '.join(cmd)}")
    print()
    
    # 执行命令
    result = os.system(' '.join(cmd))
    
    if result == 0:
        print(f"\n✓ 变体 {variant} 生成成功!")
        print(f"视频保存在: {output_dir}")
        return True
    else:
        print(f"\n✗ 变体 {variant} 生成失败")
        return False


def main():
    parser = argparse.ArgumentParser(description='使用多尺度TCN生成动作视频')
    parser.add_argument('--variant', type=str, 
                        choices=['baseline', '3-7-3-dil2', '3-5-5-dil2', 'all'],
                        default='all',
                        help='选择变体')
    parser.add_argument('--model_dir', type=str, default='save',
                        help='模型保存目录')
    parser.add_argument('--output_dir', type=str, default='videos',
                        help='视频输出目录')
    parser.add_argument('--text_prompt', type=str, 
                        default='the person walked forward and is picking up his toolbox.',
                        help='文本提示')
    parser.add_argument('--num_samples', type=int, default=3,
                        help='生成样本数量')
    
    args = parser.parse_args()
    
    # 确定要运行的变体
    if args.variant == 'all':
        variants = ['baseline', '3-7-3-dil2', '3-5-5-dil2']
    else:
        variants = [args.variant]
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 为每个变体生成视频
    results = {}
    for variant in variants:
        # 查找模型文件
        variant_model_dir = os.path.join(args.model_dir, f"ms_tcn_{variant}")
        
        # 尝试查找模型文件
        model_path = None
        if os.path.exists(variant_model_dir):
            # 查找最新的模型文件
            import glob
            model_files = glob.glob(os.path.join(variant_model_dir, "model*.pt"))
            if model_files:
                model_files.sort()
                model_path = model_files[-1]  # 使用最新的模型
        
        if model_path is None:
            # 尝试其他可能的路径
            alternative_paths = [
                os.path.join(args.model_dir, f"humanml_trans_enc_512_{variant}", "model000200000.pt"),
                os.path.join(args.model_dir, "humanml_trans_enc_512", "model000200000.pt"),
            ]
            for path in alternative_paths:
                if os.path.exists(path):
                    model_path = path
                    break
        
        if model_path is None:
            print(f"\n警告: 找不到变体 {variant} 的模型")
            print(f"搜索路径: {variant_model_dir}")
            continue
        
        # 生成视频
        variant_output_dir = os.path.join(args.output_dir, variant)
        success = generate_with_variant(
            variant=variant,
            model_path=model_path,
            output_dir=variant_output_dir,
            text_prompt=args.text_prompt,
            num_samples=args.num_samples
        )
        results[variant] = success
    
    # 打印总结
    print(f"\n{'='*60}")
    print("生成结果总结")
    print(f"{'='*60}")
    for variant, success in results.items():
        status = "✓ 成功" if success else "✗ 失败"
        print(f"{variant:15s}: {status}")
    
    print(f"\n视频保存在: {os.path.abspath(args.output_dir)}")
    print("="*60)


if __name__ == "__main__":
    main()
