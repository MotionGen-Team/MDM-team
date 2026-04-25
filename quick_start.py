"""
快速启动脚本 - 多尺度TCN实验
整合训练、生成和对比
"""
import os
import sys
import argparse


def print_header(title):
    """打印标题"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)


def check_environment():
    """检查环境是否就绪"""
    print_header("环境检查")
    
    checks = {
        "Python": sys.version.split()[0],
        "PyTorch": None,
        "CUDA": None,
        "数据集": None,
    }
    
    # 检查 PyTorch
    try:
        import torch
        checks["PyTorch"] = torch.__version__
        checks["CUDA"] = torch.cuda.is_available()
    except ImportError:
        checks["PyTorch"] = "未安装"
    
    # 检查数据集
    dataset_path = "./dataset/humanml"
    if os.path.exists(dataset_path):
        checks["数据集"] = "已找到"
    else:
        checks["数据集"] = f"未找到 ({dataset_path})"
    
    for key, value in checks.items():
        print(f"  {key:12s}: {value}")
    
    return checks


def demo_module():
    """演示多尺度TCN模块"""
    print_header("模块演示")
    
    try:
        import torch
        from model.plan_one.local_branch.multi_scale_temporal import get_multi_scale_tcn
        
        variants = ['baseline', '3-7-3-dil2', '3-5-5-dil2']
        
        print("\n测试三种变体:")
        for variant in variants:
            model = get_multi_scale_tcn(variant, latent_dim=256, dropout=0.1)
            x = torch.randn(196, 4, 256)
            
            with torch.no_grad():
                y = model(x)
            
            params = sum(p.numel() for p in model.parameters())
            print(f"  {variant:12s}: 输入{x.shape} -> 输出{y.shape} | 参数量: {params:,}")
        
        print("\n✓ 所有变体测试通过!")
        return True
        
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        return False


def train_models(variants, num_steps=1000):
    """训练模型"""
    print_header("模型训练")
    
    print(f"\n将训练以下变体 (步数: {num_steps}):")
    for v in variants:
        print(f"  - {v}")
    
    print("\n训练命令示例:")
    print(f"  python train_variants.py --variant all --num_steps {num_steps}")
    
    response = input("\n是否开始训练? (y/n): ").lower().strip()
    
    if response == 'y':
        for variant in variants:
            save_dir = f"save/ms_tcn_{variant}"
            cmd = f"python -m train.train_mdm --save_dir {save_dir} --dataset humanml --num_steps {num_steps} --use_temporal_tcn --ms_tcn_variant {variant} --save_interval 1000"
            print(f"\n训练 {variant}...")
            print(f"命令: {cmd}")
            os.system(cmd)
    else:
        print("跳过训练")


def generate_videos(variants):
    """生成视频"""
    print_header("视频生成")
    
    text_prompt = "the person walked forward and is picking up his toolbox."
    
    print(f"\n文本提示: {text_prompt}")
    print("\n将为以下变体生成视频:")
    for v in variants:
        print(f"  - {v}")
    
    print("\n生成命令示例:")
    print(f"  python generate_videos.py --variant all --text_prompt \"{text_prompt}\"")
    
    response = input("\n是否开始生成? (y/n): ").lower().strip()
    
    if response == 'y':
        cmd = f'python generate_videos.py --variant all --text_prompt "{text_prompt}"'
        os.system(cmd)
    else:
        print("跳过生成")


def main():
    parser = argparse.ArgumentParser(description='多尺度TCN快速启动')
    parser.add_argument('--step', type=str,
                        choices=['check', 'demo', 'train', 'generate', 'all'],
                        default='all',
                        help='执行步骤')
    parser.add_argument('--num_steps', type=int, default=1000,
                        help='训练步数（演示用）')
    
    args = parser.parse_args()
    
    variants = ['baseline', '3-7-3-dil2', '3-5-5-dil2']
    
    print("\n" + "="*70)
    print("  多尺度时间卷积实验 - 快速启动")
    print("="*70)
    
    if args.step in ['check', 'all']:
        env = check_environment()
        
        if env["PyTorch"] == "未安装":
            print("\n错误: 请先安装 PyTorch")
            return
    
    if args.step in ['demo', 'all']:
        if not demo_module():
            print("\n错误: 模块测试失败，请检查代码")
            return
    
    if args.step in ['train', 'all']:
        train_models(variants, args.num_steps)
    
    if args.step in ['generate', 'all']:
        generate_videos(variants)
    
    print_header("完成")
    print("\n生成的文件:")
    print("  - 模型: save/ms_tcn_*/")
    print("  - 视频: videos/*/")
    print("\n感谢使用!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
