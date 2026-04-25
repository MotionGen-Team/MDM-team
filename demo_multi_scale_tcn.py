"""
多尺度时间卷积模块演示脚本
展示三种变体的效果差异
"""
import torch
import numpy as np
import os
from model.plan_one.local_branch.multi_scale_temporal import (
    get_multi_scale_tcn,
    create_baseline_tcn,
    create_variant_3_7_3_dil2,
    create_variant_3_5_5_dil2
)


def analyze_receptive_field(kernel_size, dilation):
    """计算感受野大小"""
    return (kernel_size - 1) * dilation + 1


def print_variant_info(variant_name, kernel_sizes, dilations):
    """打印变体信息"""
    print(f"\n{'='*60}")
    print(f"变体: {variant_name}")
    print(f"{'='*60}")
    print(f"卷积核大小: {kernel_sizes}")
    print(f"Dilations:  {dilations}")
    print(f"\n各分支感受野:")
    for i, (k, d) in enumerate(zip(kernel_sizes, dilations)):
        rf = analyze_receptive_field(k, d)
        print(f"  分支 {i+1}: kernel={k}, dilation={d} -> 感受野 = {rf}")
    total_rf = sum(analyze_receptive_field(k, d) for k, d in zip(kernel_sizes, dilations))
    print(f"\n总感受野范围: {total_rf}")


def test_forward_pass(variant_name, latent_dim=256, seq_len=196, batch_size=4):
    """测试前向传播"""
    print(f"\n测试 {variant_name} 前向传播...")
    
    # 创建模型
    model = get_multi_scale_tcn(variant_name, latent_dim, dropout=0.1)
    
    # 创建随机输入
    x = torch.randn(seq_len, batch_size, latent_dim)
    
    # 前向传播
    with torch.no_grad():
        output = model(x)
    
    # 验证输出
    assert output.shape == x.shape, f"输出形状不匹配!"
    
    # 统计信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"  输入形状:  {x.shape}")
    print(f"  输出形状:  {output.shape}")
    print(f"  总参数量:  {total_params:,}")
    print(f"  可训练参数量: {trainable_params:,}")
    print(f"  ✓ 测试通过")
    
    return model, output


def compare_feature_extraction():
    """比较不同变体的特征提取能力"""
    print(f"\n{'='*60}")
    print("特征提取能力对比")
    print(f"{'='*60}")
    
    latent_dim = 256
    seq_len = 196
    batch_size = 2
    
    # 创建一个模拟的时序信号（包含不同频率的成分）
    t = torch.linspace(0, 4*np.pi, seq_len)
    
    # 低频成分
    low_freq = torch.sin(t).unsqueeze(1).unsqueeze(2).repeat(1, batch_size, latent_dim)
    # 中频成分
    mid_freq = torch.sin(3*t).unsqueeze(1).unsqueeze(2).repeat(1, batch_size, latent_dim)
    # 高频成分
    high_freq = torch.sin(10*t).unsqueeze(1).unsqueeze(2).repeat(1, batch_size, latent_dim)
    
    # 组合信号
    x = low_freq + 0.5 * mid_freq + 0.2 * high_freq
    x = x + 0.1 * torch.randn_like(x)  # 添加噪声
    
    variants = ['baseline', '3-7-3-dil2', '3-5-5-dil2']
    outputs = {}
    
    for variant in variants:
        model = get_multi_scale_tcn(variant, latent_dim, dropout=0.0)
        model.eval()
        
        with torch.no_grad():
            output = model(x)
        
        outputs[variant] = output
        
        # 计算输出变化程度
        output_var = output.var(dim=0).mean().item()
        input_var = x.var(dim=0).mean().item()
        
        print(f"\n{variant}:")
        print(f"  输入方差:  {input_var:.4f}")
        print(f"  输出方差:  {output_var:.4f}")
        print(f"  变化比例:  {output_var/input_var:.4f}")
    
    return outputs


def visualize_receptive_fields():
    """可视化不同变体的感受野"""
    print(f"\n{'='*60}")
    print("感受野可视化")
    print(f"{'='*60}")
    
    variants = {
        'baseline': ((3, 5, 3), (1, 1, 2)),
        '3-7-3-dil2': ((3, 7, 3), (1, 1, 2)),
        '3-5-5-dil2': ((3, 5, 5), (1, 1, 2)),
    }
    
    for name, (kernels, dils) in variants.items():
        print_variant_info(name, kernels, dils)


def main():
    print("\n" + "="*60)
    print("多尺度时间卷积模块演示")
    print("="*60)
    
    # 1. 展示三种变体的配置
    visualize_receptive_fields()
    
    # 2. 测试每种变体的前向传播
    print(f"\n{'='*60}")
    print("前向传播测试")
    print(f"{'='*60}")
    
    variants = ['baseline', '3-7-3-dil2', '3-5-5-dil2']
    for variant in variants:
        test_forward_pass(variant)
    
    # 3. 比较特征提取能力
    outputs = compare_feature_extraction()
    
    # 4. 总结
    print(f"\n{'='*60}")
    print("总结")
    print(f"{'='*60}")
    print("""
三种变体的特点:

1. baseline (3, 5, 3-dil2):
   - 平衡的配置
   - 适合一般性任务

2. 3-7-3-dil2:
   - 更大的中间分支感受野 (7 vs 5)
   - 适合捕捉中等范围的时序依赖

3. 3-5-5-dil2:
   - 更大的dilated分支感受野 (9 vs 5)
   - 适合捕捉长范围的时序依赖

使用建议:
- 如果任务需要捕捉长距离依赖，优先尝试 3-5-5-dil2
- 如果需要平衡性能，使用 baseline
- 如果中等范围模式重要，尝试 3-7-3-dil2
    """)
    
    print("\n演示完成!")
    print("="*60)


if __name__ == "__main__":
    main()
