import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleTemporalConv(nn.Module):
    """
    多尺度时间卷积模块
    支持不同的卷积核大小和 dilation 组合实验
    """
    def __init__(self, latent_dim, dropout=0.1, kernel_sizes=(3, 5, 3), dilations=(1, 1, 2)):
        """
        Args:
            latent_dim: 特征维度
            dropout: dropout 概率
            kernel_sizes: 每个分支的卷积核大小元组，默认 (3, 5, 3)
            dilations: 每个分支的 dilation 元组，默认 (1, 1, 2)
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.kernel_sizes = kernel_sizes
        self.dilations = dilations
        
        assert len(kernel_sizes) == len(dilations), "kernel_sizes 和 dilations 长度必须相同"
        
        # 为每个尺度创建一个卷积分支
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(
                    latent_dim, 
                    latent_dim, 
                    kernel_size=k, 
                    padding=(k - 1) * d // 2,  # 保持序列长度不变
                    dilation=d
                ),
                nn.SiLU(),
                nn.Dropout(dropout)
            )
            for k, d in zip(kernel_sizes, dilations)
        ])
        
        # 输出层归一化
        self.out_norm = nn.LayerNorm(latent_dim)
        
    def forward(self, x):
        """
        Args:
            x: [seqlen, bs, d]
        Returns:
            [seqlen, bs, d]
        """
        # 转换为 [bs, d, seqlen] 以适应 Conv1d
        x_conv = x.permute(1, 2, 0)  # [bs, d, seqlen]
        
        # 多分支卷积 + 残差连接
        outputs = []
        for branch in self.branches:
            out = branch(x_conv) + x_conv  # 残差连接
            outputs.append(out)
        
        # 多尺度特征融合（相加）
        x_fused = sum(outputs) / len(outputs)
        
        # 转换回 [seqlen, bs, d]
        x = x_fused.permute(2, 0, 1)  # [seqlen, bs, d]
        x = self.out_norm(x)
        
        return x


# ========== 预定义的变体配置 ==========

def create_baseline_tcn(latent_dim, dropout=0.1):
    """
    Baseline: (3, 5, 3-dil2)
    三个分支：kernel=3, kernel=5, kernel=3 with dilation=2
    """
    return MultiScaleTemporalConv(
        latent_dim=latent_dim,
        dropout=dropout,
        kernel_sizes=(3, 5, 3),
        dilations=(1, 1, 2)
    )


def create_variant_3_7_3_dil2(latent_dim, dropout=0.1):
    """
    变体 1: (3, 7, 3-dil2)
    三个分支：kernel=3, kernel=7, kernel=3 with dilation=2
    """
    return MultiScaleTemporalConv(
        latent_dim=latent_dim,
        dropout=dropout,
        kernel_sizes=(3, 7, 3),
        dilations=(1, 1, 2)
    )


def create_variant_3_5_5_dil2(latent_dim, dropout=0.1):
    """
    变体 2: (3, 5, 5-dil2)
    三个分支：kernel=3, kernel=5, kernel=5 with dilation=2
    """
    return MultiScaleTemporalConv(
        latent_dim=latent_dim,
        dropout=dropout,
        kernel_sizes=(3, 5, 5),
        dilations=(1, 1, 2)
    )


# ========== 工厂函数 ==========

def get_multi_scale_tcn(variant_name, latent_dim, dropout=0.1):
    """
    根据变体名称获取对应的多尺度 TCN 模块
    
    Args:
        variant_name: 变体名称，可选 'baseline', '3-7-3-dil2', '3-5-5-dil2'
        latent_dim: 特征维度
        dropout: dropout 概率
    
    Returns:
        MultiScaleTemporalConv 实例
    """
    variant_map = {
        'baseline': create_baseline_tcn,
        '3-7-3-dil2': create_variant_3_7_3_dil2,
        '3-5-5-dil2': create_variant_3_5_5_dil2,
    }
    
    if variant_name not in variant_map:
        raise ValueError(f"未知的变体名称: {variant_name}，可选: {list(variant_map.keys())}")
    
    return variant_map[variant_name](latent_dim, dropout)


# ========== 测试代码 ==========

if __name__ == "__main__":
    # 测试各个变体
    batch_size = 4
    seq_len = 196
    latent_dim = 256
    
    # 创建随机输入
    x = torch.randn(seq_len, batch_size, latent_dim)
    
    print("=" * 60)
    print("测试多尺度时间卷积模块")
    print("=" * 60)
    
    variants = ['baseline', '3-7-3-dil2', '3-5-5-dil2']
    
    for variant in variants:
        print(f"\n变体: {variant}")
        model = get_multi_scale_tcn(variant, latent_dim)
        
        # 统计参数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"  输入形状: {x.shape}")
        
        # 前向传播
        with torch.no_grad():
            output = model(x)
        
        print(f"  输出形状: {output.shape}")
        print(f"  总参数量: {total_params:,}")
        print(f"  可训练参数量: {trainable_params:,}")
        
        # 验证输出形状
        assert output.shape == x.shape, f"输出形状不匹配: {output.shape} != {x.shape}"
        print(f"  ✓ 测试通过")
    
    print("\n" + "=" * 60)
    print("所有变体测试通过！")
    print("=" * 60)
