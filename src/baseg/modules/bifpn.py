import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    """通道注意力机制，关注重要的通道特征"""
    def __init__(self, channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # 共享MLP
        reduced_channels = max(8, channels // reduction_ratio)
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(channels, reduced_channels, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(reduced_channels, channels, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # 平均池化分支
        avg_out = self.shared_mlp(self.avg_pool(x))
        # 最大池化分支
        max_out = self.shared_mlp(self.max_pool(x))
        # 融合并应用sigmoid激活
        out = self.sigmoid(avg_out + max_out)
        return x * out

class FastNormalizedFusion(nn.Module):
    """实现快速归一化融合，使用可学习权重为每个输入特征分配重要性"""
    def __init__(self, num_inputs, eps=1e-4):
        super(FastNormalizedFusion, self).__init__()
        self.weights = nn.Parameter(torch.ones(num_inputs, dtype=torch.float32), requires_grad=True)
        self.eps = eps

    def forward(self, inputs):
        # 使用GELU确保权重非负
        weights = F.gelu(self.weights)
        # 归一化权重
        norm_weights = weights / (weights.sum() + self.eps)
        
        # 加权求和
        output = 0
        for i, inp in enumerate(inputs):
            output += norm_weights[i] * inp
            
        return output

class DepthwiseSeparableConv(nn.Module):
    """深度可分离卷积块，包含深度卷积和点卷积"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size, 
            stride=stride, padding=padding, groups=in_channels, bias=bias
        )
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, 
            stride=1, padding=0, bias=bias
        )
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class ConvBlock(nn.Module):
    """改进的卷积块，使用深度可分离卷积+批归一化+激活函数+通道注意力"""
    def __init__(self, channels, use_attention=True):
        super(ConvBlock, self).__init__()
        self.conv = DepthwiseSeparableConv(channels, channels)
        self.bn = nn.BatchNorm2d(channels)
        self.act = nn.GELU()
        self.use_attention = use_attention
        if use_attention:
            self.ca = ChannelAttention(channels)
        
    def forward(self, x):
        x = self.act(self.bn(self.conv(x)))
        if self.use_attention:
            x = self.ca(x)
        return x

class BiFPNLayer(nn.Module):
    def __init__(self, channels, use_residual=True, use_attention=True, eps=1e-4):
        super().__init__()
        self.eps = eps
        self.use_residual = use_residual
        
        # 使用FastNormalizedFusion替代简单的权重参数
        self.fusion_p4_td = FastNormalizedFusion(2)  # P4, P5上采样
        self.fusion_p3_td = FastNormalizedFusion(2)  # P3, P4_td上采样
        self.fusion_p4_out = FastNormalizedFusion(3)  # P4, P4_td, P3_td下采样
        self.fusion_p5_out = FastNormalizedFusion(3)  # P5, P4_out下采样, P6
        
        # 使用改进的卷积块
        self.conv_blocks = nn.ModuleList([ConvBlock(channels, use_attention) for _ in range(4)])

    def forward(self, inputs):
        # inputs: [P3, P4, P5, P6] (高到低分辨率)
        P3, P4, P5, P6 = inputs
        
        # 上采样融合
        P5_upsampled = F.interpolate(P5, size=P4.shape[2:], mode='nearest')
        P4_td = self.conv_blocks[0](self.fusion_p4_td([P4, P5_upsampled]))
        
        P4_td_upsampled = F.interpolate(P4_td, size=P3.shape[2:], mode='nearest')
        P3_td = self.conv_blocks[1](self.fusion_p3_td([P3, P4_td_upsampled]))
        
        # 下采样融合
        P3_td_downsampled = F.max_pool2d(P3_td, 2)
        P4_out = self.conv_blocks[2](self.fusion_p4_out([P4, P4_td, P3_td_downsampled]))
        
        P4_out_downsampled = F.max_pool2d(P4_out, 2)
        P5_out = self.conv_blocks[3](self.fusion_p5_out([P5, P4_out_downsampled, P6]))
        
        # 应用残差连接
        if self.use_residual:
            P3_td = P3_td + P3
            P4_out = P4_out + P4
            P5_out = P5_out + P5
        
        return [P3_td, P4_out, P5_out, P6]

class BiFPN(nn.Module):
    def __init__(self, channels, num_layers=2, use_residual=True, use_attention=True):
        """增强型BiFPN模块
        
        Args:
            channels: 特征通道数
            num_layers: BiFPN层数
            use_residual: 是否使用残差连接
            use_attention: 是否使用通道注意力机制
        """
        super().__init__()
        self.layers = nn.ModuleList([
            BiFPNLayer(
                channels, 
                use_residual=use_residual, 
                use_attention=use_attention
            ) for _ in range(num_layers)
        ])
        
        # 初始化权重
        self._initialize_weights()
        
    def _initialize_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        
    def forward(self, inputs):
        feats = inputs
        for layer in self.layers:
            feats = layer(feats)
        return feats 