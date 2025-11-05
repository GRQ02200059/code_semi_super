# model settings
# 定义模型的标准化层配置，使用同步批归一化，允许参数更新
norm_cfg = dict(type="SyncBN", requires_grad=True)

# 定义整体模型结构
model = dict(
    # 使用自定义的编码器-解码器架构作为模型基本框架
    type="CustomEncoderDecoder",
    # 不使用数据预处理器
    data_preprocessor=None,
    
    # 定义骨干网络（特征提取器）配置
    backbone=dict(
        # 使用MixVisionTransformer作为骨干网络
        type="MixVisionTransformer",
        in_channels=12,           # 输入图像的通道数
        embed_dims=32,           # Transformer的初始嵌入维度
        num_stages=4,            # 模型的层次化阶段数
        num_layers=[2, 2, 2, 2], # 每个阶段中Transformer层的数量
        num_heads=[1, 2, 5, 8],  # 每个阶段中多头注意力的头数，随深度增加而增加
        patch_sizes=[7, 3, 3, 3],# 每个阶段的patch大小，第一层用7x7，其余用3x3
        sr_ratios=[8, 4, 2, 1],  # 每个阶段的空间规约比率，逐渐减小
        out_indices=(0, 1, 2, 3),# 输出特征的层级索引
        mlp_ratio=4,             # MLP中间层的扩展比例
        qkv_bias=True,           # 在QKV投影中使用偏置项
        drop_rate=0.0,           # dropout比率
        attn_drop_rate=0.0,      # 注意力层的dropout比率
        drop_path_rate=0.1,      # drop path的比率
    ),
    
    # 定义解码头配置
    decode_head=dict(
        # 使用自定义的SegFormer解码头
        type="CustomSegformerHead",
        in_channels=[32, 64, 160, 256],  # 每个阶段输出的特征通道数
        in_index=[0, 1, 2, 3],           # 使用的特征层索引
        channels=256,                     # 解码头输出的通道数
        dropout_ratio=0.1,                # dropout比率
        num_classes=1,                    # 分割类别数（这里是二分类）
        norm_cfg=norm_cfg,                # 使用上面定义的归一化配置
        align_corners=False,              # 上采样时是否对齐角点
    ),
    
    # 训练和测试配置
    train_cfg=dict(),                     # 训练配置（使用默认配置）
    test_cfg=dict(mode="whole"),         # 测试配置，使用整图推理模式
)
