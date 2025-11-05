# model settings
norm_cfg = dict(type="SyncBN", requires_grad=True)

model = dict(
    type="CustomEncoderDecoder",
    data_preprocessor=None,
    
    backbone=dict(
        type="SwinTransformer",
        in_channels=12,
        embed_dims=96,
        depths=[2, 2, 18, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.3,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=False,
        pretrained=None,  # 暂时不使用预训练权重
    ),
    
    decode_head=dict(
        type="CustomSegformerHead",
        in_channels=[96, 192, 384, 768],
        in_index=[0, 1, 2, 3],
        channels=256,
        num_classes=1,
        dropout_ratio=0.1,
        norm_cfg=norm_cfg,
        align_corners=False,
    ),
    
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode="whole"),
) 