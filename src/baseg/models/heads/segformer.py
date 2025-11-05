# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmseg.registry import MODELS

from baseg.models.heads import CustomBaseDecodeHead
from baseg.models.utils import resize
from baseg.models.utils import RSCSAM
import torch.nn.functional as F


@MODELS.register_module()
class CustomSegformerHead(CustomBaseDecodeHead):
    """SegFormer的全MLP解码头。

    这个解码头实现了SegFormer论文中的设计
    论文链接: `Segformer <https://arxiv.org/abs/2105.15203>` _

    主要特点：
    1. 使用轻量级的MLP进行特征处理
    2. 多尺度特征融合
    3. 简单高效的设计

    Args:
        interpolate_mode: MLP头上采样操作的插值模式
            默认使用"bilinear"双线性插值
    """

    def __init__(self, interpolate_mode="bilinear", **kwargs):
        # 初始化父类，设置输入转换模式为多选择模式（处理多尺度特征）
        super().__init__(input_transform="multiple_select", **kwargs)

        # 设置特征图上采样的插值模式
        self.interpolate_mode = interpolate_mode
        # 获取输入特征的数量（通常是4，对应backbone的4个阶段）
        num_inputs = len(self.in_channels)

        # 确保输入特征数量与索引数量匹配
        assert num_inputs == len(self.in_index)

        # RS-CSAM注意力模块（每个尺度一个）
        self.rscsam_list = nn.ModuleList([
            RSCSAM(self.in_channels[i]) for i in range(num_inputs)
        ])

        # 创建卷积层列表，用于处理每个尺度的特征
        self.convs = nn.ModuleList()
        for i in range(num_inputs):
            # 为每个输入特征创建1x1卷积，统一通道维度
            self.convs.append(
                ConvModule(
                    in_channels=self.in_channels[i],  # 当前尺度的输入通道数
                    out_channels=self.channels,       # 统一的输出通道数
                    kernel_size=1,                    # 使用1x1卷积
                    stride=1,                         # 步长为1，保持空间尺寸
                    norm_cfg=self.norm_cfg,          # 使用配置中指定的归一化
                    act_cfg=self.act_cfg,            # 使用配置中指定的激活函数
                )
            )

        # 创建特征融合的卷积层
        self.fusion_conv = ConvModule(
            in_channels=self.channels * num_inputs,  # 输入通道数是所有特征通道数的总和
            out_channels=self.channels,             # 输出为统一的通道数
            kernel_size=1,                          # 使用1x1卷积进行融合
            norm_cfg=self.norm_cfg                  # 使用配置中指定的归一化
        )


    def forward(self, inputs):
        """前向传播函数
        
        Args:
            inputs: 来自backbone的多尺度特征列表
                   包含4个不同尺度的特征图: 1/4, 1/8, 1/16, 1/32
        
        Returns:
            feat: 融合后的特征图
        """
        # 转换输入特征（可能包括排序、选择等操作）
        inputs = self._transform_inputs(inputs)
        outs = []
        
        # 先经过RS-CSAM
        for idx in range(len(inputs)):
            x = inputs[idx]
            x = self.rscsam_list[idx](x)
            conv = self.convs[idx]
            x = conv(x)
            x = resize(
                input=x,
                size=inputs[0].shape[2:],
                mode=self.interpolate_mode,
                align_corners=self.align_corners
            )
            outs.append(x)

        # 特征融合
        fused = self.fusion_conv(torch.cat(outs, dim=1))
        
        return fused
