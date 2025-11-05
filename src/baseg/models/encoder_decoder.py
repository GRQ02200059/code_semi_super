from mmseg.models.segmentors.encoder_decoder import EncoderDecoder
from mmseg.registry import MODELS
from mmseg.utils import OptSampleList
from torch import Tensor
from torch.nn import functional as F
import torch.nn as nn
from baseg.models.utils import RSCSAM


@MODELS.register_module()
class CustomEncoderDecoder(EncoderDecoder):
    """自定义的编码器-解码器分割模型。
    
    继承自MMSegmentation的EncoderDecoder类，
    实现了自定义的前向传播逻辑。
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 假设主干输出4个stage特征，channels已知
        # 这里用通道数自动推断
        if hasattr(self.backbone, 'out_channels'):
            channels = self.backbone.out_channels
        elif hasattr(self.decode_head, 'in_channels'):
            channels = self.decode_head.in_channels
        else:
            channels = [96, 192, 384, 768]  # Swin默认
        self.rscsam_list = nn.ModuleList([RSCSAM(c) for c in channels])
    
    def _forward(self, inputs: Tensor, data_samples: OptSampleList = None) -> Tensor:
        """模型的前向传播过程。

        Args:
            inputs (Tensor): 输入张量，形状为 (N, C, H, W)
                N: 批次大小
                C: 输入通道数
                H: 高度
                W: 宽度
            data_samples (List[:obj:`SegDataSample`]): 分割数据样本
                包含元信息和语义分割真值标签等信息

        Returns:
            Tensor: 如果有辅助输出，返回(output, aux)元组；
                   否则只返回主要输出。
        """
        # 使用backbone提取多尺度特征
        x = self.extract_feat(inputs)
        
        # 对每个stage特征加RSCSAM注意力
        att_x = []
        for i, f in enumerate(x):
            f = self.rscsam_list[i](f)
            att_x.append(f)
        
        # 使用解码头处理特征
        feat = self.decode_head(att_x)
        
        # 进行最终的分类预测
        out = self.decode_head.cls_seg(feat)
        
        # 将输出上采样到原始输入图像的尺寸
        out = F.interpolate(
            out,
            size=inputs.shape[2:],     # 目标尺寸为输入图像的高和宽
            mode="bilinear",           # 使用双线性插值
            align_corners=True         # 对齐角点以保持一致性
        )

        # 如果解码头有辅助输出（用于深度监督）
        if self.decode_head.has_aux_output():
            # 获取辅助预测结果
            aux = self.decode_head.cls_seg_aux(feat)
            # 将辅助输出也上采样到原始尺寸
            aux = F.interpolate(
                aux,
                size=inputs.shape[2:],
                mode="bilinear",
                align_corners=True
            )
            # 返回主输出和辅助输出的元组
            return out, aux

        # 如果没有辅助输出，只返回主输出
        return out
