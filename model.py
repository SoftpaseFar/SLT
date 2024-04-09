import torch
import torch.nn as nn
import numpy as np
from transformers import MBartForConditionalGeneration, MBartPreTrainedModel, MBartModel, MBartConfig
from torchvision.models.video import s3d, S3D_Weights
from torchvision.models import ResNet18_Weights
import torchvision


# 设置不同的输出头维度
def make_head(in_planes, planes, head_type):
    if head_type == 'linear':
        return nn.Linear(in_planes, planes, bias=False)
    else:
        return nn.Identity()


# CLIP文本编码器
class TextCLIP(nn.Module):
    def __init__(self):
        super(TextCLIP, self).__init__()


# CLIP图像编码器
class ImageCLIP(nn.Module):
    def __init__(self, config, in_planes=1024, planes=1024, head_type='linear'):
        super(ImageCLIP, self).__init__()

    def forward(self):
        return None


# 文本解码器
class TextDecoder(nn.Module):
    def __init__(self, config):
        super(TextDecoder, self).__init__()

    def forward(self):
        return None


# CLIP模型
class CLIP(nn.Module):
    def __init__(self):
        super(CLIP, self).__init__()


# 图片特征提取
class FeatureExtra(nn.Module):
    def __init__(self, frozen=True):
        super(FeatureExtra, self).__init__()
        # 获取预训练的S3D
        # self.S3D = s3d(weights=S3D_Weights.KINETICS400_V1)
        self.S3D = torchvision.models.resnet18(pretrained=True)

        # 是否冻结S3D参数
        if frozen:
            for param in self.S3D.parameters():
                param.requires_grad = False

    def forward(self, src):
        src = self.S3D(src)
        return src
