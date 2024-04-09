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
    def __init__(self, config=None, in_planes=1024, planes=1024, head_type='identy'):
        super(TextCLIP, self).__init__()
        # 获取文本编码器
        self.model_txt = MBartForConditionalGeneration.from_pretrained(config['model']['MBart_ver1']).get_encoder()
        # 设置输出头维度
        self.lm_head = make_head(in_planes, planes, head_type)

    def forward(self, tgt_input):
        txt_logits = self.model_txt(input_ids=tgt_input['input_ids'].cuda(),
                                    attention_mask=tgt_input['attention_mask'].cuda())[0]
        output = txt_logits[torch.arange(txt_logits.shape[0]), tgt_input['input_ids'].argmax(dim=-1)]
        return self.lm_head(output), txt_logits


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
    def __init__(self, config, embed_dim=1024, *args, **kwargs):
        super(CLIP, self).__init__()
        self.model_txt = TextCLIP(config, inplanes=embed_dim, planes=embed_dim)
        self.model_images = ImageCLIP(config, inplanes=embed_dim, planes=embed_dim)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))


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
