import torch
import torch.nn as nn
import numpy as np
from transformers import MBartForConditionalGeneration, MBartPreTrainedModel, MBartModel, MBartConfig
from torchvision.models.video import s3d, S3D_Weights
from torchvision.models import ResNet18_Weights
import torchvision
from definition import *
from torch.nn.utils.rnn import pad_sequence
from torch import Tensor


# CLIP文本编码器
class TextCLIP(nn.Module):
    def __init__(self, config=None, in_planes=1024):
        super(TextCLIP, self).__init__()
        # 获取文本编码器
        self.txt_encoder = MBartForConditionalGeneration.from_pretrained(config['model']['MBart_ver1']).get_encoder()

        # 设置输出头维度
        self.lm_head = nn.Identity()

    def forward(self, tgt_input):
        # 隐藏层输出
        txt_logits = self.txt_encoder(input_ids=tgt_input['input_ids'],
                                      attention_mask=tgt_input['attention_mask'])[0]
        # 获取句子编码
        output = txt_logits[:, tgt_input['input_ids'].argmax(dim=-1)]
        return self.lm_head(output), txt_logits


# CLIP图像编码器
class ImageCLIP(nn.Module):
    def __init__(self, planes=1024, frozen=False):
        super(ImageCLIP, self).__init__()
        # 获取预训练的S3D,提取视频特征和时间信息
        self.S3D = s3d(weights=S3D_Weights.KINETICS400_V1)

        # 移除S3D的分类器部分，只保留到avg pool的部分
        self.S3D.classifier = nn.Identity()
        # 是否冻结S3D参数
        if frozen:
            for param in self.S3D.parameters():
                param.requires_grad = False

        # 对接线性层，充当head
        self.fc = nn.Linear(1024, planes)

    def forward(self, src_input):
        input_ids, src_length_batch = src_input['input_ids'], src_input['src_length_batch']

        # 构建S3D的输入格式
        N = len(src_input)
        T = src_length_batch
        C, H, W = input_ids[0][0].size()

        src = torch.zeros(N, C, T, H, W)

        # 填充数据到s3d_input
        for i, video in enumerate(input_ids):
            for j, frame in enumerate(video):
                src[i, :, j, :, :] = frame

        # 获取每个视频的表示，，模型推导
        src = self.S3D(src)
        output = self.fc(src)

        return output


# 文本解码器
class TextDecoder(nn.Module):
    def __init__(self, config):
        super(TextDecoder, self).__init__()
        self.txt_decoder = MBartForConditionalGeneration.from_pretrained(
            config['model']['MBart_ver2']).get_decoder()
        self.lm_head = MBartForConditionalGeneration.from_pretrained(
            config['model']['MBart_ver2']).get_output_embeddings()


    def forward(self):
        return None


# CLIP模型
class CLIP(nn.Module):
    def __init__(self):
        super(CLIP, self).__init__()
