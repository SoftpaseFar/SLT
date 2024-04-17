import torch
import torch.nn as nn
import numpy as np
from transformers import MBartForConditionalGeneration, MBartPreTrainedModel, MBartModel, MBartConfig
from torchvision.models.video import s3d, S3D_Weights
from transformers.models.mbart.modeling_mbart import shift_tokens_right
import torchvision
from definition import *
from torch.nn.utils.rnn import pad_sequence
from torch import Tensor
from transformers import AutoConfig
from pathlib import Path


# CLIP文本编码器
class TextCLIP(nn.Module):
    def __init__(self, config=None):
        super(TextCLIP, self).__init__()
        # 获取文本编码器
        self.txt_encoder = MBartForConditionalGeneration.from_pretrained(config['model']['MBart_ver1']).get_encoder()

        # 设置输出头维度
        self.lm_head = nn.Identity()

    def forward(self, tgt_input):
        # 隐藏层输出
        logits = self.txt_encoder(input_ids=tgt_input['input_ids'],
                                  attention_mask=tgt_input['attention_mask'])[0]
        # 获取句子编码
        sentence = logits[torch.arange(logits.shape[0]), tgt_input['input_ids'].argmax(dim=-1)]
        # emotion = logits[torch.arange(logits.shape[0]), tgt_input['input_ids'].argmin(dim=-1)]
        return self.lm_head(sentence), logits


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
        N = len(src_input['input_ids'])
        T = src_length_batch
        C, H, W = input_ids[0][0].size()

        src = torch.zeros(N, C, T, H, W)
        # print(src.shape)
        # print(src_input['attention_mask'])

        # 填充数据到s3d_input
        for i, video in enumerate(input_ids):
            for j, frame in enumerate(video):
                src[i, :, j, :, :] = frame

        # 获取每个视频的表示，，模型推导
        src = self.S3D.features(src)
        src = self.S3D.avgpool(src)

        src = self.S3D.classifier(src)
        logits = torch.mean(src, dim=(3, 4))
        logits = logits.permute(0, 2, 1)
        src = torch.mean(src, dim=(2, 3, 4))

        # print(src_length_batch)
        # print(logits.shape)
        # logits = None
        # src = self.S3D(src)
        head = self.fc(src)
        # print(head.shape)

        return head, logits


# VLP阶段文本解码器
class TextDecoder(nn.Module):
    def __init__(self, config):
        super(TextDecoder, self).__init__()
        self.txt_decoder = MBartForConditionalGeneration.from_pretrained(
            config['model']['MBart_ver2']).get_decoder()
        self.lm_head = MBartForConditionalGeneration.from_pretrained(
            config['model']['MBart_ver2']).get_output_embeddings()
        self.register_buffer("final_logits_bias", torch.zeros((1, MBartForConditionalGeneration.from_pretrained(
            config['model']['MBart_ver2']).model.shared.num_embeddings)))

    def forward(self, tgt_input, masked_tgt_input, txt_encoder):
        with torch.no_grad():
            _, encoder_hidden_states = txt_encoder(masked_tgt_input)

        decoder_input_ids = shift_tokens_right(tgt_input['input_ids'], self.txt_decoder.config.pad_token_id)
        decoder_out = self.txt_decoder(
            input_ids=decoder_input_ids,
            attention_mask=tgt_input['attention_mask'],

            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=masked_tgt_input['attention_mask'],

            return_dict=True,
        )
        logits = self.lm_head(decoder_out[0]) + self.final_logits_bias

        return logits


# SLT阶段文本解码器
class SLTTextDecoder(nn.Module):
    def __init__(self, config):
        super(SLTTextDecoder, self).__init__()
        self.txt_decoder = MBartForConditionalGeneration.from_pretrained(
            config['model']['MBart_ver2']).get_decoder()
        self.lm_head = MBartForConditionalGeneration.from_pretrained(
            config['model']['MBart_ver2']).get_output_embeddings()
        self.register_buffer("final_logits_bias", torch.zeros((1, MBartForConditionalGeneration.from_pretrained(
            config['model']['MBart_ver2']).model.shared.num_embeddings)))
        # 情感层输出
        self.emo_predict = nn.Linear(2454, 60)

    def forward(self, tgt_input, encoder_hidden_states):
        decoder_input_ids = shift_tokens_right(tgt_input['input_ids'], self.txt_decoder.config.pad_token_id)
        decoder_out = self.txt_decoder(
            input_ids=decoder_input_ids,
            attention_mask=tgt_input['attention_mask'],

            encoder_hidden_states=encoder_hidden_states,
            # encoder_attention_mask=encoder_attention_mask,

            return_dict=True,
        )
        vocab_logits = self.lm_head(decoder_out[0]) + self.final_logits_bias
        emo_logits = self.emo_predict(vocab_logits[:, 0, :])
        return vocab_logits, emo_logits


# CLIP模型
class CLIP(nn.Module):
    def __init__(self, config):
        super(CLIP, self).__init__()
        self.txt_encoder = TextCLIP(config=config)
        self.img_encoder = ImageCLIP(planes=1024, frozen=False)
        # logit缩放比率，可学习参数
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        # 文本编码器的隐藏状态
        self.encoder_hidden_states = None

    # 获取文本编码器
    def get_txt_encoder(self):
        return self.txt_encoder

    # encoder_hidden_states只读
    @property
    def get_encoder_hidden_states(self):
        return self.encoder_hidden_states

    def forward(self, src_input, tgt_input):
        img_features, _ = self.img_encoder(src_input)
        txt_features, self.encoder_hidden_states = self.txt_encoder(tgt_input)

        # 特征信息归一化
        img_features = img_features / img_features.norm(dim=-1, keepdim=True)
        txt_features = txt_features / txt_features.norm(dim=-1, keepdim=True)

        # 计算相似度
        # 将logit_scale修正为正数
        logit_scale = self.logit_scale.exp()
        img_txt_s_matrix = logit_scale * img_features @ txt_features.t()
        txt_img_s_matrix = logit_scale * txt_features @ img_features.t()

        ground_truth = torch.eye(img_txt_s_matrix.shape[0],
                                 device=txt_img_s_matrix.device,
                                 dtype=img_txt_s_matrix.dtype,
                                 requires_grad=False)

        return img_txt_s_matrix, txt_img_s_matrix, ground_truth


# SLT模型
class SLT(nn.Module):
    def __init__(self, config):
        super(SLT, self).__init__()
        # 视频编码器
        self.img_encoder = ImageCLIP(planes=1024, frozen=False)
        # 文本解码器
        self.txt_decoder = SLTTextDecoder(config=config)

    def forward(self, src_input, tgt_input):
        # print(src_input['input_ids'][0].shape)
        # print(tgt_input['input_ids'].shape)
        # 视频编码
        _, encoder_hidden_states = self.img_encoder(src_input)
        # 文本解码
        vocab_logits, emo_logits = self.txt_decoder(tgt_input, encoder_hidden_states)
        return vocab_logits, emo_logits
