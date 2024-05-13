import torch
import torch.nn as nn
import numpy as np
from transformers import MBartForConditionalGeneration
from torchvision.models.video import s3d, S3D_Weights
from transformers.models.mbart.modeling_mbart import shift_tokens_right


# CLIP文本编码器
class TextCLIP(nn.Module):
    def __init__(self, config=None):
        super(TextCLIP, self).__init__()
        # 获取文本编码器
        self.txt_encoder = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-cc25").get_encoder()

        # 设置输出头维度
        self.lm_head = nn.Identity()

    def forward(self, tgt_input):
        # 隐藏层输出
        logits = self.txt_encoder(input_ids=tgt_input['input_ids'].cuda(),
                                  attention_mask=tgt_input['attention_mask'].cuda())[0]
        # 获取句子编码
        emo_voca_emb = logits[torch.arange(logits.shape[0]), tgt_input['input_ids'].argmax(dim=-1)]
        # emotion = logits[torch.arange(logits.shape[0]), tgt_input['input_ids'].argmin(dim=-1)]
        return self.lm_head(emo_voca_emb), logits


# 原始视频帧特征提取 Frames embedding -> Frames Features
class FramesFeatures(nn.Module):
    def __init__(self):
        super(FramesFeatures, self).__init__()
        self.conv1 = nn.Conv3d(in_channels=3, out_channels=64, kernel_size=(3, 3, 3), stride=(1, 1, 1),
                               padding=(1, 1, 1))
        self.conv2 = nn.Conv3d(in_channels=64, out_channels=128, kernel_size=(3, 3, 3), stride=(1, 1, 1),
                               padding=(1, 1, 1))
        self.conv3 = nn.Conv3d(in_channels=128, out_channels=256, kernel_size=(3, 3, 3), stride=(1, 1, 1),
                               padding=(1, 1, 1))
        self.conv4 = nn.Conv3d(in_channels=256, out_channels=512, kernel_size=(3, 3, 3), stride=(1, 1, 1),
                               padding=(1, 1, 1))
        self.pool = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.relu = nn.ReLU()

    def forward(self, input_ids):
        src = self.relu(self.conv1(input_ids))
        src = self.pool(src)
        src = self.relu(self.conv2(src))
        src = self.pool(src)
        src = self.relu(self.conv3(src))
        src = self.pool(src)
        src = self.relu(self.conv4(src))
        src = self.pool(src)
        # 将特征的维度进行调整，以适应后续的全连接层
        logits = src.view(src.size(0), src.size(2), -1)
        return logits


# 时间特征提取；
class TemporalFeatures(nn.Module):
    def __init__(self, input_size=512, hidden_size=1024, num_layers=1, batch_first=True):
        super(TemporalFeatures, self).__init__()
        self.gru = nn.GRU(input_size=input_size,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          batch_first=batch_first)

    def forward(self, input_ids):
        ids = input_ids.cuda()
        h0 = torch.zeros(self.gru.num_layers, ids.size(0), self.gru.hidden_size).to(ids.device)
        logits, _ = self.gru(ids, h0)

        return logits


# CLIP图像编码器
class ImageCLIP(nn.Module):
    def __init__(self, planes=1024, frozen=False):
        super(ImageCLIP, self).__init__()
        # 原始视频帧提取
        self.frames_emb = FramesFeatures()
        self.frames_tem = TemporalFeatures(input_size=512)

        # 关键点信息提取 keypoints本身具备空间信息，只需要时间建模
        self.keypoints_tem = TemporalFeatures(input_size=411)

    def forward(self, src_input):
        imgs_ids = src_input['imgs_ids'].cuda()
        keypoints_ids = src_input['keypoints_ids'].cuda()
        # 原始视频特这个提取
        src = self.frames_emb(imgs_ids)
        imgs_logits = self.frames_tem(src)
        print('imgs_logits:', imgs_logits.shape)

        # 关键点信息提取
        keypoints_logits = self.keypoints_tem(keypoints_ids)
        print('keypoints_logits:', keypoints_logits.shape)
        logits = (imgs_logits + keypoints_logits) / 2
        head, logits = None, None
        return head, logits


# VLP阶段文本解码器
class TextDecoder(nn.Module):
    def __init__(self, config):
        super(TextDecoder, self).__init__()
        self.MBart = MBartForConditionalGeneration.from_pretrained(
            "facebook/mbart-large-cc25")
        self.txt_decoder = self.MBart.get_decoder()
        self.lm_head = self.MBart.get_output_embeddings()
        self.register_buffer("final_logits_bias", torch.zeros((1, self.MBart.model.shared.num_embeddings)))
        # 情感层输出
        self.emo_predict = nn.Linear(250027, 60)

    # CLIP阶段正向反馈
    def forward_clip(self, tgt_input, masked_tgt_input, txt_encoder):
        with torch.no_grad():
            _, encoder_hidden_states = txt_encoder(masked_tgt_input)

        decoder_input_ids = shift_tokens_right(tgt_input['input_ids'], self.txt_decoder.config.pad_token_id)
        decoder_out = self.txt_decoder(
            input_ids=decoder_input_ids.cuda(),
            attention_mask=tgt_input['attention_mask'].cuda(),

            encoder_hidden_states=encoder_hidden_states[:, 1:-2, :].cuda(),
            encoder_attention_mask=masked_tgt_input['attention_mask'][:, 1:-2].cuda(),

            return_dict=True,
        )
        vocab_logits_tmp = self.lm_head(decoder_out[0]) + self.final_logits_bias
        vocab_logits = vocab_logits_tmp[:, 1:, :]
        emo_logits = self.emo_predict(vocab_logits_tmp[:, 0, :])
        return vocab_logits, emo_logits

    # SLT阶段正向反馈
    def forward_slt(self, tgt_input, encoder_hidden_states):
        decoder_input_ids = shift_tokens_right(tgt_input['input_ids'],
                                               self.txt_decoder.config.pad_token_id)
        decoder_out = self.txt_decoder(
            input_ids=decoder_input_ids.cuda(),
            attention_mask=tgt_input['attention_mask'].cuda(),

            encoder_hidden_states=encoder_hidden_states,
            # encoder_attention_mask=encoder_attention_mask,

            return_dict=True,
        )

        vocab_logits_tmp = self.lm_head(decoder_out[0]) + self.final_logits_bias
        vocab_logits = vocab_logits_tmp[:, 1:, :]
        emo_logits = self.emo_predict(vocab_logits_tmp[:, 0, :])
        return vocab_logits, emo_logits

    def forward(self, phase=None, tgt_input=None,
                masked_tgt_input=None, txt_encoder=None,
                encoder_hidden_states=None):
        if phase == 'clip':
            return self.forward_clip(tgt_input, masked_tgt_input, txt_encoder)
        elif phase == 'slt':
            return self.forward_slt(tgt_input, encoder_hidden_states)
        else:
            raise ValueError("参数错误")

    # generate方法待修订
    def generate(self, tokenizer, encoder_hidden_states, max_length=200):
        # 初始化解码器的输入，通常是一个特殊的起始 token
        input_ids = torch.tensor([[tokenizer.bos_token_id]])
        # 生成输出序列
        for _ in range(max_length):
            decoder_out = self.txt_decoder(
                input_ids=input_ids,

                encoder_hidden_states=encoder_hidden_states,
                # encoder_attention_mask=encoder_attention_mask,

                return_dict=True,
            )

            logits = self.lm_head(decoder_out[0])
            next_token_id = torch.argmax(logits[:, -1, :], dim=-1)
            input_ids = torch.cat([input_ids, next_token_id.unsqueeze(0)], dim=-1)
            if next_token_id == tokenizer.eos_token_id:
                break
        generated_sequence = input_ids
        sequence = tokenizer.batch_decode(generated_sequence[:, 1:],
                                          skip_special_tokens=True)
        return sequence


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
        # head, logits = self.kps_encoder(src_input)
        # print('head：', head.shape)
        # print('logits:', logits.shape)

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
        self.txt_decoder = TextDecoder(config=config)

    def forward(self, src_input, tgt_input):
        # print(src_input['input_ids'][0].shape)
        # print(tgt_input['input_ids'].shape)
        # 视频编码
        _, encoder_hidden_states = self.img_encoder(src_input)
        # 文本解码
        vocab_logits, emo_logits = self.txt_decoder(phase='slt', tgt_input=tgt_input,
                                                    encoder_hidden_states=encoder_hidden_states)
        return vocab_logits, emo_logits

    # generate方法待修订
    def generate(self, src_input, tokenizer):
        _, encoder_hidden_states = self.img_encoder(src_input)
        sequence = self.txt_decoder.generate(tokenizer, encoder_hidden_states=encoder_hidden_states,
                                             max_length=200)
        return sequence
