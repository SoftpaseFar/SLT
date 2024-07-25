import torch
import torch.nn as nn
import numpy as np
from transformers import MBartForConditionalGeneration
from transformers.models.mbart.modeling_mbart import shift_tokens_right


# 投影层
class ProjectionLayer(nn.Module):
    def __init__(self, input_dim=128, output_dim=1024):
        super(ProjectionLayer, self).__init__()
        self.projection = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        projected_x = self.projection(x)
        return projected_x


# 原始视频帧特征提取 Frames embedding -> Frames Features
class FramesFeatures(nn.Module):
    def __init__(self):
        super(FramesFeatures, self).__init__()
        self.conv1 = nn.Conv3d(in_channels=3, out_channels=64, kernel_size=(3, 3, 3), stride=(1, 1, 1),
                               padding=(1, 1, 1))
        self.relu = nn.ReLU()

    def forward(self, input_ids):
        # 调整维度顺序为[batch_size, channels, depth, height, width]
        input_ids = input_ids.permute(0, 2, 1, 3, 4)
        # 检查输入数据的深度维度，如果小于3，则增加深度
        if input_ids.size(2) < 3:
            padding_size = 3 - input_ids.size(2)
            input_ids = torch.nn.functional.pad(input_ids, (0, 0, 0, 0, 0, padding_size))

        src = self.relu(self.conv1(input_ids))
        src = torch.mean(src, dim=[-2, -1])
        # 将维度调整为[batch_size, depth, channels]
        features = src.permute(0, 2, 1)
        return features


# 时间特征提取；
class TemporalFeatures(nn.Module):
    def __init__(self, input_size=64, hidden_size=128, num_layers=1, batch_first=True):
        super(TemporalFeatures, self).__init__()
        self.gru = nn.GRU(input_size=input_size,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          batch_first=batch_first)

    def forward(self, input_ids):
        h0 = torch.zeros(self.gru.num_layers, input_ids.size(0), self.gru.hidden_size).to(input_ids.device)
        hidden, _ = self.gru(input_ids, h0)

        return hidden


# 图像编码器
class ImageEncoder(nn.Module):
    def __init__(self, args: dict):
        super(ImageEncoder, self).__init__()
        # 原始视频帧提取
        self.frames_emb = FramesFeatures()
        self.frames_tem = TemporalFeatures(input_size=64)

        # 关键点信息提取 keypoints本身具备空间信息，只需要时间建模
        self.keypoints_tem = TemporalFeatures(input_size=54)

        self.args = args

    def forward(self, src_input):
        imgs_ids = src_input['imgs_ids'].cuda()
        keypoints_ids = None
        if self.args['need_keypoints']:
            keypoints_ids = src_input['keypoints_ids'].cuda()
        # 原始视频特这个提取
        imgs_features = self.frames_emb(imgs_ids)
        imgs_hidden = self.frames_tem(imgs_features)

        # hidden = None
        if self.args['need_keypoints']:
            # 关键点信息提取
            keypoints_hidden = self.keypoints_tem(keypoints_ids)
            hidden = (imgs_hidden + keypoints_hidden) / 2
        else:
            hidden = imgs_hidden
        head = hidden[:, -1, :]
        logits = hidden
        return head, logits


# 文本解码器[冻结]
class TextDecoder(nn.Module):
    def __init__(self, config):
        super(TextDecoder, self).__init__()
        self.MBart = MBartForConditionalGeneration.from_pretrained(
            "facebook/mbart-large-cc25")
        self.txt_decoder = self.MBart.get_decoder()

        # 冻结解码器
        for param in self.txt_decoder.parameters():
            param.requires_grad = False

        self.lm_head = self.MBart.get_output_embeddings()
        self.register_buffer("final_logits_bias", torch.zeros((1, self.MBart.model.shared.num_embeddings)))

        # 情感层输出
        self.emo_predict = nn.Linear(250027, 3)

        # 映射层
        self.projector_128_1024 = ProjectionLayer(input_dim=128, output_dim=1024)

    # CLIP阶段正向反馈
    def forward_clip(self, tgt_input, masked_tgt_input, txt_encoder):
        with torch.no_grad():
            _, encoder_hidden_states = txt_encoder(masked_tgt_input)
            # 维度映射
            encoder_hidden_states = self.projector_128_1024(encoder_hidden_states)

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
    def forward_slt(self, tgt_input, encoder_hidden_states, encoder_attention_mask):
        decoder_input_ids = shift_tokens_right(tgt_input['input_ids'],
                                               self.txt_decoder.config.pad_token_id)
        # 维度映射
        encoder_hidden_states = self.projector_128_1024(encoder_hidden_states)

        decoder_out = self.txt_decoder(
            input_ids=decoder_input_ids.cuda(),
            attention_mask=tgt_input['attention_mask'].cuda(),

            encoder_hidden_states=encoder_hidden_states.cuda(),
            encoder_attention_mask=encoder_attention_mask.cuda(),

            return_dict=True,
        )

        vocab_logits_tmp = self.lm_head(decoder_out.last_hidden_state)

        vocab_logits = vocab_logits_tmp[:, 1:, :]
        return vocab_logits

    def forward(self, phase=None, tgt_input=None,
                masked_tgt_input=None, txt_encoder=None,
                encoder_hidden_states=None, encoder_attention_mask=None):
        if phase == 'clip':
            return self.forward_clip(tgt_input, masked_tgt_input, txt_encoder)
        elif phase == 'slt':
            return self.forward_slt(tgt_input, encoder_hidden_states, encoder_attention_mask)
        else:
            raise ValueError("参数错误")


# SLT模型
class SLT(nn.Module):
    def __init__(self, config, args: dict):
        super(SLT, self).__init__()
        # 视频编码器
        self.img_encoder = ImageEncoder(args)
        # 文本解码器
        self.txt_decoder = TextDecoder(config=config)

    def forward(self, src_input, tgt_input):
        # 视频编码
        _, encoder_hidden_states = self.img_encoder(src_input)
        # 文本解码
        vocab_logits = self.txt_decoder(phase='slt', tgt_input=tgt_input,
                                        encoder_hidden_states=encoder_hidden_states,
                                        encoder_attention_mask=src_input['attention_mask'])
        return vocab_logits
