import torch
import torch.nn as nn
from transformers import MBartTokenizerFast
import torch.nn.functional as F


class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=128, nhead=8),
            num_layers=3
        )
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=128, nhead=8),
            num_layers=3
        )

        self.tokenizer = MBartTokenizerFast.from_pretrained("facebook/mbart-large-cc25")

    def forward(self, text, src_lang='', tgt_lang=''):
        self.tokenizer.src_lang = src_lang
        self.tokenizer.tgt_lang = tgt_lang
        print('tokenizer: ')
        encoded_input = self.tokenizer(text, return_tensors="pt",
                                       padding=True, truncation=True)

        print('encoded_input: ', encoded_input)
        decoded_input = self.tokenizer('I am a good student.', return_tensors="pt",
                                       padding=True, truncation=True)

        print('decoded_input ', decoded_input)

        # 编码输入文本
        input_ids = encoded_input.input_ids
        attention_mask = encoded_input.attention_mask

        decoder_input_ids = decoded_input.input_ids
        decoder_attention_mask = decoded_input.attention_mask

        # 使用encoder生成encoder隐藏状态
        encoder_outputs = self.encoder(src=input_ids)
        # decoder_input_ids = shift_tokens_right(decoder_input_ids, self.config.pad_token_id)
        print('shift_tokens_right: ', decoder_input_ids)
        # print('self.config.pad_token_id: ', self.config.pad_token_id)

        decoder_outputs = self.decoder(
            tgt=decoder_input_ids,

            memory=encoder_outputs.last_hidden_state
        )

        # 获取 logits
        logits = self.lm_head(decoder_outputs.last_hidden_state)
        # 应用 Softmax 获取概率分布
        probabilities = F.softmax(logits, dim=-1)

        # 获取最大概率对应的 token IDs
        predicted_ids = torch.argmax(probabilities, dim=-1)

        print('forward: ', predicted_ids)


if __name__ == '__main__':
    tfm = Transformer()
    # res = mbart.generate('I love you.', src_lang="en_XX", tgt_lang="es_XX")  # 设置目标语言为中文
    tfm('I love you.', src_lang="en_XX", tgt_lang="en_XX")  # 设置目标语言为中文
    tfm('I love you.', src_lang="zh_CN", tgt_lang="en_XX")  # 设置目标语言为中文
    tfm('我爱你。', src_lang="zh_CN", tgt_lang="zh_CN")  # 设置目标语言为中文
    tfm('我爱你。', src_lang="en_XX", tgt_lang="zh_CN")  # 设置目标语言为中文
    # print(f"Translated text: {res}")
