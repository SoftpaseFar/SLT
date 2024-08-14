import torch
import torch.nn as nn
from transformers import MBartForConditionalGeneration, MBartTokenizerFast
import torch.nn.functional as F
from transformers.models.mbart.modeling_mbart import shift_tokens_right
import os


class MBart(nn.Module):
    def __init__(self):
        super(MBart, self).__init__()
        self.mbart = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-cc25")

        self.encoder = self.mbart.get_encoder()
        self.decoder = self.mbart.get_decoder()

        self.lm_head = self.mbart.lm_head

        self.tokenizer = MBartTokenizerFast.from_pretrained("facebook/mbart-large-cc25")

        self.config = self.decoder.config

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
        encoder_outputs = self.encoder(input_ids=input_ids,
                                       attention_mask=attention_mask,
                                       return_dict=True)
        decoder_input_ids = shift_tokens_right(decoder_input_ids, self.config.pad_token_id)
        print('shift_tokens_right: ', decoder_input_ids)
        # print('self.config.pad_token_id: ', self.config.pad_token_id)

        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,

            encoder_hidden_states=encoder_outputs.last_hidden_state,
            encoder_attention_mask=attention_mask,
            return_dict=True,
        )

        # 获取 logits
        logits = self.lm_head(decoder_outputs.last_hidden_state)
        # 应用 Softmax 获取概率分布
        probabilities = F.softmax(logits, dim=-1)

        # 获取最大概率对应的 token IDs
        predicted_ids = torch.argmax(probabilities, dim=-1)

        print('forward: ', predicted_ids)


def dir_is_exist(base_path, subdir):
    dir_path = os.path.join(base_path, subdir)
    return os.path.exists(dir_path) and os.path.isdir(dir_path)


if __name__ == '__main__':
    # mbart = MBart()
    # # res = mbart.generate('I love you.', src_lang="en_XX", tgt_lang="es_XX")  # 设置目标语言为中文
    # mbart('I love you.', src_lang="en_XX", tgt_lang="en_XX")  # 设置目标语言为中文
    # mbart('I love you.', src_lang="zh_CN", tgt_lang="en_XX")  # 设置目标语言为中文
    # mbart('我爱你。', src_lang="zh_CN", tgt_lang="zh_CN")  # 设置目标语言为中文
    # mbart('我爱你。', src_lang="en_XX", tgt_lang="zh_CN")  # 设置目标语言为中文
    # # print(f"Translated text: {res}")
    # loss_lambda = torch.tensor('0.1')
    # print(loss_lambda)
    input_path = '/Volumes/OneTouch/P14T/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/fullFrame-210x260px/test'
    sub_dirs = [d for d in os.listdir(input_path) if os.path.isdir(os.path.join(input_path, d))]
    print('sub_dirs: ', sub_dirs)
    # base_path = '/Volumes/OneTouch/P14T/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/fullFrame-210x260px/test/'
    base_path = ''

    if base_path:
        input_paths = [os.path.join(input_path, subdir) for subdir in sub_dirs if
                       not dir_is_exist(base_path, subdir)]
        print('1_input_paths: ', input_paths)
    else:
        input_paths = [os.path.join(input_path, subdir) for subdir in sub_dirs]
        print('2_input_paths: ', input_paths)
