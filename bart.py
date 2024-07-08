import torch
import torch.nn as nn
from transformers import BartModel, BartTokenizer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


class BartCustom(nn.Module):
    def __init__(self, device='cpu', freeze_model=True):
        super(BartCustom, self).__init__()
        self.device = device
        bart_name = "facebook/bart-large"
        self.bart_encoder = BartModel.from_pretrained(bart_name).encoder.to(self.device)
        self.bart_decoder = BartModel.from_pretrained(bart_name).decoder.to(self.device)
        self.tokenizer = BartTokenizer.from_pretrained(bart_name)
        self.smoothing_function = SmoothingFunction().method4

        # 冻结模型参数
        if freeze_model:
            for param in self.bart_encoder.parameters():
                param.requires_grad = False
            for param in self.bart_decoder.parameters():
                param.requires_grad = False

    def forward(self, input_text, target_text):
        # 编码器输入
        input_ids = self.tokenizer(input_text, return_tensors="pt")['input_ids'].to(self.device)
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id).to(input_ids.device)

        # 编码器输出
        encoder_outputs = self.bart_encoder(input_ids=input_ids, attention_mask=attention_mask)

        # 解码器输入
        decoder_input_ids = torch.full_like(input_ids, fill_value=self.tokenizer.pad_token_id).to(self.device)
        decoder_input_ids[:, 0] = self.tokenizer.pad_token_id  # 初始解码器输入
        print('input_ids: ', input_ids)
        print('decoder_input_ids: ', decoder_input_ids)

        # 存储生成的token id
        generated_ids = []

        # 解码循环
        for step in range(1, input_ids.shape[1]):
            decoder_outputs = self.bart_decoder(input_ids=decoder_input_ids,
                                                encoder_hidden_states=encoder_outputs.last_hidden_state,
                                                encoder_attention_mask=attention_mask)

            # 获取最后一个生成的token id
            next_token_logits = decoder_outputs.last_hidden_state[:, -1, :]
            next_token_id = torch.argmax(next_token_logits, dim=-1)

            # 添加到生成的序列中
            generated_ids.append(next_token_id.item())

            # 更新解码器输入以准备下一个步骤
            decoder_input_ids[:, step] = next_token_id

        # 将生成的token ids转换为文本
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        print("Generated Text: ", generated_text)
        print("Target Text: ", target_text)

        # 计算BLEU分数
        reference = [target_text.split()]
        hypothesis = generated_text.split()

        for i in range(1, 5):
            bleu_score = sentence_bleu(reference, hypothesis, weights=(1 / i,) * i,
                                       smoothing_function=self.smoothing_function)
            print(f"BLEU-{i} Score:", bleu_score)


if __name__ == '__main__':
    input_text = "A quick brown fox jumps over a lazy dog."
    target_text = "A quick brown fox jumps over a lazy dog."

    bart = BartCustom(device='cpu', freeze_model=True)
    bart(input_text, target_text)
