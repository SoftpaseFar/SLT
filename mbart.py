from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

# 其它语言 ar_AR,cs_CZ,de_DE,en_XX,es_XX,et_EE,fi_FI,
# fr_XX,gu_IN,hi_IN,it_IT,ja_XX,kk_KZ,ko_KR,lt_LT,
# lv_LV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si_LK,tr_TR,vi_VN,zh_CN

article_cn = "生活就像一块巧克力"
article_en = "Life is like a box of chocolate."

model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")


def cn2en():
    tokenizer.src_lang = "zh_CN"
    encoded_hi = tokenizer(article_cn, return_tensors="pt")
    generated_tokens = model.generate(**encoded_hi, forced_bos_token_id=tokenizer.lang_code_to_id["es_XX"])
    result = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    print(f"{article_cn}的翻译结果: {result[0]}")


def en2cn():
    tokenizer.src_lang = "en_XX"
    encoded_hi = tokenizer(article_cn, return_tensors="pt")
    generated_tokens = model.generate(**encoded_hi, forced_bos_token_id=tokenizer.lang_code_to_id["es_XX"])
    result = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    print(f"{article_en}的翻译结果: {result[0]}")


if __name__ == '__main__':
    cn2en()
    en2cn()




# --------



    def generate(self, text, src_lang='', tgt_lang=''):
        # 设置模型和tokenizer的语言
        self.tokenizer.src_lang = src_lang

        # 编码输入文本
        encoded_input = self.tokenizer(text, return_tensors="pt",
                                       padding=True, truncation=True)
        input_ids = encoded_input.input_ids
        attention_mask = encoded_input.attention_mask

        # 使用encoder生成encoder隐藏状态
        encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)

        # 初始化decoder输入（以目标语言的特殊开始标记开始）
        decoder_start_token_id = self.tokenizer.lang_code_to_id[tgt_lang]
        decoder_input_ids = torch.tensor([[decoder_start_token_id]])
        decoder_attention_mask = torch.ones(decoder_input_ids.shape, dtype=torch.long)

        print(f"Initial decoder input ids: {decoder_input_ids}")

        # 使用decoder逐步生成翻译
        translated_tokens = []
        for _ in range(100):  # 假设最大翻译长度为100
            decoder_outputs = self.decoder(
                input_ids=decoder_input_ids,
                attention_mask=decoder_attention_mask,
                encoder_hidden_states=encoder_outputs.last_hidden_state,
                encoder_attention_mask=attention_mask,
                return_dict=True,
            )

            # 获取 logits
            logits = self.lm_head(decoder_outputs.last_hidden_state)
            next_token_id = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(0)

            # 变成整数
            next_token_id_item = next_token_id.item()
            print(
                f"Next token id: {next_token_id_item}, token: {self.tokenizer.decode([next_token_id_item], skip_special_tokens=True)}")

            decoder_input_ids = torch.cat([decoder_input_ids, next_token_id], dim=-1)
            decoder_attention_mask = torch.cat([decoder_attention_mask, torch.ones((1, 1), dtype=torch.long)], dim=-1)
            if next_token_id_item == self.tokenizer.eos_token_id:
                break

            translated_tokens.append(next_token_id_item)

        # 解码翻译后的token
        translated_text = self.tokenizer.decode(translated_tokens, skip_special_tokens=True)
        return translated_text
