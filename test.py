from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-german-cased')

tgt_batch = ["Das Wetter ist heute sehr schön.", "Ich liebe es, neue Sprachen zu lernen.",
             "Können Sie mir bitte den Weg zeigen?", "Dieses Buch ist sehr interessant.",
             "Wir werden morgen früh abfahren."]

print(tgt_batch)

print(tgt_batch[0].split())


# tgt_batch = ["今天的天气非常好。", "我喜欢学习新的语言。", "你能告诉我路怎么走吗？", "这本书非常有趣。",
#              "我们明天早上会出发。"]

# with tokenizer.as_target_tokenizer():
#     tgt_input = tokenizer(tgt_batch, return_tensors="pt", padding=True, truncation=True)
#
# print(tgt_input)
