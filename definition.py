# 全局定义
UNK_TOKEN = "<unk>"
PAD_TOKEN = "<pad>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
SI_TOKEN = "<si>"
SI_IDX, PAD_IDX, UNK_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3, 4
SPECIAL_SYMBOLS = ['<si>', '<pad>', '<unk>', '<bos>', '<eos>']
WORD_MASK = "<mask>"

# 情绪词典
emotion_vocab = {
    # Positive
    'positive': 1,
    'positiv': 1,
    '积极': 1,

    # Negative
    'negative': 2,
    'negativ': 2,
    '消极': 2,

    # Complex
    'complex': 3,
    'komplex': 3,
    '复杂': 3,

    # Ambiguous
    'ambiguous': 4,
    'mehrdeutig': 4,
    '不明确': 4,
}
