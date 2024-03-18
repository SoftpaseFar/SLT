# Trasformer模型参数
class ConfTF:
    # 模型超参数
    input_dim = 512  # 输入维度
    output_dim = 512  # 输出维度
    hidden_dim = 1024  # 隐藏层维度
    num_layers = 6  # 层数
    num_heads = 8  # 注意力头数
    dropout = 0.1  # Dropout比率
    learning_rate = 1e-4  # 学习率
    batch_size = 64  # 批量大小
    num_epochs = 10  # 训练轮数

# 其他参数
