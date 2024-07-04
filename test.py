if __name__ == '__main__':
    import os
    import torch

    # 设置环境变量以同步CUDA错误
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    # 示例 logits 张量和 tgt_input 字典
    # 形状为 [batch_size, seq_length, feature_dim]
    logits = torch.randn(2, 27, 128)
    tgt_input = {'input_ids': torch.tensor(
        [[0, -1, 30086, 328, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
         [0, -1, 133, 10, 10329, 261, 16, 5, 797, 4084, 11, 5, 5897, 14, 16, 4875, 30, 30972, 2079, 235, 8, 314, 9, 5,
          4757, 4, 2]])}

    # 打印输入张量
    print('句子编码处 tgt_input["input_ids"]', tgt_input['input_ids'])

    # 计算每个样本中最小值的索引
    min_indices = tgt_input['input_ids'].argmin(dim=-1)
    print('句子编码处 tgt_input["input_ids"].argmin(dim=-1)', min_indices)

    # 打印logits的形状
    print('logits.shape[0]: ', logits.shape[0])
    print('logits 形状: ', logits.shape)

    # 打印批次索引
    batch_indices = torch.arange(logits.shape[0])
    print('torch.arange(logits.shape[0]): ', batch_indices)

    # 检查索引是否在范围内
    if not torch.all(min_indices < logits.shape[1]):
        print("Error: Some indices are out of range")
        print("min_indices: ", min_indices)
        print("logits.shape[1]: ", logits.shape[1])
    else:
        # 选择每个样本中最小值位置的 logits 特征
        emo_voca_emb = logits[batch_indices, min_indices, :]

        # 打印结果以进行验证
        print("Selected logits: ", emo_voca_emb)
