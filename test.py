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

    # 选择每个样本中最小值位置的 logits 特征
    emo_voca_emb = logits[torch.arange(logits.shape[0]), tgt_input['input_ids'].argmin(dim=-1)]

    # 打印结果以进行验证
    print("Selected logits: ", emo_voca_emb)
