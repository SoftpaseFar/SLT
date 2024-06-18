def cal_emo_accuracy(emo_pres, emo_refs):
    # 初始化计数器
    total_samples = len(emo_pres)
    correct_predictions = 0

    # 遍历每个样本，比较预测结果和参考结果
    for pred, ref in zip(emo_pres, emo_refs):
        if pred == ref:
            correct_predictions += 1

    # 计算准确率
    emo_accuracy = correct_predictions / total_samples

    return emo_accuracy
