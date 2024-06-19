import torch
import torch.nn as nn
import torch.nn.functional as F


class KLLoss(nn.Module):
    def __init__(self, error_metric=torch.nn.KLDivLoss(reduction='batchmean')):
        super(KLLoss, self).__init__()
        self.error_metric = error_metric

    def forward(self, prediction, label):
        batch_size = prediction.shape[0]
        probs1 = F.log_softmax(prediction, 1)
        probs2 = F.softmax(label * 10, 1)
        loss = self.error_metric(probs1, probs2) * batch_size
        return loss


# 返回clip_losses和tdm_losses的平均值
def compute_average(clip_losses, tdm_losses):
    avg_clip_loss = sum(clip_losses) / len(clip_losses) if clip_losses else 0
    avg_tdm_loss = sum(tdm_losses) / len(tdm_losses) if tdm_losses else 0
    return avg_clip_loss, avg_tdm_loss


# 情感交叉熵
def ce(batch_size, num_classes):
    # 假设这是情感分类的输出和真实标签
    emo_logits = torch.randn(batch_size, num_classes, requires_grad=True)
    labels_emotion = torch.randint(0, num_classes, (batch_size,))

    # 定义交叉熵损失函数
    criterion = nn.CrossEntropyLoss()

    # 计算损失
    loss_emotion = criterion(emo_logits, labels_emotion)
    return loss_emotion
