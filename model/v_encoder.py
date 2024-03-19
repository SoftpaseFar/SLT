import torch
import torch.nn as nn
import torch.nn.functional as f
from torch import Tensor
import torchvision
from torch.nn.utils.rnn import pad_sequence
from definition import *


class VEncoder(nn.Module):
    def __init__(self, emb_size, feature_size, config, ):
        super(VEncoder, self).__init__()

        self.config = config

        self.src_emb = nn.Linear(feature_size, emb_size)
        modules = [nn.BatchNorm1d(emb_size), nn.ReLU(inplace=True)]
        self.bn_ac = nn.Sequential(*modules)

        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, src: Tensor, ):
        src = self.src_emb(src)
        src = self.bn_ac(src.permute(0, 2, 1)).permute(0, 2, 1)

        return src


class TemporalConv(nn.Module):
    def __init__(self, input_size, hidden_size, conv_type=2):
        super(TemporalConv, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.conv_type = conv_type

        if self.conv_type == 0:
            self.kernel_size = ['K3']
        elif self.conv_type == 1:
            self.kernel_size = ['K5', "P2"]
        elif self.conv_type == 2:
            self.kernel_size = ['K5', "P2", 'K5', "P2"]

        modules = []
        for layer_idx, ks in enumerate(self.kernel_size):
            input_sz = self.input_size if layer_idx == 0 else self.hidden_size
            if ks[0] == 'P':
                modules.append(nn.MaxPool1d(kernel_size=int(ks[1]), ceil_mode=False))
            elif ks[0] == 'K':
                modules.append(
                    nn.Conv1d(input_sz, self.hidden_size, kernel_size=int(ks[1]), stride=1, padding=0)
                )
                modules.append(nn.BatchNorm1d(self.hidden_size))
                modules.append(nn.ReLU(inplace=True))
        self.temporal_conv = nn.Sequential(*modules)

    def forward(self, x):
        x = self.temporal_conv(x.permute(0, 2, 1))
        return x.permute(0, 2, 1)


def make_resnet(name='resnet18'):
    if name == 'resnet18':
        model = torchvision.models.resnet18(pretrained=True)
    elif name == 'resnet34':
        model = torchvision.models.resnet34(pretrained=True)
    elif name == 'resnet50':
        model = torchvision.models.resnet50(pretrained=True)
    elif name == 'resnet101':
        model = torchvision.models.resnet101(pretrained=True)
    else:
        raise Exception('There are no supported resnet model.')

    # in_channel = model.fc.in_features
    model.fc = nn.Identity()
    return model


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.resnet = make_resnet(name='resnet18')

    def forward(self, x, lengths):
        x = self.resnet(x)
        x_batch = []
        start = 0
        for length in lengths:
            end = start + length
            x_batch.append(x[start:end])
            start = end
        x = pad_sequence(x_batch, padding_value=PAD_IDX, batch_first=True)
        return x


class FeatureExtracter(nn.Module):
    def __init__(self, frozen=False):
        super(FeatureExtracter, self).__init__()
        self.conv_2d = ResNet()  # InceptionI3d()
        self.conv_1d = TemporalConv(input_size=512, hidden_size=1024, conv_type=2)

        if frozen:
            for param in self.conv_2d.parameters():
                param.requires_grad = False

    def forward(self, src: Tensor, src_length_batch):
        src = self.conv_2d(src, src_length_batch)
        src = self.conv_1d(src)

        return src
