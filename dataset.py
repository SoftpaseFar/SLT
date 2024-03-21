import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import utils
from vidaug import augmentors as va
from augmentation import *


class S2TDataset(Dataset):
    def __init__(self, path, tokenizer, config, args, phase, training_refurbish=False):
        self.config = config
        self.args = args
        self.training_refurbish = training_refurbish

        self.raw_data = utils.load_dataset_file(path)
        self.tokenizer = tokenizer
        self.img_path = config['data']['img_path']
        self.phase = phase
        self.max_length = config['data']['max_length']

        self.list = [key for key, value in self.raw_data.items()]

        sometimes = lambda aug: va.Sometimes(0.5, aug)  # Used to apply augmentor with 50% probability
        self.seq = va.Sequential([
            # va.RandomCrop(size=(240, 180)), # randomly crop video with a size of (240 x 180)
            # va.RandomRotate(degrees=10), # randomly rotates the video with a degree randomly choosen from [-10, 10]
            sometimes(va.RandomRotate(30)),
            sometimes(va.RandomResize(0.2)),
            # va.RandomCrop(size=(256, 256)),
            sometimes(va.RandomTranslate(x=10, y=10)),

            # sometimes(Brightness(min=0.1, max=1.5)),
            # sometimes(Contrast(min=0.1, max=2.0)),

        ])
        self.seq_color = va.Sequential([
            sometimes(Brightness(min=0.1, max=1.5)),
            sometimes(Color(min=0.1, max=1.5)),
            # sometimes(Contrast(min=0.1, max=2.0)),
            # sometimes(Sharpness(min=0.1, max=2.))
        ])
        # self.seq = SomeOf(self.seq_geo, self.seq_color)

    # 数据集大小
    def __len__(self):
        return len(self.raw_data)

    # 数据索引
    def __getitem__(self, index):
        key = self.list[index]
        sample = self.raw_data[key]

        tgt_sample = sample['text']
        name_sample = sample['name']
        img_sample = self.load_imgs([self.img_path + x for x in sample['imgs_path']])

        return name_sample, img_sample, tgt_sample
