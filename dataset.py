import os
import torch
import cv2
import utils
import random
import numpy as np
from PIL import Image
from definition import *
from torch.utils.data import Dataset
from torchvision import transforms
from torch.nn.utils.rnn import pad_sequence


class How2SignDataset(Dataset):
    def __init__(self, path, tokenizer, config, args, phase, training_refurbish=False):
        self.raw_data = utils.load_dataset_cvs(path)

        self.tokenizer = tokenizer
        self.config = config
        self.args = args
        self.phase = phase
        self.training_refurbish = training_refurbish

        self.videos_dir = config['data']['videos_dir']
        self.max_length = config['data']['max_length']

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, idx):
        sample = self.raw_data[idx]

        name_sample = sample['name']

        video_name = sample['video_name']
        video_path = os.path.join(self.config['data']['videos_dir'], video_name)
        imgs_sample = self.load_video(video_path)

        # length_sample = len(imgs_sample)
        tgt_sample = sample['text']

        return name_sample, imgs_sample, tgt_sample

    def load_video(self, video_path):
        data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()

        # 如果帧数超过最大长度，随机抽取max_length帧
        if len(frames) > self.max_length:
            frames = [frames[i] for i in sorted(random.sample(range(len(frames)), self.max_length))]

        imgs = torch.zeros(len(frames), 3, self.args['input_size'], self.args['input_size'])
        crop_rect, resize = utils.data_augmentation(resize=(self.args['resize'], self.args['resize']),
                                                    crop_size=self.args['input_size'], is_train=(self.phase == 'train'))

        for i, frame in enumerate(frames):
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            img = utils.adjust_img(img, self.args['resize'])
            # img = img.resize(resize)
            img = img.crop(crop_rect)
            img = data_transform(img)
            imgs[i] = img

        return imgs

    def collate_fn(self, batch):
        name_batch, imgs_batch_tmp, tgt_batch, src_length_batch, = [], [], [], []

        # 将批序列的name、imgs、tgt分别包装成列表
        for name_sample, imgs_sample, tgt_sample in batch:
            name_batch.append(name_sample)
            imgs_batch_tmp.append(imgs_sample)
            tgt_batch.append(tgt_sample)

        # 视频批序列最大长度
        imgs_batch_max_len = max([len(vid) for vid in imgs_batch_tmp])
        # 每个视频帧填充成4的倍数 + 左右填充后的长度
        imgs_batch_pad_len = torch.LongTensor([np.ceil(len(vid) / 4.0) * 4 + 16 for vid in imgs_batch_tmp])

        # 填充成一样的长度是多少
        left_pad = 8
        right_pad = int(np.ceil(imgs_batch_max_len / 4.0)) * 4 - imgs_batch_max_len + 8
        padded_max_len = imgs_batch_max_len + left_pad + right_pad

        # 将batch每个video的imgs序列填充成长度为padded_max_len
        padded_video_to_max = [torch.cat(
            (
                vid[0][None].expand(left_pad, -1, -1, -1),
                vid,
                vid[-1][None].expand(padded_max_len - len(vid) - left_pad, -1, -1, -1),
            )
            , dim=0)
            for vid in imgs_batch_tmp]
        # 原始视频帧各自填充
        src_padded_video = [padded_video_to_max[i][0:imgs_batch_pad_len[i], :, :, :] for i in
                            range(len(padded_video_to_max))]

        # 原始视频帧各自填充 长度记录
        for i in range(len(src_padded_video)):
            src_length_batch.append(len(src_padded_video[i]))
        src_length_batch = torch.tensor(src_length_batch)

        # 将一个batch的视频整合在一起:[[1,1,1],[2,2,2],[3,3,3]]=>[1,1,1,2,2,2,3,3,3]
        imgs_batch = torch.cat(src_padded_video, 0)

        # 卷积操作之后的的长度，根据这个生成输入Transformer等的掩码，暂时没变化
        new_src_lengths = src_length_batch
        new_src_lengths = new_src_lengths.long()

        # 根据卷积后的长度生成
        mask_gen = []
        for i in new_src_lengths:
            tmp = torch.ones([i]) + 7
            mask_gen.append(tmp)
        mask_gen = pad_sequence(mask_gen, padding_value=PAD_IDX, batch_first=True)
        imgs_padding_mask = (mask_gen != PAD_IDX).long()

        # 将一个batch的文本进行tokenizer
        # 对于批次中不同长度的文本进行填充
        # 截断过长的文本
        with self.tokenizer.as_target_tokenizer():
            tgt_input = self.tokenizer(tgt_batch, return_tensors="pt", padding=True, truncation=True)

        src_input = {
            'name_batch': name_batch,

            'input_ids': imgs_batch,
            'attention_mask': imgs_padding_mask,
            'src_length_batch': src_length_batch,
            'new_src_length_batch': new_src_lengths}

        # 训练阶段需要mask掉一些，用来训练解码器
        if self.training_refurbish:
            masked_tgt = utils.noise_injecting(tgt_batch, self.args['noise_rate'],
                                               random_shuffle=self.args['random_shuffle'],
                                               is_train=(self.phase == 'train'))
            with self.tokenizer.as_target_tokenizer():
                masked_tgt_input = self.tokenizer.as_target_tokenizer(masked_tgt, return_tensors="pt", padding=True,
                                                                      truncation=True)
            return src_input, tgt_input, masked_tgt_input

        # 返回一个batch视频集合 目标翻译的文本
        return src_input, tgt_input

    def __str__(self):
        return f'# total {self.phase} set: {len(self.raw_data)}.'
