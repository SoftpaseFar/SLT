import os
import re
import torch
import cv2
import utils
import random
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


# How2Sign数据集
class How2SignDataset(Dataset):
    def __init__(self, path, tokenizer, config, args, phase, training_refurbish=False):
        self.raw_data = utils.load_dataset_cvs(path)

        self.tokenizer = tokenizer
        self.config = config
        self.args = args
        self.phase = phase
        self.training_refurbish = training_refurbish

        self.videos_dir = config[self.args['dataset']]['videos_dir']
        self.max_length = config[self.args['dataset']]['max_length']

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, idx):
        sample = self.raw_data[idx]

        name_sample = sample['name']

        video_name = sample['video_name']
        video_path = os.path.join(self.config[self.args['dataset']]['videos_dir'], video_name)
        imgs_sample = self._load_video(video_path)
        # length_sample = len(imgs_sample)
        tgt_sample = sample['text']

        # 需要关键点信息
        if self.args['need_keypoints']:
            video_keypoints_name = sample['video_name'][:-4] + '.json'
            # video_keypoints_path = os.path.join(self.config[self.args['dataset']]['keypoints_dir'],
            # video_keypoints_name)
            video_keypoints_path = self.config[self.args['dataset']][
                                       'keypoints_dir'] + name_sample + '/alphapose-results.json'
            keypoints_sample = self._load_keypoints(video_keypoints_path)
            return name_sample, imgs_sample, tgt_sample, keypoints_sample

        return name_sample, imgs_sample, tgt_sample

    def _load_keypoints(self, path):
        data = utils.load_json(path)
        video_vectors = [frame_data['people'][0]['pose_keypoints_2d'] for frame_data in data.values()]
        print("video_vectors[0]:", len(video_vectors[0]))
        # 如果关键点向量数量超过最大长度，随机抽取最大长度的关键点向量，并保持顺序
        if len(video_vectors) > self.max_length:
            video_vectors = [video_vectors[i] for i in
                             sorted(random.sample(range(len(video_vectors)), self.max_length))]
        return video_vectors

    def _load_video(self, video_path):
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
            frames = [frames[i] for i in
                      sorted(random.sample(range(len(frames)), self.max_length))]

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
        imgs_batch_tmp, emo_batch_tmp, tgt_batch, src_length_batch, keypoints_batch = [], [], [], [], []

        # 将批序列的name、imgs、tgt分别包装成列表
        for _, imgs_sample, tgt_sample, *other_data in batch:
            imgs_batch_tmp.append(imgs_sample)
            # tgt_sample 加入情感占位符
            tgt_sample = '<pad>' + tgt_sample
            # 一个batch情感收集
            emo_batch_tmp.append('excited')
            tgt_batch.append(tgt_sample)
            if self.args['need_keypoints'] and other_data:
                keypoints_sample = torch.tensor(other_data[0])
                print('keypoints_sample.shape: ', keypoints_sample.shape)
                keypoints_batch.append(keypoints_sample)

        # 每个视频真实长度
        imgs_batch_len = [len(vid) for vid in imgs_batch_tmp]
        # print(imgs_batch_len)

        # 视频批序列最大长度
        imgs_batch_max_len = max(imgs_batch_len)

        # 将batch每个video的imgs序列填充成长度为imgs_batch_max_len
        imgs_batch = [torch.cat(
            (
                vid,
                torch.zeros(imgs_batch_max_len - len(vid), vid.size(1), vid.size(2), vid.size(3)).to(vid.device)
            ), dim=0)
            for vid in imgs_batch_tmp]

        imgs_batch = torch.stack(imgs_batch, dim=0)

        # 视频序列掩码
        img_padding_mask = torch.tensor(
            [[1] * length + [0] * (imgs_batch_max_len - length) for length in imgs_batch_len],
            dtype=torch.long
        )

        src_input = {
            'imgs_ids': imgs_batch,
            'attention_mask': img_padding_mask,

            'src_length_batch': imgs_batch_max_len}

        # 是否需要 need_keypoints
        if self.args['need_keypoints']:
            # 找到最大长度
            keypoints_batch_max_len = max(len(keypoints) for keypoints in keypoints_batch)
            # 将所有序列填充到最大长度
            keypoints_batch_padded = [torch.cat(
                (
                    keypoints,
                    torch.zeros(keypoints_batch_max_len - len(keypoints), keypoints.size(1)).to(keypoints.device)
                ),
                dim=0)
                for keypoints in keypoints_batch]

            # 将填充后的序列堆叠成张量
            keypoints_batch_tensor = torch.stack(keypoints_batch_padded, dim=0)
            src_input['keypoints_ids'] = keypoints_batch_tensor

        # 将一个batch的文本进行tokenizer
        # 对于批次中不同长度的文本进行填充
        # 截断过长的文本
        # with self.tokenizer.as_target_tokenizer():
        tgt_input = self.tokenizer(tgt_batch,
                                   return_tensors="pt",
                                   padding=True,
                                   truncation=True)

        print(f"正在加载数据集 {self.args['dataset']} ...")

        # 情感pad初进行情感注入
        for i, value in enumerate(utils.tokenizer(emo_batch_tmp)):
            tgt_input['input_ids'][i, 0] = value

        # 训练阶段需要mask掉一些，用来训练解码器
        if self.training_refurbish:
            masked_tgt = utils.noise_injecting(tgt_batch, self.args['noise_rate'],
                                               random_shuffle=self.args['random_shuffle'],
                                               is_train=(self.phase == 'train'))
            # with self.tokenizer.as_target_tokenizer():
            masked_tgt_input = self.tokenizer(masked_tgt,
                                              return_tensors="pt",
                                              padding=True,
                                              truncation=True)
            return src_input, tgt_input, masked_tgt_input

        # 返回一个batch视频集合 目标翻译的文本
        return src_input, tgt_input

    def __str__(self):
        return f'# total {self.phase} set: {len(self.raw_data)}.'


# PHOENIX-2014-T数据集
class P14TDataset(Dataset):
    def __init__(self, path, tokenizer, config, args, phase, training_refurbish=False):
        # 未处理原始数据
        self.data = utils.load_dataset_labels(path)
        # print('self.data: ', self.data)

        self.tokenizer = tokenizer
        self.config = config
        self.args = args
        self.phase = phase
        self.training_refurbish = training_refurbish

        self.features_path = config[args['dataset']]['features_path']
        self.max_length = config[args['dataset']]['max_length']

        # TODO 20->1000
        # print(self.data[0:20][0])
        self.raw_data = [value for item in self.data[0:20] for _, value in item.items()]
        # print(self.raw_data[0])

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, idx):
        sample: dict = self.raw_data[idx]
        # print(sample)

        name_sample = sample['name']
        imgs_path = sample['imgs_path']
        imgs_sample = self._load_imgs(imgs_path)
        # length_sample = sample['length']
        tgt_sample = sample['text']

        # 需要关键点信息
        if self.args['need_keypoints']:
            video_keypoints_path = self.config[self.args['dataset']][
                                       'keypoints_dir'] + name_sample + '/alphapose-results.json'
            keypoints_sample = self._load_keypoints(video_keypoints_path)
            return name_sample, imgs_sample, tgt_sample, keypoints_sample

        return name_sample, imgs_sample, tgt_sample

    def _load_keypoints(self, path):
        data = utils.load_json(path)
        video_vectors = [frame_data['people'][0]['pose_keypoints_2d'] for frame_data in data.values()]
        print("video_vectors[0]:", len(video_vectors[0]))
        # 如果关键点向量数量超过最大长度，随机抽取最大长度的关键点向量，并保持顺序
        if len(video_vectors) > self.max_length:
            video_vectors = [video_vectors[i] for i in
                             sorted(random.sample(range(len(video_vectors)), self.max_length))]
        return video_vectors

    def _load_imgs(self, imgs_path):
        data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        frames = []
        for path in imgs_path:
            try:
                img = cv2.imread(os.path.join(self.features_path, path))
                frames.append(img)
            except IOError as e:
                print(f"P14TDataset数据集，图片加载错误:", e)

        # 如果帧数超过最大长度，随机抽取max_length帧
        if len(frames) > self.max_length:
            frames = [frames[i] for i in
                      sorted(random.sample(range(len(frames)), self.max_length))]

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
        imgs_batch_tmp, emo_batch_tmp, tgt_batch, src_length_batch, keypoints_batch = [], [], [], [], []

        # 将批序列的name、imgs、tgt分别包装成列表
        for _, imgs_sample, tgt_sample, *other_data in batch:
            imgs_batch_tmp.append(imgs_sample)
            # tgt_sample 加入情感占位符
            tgt_sample = '<pad>' + tgt_sample
            # 一个batch情感收集
            emo_batch_tmp.append('aufgeregt')
            tgt_batch.append(tgt_sample)
            if self.args['need_keypoints'] and other_data:
                keypoints_sample = torch.tensor(other_data[0])
                print('keypoints_sample.shape: ', keypoints_sample.shape)
                keypoints_batch.append(keypoints_sample)

        # 每个视频真实长度
        imgs_batch_len = [len(vid) for vid in imgs_batch_tmp]
        # print(imgs_batch_len)

        # 视频批序列最大长度
        imgs_batch_max_len = max(imgs_batch_len)

        # 将batch每个video的imgs序列填充成长度为imgs_batch_max_len
        imgs_batch = [torch.cat(
            (
                vid,
                torch.zeros(imgs_batch_max_len - len(vid), vid.size(1), vid.size(2), vid.size(3)).to(vid.device)
            ), dim=0)
            for vid in imgs_batch_tmp]

        imgs_batch = torch.stack(imgs_batch, dim=0)

        # 视频序列掩码
        img_padding_mask = torch.tensor(
            [[1] * length + [0] * (imgs_batch_max_len - length) for length in imgs_batch_len],
            dtype=torch.long
        )

        src_input = {
            'imgs_ids': imgs_batch,
            'attention_mask': img_padding_mask,

            'src_length_batch': imgs_batch_max_len}

        # 是否需要 need_keypoints
        if self.args['need_keypoints']:
            # 找到最大长度
            keypoints_batch_max_len = max(len(keypoints) for keypoints in keypoints_batch)
            # 将所有序列填充到最大长度
            keypoints_batch_padded = [torch.cat(
                (
                    keypoints,
                    torch.zeros(keypoints_batch_max_len - len(keypoints), keypoints.size(1)).to(keypoints.device)
                ),
                dim=0)
                for keypoints in keypoints_batch]

            # 将填充后的序列堆叠成张量
            keypoints_batch_tensor = torch.stack(keypoints_batch_padded, dim=0)
            src_input['keypoints_ids'] = keypoints_batch_tensor

        # 将一个batch的文本进行tokenizer
        # 对于批次中不同长度的文本进行填充
        # 截断过长的文本
        # with self.tokenizer.as_target_tokenizer():
        tgt_input = self.tokenizer(tgt_batch,
                                   return_tensors="pt",
                                   padding=True,
                                   truncation=True)

        print(f"正在加载数据集 {self.args['dataset']} ...")

        # 情感pad初进行情感注入
        for i, value in enumerate(utils.tokenizer(emo_batch_tmp)):
            tgt_input['input_ids'][i, 0] = value

        # 训练阶段需要mask掉一些，用来训练解码器
        if self.training_refurbish:
            masked_tgt = utils.noise_injecting(tgt_batch, self.args['noise_rate'],
                                               random_shuffle=self.args['random_shuffle'],
                                               is_train=(self.phase == 'train'))
            # with self.tokenizer.as_target_tokenizer():
            masked_tgt_input = self.tokenizer(masked_tgt,
                                              return_tensors="pt",
                                              padding=True,
                                              truncation=True)
            return src_input, tgt_input, masked_tgt_input

        # 返回一个batch视频集合 目标翻译的文本
        return src_input, tgt_input

    def __str__(self):
        return f'# 阶段：{self.phase} 总共： {len(self.raw_data)}.'


# CSL-Delay数据集
class CSLDailyDataset(Dataset):
    def __init__(self, path, tokenizer, config, args, phase, training_refurbish=False):
        # 未处理原始数据
        # TODO 20->1000
        self.raw_data = utils.load_dataset_txt(path)[0:20]
        # print(self.raw_data)

        self.tokenizer = tokenizer
        self.config = config
        self.args = args
        self.phase = phase
        self.training_refurbish = training_refurbish

        self.features_path = config[args['dataset']]['features_path']
        self.max_length = config[args['dataset']]['max_length']

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, idx):
        sample: dict = self.raw_data[idx]

        name_sample = sample['name']
        video_path = os.path.join(self.features_path, name_sample)
        # print('video_path: ', video_path)
        imgs_path = sorted([os.path.join(name_sample, f) for f in os.listdir(video_path) if f.endswith('.jpg')],
                           key=lambda s: [int(text) if text.isdigit() else text.lower() for text in
                                          re.split(r'([0-9]+)', s)])

        # print('imgs_path: ', imgs_path)
        imgs_sample = self._load_imgs(imgs_path)
        # length_sample = sample['length']
        tgt_sample = sample['word']

        # 需要关键点信息
        if self.args['need_keypoints']:
            video_keypoints_path = os.path.join(self.config[self.args['dataset']]['keypoints_dir'], name_sample +
                                                '/alphapose-results.json')
            # print('video_keypoints_path: ', video_keypoints_path)
            keypoints_sample = self._load_keypoints(video_keypoints_path)
            return name_sample, imgs_sample, tgt_sample, keypoints_sample

        return name_sample, imgs_sample, tgt_sample

    def _load_keypoints(self, path):
        data = utils.load_json(path)
        video_vectors = [frame_data['people'][0]['pose_keypoints_2d'] for frame_data in data.values()]
        print("video_vectors[0]:", len(video_vectors[0]))
        # 如果关键点向量数量超过最大长度，随机抽取最大长度的关键点向量，并保持顺序
        if len(video_vectors) > self.max_length:
            video_vectors = [video_vectors[i] for i in
                             sorted(random.sample(range(len(video_vectors)), self.max_length))]
        return video_vectors

    def _load_imgs(self, imgs_path):
        data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        frames = []
        for path in imgs_path:
            try:
                img = cv2.imread(os.path.join(self.features_path, path))
                frames.append(img)
            except IOError as e:
                print(f"CSLDailyDataset数据集，图片加载错误:", e)

        # 如果帧数超过最大长度，随机抽取max_length帧
        if len(frames) > self.max_length:
            frames = [frames[i] for i in
                      sorted(random.sample(range(len(frames)), self.max_length))]

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
        imgs_batch_tmp, emo_batch_tmp, tgt_batch, src_length_batch, keypoints_batch = [], [], [], [], []

        # 将批序列的name、imgs、tgt分别包装成列表
        for _, imgs_sample, tgt_sample, *other_data in batch:
            imgs_batch_tmp.append(imgs_sample)
            # tgt_sample 加入情感占位符
            tgt_sample = '<pad>' + tgt_sample
            # 一个batch情感收集
            emo_batch_tmp.append('高兴')
            tgt_batch.append(tgt_sample)
            if self.args['need_keypoints'] and other_data:
                keypoints_sample = torch.tensor(other_data[0])
                print('keypoints_sample.shape: ', keypoints_sample.shape)
                keypoints_batch.append(keypoints_sample)

        # 每个视频真实长度
        imgs_batch_len = [len(vid) for vid in imgs_batch_tmp]
        # print(imgs_batch_len)

        # 视频批序列最大长度
        imgs_batch_max_len = max(imgs_batch_len)

        # 将batch每个video的imgs序列填充成长度为imgs_batch_max_len
        imgs_batch = [torch.cat(
            (
                vid,
                torch.zeros(imgs_batch_max_len - len(vid), vid.size(1), vid.size(2), vid.size(3)).to(vid.device)
            ), dim=0)
            for vid in imgs_batch_tmp]

        imgs_batch = torch.stack(imgs_batch, dim=0)

        # 视频序列掩码
        img_padding_mask = torch.tensor(
            [[1] * length + [0] * (imgs_batch_max_len - length) for length in imgs_batch_len],
            dtype=torch.long
        )

        src_input = {
            'imgs_ids': imgs_batch,
            'attention_mask': img_padding_mask,

            'src_length_batch': imgs_batch_max_len}

        # 是否需要 need_keypoints
        if self.args['need_keypoints']:
            # 找到最大长度
            keypoints_batch_max_len = max(len(keypoints) for keypoints in keypoints_batch)
            # 将所有序列填充到最大长度
            keypoints_batch_padded = [torch.cat(
                (
                    keypoints,
                    torch.zeros(keypoints_batch_max_len - len(keypoints), keypoints.size(1)).to(keypoints.device)
                ),
                dim=0)
                for keypoints in keypoints_batch]

            # 将填充后的序列堆叠成张量
            keypoints_batch_tensor = torch.stack(keypoints_batch_padded, dim=0)
            src_input['keypoints_ids'] = keypoints_batch_tensor

        # 将一个batch的文本进行tokenizer
        # 对于批次中不同长度的文本进行填充
        # 截断过长的文本
        # with self.tokenizer.as_target_tokenizer():
        tgt_input = self.tokenizer(tgt_batch,
                                   return_tensors="pt",
                                   padding=True,
                                   truncation=True)

        print(f"正在加载数据集 {self.args['dataset']} ...")

        # 情感pad初进行情感注入
        for i, value in enumerate(utils.tokenizer(emo_batch_tmp)):
            tgt_input['input_ids'][i, 0] = value

        # 训练阶段需要mask掉一些，用来训练解码器
        if self.training_refurbish:
            masked_tgt = utils.noise_injecting(tgt_batch, self.args['noise_rate'],
                                               random_shuffle=self.args['random_shuffle'],
                                               is_train=(self.phase == 'train'))
            # with self.tokenizer.as_target_tokenizer():
            masked_tgt_input = self.tokenizer(masked_tgt,
                                              return_tensors="pt",
                                              padding=True,
                                              truncation=True)
            return src_input, tgt_input, masked_tgt_input

        # 返回一个batch视频集合 目标翻译的文本
        return src_input, tgt_input

    def __str__(self):
        return f'# 阶段：{self.phase} 总共： {len(self.raw_data)}.'
