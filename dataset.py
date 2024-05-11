import os
import torch
import cv2
import utils
import random
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


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
        imgs_sample = self._load_video(video_path)
        # length_sample = len(imgs_sample)
        tgt_sample = sample['text']

        # 需要关键点信息
        if self.args['need_keypoints']:
            video_keypoints_name = sample['video_name'][:-4] + '.json'
            video_keypoints_path = os.path.join(self.config['data']['keypoints_dir'], video_keypoints_name)
            keypoints_sample = self._load_keypoints(video_keypoints_path)
            return name_sample, imgs_sample, tgt_sample, keypoints_sample

        return name_sample, imgs_sample, tgt_sample

    def _load_keypoints(self, path):
        video_vectors = utils.load_json(path)
        print('video_vectors:', video_vectors)
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
                keypoints_sample = other_data[0]
                src_length_batch.append(keypoints_sample)

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
            )
            , dim=0)
            for vid in imgs_batch_tmp]
        # # 计算S3D操作后的视频掩码 TODO 考虑掩码
        # s3d_after_length = ((torch.tensor(imgs_batch_len) / 2 - 3) / 2 + 1 - 2) / 2 + 1 - 1
        # s3d_after_length = s3d_after_length.long()
        # print(s3d_after_length)
        # mask_gen = []
        # for i in s3d_after_length:
        #     tmp = torch.ones([i]) + 7
        #     mask_gen.append(tmp)
        # mask_gen = pad_sequence(mask_gen, padding_value=PAD_IDX, batch_first=True)
        # img_padding_mask = (mask_gen != PAD_IDX).long()

        src_input = {
            'input_ids': imgs_batch,
            # 'attention_mask': img_padding_mask,

            'src_length_batch': imgs_batch_max_len}

        # 是否需要 need_keypoints
        if self.args['need_keypoints']:
            src_input['keypoints_ids'] = keypoints_batch

        # 将一个batch的文本进行tokenizer
        # 对于批次中不同长度的文本进行填充
        # 截断过长的文本
        # with self.tokenizer.as_target_tokenizer():
        tgt_input = self.tokenizer(tgt_batch,
                                   return_tensors="pt",
                                   padding=True,
                                   truncation=True)
        # tgt_input_attention_mask = []
        # tgt_input = {
        #     'input_ids': tgt_input_ids,
        #     'attention_mask': tgt_input_attention_mask
        # }

        print(f"正在加载数据集...")

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
