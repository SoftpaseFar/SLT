import pandas as pd
import numpy as np
import random
from definition import *
import torch
from pathlib import Path
import matplotlib.pyplot as plt
import os
import json
import logging
from datetime import datetime
import gzip
import pickle
from colorama import init, Back
import yaml


# -------
# 加载CSV文件
def load_annotations(file_path):
    return pd.read_csv(file_path, sep='\t')


# 列表返回name/text/video_name
def load_dataset_cvs(path):
    annotations = load_annotations(path)
    data_raws = [{
        'name': row['SENTENCE_NAME'],
        'text': row['SENTENCE'],
        'video_name': '' + row['SENTENCE_NAME'] + '.mp4'
    } for _, row in annotations.iterrows()]

    return data_raws


# ------
# 为图像生成裁剪区域
def data_augmentation(resize=(256, 256), crop_size=224, is_train=True):
    if is_train:
        left, top = np.random.randint(0, resize[0] - crop_size), np.random.randint(0, resize[1] - crop_size)
    else:
        left, top = (resize[0] - crop_size) // 2, (resize[1] - crop_size) // 2

    return (left, top, left + crop_size, top + crop_size), resize


# 调整图像适合模型使用
def adjust_img(img, resize_length):
    # 确定缩放因子和新尺寸
    scale_factor = resize_length / min(img.width, img.height)
    new_width = int(img.width * scale_factor)
    new_height = int(img.height * scale_factor)

    # 缩放图像
    img_resized = img.resize((new_width, new_height))

    # 计算裁剪坐标
    if new_width > new_height:
        # 宽度更长，从宽度中心裁剪
        left = (new_width - resize_length) // 2
        top = 0
        right = left + resize_length
        bottom = resize_length
    else:
        # 高度更长，从高度中心裁剪
        left = 0
        top = (new_height - resize_length) // 2
        right = resize_length
        bottom = top + resize_length

    # 裁剪图像
    img_cropped = img_resized.crop((left, top, right, bottom))

    return img_cropped


# 显示数据集检索的图片集合
def show_video_frames(imgs_sample):
    # 将视频帧序列转换为图像网格
    num_frames = len(imgs_sample)
    fig, axes = plt.subplots(1, num_frames, figsize=(num_frames * 2, 2))

    # 显示每一帧图像
    for i in range(num_frames):
        # 将图像数据转换为NumPy数组，并确保数据范围在[0, 1]之间
        img_np = imgs_sample[i].permute(1, 2, 0).numpy()
        img_np = img_np.clip(0, 1)  # 将图像数据裁剪到[0, 1]范围内
        axes[i].imshow(img_np)
        axes[i].axis('off')

    plt.show()


# ------
# 生成采样点
def sampler_func(clip, sn, random_choice=True):
    # 随机选择的方式生成
    if random_choice:
        f = lambda n: [(lambda n, arr: n if arr == [] else np.random.choice(arr))(n * i / sn,
                                                                                  range(int(n * i / sn),
                                                                                        max(int(n * i / sn) + 1,
                                                                                            int(n * (
                                                                                                    i + 1) / sn))))
                       for i in range(sn)]
    # 均匀分布的方式生成
    else:
        f = lambda n: [(lambda n, arr: n if arr == [] else int(np.mean(arr)))(n * i / sn, range(int(n * i / sn),
                                                                                                max(int(
                                                                                                    n * i / sn) + 1,
                                                                                                    int(n * (
                                                                                                            i + 1) / sn))))
                       for i in range(sn)]
    return f(clip)


# 文本注入噪声，根据noise_rate随机mask掉
def noise_injecting(raw_gloss, noise_rate=0.15, random_shuffle=False, is_train=True):
    res_gloss = []

    for gloss in raw_gloss:
        text = gloss.split()

        if is_train:
            index = sampler_func(len(text), int(len(text) * (1. - noise_rate)), random_choice=is_train)
            noise_gloss = []
            noise_idx = []
            for i, d in enumerate(text):
                if i in index:
                    noise_gloss.append(d)
                else:
                    noise_gloss.append(WORD_MASK)
                    noise_idx.append(i)
        else:
            noise_gloss = text

        # 对noise_gloss随机洗牌
        if is_train and random_shuffle and random.uniform(0, 1) > 0.5:
            random.shuffle(noise_gloss)

        res_gloss.append(' '.join(noise_gloss))
    return res_gloss


# ------
# 日志
# 类型： DEBUG、INFO、WARNING、ERROR、CRITICAL 等
def log(msg, config, file_name='', log_type="INFO", console=True, log_level=logging.INFO, **kwargs):
    # 设置日志级别
    logging.basicConfig(level=log_level)

    # 获取当前时间
    current_time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

    # 构造日志信息
    log_message = (f"[时间：{current_time}] [类型：{log_type}]\n"
                   f"{msg}\n")

    # 添加额外的元数据
    for key, value in kwargs.items():
        log_message += (f"[其他："
                        f"{key}: {value}]")

    # 写入日志文件
    if config['log']['need_save']:
        # 如果不存在，则创建目录
        if not os.path.exists(config['log']['save_path']):
            os.makedirs(config['log']['save_path'])
        file_name = current_time + '_' + log_type + '_' + file_name
        file_path = os.path.join(config['log']['save_path'], file_name + '.txt')
        with open(file_path, "a") as file:
            file.write(log_message)

    # 打印到控制台
    if console:
        print(msg)


# ------
# 保存检查点
def save_checkpoint(state, args, filename):
    save_path = Path(args['checkpoints_dir']) / filename
    save_dir = save_path.parent
    save_dir.mkdir(parents=True, exist_ok=True)
    try:
        # 保存checkpoint
        torch.save(state, save_path)
        print(f"模型检查点保存在：{save_path}")
    except IOError as e:
        print(f"保存失败，错误：{e}")


# ------
# 情感tokenizer
def tokenizer(words):
    emo_vocab = {
        # 正面情绪
        "happy": 1,
        "joy": 2,
        "excited": 3,
        "elated": 4,
        "thrilled": 5,
        "delighted": 6,
        "pleased": 7,
        "content": 8,
        "satisfied": 9,
        "proud": 10,
        "optimistic": 11,
        "hopeful": 12,
        "grateful": 13,
        "inspired": 14,
        "cheerful": 15,
        "blissful": 16,
        "euphoric": 17,

        # 负面情绪
        "sad": 18,
        "angry": 19,
        "fear": 20,
        "disappointed": 21,
        "frustrated": 22,
        "hurt": 23,
        "depressed": 24,
        "anxious": 25,
        "lonely": 26,
        "miserable": 27,
        "gloomy": 28,
        "hopeless": 29,
        "overwhelmed": 30,
        "embarrassed": 31,
        "jealous": 32,
        "dismayed": 33,
        "disgusted": 34,

        # 中性/复杂情绪
        "love": 35,
        "hate": 36,
        "indifferent": 37,
        "curious": 38,
        "surprised": 39,
        "bored": 40,
        "confused": 41,
        "amused": 42,
        "calm": 43,
        "relaxed": 44,
        "sympathetic": 45,
        "nervous": 46,
        "alarmed": 47,
        "apathetic": 48,
        "inquisitive": 49,
        "astonished": 50,

        # 情感不明确或无情感
        "neutral": 51,
        "uncertain": 52,
        "ambiguous": 53,
        "mixed": 54,
        "varying": 55,
        "complex": 56,
        "subtle": 57,
        "unclassified": 58,
        "noncommittal": 59,
        "undetermined": 60
    }
    return [emo_vocab.get(word, 0) for word in words]


# 情感ids解码
def batch_decode(ids):
    print('自定义加码器:', ids)
    return 'excited'


# ------
# 将视频多个json合并成一个表征视频的vectors
def gen_videos_vectors(pending_dir='', output_dir=''):
    # 如果存在output_dir，清空其下的所有内容 [这个逻辑也可以不要]

    # 遍历 pending_keypoints 目录下的所有子目录
    for subdir in os.listdir(pending_dir):
        output_filename = subdir + '.json'
        subdir_path = os.path.join(pending_dir, subdir)
        output_file_path = os.path.join(output_dir, output_filename)
        video_keypoints_vectors = merge_json_from_subdir(subdir_path)
        save_video_keypoints_vectors(output_file_path, video_keypoints_vectors)
    print('处理成功')


# 把subdir_path的json合并成一个vector
def merge_json_from_subdir(subdir_path):
    video_keypoints_vectors = []
    # 合并一个目录下所有的json文件
    for filename in os.listdir(subdir_path):
        frame_keypoints_vectors = []
        if filename.endswith('.json'):
            with open(os.path.join(subdir_path, filename), 'r') as f:
                data = json.load(f)
                if 'people' in data and len(data['people']) > 0:
                    frame_keypoints_vectors.extend(data['people'][0]['pose_keypoints_2d'])
                    frame_keypoints_vectors.extend(data['people'][0]['face_keypoints_2d'])
                    frame_keypoints_vectors.extend(data['people'][0]['hand_left_keypoints_2d'])
                    frame_keypoints_vectors.extend(data['people'][0]['hand_right_keypoints_2d'])
        video_keypoints_vectors.append(frame_keypoints_vectors)
    return video_keypoints_vectors


# 保存合并后的大json文件
def save_video_keypoints_vectors(output_file_path, vectors):
    with open(output_file_path, 'w') as save_file_path:
        json.dump(vectors, save_file_path)
        print(output_file_path + ' 保存成功。')


# 读取json文件
def load_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


# -------
# 加载 labels 文件
def load_dataset_labels(path):
    with gzip.open(path, "rb") as f:
        loaded_object = pickle.load(f)
        return loaded_object


if __name__ == '__main__':
    # keypoints预处理
    gen_videos_vectors('./data/How2Sign/pending_keypoints', './data/How2Sign/keypoints')

    # # log 测试
    # init()  # 初始化 colorama
    # # 加载参数
    # with open('./config.yaml', 'r+', encoding='utf-8') as f:
    #     config = yaml.load(f, Loader=yaml.FullLoader)
    # log(f"{Back.GREEN} Training - Epoch: {5 + 1}, CLIP loss: {0.666}, TDM Loss: {0.666} {Back.RESET}", config, 'test')
