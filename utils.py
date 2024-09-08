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
import csv
from colorama import init, Back
from transformers import MBartForConditionalGeneration, MBartConfig
import yaml
import gc
import re


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
def log_(msg, config, file_name='', log_type="INFO", console=True, log_level=logging.INFO, **kwargs):
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


def write_log(filename, phase, **kwargs):
    # 获取当前时间
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    # 将kwargs解包并用空格隔开
    content = '[' + current_time + ']|' + phase + '|' + '|'.join(f"{key}={value}" for key, value in kwargs.items())
    # 追加写入到文件
    with open(filename, 'a', encoding='utf-8') as file:
        file.write(content + '\n')
    print(f"{Back.GREEN}保存成功{Back.RESET}")


def log(phase, **kwargs):
    if 'vlp' in phase:
        write_log('trash/vlp.txt', phase, **kwargs)
    elif 'slt' in phase:
        write_log('trash/slt.txt', phase, **kwargs)
    elif 'pred' in phase:
        write_log('trash/pred.txt', phase, **kwargs)
    else:
        print(f"{Back.RED}保存失败{Back.RESET}")


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
    try:
        # 将情感词转换为小写并提取编码
        return [emotion_vocab[word.lower()] for word in words]
    except KeyError as e:
        raise ValueError(f"无法识别情感词: {e}")


# 情感ids解码
def batch_decode(ids):
    print('情感批量解码:', ids)
    return 'positive'


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
                    # frame_keypoints_vectors.extend(data['people'][0]['face_keypoints_2d'])
                    # frame_keypoints_vectors.extend(data['people'][0]['hand_left_keypoints_2d'])
                    # frame_keypoints_vectors.extend(data['people'][0]['hand_right_keypoints_2d'])
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


# 从config加载MBart模型
# def load_mbart_from_conf(config_path):
#     config_dict = load_json(config_path)
#     # 使用加载的配置初始化配置对象
#     config = MBartConfig.from_dict(config_dict)
#     # 从预训练模型加载权重并应用新的配置
#     mbart = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-cc25", config=config)
#     return mbart


# -------
# 加载数据集标签的函数
def load_dataset_labels(path):
    with gzip.open(path, "rb") as f:
        loaded_object = pickle.load(f)
        return loaded_object


# 保存数据集标签的函数
def save_dataset_labels(data, path):
    with gzip.open(path, "wb") as f:
        pickle.dump(data, f)


# -------
# 加载 txt 文件
def load_dataset_txt(path):
    # 读取文件并转换为字典列表
    with open(path, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file, delimiter='|')
        data_list = [row for row in reader]
        return data_list


# -------
# 清理CUDA缓存
def clear_cuda_cache():
    torch.backends.cuda.max_split_size_mb = 1024
    # 清理未使用的CUDA缓存
    torch.cuda.empty_cache()
    # 重置所有峰值显存统计信息
    torch.cuda.reset_peak_memory_stats()

    gc.collect()


# -------
# 情感准确率计算
def get_first_word(text):
    # 使用正则表达式匹配第一个单词
    match = re.search(r'\b\w+\b', text)
    return match.group(0) if match else ''


def compare_first_words(hyp, ref):
    hyp_first_word = get_first_word(hyp)
    ref_first_word = get_first_word(ref)
    return 1 if hyp_first_word == ref_first_word else 0


def calculate_ratio_of_ones(emo_collection):
    total = len(emo_collection)
    if total == 0:
        return 0  # 避免除以零
    count_of_ones = emo_collection.count(1)
    ratio_of_ones = count_of_ones / total
    return ratio_of_ones


# 文本修正, 提高BLEU4分数
# 去除重复
def remove_duplicates(text):
    words = text.split()
    deduped_words = []
    seen = set()
    for word in words:
        if word not in seen:
            deduped_words.append(word)
            seen.add(word)
    return ' '.join(deduped_words)


if __name__ == '__main__':
    log('vlp', loss_1=1, loss_2=2, loss_3=3)
    log('vlp', loss_1=1, loss_2=2, loss_3=3)
    log('slt', loss_1=1, loss_2=2, loss3=3)
    log('slt', loss_1=1, loss_2=2, loss3=3)

    # # keypoints预处理
    # gen_videos_vectors('./data/How2Sign/pending_keypoints', './data/How2Sign/keypoints')

    # res = load_dataset_labels('./data/Phonexi2014T/labels.test')
    # print(res)

    # # log 测试
    # init()  # 初始化 colorama
    # # 加载参数
    # with open('./config.yaml', 'r+', encoding='utf-8') as f:
    #     config = yaml.load(f, Loader=yaml.FullLoader)
    # log(f"{Back.GREEN} Training - Epoch: {5 + 1}, CLIP loss: {0.666}, TDM Loss: {0.666} {Back.RESET}", config, 'test')
