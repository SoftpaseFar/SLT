import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from definition import *


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
def log():
    pass


if __name__ == '__main__':
    # res = load_dataset_cvs('./data/How2Sign/test.csv')
    res = load_dataset_cvs('./data/How2Sign/test.csv')
    print(res)
    pass
