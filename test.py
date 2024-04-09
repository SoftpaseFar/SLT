import gzip
import pickle
import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from definition import *


# definition
def load_dataset_file(filename):
    with gzip.open(filename, "rb") as f:
        loaded_object = pickle.load(f)
        return dict(loaded_object)


import cv2


def get_first_frame_dimensions(video_path):
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)

    # 检查视频是否成功打开
    if not cap.isOpened():
        print("Error opening video file")
        return

    # 读取第一帧
    ret, frame = cap.read()
    if not ret:
        print("Cannot read the video file or no frames to read.")
        return

    height, width = frame.shape[:2]

    # 打印出宽度和高度
    print(f"Width: {width}, Height: {height}")

    # 释放视频文件
    cap.release()


if __name__ == '__main__':
    # 调用函数
    # get_first_frame_dimensions('./data/How2Sign/videos/-g0sqksgyc4_3-2-rgb_front.mp4')
    # new_src_lengths = [5, 2, 1]
    # mask_gen = []
    # for i in new_src_lengths:
    #     tmp = torch.ones([i]) + 7
    #     mask_gen.append(tmp)
    # print(mask_gen)
    # mask_gen = pad_sequence(mask_gen, padding_value=PAD_IDX, batch_first=True)
    # print(mask_gen)
    # img_padding_mask = (mask_gen != PAD_IDX).long()
    # print(img_padding_mask)

    pass
