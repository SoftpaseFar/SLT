import gzip
import pickle
import torch
import mediapipe as mp
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from definition import *
from colorama import init, Fore, Back, Style
from termcolor import colored


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


def extract_keypoints_from_video(video_path):
    # 初始化MediaPipe Pose模块
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    # 存储所有帧的关键点
    keypoints_all_frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # 将BGR图像转换为RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # 处理图像并提取姿态关键点
        results = pose.process(image)
        keypoints_all_frames.append(results)
        # 收集关键点
        # if results.pose_landmarks:
        #     keypoints = [(landmark.x, landmark.y, landmark.z) for landmark in results.pose_landmarks.landmark]
        #     keypoints_all_frames.append(keypoints)
    cap.release()

    return keypoints_all_frames


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

    # x_batch = [torch.zeros(1, 2, 2),
    #            torch.zeros(2, 2, 2),
    #            torch.zeros(3, 2, 2)]
    # x = pad_sequence(x_batch, padding_value=-1, batch_first=True)
    # print(x)

    # x = torch.tensor([[1, 2, 3],
    #                   [4, 5, 6],
    #                   [7, 8, 9]])
    # x = x.view(x.size(0), -1)
    # print(x)

    # tgt_input = {
    #     'input_ids': torch.tensor([[1, 2, 3], [4, 5, 6]]),  # 示例的input_ids
    #     'attention_mask': torch.tensor([[1, 1, 1], [1, 1, 1]])  # 示例的attention_mask
    # }
    # txt_logits = torch.randn(2, 3, 10)
    # print(txt_logits)
    # output = txt_logits[:, [2, 2]]
    # print(output)

    # res = extract_keypoints_from_video('./data/How2Sign/videos/-g0sqksgyc4_3-2-rgb_front.mp4')
    # print(res[0])

    # 初始化 colorama
    # init()
    #
    # # 打印红色文本
    # print(Fore.RED + 'This is red text')
    # # 打印绿色背景
    # print(Back.GREEN + 'This has a green background')
    # # 重置颜色
    # print(Style.RESET_ALL + 'Back to normal')
    # 打印红色文本
    print(colored('This is red text', 'red'))
    # 打印绿色背景
    print(colored('This has a green background', 'green', 'on_red'))

    # 重置颜色和样式（termcolor 中不需要额外的操作）
    print(colored('Back to normal', 'white'))
    pass
