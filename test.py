import utils

if __name__ == '__main__':
    # 给定的关键点信息
    keypoints_info = utils.load_frame_keypoints(
        './data/How2Sign/keypoints/-fZc293MpJk_0-1-rgb_front/-fZc293MpJk_0-1-rgb_front_000000000000_keypoints.json')

    # 关键点信息可视化
    utils.show_frame_keypoints(keypoints_info)

    pass
