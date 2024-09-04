import zipfile
import csv
import os
import argparse


def main():
    # 设置命令行参数解析
    parser = argparse.ArgumentParser(description='从zip文件中解压指定的文件')
    parser.add_argument('-zip_path', default='', help='zip文件路径')
    parser.add_argument('-csv_path', default='', help='包含文件名的csv文件路径')
    parser.add_argument('-output_path', default='',
                        help='解压输出目录路径')
    args = parser.parse_args()

    # 从命令行获取文件路径，如果未提供，则使用默认值
    zip_file_path = args.zip_path
    csv_file_path = args.csv_path
    output_directory = args.output_path

    # 如果输出目录不存在，则创建该目录
    os.makedirs(output_directory, exist_ok=True)

    # 读取CSV文件中的SENTENCE_NAME字段，并生成所需的文件名集合
    filenames_to_extract = set()
    with open(csv_file_path, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file, delimiter='\t')  # 假设CSV文件是使用制表符分隔
        for row in reader:
            sentence_name = 'raw_videos/' + row['SENTENCE_NAME'].strip() + '.mp4'  # 生成文件名
            filenames_to_extract.add(sentence_name)

    print('需要解压文件数量：', len(filenames_to_extract))

    # 打开zip文件
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        # 获取zip中的所有文件
        all_files = zip_ref.namelist()
        # print('all_files: ', all_files)

        # 过滤需要解压的文件
        files_to_extract = [file for file in all_files if file in filenames_to_extract]
        print('文件数量校验：', len(files_to_extract))

        # 找出不在zip文件中的文件名
        files_not_found = [file for file in filenames_to_extract if file not in all_files]
        print('不存在的文件数量：', len(files_not_found))
        print('不存在的文件：', files_not_found)

        print('正在解压...')
        unzip_failed = []
        # 解压需要的文件到指定目录
        for file in files_to_extract:
            try:
                zip_ref.extract(file, output_directory)
                print('解压：', file)
            except Exception as e:
                print(f"解压文件 '{file}' 时发生错误: {e}")
                unzip_failed.append(file)
                continue

        print(f"解压缩完成，{len(files_to_extract) - len(unzip_failed)} 个文件已解压到 {output_directory}")
        print('加压失败文件数量：', len(unzip_failed))
        print('加压失败文件：', unzip_failed)


# 判断当前模块是否为主模块
if __name__ == '__main__':
    main()

# 不存在的文件： ['raw_videos/1mdMz4RkRdA_19-8-rgb_front.mp4', 'raw_videos/1mdMz4RkRdA_28-8-rgb_front.mp4', 'raw_videos/CH7AviIr0-0_15-8-rgb_front.mp4', 'raw_videos/bpyAuV3jNIc_11-8-rgb_front.mp4', 'raw_videos/1mdMz4RkRdA_21-8-rgb_front.mp4', 'raw_videos/1P6n7Tv8oco_24-8-rgb_front.mp4', 'raw_videos/B_ye00IgI4w_16-5-rgb_front.mp4', 'raw_videos/f1R7MZSlOOg_11-5-rgb_front.mp4', 'raw_videos/CH7AviIr0-0_19-8-rgb_front.mp4', 'raw_videos/1mdMz4RkRdA_23-8-rgb_front.mp4', 'raw_videos/dzctDQsw2dk_12-8-rgb_front.mp4']
# 加压失败文件： ['raw_videos/077IIb5uuCs_7-5-rgb_front.mp4', 'raw_videos/1r9BuA9JwsY_0-11-rgb_front.mp4']
# 加压失败文件： ['raw_videos/077IIb5uuCs_7-5-rgb_front.mp4', 'raw_videos/1r9BuA9JwsY_0-11-rgb_front.mp4']
# python filter.py -zip_path ./videos_zip/val_rgb_front_clips.zip -csv_path ./emo_val.csv -output_path ~/autodl-tmp/How2Sign/