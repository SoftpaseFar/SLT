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
            sentence_name = row['SENTENCE_NAME'].strip() + '.mp4'  # 生成文件名
            filenames_to_extract.add(sentence_name)

    print('需要解压文件数量：', len(filenames_to_extract))

    # 打开zip文件
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        # 获取zip中的所有文件
        all_files = zip_ref.namelist()

        # 过滤需要解压的文件
        files_to_extract = [file for file in all_files if file in filenames_to_extract]
        print('文件数量校验：', len(files_to_extract))

        print('正在解压...')
        # 解压需要的文件到指定目录
        for file in files_to_extract:
            print('解压：', file)
            zip_ref.extract(file, output_directory)

    print(f"解压缩完成，{len(files_to_extract)} 个文件已解压到 {output_directory}")


# 判断当前模块是否为主模块
if __name__ == '__main__':
    main()
