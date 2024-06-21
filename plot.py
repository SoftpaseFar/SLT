# 绘制论文图表
from typing import List, Dict
import matplotlib.pyplot as plt
import os

# 图表数据路径
vlp_data_path = './log/vlp.txt'
slt_data_path = './log/slt.txt'

# 保存图片的路径
imgs_save_path = './log/show/'

# 横坐标间隔
interval = 4


# 解析数据
def parse_log(phase: str, file_path: str) -> Dict[str, List[Dict[str, float]]]:
    # Initialize the containers for the different types
    res_data = None
    if phase == 'vlp':
        res_data = {
            'vlp_train': [],
            'vlp_val': [],
            'vlp_test': []
        }
    elif phase == 'slt':
        res_data = {
            'slt_train': [],
            'slt_val': [],
            'slt_test': []
        }
    else:
        print('phase参数不正确...')
        exit(1)

    # Open the file and parse each line
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split('|')
            # Skip any malformed lines
            if (phase == 'vlp' and len(parts) < 5) or (phase == 'slt' and len(parts) < 4):
                continue

            # Extract the type from the second part
            log_type = parts[1]
            # Skip unknown types
            if log_type not in res_data:
                continue

            # Extract the key-value pairs
            log_entry = {
                'timestamp': parts[0].strip('[]')
            }
            for part in parts[2:]:
                key, value = part.split('=')
                log_entry[key] = float(value) if '.' in value else int(value)

            # Append the dictionary to the appropriate list
            res_data[log_type].append(log_entry)

    return res_data


# 绘制vlp图表
def plot_vlp(title, data: List[Dict[str, float]]):
    epochs = [entry['epoch'] for entry in data]
    clip_loss = [entry['clip_loss'] for entry in data]
    tdm_loss = [entry['tdm_loss'] for entry in data]
    train_loss, val_loss = None, None
    if 'train' in title:
        train_loss = [entry['train_loss'] for entry in data]
    elif 'dev' in title:
        val_loss = [entry['val_loss'] for entry in data]

    plt.figure(figsize=(10, 6))

    plt.plot(epochs, clip_loss, label='clip_loss', color='blue', linewidth=2.5)
    plt.plot(epochs, tdm_loss, label='tdm_loss', color='orange', linewidth=2.5)
    if 'train' in title:
        plt.plot(epochs, train_loss, label='train_loss', color='green', linewidth=2.5)
    elif 'dev' in title:
        plt.plot(epochs, val_loss, label='val_loss', color='green', linewidth=2.5)

    plt.xlabel('#Epoch', fontsize=18)
    plt.ylabel('Loss', fontsize=18)
    plt.title(title, fontsize=18)
    plt.legend()
    plt.grid(False)

    # 设置x轴为整数和固定间隔
    epoch_nums = len(epochs)
    plt.xticks(range(0, epoch_nums + 1, interval))

    # 保存图表
    plt.savefig(os.path.join(imgs_save_path, f'{title}.png'))

    # 图表展示
    # plt.show()


# 绘制slt图表
def plot_slt(title, slt_train: List[Dict[str, float]], slt_val: List[Dict[str, float]]):
    # Extracting data for the first plot
    epochs_train = [entry['epoch'] for entry in slt_train]
    avg_vocab_emo_loss_train = [entry['avg_vocab_emo_loss'] for entry in slt_train]

    epochs_val = [entry['epoch'] for entry in slt_val]
    avg_vocab_emo_loss_val = [entry['avg_vocab_emo_loss'] for entry in slt_val]

    epoch_nums = len(epochs_train)

    # Plotting avg_vocab_emo_loss for slt_train and slt_val
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_train, avg_vocab_emo_loss_train, label='slt_train avg_vocab_emo_loss', color='blue', linewidth=2.5)
    plt.plot(epochs_val, avg_vocab_emo_loss_val, label='slt_val avg_vocab_emo_loss', color='orange', linewidth=2.5)
    plt.xlabel('#Epochs', fontsize=18)
    plt.ylabel('avg_vocab_emo_loss', fontsize=18)
    plt.title(title[0], fontsize=18)
    plt.legend()
    plt.grid(False)

    # 设置x轴为整数和固定间隔
    plt.xticks(range(0, epoch_nums + 1, interval))

    # 保存图表
    plt.savefig(os.path.join(imgs_save_path, f'{title[0]}.png'))

    # 图表展示
    # plt.show()

    # Extracting data for the second, third, and fourth plots
    vocab_bleu_s = [entry['vocab_bleu_s'] for entry in slt_val]
    emo_accuracy = [entry['emo_accuracy'] for entry in slt_val]
    integrated_score = [entry['integrated_score'] for entry in slt_val]

    # Plotting vocab_bleu_s for slt_val
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_val, vocab_bleu_s, label='vocab_bleu_s', color='green', linewidth=2.5)
    plt.xlabel('#Epochs', fontsize=18)
    plt.ylabel('vocab_bleu_s', fontsize=18)
    plt.title(title[1], fontsize=18)
    plt.legend()
    plt.grid(False)

    # 设置x轴为整数和固定间隔
    plt.xticks(range(0, epoch_nums + 1, interval))

    # 保存图表
    plt.savefig(os.path.join(imgs_save_path, f'{title[1]}.png'))

    # 图表展示
    # plt.show()

    # Plotting emo_accuracy for slt_val
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_val, emo_accuracy, label='emo_accuracy', color='red', linewidth=2.5)
    plt.xlabel('#Epochs', fontsize=18)
    plt.ylabel('emo_accuracy', fontsize=18)
    plt.title(title[2], fontsize=18)
    plt.legend()
    plt.grid(False)

    # 设置x轴为整数和固定间隔
    plt.xticks(range(0, epoch_nums + 1, interval))

    # 保存图表
    plt.savefig(os.path.join(imgs_save_path, f'{title[2]}.png'))

    # 图表展示
    # plt.show()

    # Plotting integrated_score for slt_val
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_val, integrated_score, label='integrated_score', color='purple', linewidth=2.5)
    plt.xlabel('#Epochs', fontsize=18)
    plt.ylabel('integrated_score', fontsize=18)
    plt.title(title[3], fontsize=18)
    plt.legend()
    plt.grid(False)

    # 设置x轴为整数和固定间隔
    plt.xticks(range(0, epoch_nums + 1, interval))

    # 保存图表
    plt.savefig(os.path.join(imgs_save_path, f'{title[3]}.png'))

    # 图表展示
    # plt.show()


def show_example_for_vlp():
    # Example usage
    vlp_data = parse_log('vlp', vlp_data_path)

    # Accessing the values
    vlp_train = vlp_data['vlp_train']
    vlp_val = vlp_data['vlp_val']
    vlp_test = vlp_data.get('vlp_test', [])
    print('vlp_train: ', vlp_train)
    print('vlp_val: ', vlp_val)
    print('vlp_test: ', vlp_test)

    # plt
    plot_vlp('vlp_train', vlp_train)
    plot_vlp('vlp_dev', vlp_val)
    print('vlp, 绘制并保存成功.')


def show_example_for_slt():
    # Example usage
    slt_data = parse_log('slt', slt_data_path)

    # Accessing the values
    slt_train = slt_data['slt_train']
    slt_val = slt_data['slt_val']
    slt_test = slt_data.get('slt_test', [])
    print('slt_train: ', slt_train)
    print('slt_val: ', slt_val)
    print('slt_test: ', slt_test)

    # plt
    title = ['avg_vocab_emo_loss', 'vocab_bleu_s', 'emo_accuracy', 'integrated_score']
    plot_slt(title, slt_train, slt_val)
    print('slt, 绘制并保存成功.')


if __name__ == '__main__':
    show_example_for_vlp()
    show_example_for_slt()
