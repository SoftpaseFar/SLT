import os
import torch
import yaml
import argparse
from pathlib import Path
from transformers import MBartTokenizer
import numpy as np
import random
from model import TextCLIP, TextDecoder
from dataset import How2SignDataset
from torch.utils.data import DataLoader


def get_args_parser():
    a_parser = argparse.ArgumentParser('VLP scripts', add_help=False)
    a_parser.add_argument('--batch_size', default=3, type=int)
    a_parser.add_argument('--epochs', default=10, type=int)

    a_parser.add_argument('--config', type=str, default='./config.yaml')
    a_parser.add_argument('--device', default='cpu')
    # a_parser.add_argument('--device', default='cuda')
    a_parser.add_argument('--resize', default=256, type=int)
    a_parser.add_argument('--seed', default=0, type=int)
    a_parser.add_argument('--pin_mem', action='store_true', default=True)
    a_parser.add_argument('--num_workers', default=8, type=int)
    # a_parser.add_argument('--num_workers', default=2, type=int)
    a_parser.add_argument('--output_dir', default='./output/test/o_1')
    a_parser.add_argument('--input_size', default=224, type=int)

    a_parser.add_argument('--training_refurbish', default=True, type=bool)
    a_parser.add_argument('--noise_rate', default=0.15, type=float)
    a_parser.add_argument('--random_shuffle', default=False, type=bool)
    return a_parser


def main(args, config):
    # 获取设备
    device = torch.device(args['device'])
    print("starting on...", device, sep=' ')

    # 设置随机种子
    seed = args['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # 禁用CUDA自动调优，使实验结果一致
    # cudnn.benchmark = False

    # 加载分词器
    tokenizer = MBartTokenizer.from_pretrained(config['model']['tokenizer'])

    # 加载训练数据集
    # 训练数据集
    train_data = How2SignDataset(path=config['data']['train_label_path'],
                                 tokenizer=tokenizer,
                                 config=config,
                                 args=args,
                                 phase='train',
                                 training_refurbish=True)
    train_dataloader = DataLoader(train_data,
                                  batch_size=args['batch_size'],
                                  num_workers=args['num_workers'],
                                  collate_fn=train_data.collate_fn,
                                  pin_memory=args['pin_mem'],
                                  drop_last=True)

    # 验证数据集
    val_data = How2SignDataset(path=config['data']['val_label_path'],
                               tokenizer=tokenizer,
                               config=config,
                               args=args,
                               phase='val',
                               training_refurbish=False)
    train_dataloader = DataLoader(val_data,
                                  batch_size=args['batch_size'],
                                  num_workers=args['num_workers'],
                                  collate_fn=val_data.collate_fn,
                                  pin_memory=args['pin_mem'],
                                  drop_last=True)

    # 测试数据集
    test_data = How2SignDataset(path=config['data']['test_label_path'],
                                tokenizer=tokenizer,
                                config=config,
                                args=args,
                                phase='test',
                                training_refurbish=False)
    test_dataloader = DataLoader(test_data,
                                 batch_size=args['batch_size'],
                                 num_workers=args['num_workers'],
                                 collate_fn=test_data.collate_fn,
                                 pin_memory=args['pin_mem'],
                                 drop_last=True)

    # 测试代码
    # 手动调用collate_fn函数
    train_batch = [train_data[i] for i in range(args['batch_size'])]  # 获取一个batch的数据
    src_input, tgt_input, masked_tgt_input = train_data.collate_fn(train_batch)  # 调用collate_fn函数

    # 模型推导
    txt_encoder = TextCLIP(config=config)
    txt_encoder.to(device)
    le_head, le_logits = txt_encoder(tgt_input)
    print(le_head.shape, le_logits.shape)
    print("==================================")
    txt_decoder = TextDecoder(config=config)
    txt_decoder.to(device)
    ld_logits = txt_decoder(tgt_input, masked_tgt_input, txt_encoder)
    print(ld_logits.shape)


if __name__ == '__main__':
    # 禁用分词器的并行处理
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # 加载参数
    parser = argparse.ArgumentParser('VLP scripts', parents=[get_args_parser()])
    args = parser.parse_args()
    with open(args.config, 'r+', encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # 创建输出文件夹
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # 开始训练
    main(vars(args), config)
