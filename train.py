import torch
import os
import yaml
import argparse
from transformers import AutoTokenizer
import numpy as np
import random
import torch.backends.cudnn as cudnn
from dataset import S2TDataset
from torch.utils.data import DataLoader


def get_args_parser():
    a_parser = argparse.ArgumentParser('VLP scripts', add_help=False)
    a_parser.add_argument('--batch-size', default=16, type=int)
    a_parser.add_argument('--epochs', default=10, type=int)
    a_parser.add_argument('--config', type=str, default='./config.yaml')
    a_parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    a_parser.add_argument('--resize', default=256, type=int)
    a_parser.add_argument('--seed', default=0, type=int)
    a_parser.set_defaults(pin_mem=True)
    a_parser.add_argument('--num_workers', default=8, type=int)
    return a_parser


def main(argus, conf):
    # 获取设备
    device = torch.device(args.device)

    # 设置随机种子
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    cudnn.benchmark = False

    # 数据集准备
    print(f"Creating dataset:")
    tokenizer = AutoTokenizer.from_pretrained(config['model']['tokenizer'])

    # 练练数据
    train_data = S2TDataset(path=config['data']['train_label_path'], tokenizer=tokenizer, config=config, args=args,
                            phase='train')
    train_dataloader = DataLoader(train_data,
                                  batch_size=args.batch_size,
                                  num_workers=args.num_workers,
                                  collate_fn=train_data.collate_fn,
                                  pin_memory=args.pin_mem)

    # 验证数据
    dev_data = S2TDataset(path=config['data']['dev_label_path'], tokenizer=tokenizer, config=config, args=args,
                          phase='val')

    dev_dataloader = DataLoader(dev_data,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                collate_fn=dev_data.collate_fn,
                                pin_memory=args.pin_mem)

    # 测试数据
    test_data = S2TDataset(path=config['data']['test_label_path'], tokenizer=tokenizer, config=config, args=args,
                           phase='test')
    test_dataloader = DataLoader(test_data,
                                 batch_size=args.batch_size,
                                 num_workers=args.num_workers,
                                 collate_fn=test_data.collate_fn,
                                 pin_memory=args.pin_mem)

    model = gloss_free_model(config, args)
    model.to(device)
    
    pass


if __name__ == '__main__':
    # 禁用分词器的并行处理
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    parser = argparse.ArgumentParser('VLP scripts', parents=[get_args_parser()])
    args = parser.parse_args()
    print(args)

    with open(args.config, 'r+', encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    main(vars(args), config)
