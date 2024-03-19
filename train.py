import torch
import os
import yaml
import argparse
from transformers import MBartForConditionalGeneration, MBartTokenizer,MBartConfig


def get_args_parser():
    a_parser = argparse.ArgumentParser('VLP scripts', add_help=False)
    a_parser.add_argument('--batch-size', default=16, type=int)
    a_parser.add_argument('--epochs', default=10, type=int)
    a_parser.add_argument('--config', type=str, default='./config.yaml')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    return a_parser


def main(argus, conf):
    # 获取设备
    device = torch.device(args.device)

    # 数据集准备
    print(f"Creating dataset:")
    tokenizer = MBartTokenizer.from_pretrained(config['model']['tokenizer'])
    # train_data = S2T_Dataset(path=config['data']['train_label_path'], tokenizer=tokenizer, config=config, args=args,
    #                          phase='train')
    # print(train_data)


if __name__ == '__main__':
    # 禁用分词器的并行处理
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    parser = argparse.ArgumentParser('VLP scripts', parents=[get_args_parser()])
    args = parser.parse_args()
    print(args)

    with open(args.config, 'r+', encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    main(vars(args), config)
