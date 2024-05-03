import os
import math
import sys
import torch
import yaml
import argparse
from pathlib import Path
from transformers import MBartTokenizer
import numpy as np
import random
from model import SLT
from dataset import How2SignDataset
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler as scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler
from definition import *
import utils
from sacrebleu.metrics import BLEU


def get_args_parser():
    a_parser = argparse.ArgumentParser('VLP scripts', add_help=False)
    a_parser.add_argument('--batch_size', default=3, type=int)
    a_parser.add_argument('--epochs', default=6, type=int)

    a_parser.add_argument('--config', type=str, default='./config.yaml')
    a_parser.add_argument('--device', default='cpu')
    # a_parser.add_argument('--device', default='cuda')
    a_parser.add_argument('--resize', default=256, type=int)
    a_parser.add_argument('--seed', default=0, type=int)
    a_parser.add_argument('--pin_mem', action='store_true', default=True)
    a_parser.add_argument('--num_workers', default=8, type=int)
    # a_parser.add_argument('--num_workers', default=2, type=int)
    a_parser.add_argument('--checkpoints_dir', default='./checkpoints/')
    a_parser.add_argument('--log_dir', default='./log/')
    a_parser.add_argument('--input_size', default=224, type=int)

    a_parser.add_argument('--training_refurbish', default=True, type=bool)
    a_parser.add_argument('--noise_rate', default=0.15, type=float)
    a_parser.add_argument('--random_shuffle', default=False, type=bool)
    a_parser.add_argument('--loss_lambda', type=float, default=1.0, metavar='RATE')

    # * Optimize参数
    a_parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER')
    a_parser.add_argument('--opt-eps', default=1.0e-09, type=float, metavar='EPSILON')
    a_parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA')
    a_parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM')
    a_parser.add_argument('--momentum', type=float, default=0.9, metavar='M')
    a_parser.add_argument('--weight-decay', type=float, default=0.0)

    # * Learning rate 参数
    a_parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER')
    a_parser.add_argument('--lr', type=float, default=1.0e-3, metavar='LR')
    a_parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct')
    a_parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT')
    a_parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV')
    a_parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR')
    a_parser.add_argument('--min-lr', type=float, default=1.0e-08, metavar='LR')
    a_parser.add_argument('--decay-epochs', type=float, default=30, metavar='N')
    a_parser.add_argument('--warmup-epochs', type=int, default=0, metavar='N')
    a_parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N')
    a_parser.add_argument('--patience-epochs', type=int, default=10, metavar='N')
    a_parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE')

    a_parser.add_argument('--save_interval', default=1, type=int)
    a_parser.add_argument('--patience', default=10, type=int)
    a_parser.add_argument('--save_model', default=True, type=bool)

    a_parser.add_argument('--finetune', default=False, type=bool)
    a_parser.add_argument('--eval', default=False, type=bool)
    return a_parser


def main(args_, config):
    # 转成字典方便操作
    args = vars(args_)
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
    tokenizer = MBartTokenizer.from_pretrained("facebook/mbart-large-cc25", vocab_size=2454)

    # 加载训练数据集
    # 训练数据集
    train_data = How2SignDataset(path=config['data']['train_label_path'],
                                 tokenizer=tokenizer,
                                 config=config,
                                 args=args,
                                 phase='train',
                                 training_refurbish=False)
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
    val_dataloader = DataLoader(val_data,
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

    # model

    # 模型微调,参数加载 TODO
    if args['finetune']:
        pass

    # SLT
    slt_model = SLT(config=config)
    slt_model.to(device)

    # 优化器 学习率调度器
    optimizer = create_optimizer(args_, slt_model)
    lr_scheduler = scheduler.CosineAnnealingLR(
        optimizer=optimizer,
        eta_min=1e-8,
        T_max=args['epochs'],
    )
    slt_train_dict = dict(
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        slt_model=slt_model
    )

    # 损失函数 loss缩放器
    criterion = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX, label_smoothing=0.2)
    loss_scaler = NativeScaler()

    # 数据增强 TODO

    # 评估模式 TODO
    if args['eval']:
        pass

    # 开始训练
    print(f"开始训练，共训练 {args['epochs']} 轮.")

    # 优化指标
    max_accuracy = 0.0

    for epoch in range(args['epochs']):
        # 训练一个epoch
        train_stats = train_one_epoch(args, epoch,
                                      train_dataloader,
                                      slt_train_dict,
                                      criterion, loss_scaler)
        print(f"Training - Epoch: {epoch + 1}, Vocab Emo Loss: {train_stats['vocab_emo_loss']}")

        # 评估一个epoch
        val_stats = evaluate_one_epoch(args, epoch,
                                       val_dataloader,
                                       slt_train_dict,
                                       criterion,
                                       tokenizer)
        print(f"Evaluation - Epoch: {epoch + 1}, total_loss: {val_stats['total_loss']}，bleu_s： {val_stats['bleu_s']}")

        if max_accuracy < val_stats["bleu_s"]:
            max_accuracy = val_stats["bleu_s"]
            # 保存模型
            if args['save_model'] and epoch % args['save_interval'] == 0:
                utils.save_checkpoint(state={
                    'epoch': epoch + 1,
                    'slt_train_dict': dict(
                        optimizer=slt_train_dict['optimizer'].state_dict(),
                        lr_scheduler=slt_train_dict['lr_scheduler'].state_dict(),
                        slt_model=slt_train_dict['slt_model'].state_dict()
                    ),
                    'train_stats': train_stats,
                    'val_stats': val_stats,
                    'max_accuracy': max_accuracy
                }, args=args, filename=f"slt_checkpoint_{epoch + 1}.pth.tar")

        print(f'当前最优 Blue-4分数: {max_accuracy:.2f}%')

        # 其他逻辑 TODO
        print("其他逻辑...")


# 训练一个epoch
def train_one_epoch(args, epoch,
                    dataloader,
                    slt_train_dict,
                    criterion, loss_scaler: NativeScaler()):
    print(f"Epoch {epoch + 1} train...")

    vocab_emo_losses = []

    # 开启训练模式
    slt_train_dict['slt_model'].train(True)

    for step, (src_input, tgt_input) in enumerate(dataloader):
        print(f"Epoch {epoch + 1} train, Step {step}...")
        vocab_logits, emo_logits = slt_train_dict['slt_model'](src_input, tgt_input)

        vocab_masked_lm_loss = criterion(vocab_logits.view(-1, vocab_logits.shape[-1]),
                                         tgt_input['input_ids'][:, 1:, :].view(-1)) * args['loss_lambda']
        emo_masked_lm_loss = criterion(emo_logits, tgt_input['input_ids'][:, 0, :].view(-1)) * args[
            'loss_lambda']

        vocab_emo_loss = (vocab_masked_lm_loss + emo_masked_lm_loss) / 2
        # 梯度清零 梯度回传 更新梯度
        slt_train_dict['optimizer'].zero_grad()
        loss_scaler.scale(vocab_emo_loss).backward()
        loss_scaler.step(vocab_emo_loss['optimizer'])
        loss_scaler.update()
        vocab_emo_losses.append(vocab_emo_loss.item())

        # 梯度爆炸
        if not math.isfinite(vocab_emo_loss.item()):
            print("CLIP Loss: {}, 结束训练".format(vocab_emo_loss.item()))
            sys.exit(1)

    # 更新学习率
    slt_train_dict['lr_scheduler'].step()

    avg_vocab_emo_loss = sum(vocab_emo_losses) / len(vocab_emo_losses) if vocab_emo_losses else 0

    # 用于返回的状态字典
    train_stats = {'vocab_emo_loss': avg_vocab_emo_loss}
    return train_stats


# 评估一个epoch
def evaluate_one_epoch(args, epoch,
                       dataloader,
                       slt_train_dict,
                       criterion,
                       tokenizer):
    print(f"Epoch {epoch + 1} val...")
    # 设置模型为评估模式
    slt_train_dict['slt_model'].eval()

    with (torch.no_grad()):
        # 生成序列 参考序列
        tgt_pres = []
        tgt_refs = []

        # 整体损失
        total_loss = 0.0

        for step, (src_input, tgt_input) in enumerate(dataloader):
            print(f"Epoch {epoch + 1} val, Step {step}...")
            # 计算损失

            vocab_logits, emo_logits = slt_train_dict['slt_model'](src_input, tgt_input)

            vocab_masked_lm_loss = criterion(vocab_logits[:, 1:, :].view(-1, vocab_logits.shape[-1]),
                                             tgt_input['input_ids'][:, 1:, :].view(-1)) * args['loss_lambda']
            emo_masked_lm_loss = criterion(emo_logits, tgt_input['input_ids'][:, 0, :].view(-1)) * args[
                'loss_lambda']

            vocab_emo_loss = (vocab_masked_lm_loss + emo_masked_lm_loss) / 2
            total_loss += vocab_emo_loss
            # 使用 tokenizer 解码每个样本
            one_batch_tgt_pres = tokenizer.batch_decode(torch.argmax(vocab_logits[:, 1:, :], dim=-1),
                                                        skip_special_tokens=True)
            one_batch_tgt_refs = tokenizer.batch_decode(tgt_input['input_ids'][:, 1:],
                                                        skip_special_tokens=True)

            print(f"Epoch {epoch + 1} val, Step {step}, one_batch_tgt_pres: {one_batch_tgt_pres}")
            print(f"Epoch {epoch + 1} val, Step {step}, one_batch_tgt_refs: {one_batch_tgt_refs}")

            tgt_pres.extend(one_batch_tgt_pres)
            tgt_refs.extend(one_batch_tgt_refs)

        # 评测指标计算
        bleu = BLEU()
        bleu_s = bleu.corpus_score(tgt_pres, [tgt_refs]).score
        print(f"Epoch {epoch + 1} val, bleu_s:{bleu_s}")
    val_stats = {
        'total_loss': total_loss,
        'bleu_s': bleu_s
    }

    return val_stats


if __name__ == '__main__':
    # 禁用分词器的并行处理
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # 设置 Hugging Face 模型中心镜像源的地址
    os.environ["HF_HOME"] = "https://mirrors.tuna.tsinghua.edu.cn/hugging-face-models/"
    os.environ['HF_ENDPOINT'] = "https://mirrors.tuna.tsinghua.edu.cn/hugging-face-models/"
    # 加载参数
    parser = argparse.ArgumentParser('VLP scripts', parents=[get_args_parser()])
    args = parser.parse_args()
    with open(args.config, 'r+', encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # 创建输出文件夹
    if args.checkpoints_dir:
        Path(args.checkpoints_dir).mkdir(parents=True, exist_ok=True)

    # 开始训练
    main(args, config)
