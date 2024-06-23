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
# How2SignDataset、P14TDataset、CSLDailyDataset有用 动态加载
from dataset import How2SignDataset
from dataset import P14TDataset
from dataset import CSLDailyDataset
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler as scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler
from definition import *
import utils
from sacrebleu.metrics import BLEU
import multiprocessing
from colorama import init, Back
import metrics
import torch.nn.functional as F


def get_args_parser():
    a_parser = argparse.ArgumentParser('VLP scripts', add_help=False)
    a_parser.add_argument('--batch_size', default=1, type=int)
    a_parser.add_argument('--epochs', default=20, type=int)

    a_parser.add_argument('--config', type=str, default='./config.yaml')
    a_parser.add_argument('--device', default='cuda')
    # a_parser.add_argument('--device', default='cuda')
    a_parser.add_argument('--resize', default=256, type=int)
    a_parser.add_argument('--seed', default=0, type=int)
    a_parser.add_argument('--pin_mem', action='store_true', default=True)
    a_parser.add_argument('--num_workers', default=4, type=int)
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

    a_parser.add_argument('--finetune', default=True, type=bool)

    a_parser.add_argument('--need_keypoints', default=True, type=bool)

    a_parser.add_argument('--lambda', type=float, default=0.1, metavar='RATE')

    a_parser.add_argument('--dataset', default='How2SignDataset', type=str,
                          choices=['How2SignDataset', 'P14TDataset', 'CSLDailyDataset'])
    # a_parser.add_argument('--language', default='ch', type=str,
    # choices=['en', 'de', 'ch'])
    return a_parser


def main(args_, config):
    # 转成字典方便操作
    args = vars(args_)
    # 获取设备
    device = torch.device(args['device'])
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("starting on...", device, sep=' ')

    # 设置随机种子
    seed = args['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # 禁用CUDA自动调优，使实验结果一致
    # cudnn.benchmark = False

    # 加载分词器
    tokenizer = MBartTokenizer.from_pretrained("facebook/mbart-large-cc25")
    lang = {
        'How2SignDataset': 'en_XX',
        'P14TDataset': 'de_DE',
        'CSLDailyDataset': 'zh_CN'
    }
    tokenizer.src_lang = lang[args['dataset']]

    # 加载训练数据集
    # 训练数据集
    train_data = eval(args['dataset'])(path=config[args['dataset']]['train_label_path'],
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
    val_data = eval(args['dataset'])(path=config[args['dataset']]['dev_label_path'],
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
    test_data = eval(args['dataset'])(path=config[args['dataset']]['test_label_path'],
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

    # SLT Model
    slt_model = SLT(config=config)
    slt_model.to(device)

    # VLP阶段权重加载
    if args['finetune']:
        try:
            print("加载VLP模型权重...")
            # 加载模型1的检查点
            checkpoint = torch.load(config['model']['vlp_cps'])

            # 获取模型1的权重参数
            clip_model_state_dict = checkpoint['clip_train_dict']['clip_model']
            txt_decoder_state_dict = checkpoint['td_train_dict']['txt_decoder']

            # 将模型1的权重参数加载到模型2中
            # 严格模式设置为False以允许不匹配的参数
            slt_model.load_state_dict(clip_model_state_dict, strict=False)
            slt_model.load_state_dict(txt_decoder_state_dict, strict=False)
        except IOError as e:
            print("模型文件不存在或者加载失败...")
        else:
            print("VLP模型权重应用到SLT...，加载完成.")

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
    # criterion = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX, label_smoothing=0.2)
    criterion = dict(
        loss_vocab=torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX,
                                             label_smoothing=0.2),
        loss_emo=torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    )
    loss_scaler = NativeScaler()

    # 数据增强、内存优化、日志优化 Maybe->TODO

    # 开始训练
    print(f"开始训练，共训练{Back.GREEN} {args['epochs']} {Back.RESET}轮.")

    # 优化指标
    max_accuracy = 0.0

    for epoch in range(args['epochs']):
        # 在需要释放内存的地方调用
        torch.cuda.empty_cache()

        # 训练一个epoch
        train_stats = train_one_epoch(args, epoch,
                                      train_dataloader,
                                      slt_train_dict,
                                      criterion, loss_scaler)

        print(
            f"SLT阶段，在训练集上："
            f"avg_vocab_emo_loss={train_stats['avg_vocab_emo_loss']}")
        utils.log('slt_train', epoch=epoch + 1,
                  avg_vocab_emo_loss=train_stats['avg_vocab_emo_loss'])

        # 清理CUDA缓存
        utils.clear_cuda_cache()

        # 评估一个epoch
        val_stats = evaluate_one_epoch(args, epoch,
                                       val_dataloader,
                                       slt_train_dict,
                                       criterion,
                                       tokenizer)
        print(
            f"SLT阶段，在验证集上："
            f"avg_vocab_emo_loss={val_stats['avg_vocab_emo_loss']},"
            f"emo_accuracy={val_stats['emo_accuracy']},"
            f"vocab_bleu_s={val_stats['vocab_bleu_s']},"
            f"integrated_score={val_stats['integrated_score']},"
        )
        utils.log('slt_val', epoch=epoch + 1,
                  avg_vocab_emo_loss=val_stats['avg_vocab_emo_loss'],
                  emo_accuracy=val_stats['emo_accuracy'],
                  vocab_bleu_s=val_stats['vocab_bleu_s'],
                  integrated_score=val_stats['integrated_score']
                  )

        # 清理CUDA缓存
        utils.clear_cuda_cache()

        if max_accuracy < val_stats["integrated_score"]:
            max_accuracy = val_stats["integrated_score"]
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
                }, args=args, filename=f"slt_checkpoint.pth.tar")
            print(f"vocab_bleu: {Back.GREEN}"
                  f"{val_stats['vocab_bleu']}"
                  f"{Back.RESET}")
        else:
            print(f"在val数据集上无提升，对于第{epoch + 1}轮. ")

        print(f'当前最优{Back.GREEN} Blue-4分数: {max_accuracy:.2f}%{Back.RESET}')

    # 测试集评估
    test_stats = evaluate_one_epoch(args, epoch=-1,
                                    dataloader=test_dataloader,
                                    slt_train_dict=slt_train_dict,
                                    criterion=criterion,
                                    tokenizer=tokenizer)
    print(
        f"SLT阶段，在测试集上："
        f"avg_vocab_emo_loss={test_stats['avg_vocab_emo_loss']},"
        f"emo_accuracy={test_stats['emo_accuracy']},"
        f"vocab_bleu_s={test_stats['vocab_bleu_s']},"
        f"integrated_score={test_stats['integrated_score']},"
    )
    utils.log('slt_test',
              avg_vocab_emo_loss=test_stats['avg_vocab_emo_loss'],
              emo_accuracy=test_stats['emo_accuracy'],
              vocab_bleu_s=test_stats['vocab_bleu_s'],
              integrated_score=test_stats['integrated_score']
              )


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
        print(f"Epoch {epoch + 1} train, Step {step + 1}...")
        # 解码器损失权重分配
        vocab_weight = (len(tgt_input['input_ids']) - 1) / len(tgt_input['input_ids']) - 1
        emo_weight = 1 / len(tgt_input['input_ids'])
        masked_lm_loss_weight = torch.tensor([vocab_weight, emo_weight], device=args['device'])

        vocab_logits, emo_logits = slt_train_dict['slt_model'](src_input, tgt_input)

        loss_lambda = torch.tensor(args['loss_lambda'], device=args['device'])
        # loss_lambda = torch.tensor(args['loss_lambda'])
        vocab_lm_loss = criterion['loss_vocab'](vocab_logits.reshape(-1, vocab_logits.shape[-1]),
                                                tgt_input['input_ids'][:, 1:].cuda().reshape(-1)) * loss_lambda
        emo_lm_loss = criterion['loss_emo'](emo_logits, tgt_input['input_ids'][:, 0].cuda().reshape(-1)) * loss_lambda

        # vocab_emo_loss = (vocab_lm_loss + emo_masked_lm_loss) / 2

        print(
            f"{Back.GREEN}"
            f"Training - Epoch: {epoch + 1}, vocab_lm_loss: {vocab_lm_loss}, "
            f"emo_lm_loss: {emo_lm_loss}"
            f"{Back.RESET}")

        vocab_emo_loss = torch.stack([vocab_lm_loss, emo_lm_loss])
        vocab_emo_loss = torch.mean(vocab_emo_loss * masked_lm_loss_weight)
        # 梯度清零 梯度回传 更新梯度
        slt_train_dict['optimizer'].zero_grad()
        # 使用loss_scaler 的__call__方法进行损失的缩放和梯度更新
        loss_scaler(vocab_emo_loss, slt_train_dict['optimizer'])
        vocab_emo_losses.append(vocab_emo_loss.item())

        # 梯度爆炸
        if not math.isfinite(vocab_emo_loss.item()):
            print("CLIP Loss: {}, 结束训练".format(vocab_emo_loss.item()))
            sys.exit(1)

    # 更新学习率
    slt_train_dict['lr_scheduler'].step(epoch)

    avg_vocab_emo_loss = sum(vocab_emo_losses) / len(vocab_emo_losses) if vocab_emo_losses else 0

    # 用于返回的状态字典
    train_stats = {'avg_vocab_emo_loss': avg_vocab_emo_loss}
    return train_stats


# 评估一个epoch
def evaluate_one_epoch(args, epoch,
                       dataloader,
                       slt_train_dict,
                       criterion,
                       tokenizer):
    # -1 代表在测试数据集上
    if epoch >= 0:
        print(f"Epoch {epoch + 1} val...")

    # 设置模型为评估模式
    slt_train_dict['slt_model'].eval()

    with (torch.no_grad()):
        # 情感部分 生成情感 参考情感
        emo_pres = []
        emo_refs = []

        # 翻译部分 生成序列 参考序列
        tgt_pres = []
        tgt_refs = []

        # 整体损失
        vocab_emo_losses = []

        for step, (src_input, tgt_input) in enumerate(dataloader):
            # -1 代表在测试数据集上
            if epoch >= 0:
                print(f"Epoch {epoch + 1} val, Step {step}...")
            else:
                print(f"Step {step}...")

            # 解码器损失权重分配
            vocab_weight = (len(tgt_input['input_ids']) - 1) / len(tgt_input['input_ids']) - 1
            emo_weight = 1 / len(tgt_input['input_ids'])
            masked_lm_loss_weight = torch.tensor([vocab_weight, emo_weight], device=args['device'])

            # 计算损失
            vocab_logits, emo_logits = slt_train_dict['slt_model'](src_input, tgt_input)

            loss_lambda = torch.tensor(args['loss_lambda'], device=args['device'])
            # loss_lambda = torch.tensor(args['loss_lambda'])
            vocab_lm_loss = criterion['loss_vocab'](vocab_logits.reshape(-1, vocab_logits.shape[-1]),
                                                    tgt_input['input_ids'][:, 1:].cuda().reshape(-1)) * loss_lambda
            emo_lm_loss = criterion['loss_emo'](emo_logits,
                                                tgt_input['input_ids'][:, 0].cuda().reshape(-1)) * loss_lambda

            # vocab_emo_loss = (vocab_lm_loss + emo_masked_lm_loss) / 2

            print(
                f"{Back.GREEN}"
                f"Evaluation - Epoch: {epoch + 1}, vocab_lm_loss: {vocab_lm_loss}, "
                f"emo_lm_loss: {emo_lm_loss}"
                f"{Back.RESET}")

            vocab_emo_loss = torch.stack([vocab_lm_loss, emo_lm_loss])
            vocab_emo_loss = torch.mean(vocab_emo_loss * masked_lm_loss_weight)
            vocab_emo_losses.append(vocab_emo_loss.item())

            avg_vocab_emo_loss = sum(vocab_emo_losses) / len(vocab_emo_losses) if vocab_emo_losses else 0

            # 情感准确率计算数据准备
            # 使用 tokenizer 解码每个样本
            one_batch_emo_pres = utils.batch_decode(torch.argmax(vocab_logits[:, 0, :], dim=-1))
            one_batch_emo_refs = utils.batch_decode(tgt_input['input_ids'][:, 0])

            print(f"Epoch {epoch + 1} val, Step {step}, one_batch_emo_pres: {one_batch_emo_pres}")
            print(f"Epoch {epoch + 1} val, Step {step}, one_batch_emo_refs: {one_batch_emo_refs}")
            emo_pres.extend(one_batch_emo_pres)
            emo_refs.extend(one_batch_emo_refs)

            # BLEU分数计算数据准备
            # 使用 tokenizer 解码每个样本
            # one_batch_tgt_pres = tokenizer.batch_decode(torch.argmax(vocab_logits[:, 1:, :], dim=-1),
            #                                             skip_special_tokens=True)
            # 应用 Softmax 获取概率分布
            probabilities = F.softmax(vocab_logits[:, 1:, :], dim=-1)
            # 获取最大概率对应的 token IDs
            predicted_ids = torch.argmax(probabilities, dim=-1)
            print('predicted_ids: ', predicted_ids)
            one_batch_tgt_pres = tokenizer.batch_decode(predicted_ids)
            one_batch_tgt_refs = tokenizer.batch_decode(tgt_input['input_ids'][:, 1:],
                                                        skip_special_tokens=True)

            print(f"Epoch {epoch + 1} val, Step {step}, one_batch_tgt_pres: {one_batch_tgt_pres}")
            print(f"Epoch {epoch + 1} val, Step {step}, one_batch_tgt_refs: {one_batch_tgt_refs}")

            tgt_pres.extend(one_batch_tgt_pres)
            tgt_refs.extend(one_batch_tgt_refs)

        # 情感评估指标计算
        emo_accuracy = metrics.cal_emo_accuracy(emo_pres, emo_refs)

        # 翻译评测指标计算
        bleu = BLEU()
        vocab_bleu = bleu.corpus_score(tgt_pres, [tgt_refs])
        vocab_bleu_s = vocab_bleu.score

        integrated_score = emo_accuracy * args['lambda'] + vocab_bleu_s * (1 - args['lambda'])

    val_stats = {
        'avg_vocab_emo_loss': avg_vocab_emo_loss,
        'emo_accuracy': emo_accuracy,
        'vocab_bleu_s': vocab_bleu_s,
        'integrated_score': integrated_score,
        'vocab_bleu': vocab_bleu
    }

    return val_stats


if __name__ == '__main__':
    # 禁用分词器的并行处理
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # 设置进程启动方法为 'spawn'
    multiprocessing.set_start_method('spawn', force=True)

    # 初始化 colorama
    init()

    # 加载参数
    parser = argparse.ArgumentParser('VLP scripts', parents=[get_args_parser()])
    args = parser.parse_args()
    with open(args.config, 'r+', encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # 创建输出文件夹
    if args.checkpoints_dir:
        Path(args.checkpoints_dir).mkdir(parents=True, exist_ok=True)

    # 清理CUDA缓存
    utils.clear_cuda_cache()

    # 开始训练
    main(args, config)
