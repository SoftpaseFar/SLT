import os
import math
import sys
import torch
import yaml
import random
import loss
import utils
import argparse
import numpy as np
from pathlib import Path
from transformers import MBartTokenizer
from model import CLIP, TextDecoder
# How2SignDataset、P14TDataset、CSLDailyDataset有用 动态加载
from dataset import How2SignDataset
from dataset import P14TDataset
from dataset import CSLDailyDataset
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler as scheduler
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
from timm.optim import AdamW
from timm.utils import NativeScaler
from loss import KLLoss
from definition import *
import multiprocessing
from colorama import init, Back
import torch.nn.functional as f


def get_args_parser():
    a_parser = argparse.ArgumentParser('VLP scripts', add_help=False)
    a_parser.add_argument('--batch_size', default=2, type=int)
    a_parser.add_argument('--epochs', default=20, type=int)

    a_parser.add_argument('--config', type=str, default='./config.yaml')
    a_parser.add_argument('--device', default='cuda')
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

    a_parser.add_argument('--need_keypoints', default=True, type=bool)

    a_parser.add_argument('--dataset', default='How2SignDataset', type=str,
                          choices=['How2SignDataset', 'P14TDataset', 'CSLDailyDataset'])
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
    tokenizer = MBartTokenizer.from_pretrained("facebook/mbart-large-cc25")

    # 加载训练数据集
    # 训练数据集
    train_data = eval(args['dataset'])(path=config[args['dataset']]['train_label_path'],
                                       tokenizer=tokenizer,
                                       config=config,
                                       args=args,
                                       phase='train',
                                       training_refurbish=True)

    # # 测试代码
    # print(train_data[0])
    # return 666
    #
    # # 测试结束

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
                                     training_refurbish=True)

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
                                      training_refurbish=True)
    test_dataloader = DataLoader(test_data,
                                 batch_size=args['batch_size'],
                                 num_workers=args['num_workers'],
                                 collate_fn=test_data.collate_fn,
                                 pin_memory=args['pin_mem'],
                                 drop_last=True)

    # CLIP Model
    clip_model = CLIP(config=config)
    clip_model.to(device)

    # 优化器 学习率调度器
    optimizer_clip = create_optimizer(args_, clip_model)
    lr_scheduler_clip, _ = create_scheduler(args_, optimizer_clip)
    clip_train_dict = dict(
        optimizer=optimizer_clip,
        lr_scheduler=lr_scheduler_clip,
        clip_model=clip_model
    )

    # 解码器
    txt_decoder = TextDecoder(config=config)
    txt_decoder.to(device)
    optimizer_td = AdamW(txt_decoder.parameters(), lr=1e-3, weight_decay=0, betas=(0.9, 0.98))
    lr_scheduler_td = scheduler.CosineAnnealingLR(
        optimizer=optimizer_td,
        eta_min=1e-8,
        T_max=args['epochs'],
    )

    td_train_dict = dict(
        optimizer=optimizer_td,
        lr_scheduler=lr_scheduler_td,
        txt_decoder=txt_decoder
    )

    # 损失函数 缩放管理器
    criterion = dict(
        loss_kl=KLLoss(),
        loss_ce=torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX,
                                          label_smoothing=0.2)
    )
    loss_scaler = NativeScaler()

    # 开始训练
    print(f"开始训练，共训练{Back.GREEN} {args['epochs']} {Back.RESET}轮.")

    # 初始化损失值和早停计数器
    min_loss = np.inf
    patience = args['patience']
    patience_counter = 0

    for epoch in range(args['epochs']):
        # 在需要释放内存的地方调用
        torch.cuda.empty_cache()

        # 训练一个epoch
        train_stats = train_one_epoch(args, epoch, train_dataloader,
                                      clip_train_dict, td_train_dict,
                                      criterion, loss_scaler)
        train_loss = (train_stats['clip_loss'] + train_stats['tdm_loss']) / 2
        print(
            f"VLP阶段，在训练集上："
            f"clip_loss={train_stats['clip_loss']}, tdm_loss={train_stats['tdm_loss']}, train_loss={train_loss}")
        utils.log('vlp_train', epoch=epoch + 1,
                  clip_loss=train_stats['clip_loss'],
                  tdm_loss=train_stats['tdm_loss'],
                  train_loss=train_loss)

        # 评估一个epoch
        val_stats = evaluate_one_epoch(args, epoch, val_dataloader,
                                       clip_train_dict, td_train_dict,
                                       criterion)
        val_loss = (val_stats['clip_loss'] + val_stats['tdm_loss']) / 2
        utils.log('vlp_val', epoch=epoch + 1,
                  clip_loss=val_stats['clip_loss'],
                  tdm_loss=val_stats['tdm_loss'],
                  val_loss=val_loss)
        print(
            f"VLP阶段，在验证集上："
            f"clip_loss={val_stats['clip_loss']}, tdm_loss={val_stats['tdm_loss']}, val_loss={val_loss}")

        # 检查是否有新的最低验证损失
        if val_loss < min_loss:
            min_loss = val_loss
            # 重置早停计数器
            patience_counter = 0
            print(f"最新val损失 {min_loss:.4f} 在第{epoch + 1}轮, 保存模型中... ")
            # 保存最佳模型
            if args['save_model'] and epoch % args['save_interval'] == 0:
                utils.save_checkpoint(state={
                    'epoch': epoch + 1,
                    'clip_train_dict': dict(
                        optimizer=clip_train_dict['optimizer'].state_dict(),
                        lr_scheduler=clip_train_dict['lr_scheduler'].state_dict(),
                        clip_model=clip_train_dict['clip_model'].state_dict()
                    ),
                    'td_train_dict': dict(
                        optimizer=td_train_dict['optimizer'].state_dict(),
                        lr_scheduler=td_train_dict['lr_scheduler'].state_dict(),
                        txt_decoder=td_train_dict['txt_decoder'].state_dict()
                    ),
                    'train_stats': train_stats,
                    'val_stats': val_stats,
                    'best_loss': val_loss
                }, args=args, filename=f"vlp_checkpoint.pth.tar")

        else:
            patience_counter += 1
            print(f"在val损失上无提升，对于第{epoch + 1}轮. ")

        # 检查是否达到早停条件
        if patience_counter >= patience:
            print(f"训练早停，patience：{patience}，第{epoch + 1}轮.")
            break

    # 测试集评估
    print("在-测试数据集-查看效果，请稍等...")
    # 评估一个epoch
    test_stats = evaluate_one_epoch(args, epoch=-1,
                                    dataloader=test_dataloader,
                                    clip_train_dict=clip_train_dict,
                                    td_train_dict=td_train_dict,
                                    criterion=criterion)
    test_loss = (test_stats['clip_loss'] + test_stats['tdm_loss']) / 2
    print(
        f"VLP阶段，在测试集上：clip_loss={test_stats['clip_loss']}, tdm_loss={test_stats['tdm_loss']}, test_loss={test_loss}")
    utils.log('vlp_test',
              clip_loss=test_stats['clip_loss'],
              tdm_loss=test_stats['tdm_loss'],
              test_loss=test_loss)


def train_one_epoch(args, epoch, dataloader,
                    clip_train_dict, td_train_dict,
                    criterion, loss_scaler: NativeScaler()):
    print(f"Epoch {epoch + 1} train...")

    # 状态记录表
    clip_losses, tdm_losses = [], []

    # 开启训练模式
    clip_train_dict['clip_model'].train(True)
    td_train_dict['txt_decoder'].train(True)

    clip_loss = criterion['loss_kl']
    tdm_loss = criterion['loss_ce']
    for step, (src_input, tgt_input, masked_tgt_input) in enumerate(dataloader):
        print(f"Epoch {epoch + 1} train, Step {step + 1}...")

        # 解码器损失权重分配
        vocab_weight = (len(tgt_input['input_ids']) - 1) / len(tgt_input['input_ids']) - 1
        emo_weight = 1 / len(tgt_input['input_ids'])
        masked_lm_loss_weight = torch.tensor([vocab_weight, emo_weight], device=args['device'])

        # 刷新梯度
        clip_train_dict['optimizer'].zero_grad()
        # 采用自动混合精度
        with torch.cuda.amp.autocast():
            img_txt_s_matrix, txt_img_s_matrix, ground_truth = clip_train_dict['clip_model'](src_input,
                                                                                             tgt_input)
            loss_i_t = clip_loss(img_txt_s_matrix, ground_truth)
            loss_t_i = clip_loss(txt_img_s_matrix, ground_truth)
            clip_total_loss = (loss_i_t + loss_t_i) / 2.
        # 根据梯度模型参数
        loss_scaler(clip_total_loss, clip_train_dict['optimizer'])
        clip_losses.append(clip_total_loss.item())

        # 5个step 更新解码器
        if step % 5 == 0:
            td_train_dict['optimizer'].zero_grad()
            with torch.cuda.amp.autocast():
                tdm_logits, emo_logits = td_train_dict['txt_decoder'](phase='clip', tgt_input=tgt_input,
                                                                      masked_tgt_input=masked_tgt_input,
                                                                      txt_encoder=clip_train_dict[
                                                                          'clip_model'].get_txt_encoder())
                loss_lambda = torch.tensor(args['loss_lambda'], device=args['device'])
                vocab_masked_lm_loss = tdm_loss(tdm_logits.reshape(-1, tdm_logits.shape[-1]),
                                                tgt_input['input_ids'][:, 1:].cuda().reshape(-1)) * loss_lambda

                # 将 logits 转换为概率分布
                emo_probs = f.softmax(emo_logits, dim=-1)
                print('emo_logits: ', emo_probs)
                print('tgt_input[:, 0]: ', tgt_input['input_ids'][:, 0].cuda().reshape(-1))
                emo_masked_lm_loss = tdm_loss(emo_logits, tgt_input['input_ids'][:, 0].cuda().reshape(-1)) * loss_lambda

                print(
                    f"{Back.GREEN}"
                    f"Evaluation - Epoch: {epoch + 1}, vocab_masked_lm_loss: {vocab_masked_lm_loss}, "
                    f"emo_masked_lm_loss: {emo_masked_lm_loss}"
                    f"{Back.RESET}")

                masked_lm_loss = torch.stack([vocab_masked_lm_loss, emo_masked_lm_loss])
                masked_lm_loss = torch.mean(masked_lm_loss * masked_lm_loss_weight)
                # 根据梯度模型参数
                loss_scaler(masked_lm_loss, td_train_dict['optimizer'])
                tdm_losses.append(masked_lm_loss.item())

        # 梯度爆炸
        if not math.isfinite(clip_total_loss.item()):
            print("CLIP Loss: {}, 结束训练".format(clip_total_loss.item()))
            sys.exit(1)
        if not math.isfinite(masked_lm_loss.item()):
            print("TDM Loss: {}, 结束训练".format(masked_lm_loss.item()))
            sys.exit(1)

    # 更新学习率
    clip_train_dict['lr_scheduler'].step(epoch)
    td_train_dict['lr_scheduler'].step()

    avg_clip_loss, avg_tdm_loss = loss.compute_average(clip_losses, tdm_losses)

    # 用于返回的状态字典
    train_stats = {'clip_loss': avg_clip_loss,
                   'tdm_loss': avg_tdm_loss}
    return train_stats


# 评估一个epoch
def evaluate_one_epoch(args, epoch, dataloader,
                       clip_train_dict, td_train_dict,
                       criterion):
    # -1 代表在测试数据集上
    if epoch >= 0:
        print(f"Epoch {epoch + 1} val...")

    # 状态记录表
    clip_losses, tdm_losses = [], []
    # 设置模型为评估模式
    clip_train_dict['clip_model'].eval()
    td_train_dict['txt_decoder'].eval()

    with torch.no_grad():
        for step, (src_input, tgt_input, masked_tgt_input) in enumerate(dataloader):
            # -1 代表在测试数据集上
            if epoch >= 0:
                print(f"Step {step + 1}...")
            else:
                print(f"Epoch {epoch + 1} val, Step {step + 1}...")

            # 解码器损失权重分配
            vocab_weight = (len(tgt_input['input_ids']) - 1) / len(tgt_input['input_ids']) - 1
            emo_weight = 1 / len(tgt_input['input_ids'])
            masked_lm_loss_weight = torch.tensor([vocab_weight, emo_weight], device=args['device'])

            # 采用自动混合精度
            with torch.cuda.amp.autocast():
                # clip 部分
                img_txt_s_matrix, txt_img_s_matrix, ground_truth = clip_train_dict['clip_model'](src_input,
                                                                                                 tgt_input)
                loss_i_t = criterion['loss_kl'](img_txt_s_matrix, ground_truth)
                loss_t_i = criterion['loss_kl'](txt_img_s_matrix, ground_truth)
                clip_total_loss = (loss_i_t + loss_t_i) / 2.
                clip_losses.append(clip_total_loss.item())

                # mask 部分
                tdm_logits, emo_logits = td_train_dict['txt_decoder'](phase='clip', tgt_input=tgt_input,
                                                                      masked_tgt_input=masked_tgt_input,
                                                                      txt_encoder=clip_train_dict[
                                                                          'clip_model'].get_txt_encoder())
                loss_lambda = torch.tensor(args['loss_lambda'], device=args['device'])
                vocab_masked_lm_loss = criterion['loss_ce'](tdm_logits.reshape(-1, tdm_logits.shape[-1]),
                                                            tgt_input['input_ids'][:, 1:].cuda().reshape(
                                                                -1)) * loss_lambda

                emo_masked_lm_loss = criterion['loss_ce'](emo_logits,
                                                          tgt_input['input_ids'][:, 0].cuda().reshape(-1)) * loss_lambda

                print(
                    f"{Back.GREEN}"
                    f"Evaluation - Epoch: {epoch + 1}, vocab_masked_lm_loss: {vocab_masked_lm_loss}, "
                    f"emo_masked_lm_loss: {emo_masked_lm_loss}"
                    f"{Back.RESET}")

                masked_lm_loss = torch.stack([vocab_masked_lm_loss, emo_masked_lm_loss])
                masked_lm_loss = torch.mean(masked_lm_loss * masked_lm_loss_weight)

                tdm_losses.append(masked_lm_loss.item())

    avg_clip_loss, avg_tdm_loss = loss.compute_average(clip_losses, clip_losses)

    eval_stats = {'clip_loss': avg_clip_loss, 'tdm_loss': avg_tdm_loss}
    return eval_stats


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

    # 开始训练
    main(args, config)
