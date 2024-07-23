import os
import math
import sys
import torch
import yaml
import argparse
from pathlib import Path
from transformers import MBartTokenizer, AutoTokenizer
import numpy as np
import random
from model_v2 import SLT
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
from rouge import Rouge
from torch.cuda.amp import GradScaler, autocast


def get_args_parser():
    a_parser = argparse.ArgumentParser('VLP scripts', add_help=False)
    a_parser.add_argument('--batch_size', default=1, type=int)
    a_parser.add_argument('--epochs', default=40, type=int)

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
    a_parser.add_argument('--loss_lambda', type=float, default=0.1, metavar='RATE')

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

    a_parser.add_argument('--need_keypoints', default=False, type=bool)

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
    # tokenizer = MBartTokenizer.from_pretrained("facebook/mbart-large-cc25")
    tokenizer = MBartTokenizer.from_pretrained("facebook/mbart-large-cc25")
    # lang = {
    #     'How2SignDataset': 'en_XX',
    #     'P14TDataset': 'de_DE',
    #     'CSLDailyDataset': 'zh_CN'
    # }
    # tokenizer.src_lang = lang[args['dataset']]
    tokenizer.src_lang = 'en_XX'
    tokenizer.tgt_lang = 'en_XX'

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
    slt_model = SLT(config=config, args=args)
    slt_model.to(device)

    optimizer = create_optimizer(args_, slt_model)
    criterion = torch.nn.CrossEntropyLoss()
    scaler = GradScaler()

    lr_scheduler = scheduler.CosineAnnealingLR(optimizer, eta_min=args['min_lr'], T_max=args['epochs'])

    best_loss = float('inf')
    for epoch in range(args['epochs']):
        train_loss = train_one_epoch(slt_model, train_dataloader, optimizer, criterion, device, scaler)
        utils.log('slt_train', epoch=epoch + 1,
                  train_loss=train_loss
                  )

        val_loss, bleu, rouge = evaluate(slt_model, val_dataloader, criterion, device, tokenizer)

        print(
            f"Epoch [{epoch + 1}/{args['epochs']}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, BLEU: {bleu:.2f}, ROUGE: {rouge:.2f}")
        utils.log('slt_val', epoch=epoch + 1,
                  val_loss=val_loss,
                  bleu=bleu,
                  rouge=rouge
                  )

        lr_scheduler.step()

        if val_loss < best_loss:
            best_loss = val_loss
            if args['save_model']:
                torch.save(slt_model.state_dict(), os.path.join(args['checkpoints_dir'], 'best_model.pth'))

    print("Training completed. Evaluating on test set...")
    test_loss, test_bleu, test_rouge = evaluate(slt_model, test_dataloader, criterion, device, tokenizer)
    print(f"Test Loss: {test_loss:.4f}, Test BLEU: {test_bleu:.2f}, Test ROUGE: {test_rouge:.2f}")
    utils.log('slt_test',
              test_loss=test_loss,
              test_bleu=test_bleu,
              test_rouge=test_rouge,
              )


def train_one_epoch(model, dataloader, optimizer, criterion, device, scaler: NativeScaler):
    model.train()
    running_loss = 0.0
    for batch in dataloader:
        try:
            optimizer.zero_grad()
            src_input, tgt_input = batch
            # print(src_input)
            # print(tgt_input)
            # inputs, targets = batch['inputs'].to(device), batch['targets'].to(device)
            with autocast():
                vocab_logits, emo_logits = model(src_input, tgt_input)
                # print('vocab_logits: ', vocab_logits)
                # print('emo_logits: ', emo_logits)
                # print(" tgt_input['input_ids']", tgt_input['input_ids'])
                # 调整形状以适应CrossEntropyLoss的输入要求
                # [batch_size * seq_len, vocab_size]
                vocab_logits_flat = vocab_logits.view(-1, vocab_logits.size(-1)).to(device)
                # print('vocab_logits_flat.shape: ', vocab_logits_flat.shape)
                # print('vocab_logits_flat:', vocab_logits_flat)

                # [batch_size * seq_len]
                tgt_input_flat = tgt_input['input_ids'][:, 1:].contiguous().view(-1).to(device)
                # print('tgt_input_flat.shape: ', tgt_input_flat.shape)
                # print('tgt_input_flat: ', tgt_input_flat)

                loss = criterion(vocab_logits_flat, tgt_input_flat)
                print('loss: ', loss)
            scaler.scale(loss).backward()  # 使用 GradScaler 的 scale 方法
            scaler.step(optimizer)  # 使用 GradScaler 的 step 方法
            scaler.update()  # 使用 GradScaler 的 update 方法
            # print("src_input: ", src_input)

            running_loss += loss.item() * src_input['imgs_ids'].size(0)
        except Exception as e:
            print("数据错误，摒弃本数据。", e)
            continue
    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss


def evaluate(model, dataloader, criterion, device, tokenizer):
    model.eval()
    running_loss = 0.0
    references = []
    hypotheses = []
    with torch.no_grad():
        for batch in dataloader:
            try:
                # 获取输入和目标数据
                src_input, tgt_input = batch

                # 模型前向传播
                vocab_logits, emo_logits = model(src_input, tgt_input)

                # 计算损失
                vocab_logits_flat = vocab_logits.view(-1, vocab_logits.size(-1)).to(device)
                tgt_input_flat = tgt_input['input_ids'][:, 1:].contiguous().view(-1).to(device)
                loss = criterion(vocab_logits_flat, tgt_input_flat)
                print('val loss:', loss)

                # 累加损失
                running_loss += loss.item() * src_input['imgs_ids'].size(0)
                print('val running_loss: ', running_loss)

                # 解码预测结果和参考答案
                hypotheses_batch = tokenizer.batch_decode(vocab_logits.argmax(dim=-1), skip_special_tokens=True)
                references_batch = tokenizer.batch_decode(tgt_input['input_ids'], skip_special_tokens=True)

                # 打印解码结果（用于调试）
                print('hypotheses_batch: ', hypotheses_batch)
                print('references_batch: ', references_batch)

                # 收集解码后的预测和参考答案
                hypotheses.extend(hypotheses_batch)
                references.extend(references_batch)

            except Exception as e:
                print("数据错误，摒弃本数据。", e)
                continue

    epoch_loss = running_loss / len(dataloader.dataset)

    # 计算 BLEU 和 ROUGE 分数
    bleu = BLEU().corpus_score(hypotheses, [references])
    rouge = Rouge().get_scores(hypotheses, references, avg=True)

    # 解析 BLEU 分数
    bleu1 = bleu.precisions[0]
    bleu2 = bleu.precisions[1]
    bleu3 = bleu.precisions[2]
    bleu4 = bleu.precisions[3]

    # 解析 ROUGE-L 分数
    rouge_l = rouge['rouge-l']['f']

    print(f"epoch_loss: {epoch_loss}")
    print(f"BLEU-1: {bleu1}")
    print(f"BLEU-2: {bleu2}")
    print(f"BLEU-3: {bleu3}")
    print(f"BLEU-4: {bleu4}")
    print(f"ROUGE-L: {rouge_l}")

    return epoch_loss, bleu4, rouge_l


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
