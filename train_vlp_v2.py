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
from model_v2 import CLIP
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
from rouge import Rouge
from torch.cuda.amp import GradScaler, autocast
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
import torch
import torch.nn.functional as F
from torch.optim import lr_scheduler
from loss import KLLoss


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
    a_parser.add_argument('--num_workers', default=1, type=int)
    # a_parser.add_argument('--num_workers', default=2, type=int)
    a_parser.add_argument('--checkpoints_dir', default='./checkpoints/')
    a_parser.add_argument('--log_dir', default='./log/')
    a_parser.add_argument('--input_size', default=224, type=int)

    a_parser.add_argument('--training_refurbish', default=False, type=bool)
    a_parser.add_argument('--noise_rate', default=0.15, type=float)
    a_parser.add_argument('--random_shuffle', default=False, type=bool)
    a_parser.add_argument('--loss_lambda', type=float, default=0.1, metavar='RATE')

    # * Optimize参数
    a_parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER')
    a_parser.add_argument('--opt-eps', default=1.0e-09, type=float, metavar='EPSILON')
    a_parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA')
    a_parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM')
    a_parser.add_argument('--momentum', type=float, default=0.9, metavar='M')
    a_parser.add_argument('--weight-decay', type=float, default=0.01)

    # * Learning rate 参数
    a_parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER')
    a_parser.add_argument('--lr', type=float, default=1.0e-4, metavar='LR')
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

    a_parser.add_argument('--dataset', default='P14TDataset', type=str,
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
    # tokenizer = MBartTokenizer.from_pretrained("./data/pretrain_models/MBart_proun", local_files_only=True)
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

    # CLIP Model
    vlp_model = CLIP(config=config, args=args)
    # 权重加载
    if args['finetune']:
        try:
            print("加载SLT模型权重...")
            # 加载模型的检查点
            checkpoint_path = os.path.join(args['checkpoints_dir'], 'best_model.pth')
            checkpoint = torch.load(checkpoint_path)
            vlp_model.load_state_dict(checkpoint)
            print("模型权重加载成功")
        except Exception as e:
            print("加载模型权重时出现错误:", e)

    # 移动到设备上
    vlp_model.to(device)

    optimizer = create_optimizer(args_, vlp_model)

    criterion = dict(
        vocab_criterion=torch.nn.CrossEntropyLoss(),
        clip_criterion=KLLoss()
    )

    scaler = GradScaler()

    lr_scheduler = scheduler.CosineAnnealingLR(optimizer, eta_min=args['min_lr'], T_max=args['epochs'])

    best_loss = float('inf')
    for epoch in range(args['epochs']):
        torch.cuda.empty_cache()
        try:
            train_loss = train_one_epoch(vlp_model, train_dataloader, optimizer, criterion, device, scaler)
            utils.log('vlp_train', epoch=epoch + 1,
                      train_loss=train_loss
                      )

            val_loss, bleu1, bleu2, bleu3, bleu4, rouge_l, emo_accuracy = evaluate(vlp_model, val_dataloader,
                                                                                   criterion, device, tokenizer)

            print(
                f"Epoch [{epoch + 1}/{args['epochs']}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, BLEU-4: {bleu4:.2f}, ROUGE-l: {rouge_l:.2f}, Accuracy: {emo_accuracy:.2f}")
            utils.log('vlp_val', epoch=epoch + 1,
                      val_loss=val_loss,
                      )

            lr_scheduler.step()

            if val_loss < best_loss:
                best_loss = val_loss
                if args['save_model']:
                    torch.save(vlp_model.state_dict(), os.path.join(args['checkpoints_dir'], 'vlp_best_model.pth'))
        except Exception as e:
            print(f"Epoch {epoch + 1} 出现错误。", e)
            continue


def train_one_epoch(model, dataloader, optimizer, criterion, device, scaler: NativeScaler):
    model.train()
    running_loss = 0.0
    clip_losses, vocab_losses = [], []
    for step, batch in enumerate(dataloader):
        print('---step---: ', step)
        try:
            optimizer.zero_grad()
            src_input, tgt_input, masked_tgt_input = batch
            with autocast():
                vocab_logits = model(src_input, tgt_input)
                vocab_logits_flat = vocab_logits.view(-1, vocab_logits.size(-1)).to(device)
                tgt_input_flat = tgt_input['input_ids'][:, 1:].contiguous().view(-1).to(device)
                loss = criterion(vocab_logits_flat, tgt_input_flat)
                print('loss: ', loss)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
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
    emo_collection = []
    with torch.no_grad():
        for step, batch in enumerate(dataloader):
            print('---step---: ', step)
            try:
                src_input, tgt_input = batch

                vocab_logits = model(src_input, tgt_input)
                vocab_logits_flat = vocab_logits.view(-1, vocab_logits.size(-1)).to(device)
                tgt_input_flat = tgt_input['input_ids'][:, 1:].contiguous().view(-1).to(device)
                loss = criterion(vocab_logits_flat, tgt_input_flat)

                running_loss += loss.item() * src_input['imgs_ids'].size(0)

                hypotheses_batch = tokenizer.batch_decode(vocab_logits.argmax(dim=-1), skip_special_tokens=True)
                references_batch = tokenizer.batch_decode(tgt_input['input_ids'], skip_special_tokens=True)

                for hyp, ref in zip(hypotheses_batch, references_batch):
                    if not hyp.strip():
                        hyp = "neutral -<empty>-"
                    print('hyp: ', hyp)
                    print('ref: ', ref)
                    emo_collection.append(utils.compare_first_words(hyp, ref))
                    hyp = utils.remove_duplicates(hyp)
                    ref = utils.remove_duplicates(ref)
                    hypotheses.append(hyp)
                    references.append(ref)

            except Exception as e:
                print("数据错误，摒弃本数据。", e)
                continue
    # 情感准确率
    emo_accuracy = utils.calculate_ratio_of_ones(emo_collection)

    # 计算 LOSS
    epoch_loss = running_loss / len(dataloader.dataset)

    # 计算 BLEU 和 ROUGE 分数
    bleu = BLEU().corpus_score(hypotheses, [references])
    # 计算 BLEU 分数

    rouge = Rouge().get_scores(hypotheses, references, avg=True)
    # smoothing_function = SmoothingFunction().method4
    # bleu1 = corpus_bleu(references, hypotheses, weights=(1, 0, 0, 0), smoothing_function=smoothing_function)
    # bleu2 = corpus_bleu(references, hypotheses, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing_function)
    # bleu3 = corpus_bleu(references, hypotheses, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothing_function)
    # bleu4 = corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25),
    # smoothing_function=smoothing_function)

    # 解析 BLEU 和 ROUGE 分数
    bleu1 = bleu.precisions[0]
    bleu2 = bleu.precisions[1]
    bleu3 = bleu.precisions[2]
    bleu4 = bleu.precisions[3]
    rouge_l = rouge['rouge-l']['f']

    print(f"epoch_loss: {epoch_loss}")
    print(f"BLEU-1: {bleu1}")
    print(f"BLEU-2: {bleu2}")
    print(f"BLEU-3: {bleu3}")
    print(f"BLEU-4: {bleu4}")
    print(f"ROUGE-L: {rouge_l}")

    return epoch_loss, bleu1, bleu2, bleu3, bleu4, rouge_l, emo_accuracy


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
