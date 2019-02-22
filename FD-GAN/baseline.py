#-*- coding: utf-8 -*-
from __future__ import print_function, absolute_import
import argparse
import os.path as osp

import numpy as np
import os, sys
from bisect import bisect_right
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
from torch.backends import cudnn
from torch.utils.data import DataLoader

from reid import datasets
from reid import models
from reid.utils.data import transforms as T
from reid.utils.data.preprocessor import Preprocessor
from reid.utils.logging import Logger
from reid.utils.serialization import load_checkpoint, save_checkpoint, copy_state_dict

from reid.utils.data.sampler import RandomPairSampler
from reid.models.embedding import EltwiseSubEmbed
from reid.models.multi_branch import SiameseNet
from reid.evaluators import CascadeEvaluator
from reid.trainers import SiameseTrainer

# split_id = 0
def get_data(name, split_id, data_dir, height, width, batch_size, workers,
             combine_trainval, np_ratio):
    print('============get_data============')
    print('name')
    print(name)
    print('split_id')
    print(split_id)
    print('data_dir')
    print(data_dir)

    print('height')
    print(height)
    print('width')
    print(width)

    print('batch_size')
    print(batch_size)
    print('workers')
    print(workers)
    print('combine_trainval')
    print(combine_trainval)

    print('np_ratio')
    print(np_ratio)
    
    # root = /home/aibc/Documents/FD-GAN/datasets/market1501
    root = osp.join(data_dir, name)
    print('root')   
    print(root)

    dataset = datasets.create(name, root, split_id=split_id)
    print('dataset')
    print(dataset)

    # 正常一个张量图像的平均值和标准偏差。
    # mean(序列):对每个通道的序列。
    # std序列):为每个通道序列的标准差。
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    # combine_trainval : TURE
    # train_set :::dataset.train  (前651)
    train_set = dataset.trainval if combine_trainval else dataset.train
    # print('train_set')
    # print(train_set)
    
    # Compose 一起组成几个转换
    # RandomSizedRectCrop 随机大小的矩形作物 
    # RandomSizedEarser 随机大小的抹音器
    # RandomHorizontalFlip 随机水平翻转
    train_transformer = T.Compose([
        T.RandomSizedRectCrop(height, width),
        T.RandomSizedEarser(),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalizer,
    ])

    # RectScale矩形规模
    test_transformer = T.Compose([
        T.RectScale(height, width),
        T.ToTensor(),
        normalizer,
    ])

    # print('test_transformer')
    # print(test_transformer)

    # DataLoader数据加载程序。结合数据集和取样器,并提供单一或多进程数据集迭代器。
    # DataLoader是一个比较重要的类，它为我们提供的常用操作有：batch_size(每个batch的大小), shuffle(是否进行shuffle操作), 
    # num_workers 4 (加载数据的时候使用几个子进程)
    # pin_memory(bool,可选):如果' 'Tuer' ',数据加载程序将张量复制到CUDA固定内存返回之前。
    # 取样器(取样器,可选):画样本数据集定义了策略。如果指定了”、“shuffle”“必须是假的。
    #  shuffle=True, # 是否随机打乱顺序
    # num_workers=4, # 多线程读取数据的线程数
    train_loader = DataLoader(
        Preprocessor(train_set, root=dataset.images_dir,
                     transform=train_transformer),
        sampler=RandomPairSampler(train_set, neg_pos_ratio=np_ratio),
        batch_size=batch_size, num_workers=workers, pin_memory=False)

    val_loader = DataLoader(
        Preprocessor(dataset.val, root=dataset.images_dir,
                     transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=False)

    test_loader = DataLoader(
        Preprocessor(list(set(dataset.query) | set(dataset.gallery)),
                     root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=False)

    # print('train_loader')
    # print(train_loader)
    # print('val_loader')
    # print(val_loader)
    # print('test_loader')
    # print(test_loader)

    

    return dataset, train_loader, val_loader, test_loader


def main(args):
    # print('main')
    # print('args.dataset')
    # print(args.dataset)
    # print('args.split')
    # print(args.split)
    # print('args.data_dir')
    # print(args.data_dir)
    # print('args.logs_dir')
    # print(args.logs_dir)
    # print('maargs.heightin')
    # print(args.height)
    # print('args.width')
    # print(args.width)
    # print('args.batch_size')
    # print(args.batch_size)
    # print('args.workers')
    # print(args.workers)
    # print('args.combine_trainval')
    # print(args.combine_trainval)
    # print('args.workers')
    # print(args.np_ratio)
    
    #随机数
    np.random.seed(args.seed)
     #为CPU设置种子用于生成随机数，以使得结果是确定的 
    torch.manual_seed(args.seed)
    # 增加运行效率
    cudnn.benchmark = True

    # Redirect print to both console and log file
    # 重定向打印到控制台和日志文件

    # print('args.evaluate')
    # print(args.evaluate)

    if not args.evaluate:
        # Logger（）打印日志
        sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))
    else:
        log_dir = osp.dirname(args.resume)
        sys.stdout = Logger(osp.join(log_dir, 'log_test.txt'))
    # print("==========\nArgs:{}\n==========".format(args))

    # Create data loaders       创建数据加载器
    if args.height is None or args.width is None:
        args.height, args.width = (256, 128)
    dataset, train_loader, val_loader, test_loader = \
        get_data(args.dataset, args.split, args.data_dir, args.height,
                 args.width, args.batch_size, args.workers,
                 args.combine_trainval, args.np_ratio)

    # Create model  创建模型    resnet50
    base_model = models.create(args.arch, cut_at_pooling=True)
    # print('base_model')
    # print(base_model)
    embed_model = EltwiseSubEmbed(use_batch_norm=True, use_classifier=True,
                                      num_features=2048, num_classes=2)
    print('embed_model')
    print(embed_model)
    model = SiameseNet(base_model, embed_model)
    print('model')
    print(model)
    # 实现数据在模块级并行性。
    model = nn.DataParallel(model).cuda()
    print('model')
    print(model)

    # Evaluator 评估
    # DataParallel 实现数据在模块级并行性。
    evaluator = CascadeEvaluator(
        torch.nn.DataParallel(base_model).cuda(),
        embed_model,
        embed_dist_fn=lambda x: F.softmax(Variable(x), dim=1).data[:, 0])

    # print('evaluator')
    # print(evaluator)
    # Load from checkpoint
    print('args.evaluate')
    print(args.evaluate)
    best_mAP = 0
    if args.resume:
        checkpoint = load_checkpoint(args.resume)
        if 'state_dict' in checkpoint.keys():
            print('state_dict')
            checkpoint = checkpoint['state_dict']
        # print('checkpoint')
        # print(checkpoint)
        model.load_state_dict(checkpoint)

        print("Test the loaded model:")
        top1, mAP = evaluator.evaluate(test_loader, dataset.query, dataset.gallery, rerank_topk=100, dataset=args.dataset)
        print('top1')
        print(top1)
        print('mAP')
        print(mAP)
        best_mAP = mAP

    # args.evaluate : Ture
    if args.evaluate:
        return

    # Criterion 标准
    criterion = nn.CrossEntropyLoss().cuda()
    # Optimizer
    param_groups = [
        {'params': model.module.base_model.parameters(), 'lr_mult': 1.0},
        {'params': model.module.embed_model.parameters(), 'lr_mult': 10.0}]
    optimizer = torch.optim.SGD(param_groups, args.lr, momentum=args.momentum,
                                weight_decay=args.weight_decay)
    # Trainer
    trainer = SiameseTrainer(model, criterion)

    # Schedule learning rate
    def adjust_lr(epoch):
        print('adjust_lr')
        lr = args.lr * (0.1 ** (epoch // args.step_size))
        for g in optimizer.param_groups:
            g['lr'] = lr * g.get('lr_mult', 1)

    print('args.eval_step')
    print(args.eval_step)
    # Start training
    for epoch in range(0, args.epochs):
        adjust_lr(epoch)
        trainer.train(epoch, train_loader, optimizer, base_lr=args.lr)

        if epoch % args.eval_step==0:
            print('ififififif')
            mAP = evaluator.evaluate(val_loader, dataset.val, dataset.val, top1=False)
            is_best = mAP > best_mAP
            best_mAP = max(mAP, best_mAP)
            save_checkpoint({
                'state_dict': model.state_dict()
            }, is_best, fpath=osp.join(args.logs_dir, 'checkpoint.pth.tar'))

            print('\n * Finished epoch {:3d}  mAP: {:5.1%}  best: {:5.1%}{}\n'.
                  format(epoch, mAP, best_mAP, ' *' if is_best else ''))

    # Final test
    print('Test with best model:')
    checkpoint = load_checkpoint(osp.join(args.logs_dir, 'model_best.pth.tar'))
    model.load_state_dict(checkpoint['state_dict'])
    evaluator.evaluate(test_loader, dataset.query, dataset.gallery, dataset=args.dataset)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Siamese reID baseline")
    # data
    parser.add_argument('-d', '--dataset', type=str, default='market1501',
                        choices=datasets.names())
    parser.add_argument('-b', '--batch_size', type=int, default=256)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--split', type=int, default=0)
    parser.add_argument('--height', type=int,
                        help="input height, default: 256 for resnet")
    parser.add_argument('--width', type=int,
                        help="input width, default: 128 for resnet")
    parser.add_argument('--combine_trainval', action='store_true',
                        help="train and val sets together for training, "
                             "val set alone for validation")
    # model
    parser.add_argument('-a', '--arch', type=str, default='resnet50',
                        choices=models.names())
    # optimizer
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--np_ratio', type=int, default=3)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--step_size', type=int, default=40)
    # training configs
    parser.add_argument('--resume', type=str, default='', metavar='PATH')
    parser.add_argument('--evaluate', action='store_true',
                        help="evaluation only")
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--eval_step', type=int, default=20, help="evaluation step")
    parser.add_argument('--seed', type=int, default=1)
    # misc
    working_dir = osp.dirname(osp.abspath('/home/aibc/Documents/FD-GAN/data'))
    parser.add_argument('--data_dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'datasets'))
    parser.add_argument('--logs_dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'checkpoints'))
    print('__name__')
    print('working_dir')
    print(working_dir)
    
    main(parser.parse_args())
