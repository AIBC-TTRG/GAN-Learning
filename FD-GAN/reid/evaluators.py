#-*- coding: utf-8 -*-
from __future__ import print_function, absolute_import
import time
from collections import OrderedDict

import torch
import numpy as np
from torch.utils.data import DataLoader

from .evaluation_metrics import cmc, mean_ap
from .feature_extraction import extract_cnn_feature
from .utils.meters import AverageMeter
from .utils import to_numpy
from torch.autograd import Variable
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
cudnn.enabled = True
cudnn.benchmark = True


def extract_embeddings(model, features, alpha, query=None, topk_gallery=None, rerank_topk=0, print_freq=500):
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    end = time.time()
    # pairwise_score <class 'torch.autograd.variable.Variable'> torch.Size([16483, 100, 2])
    pairwise_score = Variable(torch.zeros(len(query), rerank_topk, 2).cuda())
    # print('pairwise_score')
    # print(pairwise_score)
    # print(type(pairwise_score))
    # print(pairwise_score.shape)
#      seq表示要连接的两个序列，以元组的形式给出，例如:seq=(a,b),  a,b 为两个可以连接的序列
# dim 表示以哪个维度连接，dim=0, 横向连接         dim=1,纵向连接

    # probe_feature <class 'torch.FloatTensor'> torch.Size([16483, 2048])
    probe_feature = torch.cat([features[f].unsqueeze(0) for f, _, _ in query], 0)
    # print('probe_feature')
    # print(probe_feature)
    # print(type(probe_feature))
    # print(probe_feature.shape)
    for i in range(len(query)):
        gallery_feature = torch.cat([features[f].unsqueeze(0) for f, _, _ in topk_gallery[i]], 0)
        pairwise_score[i, :, :] = model(Variable(probe_feature[i].view(1, -1).cuda(), volatile=True),
                                        Variable(gallery_feature.cuda(), volatile=True))
        # print('pairwise_score')
        # print(pairwise_score)

        batch_time.update(time.time() - end)
        end = time.time()

        if (i + 1) % print_freq == 0:
         print('Extract Embedding: [{}/{}]\t'
               'Time {:.3f} ({:.3f})\t'
               'Data {:.3f} ({:.3f})\t'.format(
               i + 1, len(query),
               batch_time.val, batch_time.avg,
               data_time.val, data_time.avg))

    return pairwise_score.view(-1,2)


def extract_features(model, data_loader, print_freq=1, metric=None):
    print('====extract_features=======')
    model.eval()
    # print('data_loader')
    # print(data_loader)
    batch_time = AverageMeter()
    data_time = AverageMeter()

    # OrderedDict()记得插入顺序的字典
    features = OrderedDict()
    labels = OrderedDict()
    # print('batch_time')
    # print(batch_time)
    # print('data_time')
    # print(data_time)
    # print('features')
    # print(features)
    # print('labels')
    # print(labels)


    end = time.time()
    # print('end')
    # print(end)
    # enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，
    # 同时列出数据和数据下标，一般用在 for 循环当中。
    for i, (imgs, fnames, pids, _) in enumerate(data_loader):
        data_time.update(time.time() - end)
        # print('imgs')
        # print(imgs)
        # print('fnames')
        # # print(fnames)
        # print(type(fnames))
        # print('pids')
        # print(pids)
        # imgs  [torch.FloatTensor of size 16x3x256x128]
        # type(fnames) <type 'tuple'>
        # pids  [torch.LongTensor of size 16]
        # outputs  [torch.FloatTensor of size 16x2048]
        outputs = extract_cnn_feature(model, imgs)
        # print('outputs')
        # print(outputs)
        # print('outputs')
        # print(type(outputs))
        # print(outputs)
        # zip函数接受任意多个（包括0个和1个）序列作为参数，返回一个tuple列表
        for fname, output, pid in zip(fnames, outputs, pids):
            features[fname] = output
            labels[fname] = pid

        batch_time.update(time.time() - end)
        end = time.time()

        if (i + 1) % print_freq == 0:
            print('Extract Features: [{}/{}]\t'
                  'Time {:.3f} ({:.3f})\t'
                  'Data {:.3f} ({:.3f})\t'
                  .format(i + 1, len(data_loader),
                          batch_time.val, batch_time.avg,
                          data_time.val, data_time.avg))
    return features, labels


def pairwise_distance(features, query=None, gallery=None, metric=None):
    print('pairwise_distance')
    if query is None and gallery is None:
        n = len(features)
        print('nnnn')
        print(n)
        x = torch.cat(list(features.values()))
        x = x.view(n, -1)
        if metric is not None:
            x = metric.transform(x)
        dist = torch.pow(x, 2).sum(dim=1, keepdim=True) * 2
        dist = dist.expand(n, n) - 2 * torch.mm(x, x.t())
        return dist

    x = torch.cat([features[f].unsqueeze(0) for f, _, _ in query], 0)
    y = torch.cat([features[f].unsqueeze(0) for f, _, _ in gallery], 0)
    m, n = x.size(0), y.size(0)
    x = x.view(m, -1)
    y = y.view(n, -1)
    if metric is not None:
        x = metric.transform(x)
        y = metric.transform(y)
    dist = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
           torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist.addmm_(1, -2, x, y.t())
    print('dist')
    return dist


def evaluate_all(distmat, query=None, gallery=None,
                 query_ids=None, gallery_ids=None,
                 query_cams=None, gallery_cams=None,
                 cmc_topk=(1, 5, 10), dataset=None, top1=True):
    print('evaluate_all======')
    if query is not None and gallery is not None:
        print('111111111111')
        # query_ids <type 'list'>  len(query_ids) 16483
        # gallery_ids <type 'list'>  len(gallery_ids) 19281
        # query_cams <type 'list'>  len(query_cams) 16483
        # gallery_cams <type 'list'>  len(gallery_cams) 19281
        query_ids = [pid for _, pid, _ in query]
        gallery_ids = [pid for _, pid, _ in gallery]
        query_cams = [cam for _, _, cam in query]
        gallery_cams = [cam for _, _, cam in gallery]
        # print('query_ids')
        # print(query_ids)
        # print('gallery_ids')
        # print(gallery_ids)
        # print('query_cams')
        # print(query_cams)
        # print('gallery_cams')
        # print(gallery_cams)
        
    else:
        assert (query_ids is not None and gallery_ids is not None
                and query_cams is not None and gallery_cams is not None)

    # Compute mean AP  平均AP值 
    mAP = mean_ap(distmat, query_ids, gallery_ids, query_cams, gallery_cams)
    print('Mean AP: {:4.1%}'.format(mAP))
    # print('Mean AP: ')


    if top1:
      # Compute all kinds of CMC scores     计算各种CMC的分数
      if not dataset:
        print('dataset++++++')
        cmc_configs = {
            'allshots': dict(separate_camera_set=False,
                             single_gallery_shot=False,
                             first_match_break=False),
            'cuhk03': dict(separate_camera_set=True,
                           single_gallery_shot=True,
                           first_match_break=False),
            'market1501': dict(separate_camera_set=False,
                               single_gallery_shot=False,
                               first_match_break=True)}
        cmc_scores = {name: cmc(distmat, query_ids, gallery_ids,
                                query_cams, gallery_cams, **params)
                      for name, params in cmc_configs.items()}
        
        print('cmc_scores')
        print(cmc_scores)

        print('CMC Scores{:>12}{:>12}{:>12}'
              .format('allshots', 'cuhk03', 'market1501'))
        for k in cmc_topk:
            print('  top-{:<4}{:12.1%}{:12.1%}{:12.1%}'
                  .format(k, cmc_scores['allshots'][k - 1],
                          cmc_scores['cuhk03'][k - 1],
                          cmc_scores['market1501'][k - 1]))

        # Use the allshots cmc top-1 score for validation criterion 使用allshots cmc(得分为验证标准
        return cmc_scores['allshots'][0]
      else:

        if (dataset == 'cuhk03'):
          print('cuhk03')
          cmc_configs = {
              'cuhk03': dict(separate_camera_set=True,
                                single_gallery_shot=True,
                                first_match_break=False),
              }
          cmc_scores = {name: cmc(distmat, query_ids, gallery_ids,
                                  query_cams, gallery_cams, **params)
                        for name, params in cmc_configs.items()}

          print('CMC Scores{:>12}'.format('cuhk03'))
          for k in cmc_topk:
              print('  top-{:<4}{:12.1%}'
                    .format(k,
                            cmc_scores['cuhk03'][k - 1]))
          # Use the allshots cmc top-1 score for validation criterion
          return cmc_scores['cuhk03'][0], mAP
        else:
          print('market1501')
          cmc_configs = {
              'market1501': dict(separate_camera_set=False,
                                 single_gallery_shot=False,
                                 first_match_break=True)
                      }
          cmc_scores = {name: cmc(distmat, query_ids, gallery_ids,
                                  query_cams, gallery_cams, **params)
                        for name, params in cmc_configs.items()}

        #   print('cmc_scores')
        #   print(cmc_scores)
          print('CMC Scores{:>12}'.format('market1501'))
          for k in cmc_topk:
              print('  top-{:<4}{:12.1%}'
                    .format(k,
                            cmc_scores['market1501'][k-1]))
        #   print(cmc_scores['market1501'][0])
          return cmc_scores['market1501'][0], mAP
    else:
      return mAP

class CascadeEvaluator(object):
    def __init__(self, base_model, embed_model, embed_dist_fn=None):
        print('CascadeEvaluator__init__')
        super(CascadeEvaluator, self).__init__()
        self.base_model = base_model
        self.embed_model = embed_model
        self.embed_dist_fn = embed_dist_fn 

    def evaluate(self, data_loader, query, gallery, alpha=0, cache_file=None,
                 rerank_topk=75, second_stage=True, dataset=None, top1=True):
        # data_loader :: test_loader
        # Extract features image by image   提取图像的图像特征
        # print('data_loader')
        # print(data_loader)
        print('========CascadeEvaluator_evaluate========')
        features, _ = extract_features(self.base_model, data_loader)
        # print('features')
        # print(features)
        
        # Compute pairwise distance and evaluate for the first stage
        # 计算两两距离和第一阶段评估
        # <class 'torch.FloatTensor'>
        distmat = pairwise_distance(features, query, gallery)
        
        # distmat::killed
        print('distmat')
        # print(distmat)
        
        print("First stage evaluation:")
        

        if second_stage:
            print('second_stage===========')
            evaluate_all(distmat, query=query, gallery=gallery, dataset=dataset, top1=top1)

            # Sort according to the first stage distance    根据第一阶段的距离
            distmat = to_numpy(distmat)
            #rank_indices  <type 'numpy.ndarray'>  317808723  (16483, 19281)
            rank_indices = np.argsort(distmat, axis=1)
            # print('rank_indices')
            # print(type(rank_indices))
            # print(rank_indices.shape)
            # print(rank_indices)

            # Build a data loader for topk predictions for each query
            # 建立一个数据加载程序为每个查询topk预测
            # topk_gallery <type 'list'>  16483
            topk_gallery = [[] for i in range(len(query))]
            # print('topk_gallery')
            # print(type(topk_gallery))
            # print(len(topk_gallery))
            # print(topk_gallery)
            for i, indices in enumerate(rank_indices):
                # rerank_topk ::75
                for j in indices[:rerank_topk]:
                    gallery_fname_id_pid = gallery[j]
                    topk_gallery[i].append(gallery_fname_id_pid)

            embeddings = extract_embeddings(self.embed_model, features, alpha,
                                    query=query, topk_gallery=topk_gallery, rerank_topk=rerank_topk)

            print('self.embed_dist_fn')
            print(self.embed_dist_fn)
            if self.embed_dist_fn is not None:
                print('fffffffff')
                embeddings = self.embed_dist_fn(embeddings.data)

            # Merge two-stage distances     合并两级的距离
            for k, embed in enumerate(embeddings):
                i, j = k // rerank_topk, k % rerank_topk
                distmat[i, rank_indices[i, j]] = embed
            for i, indices in enumerate(rank_indices):
                bar = max(distmat[i][indices[:rerank_topk]])
                gap = max(bar + 1. - distmat[i, indices[rerank_topk]], 0)
                if gap > 0:
                    distmat[i][indices[rerank_topk:]] += gap
            print("Second stage evaluation:")
        return evaluate_all(distmat, query, gallery, dataset=dataset, top1=top1)