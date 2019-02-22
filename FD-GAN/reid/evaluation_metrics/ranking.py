#-*- coding: utf-8 -*-
from __future__ import absolute_import
from collections import defaultdict

import numpy as np
from sklearn.metrics import average_precision_score

from ..utils import to_numpy


def _unique_sample(ids_dict, num):
    mask = np.zeros(num, dtype=np.bool)
    for _, indices in ids_dict.items():
        i = np.random.choice(indices)
        mask[i] = True
    return mask


def cmc(distmat, query_ids=None, gallery_ids=None,
        query_cams=None, gallery_cams=None, topk=100,
        separate_camera_set=False,
        single_gallery_shot=False,
        first_match_break=False):
    # print('cmc====')
    # distmat (16483, 19281)
    distmat = to_numpy(distmat)
    # m : 16483 n : 19281  
    m, n = distmat.shape
    # print('distmat')
    # print(type(distmat))
    # print(np.shape(distmat))
    # print('mmmmmmmmmm')
    # print(m)
    # print('nnnnnnnnnnn')
    # print(n)
    # Fill up default values    填满的默认值
    if query_ids is None:
        # print('aaaaaaaaaaaaaaaa')
        query_ids = np.arange(m)
    if gallery_ids is None:
        # print('aaaaaaaaaaaaaaaa')
        gallery_ids = np.arange(n)
    if query_cams is None:
        # print('aaaaaaaaaaaaaaaa')
        query_cams = np.zeros(m).astype(np.int32)
    if gallery_cams is None:
        # print('aaaaaaaaaaaaaaaa')
        gallery_cams = np.ones(n).astype(np.int32)
    # Ensure numpy array    确保numpy数组
    # np.asarray 结构数据转化为ndarray。
    # query_ids  <type 'numpy.ndarray'> shape ：(16483,)
    # gallery_ids  <type 'numpy.ndarray'> shape ：(19281,)
    # query_cams  <type 'numpy.ndarray'> shape ：(16483,)
    # gallery_cams  <type 'numpy.ndarray'> shape ：(19281,)
    query_ids = np.asarray(query_ids)
    gallery_ids = np.asarray(gallery_ids)
    query_cams = np.asarray(query_cams)
    gallery_cams = np.asarray(gallery_cams)
    # print('gallery_ids')
    # print(type(gallery_ids))
    # print(np.shape(gallery_ids))
    # print(gallery_ids)
    # print('query_cams')
    # print(type(query_cams))
    # print(np.shape(query_cams))
    # print(query_cams)
    # print('gallery_cams')
    # print(type(gallery_cams))
    # print(np.shape(gallery_cams))
    # print(gallery_cams)
    # Sort and find correct matches     排序,找到正确的匹配
    # argsort函数返回的是数组值从小到大的索引值
    indices = np.argsort(distmat, axis=1)
    # matches <type 'numpy.ndarray'> (16483, 19281)
    matches = (gallery_ids[indices] == query_ids[:, np.newaxis])
    # print('matches')
    # print(matches)
    # print(type(matches))
    # print(np.shape(matches))
    # Compute CMC for each query    计算CMC为每个查询
    ret = np.zeros(topk)
    num_valid_queries = 0
    for i in range(m):
        # Filter out the same id and same camera    过滤掉相同的id和相同的cam
        valid = ((gallery_ids[indices[i]] != query_ids[i]) |
                 (gallery_cams[indices[i]] != query_cams[i]))
        # valid  True Flase
        # print('valid')
        # print(valid)
        if separate_camera_set:
            # Filter out samples from same camera   过滤掉样本相同的cam
            valid &= (gallery_cams[indices[i]] != query_cams[i])
        if not np.any(matches[i, valid]): continue
        if single_gallery_shot:
            repeat = 10
            gids = gallery_ids[indices[i][valid]]
            inds = np.where(valid)[0]
            ids_dict = defaultdict(list)
            for j, x in zip(inds, gids):
                ids_dict[x].append(j)
        else:
            repeat = 1

        # repeat = 1
        # print('repeat')
        # print(repeat)
        for _ in range(repeat):
            if single_gallery_shot:
                # Randomly choose one instance for each id  随机选择一个实例为每个id
                sampled = (valid & _unique_sample(ids_dict, len(valid)))
                index = np.nonzero(matches[i, sampled])[0]
            else:
                index = np.nonzero(matches[i, valid])[0]
            delta = 1. / (len(index) * repeat)
            for j, k in enumerate(index):
                if k - j >= topk: break
                if first_match_break:
                    ret[k - j] += 1
                    break
                ret[k - j] += delta
        num_valid_queries += 1
    # num_valid_queries 16483
    # print('num_valid_queries')
    # print(num_valid_queries)
    if num_valid_queries == 0:
        raise RuntimeError("No valid query")
    # ret.cumsum() 累加
    return ret.cumsum() / num_valid_queries


def mean_ap(distmat, query_ids=None, gallery_ids=None,
            query_cams=None, gallery_cams=None):
    distmat = to_numpy(distmat)
    m, n = distmat.shape
    # Fill up default values
    if query_ids is None:
        query_ids = np.arange(m)
    if gallery_ids is None:
        gallery_ids = np.arange(n)
    if query_cams is None:
        query_cams = np.zeros(m).astype(np.int32)
    if gallery_cams is None:
        gallery_cams = np.ones(n).astype(np.int32)
    # Ensure numpy array    确保numpy数组
    query_ids = np.asarray(query_ids)
    gallery_ids = np.asarray(gallery_ids)
    query_cams = np.asarray(query_cams)
    gallery_cams = np.asarray(gallery_cams)
    # Sort and find correct matches     排序,找到正确的匹配
    indices = np.argsort(distmat, axis=1)
    matches = (gallery_ids[indices] == query_ids[:, np.newaxis])
    # Compute AP for each query     为每个查询计算AP
    aps = []
    for i in range(m):
        # Filter out the same id and same camera        过滤掉相同的id和相同的相机
        valid = ((gallery_ids[indices[i]] != query_ids[i]) |
                 (gallery_cams[indices[i]] != query_cams[i]))
        y_true = matches[i, valid]
        y_score = -distmat[i][indices[i]][valid]
        if not np.any(y_true): continue
        aps.append(average_precision_score(y_true, y_score))
    if len(aps) == 0:
        raise RuntimeError("No valid query")
    return np.mean(aps)
