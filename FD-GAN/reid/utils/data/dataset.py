#-*- coding: utf-8 -*-
from __future__ import print_function
import os.path as osp

import numpy as np

from ..serialization import read_json

def _pluck(identities, indices, relabel=False):
    ret = []
    query = {}
    # enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，
    # 同时列出数据和数据下标，一般用在 for 循环当中。
    for index, pid in enumerate(indices):
        pid_images = identities[pid]
        if relabel:
            if index not in query.keys():
                query[index] = []
        else:
            if pid not in query.keys():
                query[pid] = []
        for camid, cam_images in enumerate(pid_images):
            for fname in cam_images:
                # os.path.splitext(path)  #分割路径，返回路径名和文件扩展名的元组
                name = osp.splitext(fname)[0]   #文件名
                # name.split 以_分割开
                # map() 会根据提供的函数对指定序列做映射
                x, y, _ = map(int, name.split('_'))
                assert pid == x and camid == y
                if relabel:
                    ret.append((fname, index, camid))
                    query[index].append(fname)
                else:
                    ret.append((fname, pid, camid))
                    query[pid].append(fname)
    return ret, query

class Dataset(object):
    def __init__(self, root, split_id=0):
        self.root = root
        self.split_id = split_id
        self.meta = None
        self.split = None
        self.train, self.val, self.trainval = [], [], []
        self.query, self.gallery = [], []
        self.num_train_ids, self.num_val_ids, self.num_trainval_ids = 0, 0, 0

    def __len__(self):
        return

    @property
    def images_dir(self):
        return osp.join(self.root, 'images')

    @property
    def poses_dir(self):
        return osp.join(self.root, 'poses')

    def load(self, num_val=0.3, verbose=True):
        print('Market1501__load__')
        splits = read_json(osp.join(self.root, 'splits.json'))
        # print('***************')
        # print(splits)
        # print('============')
        # print(len(splits))
        # print('============')
        # print(self.split_id)
        
        if self.split_id >= len(splits):
            raise ValueError("split_id exceeds total splits {}"
                             .format(len(splits)))
        self.split = splits[self.split_id]
        # print('-------------')
        # print(self.split)

        # sorted进行排序（从2.4开始），返回副本，原始输入不变
        # np.asarray()将结构数据转化为ndarray
        trainval_pids = sorted(np.asarray(self.split['trainval']))
        # print('-------------')
        # print(trainval_pids)

        # num = 751
        num = len(trainval_pids)
        # print('num---')
        # print(num)
        # print('num_val---')
        # print(num_val)

        # isinstance() 函数来判断一个对象是否是一个已知的类型
        if isinstance(num_val, float):
            # round() 方法返回浮点数x的四舍五入值
            num_val = int(round(num * num_val))
            print('num_val---')
            # print(num_val)
        if num_val >= num or num_val < 0:
            raise ValueError("num_val exceeds total identities {}"
                             .format(num))
        # num_val = 100  train_pids = 数组前651  val_pids 数组后100
        train_pids = sorted(trainval_pids[:-num_val])
        val_pids = sorted(trainval_pids[-num_val:])
        # print('-------------')
        # print(train_pids)
        # print('----val_pids------')
        # print(val_pids)

        self.meta = read_json(osp.join(self.root, 'meta.json'))
        identities = self.meta['identities']
        
        # _pluck（）  return :ret, query
        self.train, self.train_query = _pluck(identities, train_pids, relabel=True)
        self.val, self.val_query = _pluck(identities, val_pids, relabel=True)
        self.trainval, self.trainval_query = _pluck(identities, trainval_pids, relabel=True)
        self.query, self.query_query = _pluck(identities, self.split['query'])
        self.gallery, self.gallery_query = _pluck(identities, self.split['gallery'])
        self.num_train_ids = len(train_pids)
        self.num_val_ids = len(val_pids)
        self.num_trainval_ids = len(trainval_pids)
        # print('-self.meta-------')
        # print(self.meta)
        # print('---identities-----')
        # print(identities)
        # print('-----self.train----')
        # print(self.train)
        # print('----self.train_query------')
        # print(self.train_query)
        # print('----self.val------')
        # print(self.val)
        # print('----self.val_query------')
        # print(self.val_query)
        # print('------self.trainval---')
        # print(self.trainval)
        # print('----self.trainval_query------')
        # print(self.trainval_query)
        # print('-----self.query-----')
        # print(self.query)
        # print('----self.query_query------')
        # print(self.query_query)
        # print('----self.gallery------')
        # print(self.gallery)
        # print('----self.gallery_query------')
        # print(self.gallery_query)

        # print('----num_train_ids------')
        # print(self.num_train_ids)
        # print('----num_val_ids------')
        # print(self.num_val_ids)
        # print('----num_trainval_ids------')
        # print(self.num_trainval_ids)

        if verbose:
            print(self.__class__.__name__, "dataset loaded")
            print("  subset   | # ids | # images")
            print("  ---------------------------")
            print("  train    | {:5d} | {:8d}"
                  .format(self.num_train_ids, len(self.train)))
            print("  val      | {:5d} | {:8d}"
                  .format(self.num_val_ids, len(self.val)))
            print("  trainval | {:5d} | {:8d}"
                  .format(self.num_trainval_ids, len(self.trainval)))
            print("  query    | {:5d} | {:8d}"
                  .format(len(self.split['query']), len(self.query)))
            print("  gallery  | {:5d} | {:8d}"
                  .format(len(self.split['gallery']), len(self.gallery)))

    def _check_integrity(self):
        return osp.isdir(osp.join(self.root, 'images')) and \
               osp.isfile(osp.join(self.root, 'meta.json')) and \
               osp.isfile(osp.join(self.root, 'splits.json')) and \
               osp.isdir(osp.join(self.root, 'poses'))
