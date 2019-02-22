#-*- coding: utf-8 -*-
from __future__ import absolute_import
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data.sampler import (
    Sampler, SequentialSampler, RandomSampler, SubsetRandomSampler,
    WeightedRandomSampler)

def _choose_from(start, end, excluded_range=None, size=1, replace=False):
    num = end - start + 1
    if excluded_range is None:
        return np.random.choice(num, size=size, replace=replace) + start
    ex_start, ex_end = excluded_range
    num_ex = ex_end - ex_start + 1
    num -= num_ex
    inds = np.random.choice(num, size=size, replace=replace) + start
    inds += (inds >= ex_start) * num_ex
    return inds

class RandomPairSampler(Sampler):
    def __init__(self, data_source, neg_pos_ratio=1):
        print('RandomPairSampler__init__')
        super(RandomPairSampler, self).__init__(data_source)
        self.data_source = data_source
        self.num_samples = len(data_source)
        self.neg_pos_ratio = neg_pos_ratio
        # Sort by pid       按pid排序
        indices = np.argsort(np.asarray(data_source)[:, 1])
        self.index_map = dict(zip(np.arange(self.num_samples), indices))
        # print('indices')
        # print(indices)

        # print('self.index_map')
        # print(self.index_map)
        # Get the range of indices for each pid     为每个pid得到指标的范围
        self.index_range = defaultdict(lambda: [self.num_samples, -1])
        for i, j in enumerate(indices):
            _, pid, _ = data_source[j]
            # print('data_source[j]')
            # print(data_source[j])
            self.index_range[pid][0] = min(self.index_range[pid][0], i)
            self.index_range[pid][1] = max(self.index_range[pid][1], i)

    def __iter__(self):
        print('RandomPairSampler____iter____')
        indices = np.random.permutation(self.num_samples)
        for i in indices:
            # anchor sample
            anchor_index = self.index_map[i]
            _, pid, _ = self.data_source[anchor_index]
            # positive sample   正样本
            start, end = self.index_range[pid]
            pos_index = _choose_from(start, end, excluded_range=(i, i))[0]
            yield anchor_index, self.index_map[pos_index]
            # negative samples  负样本
            neg_indices = _choose_from(0, self.num_samples - 1,
                                       excluded_range=(start, end),
                                       size=self.neg_pos_ratio)
            for neg_index in neg_indices:
                yield anchor_index, self.index_map[neg_index]

    def __len__(self):
        print('RandomPairSampler_____len_____')
        return self.num_samples * (1 + self.neg_pos_ratio)