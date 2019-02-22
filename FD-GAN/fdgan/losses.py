#-*- coding: utf-8 -*-
from __future__ import absolute_import
import os, sys
import functools
import random

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
from torch.nn import init

class GANLoss(nn.Module):
    def __init__(self, smooth=False):
        print('GANLoss__init__')
        super(GANLoss, self).__init__()
        self.smooth = smooth

    def get_target_tensor(self, input, target_is_real):
        print('GANLoss__get_target_tensor__')
        real_label = 1.0
        fake_label = 0.0
        if self.smooth:
            print('1111111111111')
            # random.uniform() 方法将随机生成下一个实数，它在 [x, y) 范围内。
            real_label = random.uniform(0.7,1.0)
            fake_label = random.uniform(0.0,0.3)
        if target_is_real:
            print('2222222222222222222222')
            # torch.ones_like返回填充了标量值 1 的张量，大小与输入相同
            target_tensor = torch.ones_like(input).fill_(real_label)
        else:
            # torch.zeros_like返回填充了标量值 1 的张量，大小与输入相同
            target_tensor = torch.zeros_like(input).fill_(fake_label)
        return target_tensor

    def __call__(self, input, target_is_real):
        print('GANLoss____call____')
        target_tensor = self.get_target_tensor(input, target_is_real)
        input = F.sigmoid(input)
        # binary_cross_entropy亦称作对数损失
        return F.binary_cross_entropy(input, target_tensor)