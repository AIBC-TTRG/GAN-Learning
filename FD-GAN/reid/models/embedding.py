#-*- coding: utf-8 -*-
import math
import copy
from torch import nn
import torch
import torch.nn.functional as F

class EltwiseSubEmbed(nn.Module):
    def __init__(self, nonlinearity='square', use_batch_norm=False,
                 use_classifier=False, num_features=0, num_classes=0):
        print('EltwiseSubEmbed__init__')
        super(EltwiseSubEmbed, self).__init__()
        self.nonlinearity = nonlinearity
        if nonlinearity is not None and nonlinearity not in ['square', 'abs']:
            raise KeyError("Unknown nonlinearity:", nonlinearity)
        self.use_batch_norm = use_batch_norm
        self.use_classifier = use_classifier
        print('self.use_batch_norm')
        print(self.use_batch_norm)
        print('self.use_classifier')
        print(self.use_classifier)
        if self.use_batch_norm:
            print('1111111111')
            self.bn = nn.BatchNorm1d(num_features)
            # m.weight.data是卷积核参数, m.bias.data是偏置项参数
            self.bn.weight.data.fill_(1)
            self.bn.bias.data.zero_()
            # print('self.bn.weight.data.fill_(1)')
            # print(self.bn.weight.data.fill_(1))
        if self.use_classifier:
            print('22222222')
            assert num_features > 0 and num_classes > 0
            self.classifier = nn.Linear(num_features, num_classes)
            self.classifier.weight.data.normal_(0, 0.001)
            self.classifier.bias.data.zero_()

    def forward(self, x1, x2):
        # print('EltwiseSubEmbed_forward_')
        x = x1 - x2
        if self.nonlinearity == 'square':
            x = x.pow(2)
        elif self.nonlinearity == 'abs':
            x = x.abs()
        if self.use_batch_norm:
            x = self.bn(x)
        # x.size()  具体位置一般都是在调用分类器之前。分类器是一个简单的nn.Linear()结构，输入输出都是维度为一的值
        if self.use_classifier:
            x = x.view(x.size(0),-1)
            x = self.classifier(x)
        else:
            x = x.sum(1)

        return x