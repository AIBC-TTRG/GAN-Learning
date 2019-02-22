#-*- coding: utf-8 -*-
from __future__ import absolute_import
from collections import OrderedDict

from torch.autograd import Variable

from ..utils import to_torch


def extract_cnn_feature(model, inputs, modules=None):
    print('======extract_cnn_feature==========')
    model.eval()
    inputs = to_torch(inputs)
    inputs = Variable(inputs, volatile=True).cuda()
    if modules is None:
        # print('sssss')
        # print('ResNet__forward_')
        outputs = model(inputs)
        outputs = outputs.data.cpu()
        # print('outputs')
        # print(outputs)
        
        return outputs
    
    # Register forward hook for each module 为每个模块注册向前钩
    outputs = OrderedDict()
    # print('outputs')
    # print(outputs)
    handles = []
    for m in modules:
        print('7777777')
        outputs[id(m)] = None
        def func(m, i, o): outputs[id(m)] = o.data.cpu()
        handles.append(m.register_forward_hook(func))
    model(inputs)
    for h in handles:
        # remove() 函数用于移除列表中某个值的第一个匹配项。
        h.remove()
    print('fanhuilo')
    return list(outputs.values())
