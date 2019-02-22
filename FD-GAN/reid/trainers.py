from __future__ import print_function, absolute_import
import time

import torch
from torch.autograd import Variable

from .evaluation_metrics import accuracy
from .utils.meters import AverageMeter


class BaseTrainer(object):
    def __init__(self, model, criterion, num_classes=0, num_instances=4):
        print('BaseTrainer__init__')
        super(BaseTrainer, self).__init__()
        self.model = model
        self.criterion = criterion
        self.num_classes = num_classes
        self.num_instances = num_instances

    def train(self, epoch, data_loader, optimizer, base_lr=0.1, print_freq=1):
        
        print('BaseTrainer_train_')
        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        precisions = AverageMeter()

        end = time.time()
        for i, inputs in enumerate(data_loader):
            data_time.update(time.time() - end)

            print('inputs')
            print(inputs)
            inputs, targets = self._parse_data(inputs)
            
            print('inputs')
            print(inputs)
            print('targets')
            print(targets)
            loss, prec1 = self._forward(inputs, targets)
            print('loss')
            print(loss)
            print('prec1')
            print(prec1)

            losses.update(loss.data[0], targets.size(0))
            precisions.update(prec1, targets.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()
            # print('precisions.val :%.2f%%' %(precisions.val))
            # print('#precisions.val = %.2f%%' % (precisions.val*100))
            # print('precisions.val')
            # print(type(precisions.val))
            # print(precisions.val)
            # print('precisions.avg :%.2f%%'%(precisions.avg))
            # print('precisions.avg')
            # print(type(precisions.avg))
            # print(precisions.avg)

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\t'
                    #   'Prec {:.2f} ({:.2f})\t'
                      .format(epoch, i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg
                               ))
                print('#Prec.val = %.2f%%' % (precisions.val*100))
                print('#Prec.avg = %.2f%%' % (precisions.val*100))
                            #   precisions.val, precisions.avg

    def _parse_data(self, inputs):
        print('_parse_data')
        raise NotImplementedError

    def _forward(self, inputs, targets):
        print('_parse_data') 
        raise NotImplementedError

class SiameseTrainer(BaseTrainer):
    def _parse_data(self, inputs):
        print('SiameseTrainer_parse_data')
        (imgs1, _, pids1, _), (imgs2, _, pids2, _) = inputs
        inputs = [Variable(imgs1), Variable(imgs2)]
        targets = Variable((pids1 == pids2).long().cuda())
        return inputs, targets

    def _forward(self, inputs, targets):
        print('SiameseTrainer__forward')
        _, _, outputs = self.model(*inputs)
        print('outputs')
        print(outputs)
        loss = self.criterion(outputs, targets)
        prec1 = accuracy(outputs.data, targets.data)
        print('outputs.data')
        print(outputs.data)
        print('targets.data')
        print(targets.data)
        return loss, prec1[0]