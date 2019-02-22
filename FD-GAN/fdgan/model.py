#-*- coding: utf-8 -*-
import os,sys
import itertools
import numpy as np
import math
import random
import copy
from collections import OrderedDict

import torch
from torch.autograd import Variable
from torch.nn import functional as F

import fdgan.utils.util as util
from fdgan.networks import get_norm_layer, init_weights, CustomPoseGenerator, NLayerDiscriminator, \
                            remove_module_key, set_bn_fix, get_scheduler, print_network
from fdgan.losses import GANLoss
from reid.models import create
from reid.models.embedding import EltwiseSubEmbed
from reid.models.multi_branch import SiameseNet

class FDGANModel(object):

    def __init__(self, opt):
        print('FDGANModel__init__')
        self.opt = opt
        self.save_dir = os.path.join(opt.checkpoints, opt.name)
        self.norm_layer = get_norm_layer(norm_type=opt.norm)
        print('self.save_dir')
        print(self.save_dir)
        print('self.norm_layer')
        print(self.norm_layer)

        self._init_models()
        self._init_losses()
        self._init_optimizers()

        print('---------- Networks initialized -------------')
        print('net_E========================')
        print_network(self.net_E)
        print('net_G========================')
        print_network(self.net_G)
        print('net_Di========================')
        print_network(self.net_Di)
        print('net_Dp========================')
        print_network(self.net_Dp)
        print('-----------------------------------------------')

    # 初始化模型
    def _init_models(self):
        print('FDGANModel___init_models_')
        # print('self.opt.pose_feature_size')
        # print(self.opt.pose_feature_size)
        # print('self.opt.noise_feature_size')
        # print(self.opt.noise_feature_size)
        # print('self.opt.drop')
        # print(self.opt.drop)
        # print('self.norm_layer')
        # print(self.norm_layer)
        # print('self.opt.fuse_mode')
        # print(self.opt.fuse_mode)
        # print('self.opt.connect_layers')
        # print(self.opt.connect_layers)
        # self.opt.pose_feature_size 128  self.opt.noise_feature_size 256
        # self.opt.drop  0.2  self.opt.fuse_mode  cat  self.opt.connect_layers 0
        self.net_G = CustomPoseGenerator(self.opt.pose_feature_size, 2048, self.opt.noise_feature_size,
                                dropout=self.opt.drop, norm_layer=self.norm_layer, fuse_mode=self.opt.fuse_mode, connect_layers=self.opt.connect_layers)
        e_base_model = create(self.opt.arch, cut_at_pooling=True)
        e_embed_model = EltwiseSubEmbed(use_batch_norm=True, use_classifier=True, num_features=2048, num_classes=2)
        self.net_E = SiameseNet(e_base_model, e_embed_model)

        di_base_model = create(self.opt.arch, cut_at_pooling=True)
        di_embed_model = EltwiseSubEmbed(use_batch_norm=True, use_classifier=True, num_features=2048, num_classes=1)
        # Di鉴别器 身份 DP鉴别器  姿势
        self.net_Di = SiameseNet(di_base_model, di_embed_model)
        self.net_Dp = NLayerDiscriminator(3+18, norm_layer=self.norm_layer)

        if self.opt.stage==1:
            print('11111111111111111111111111111111111')
            # 初始换权重net_G net_Dp
            init_weights(self.net_G)
            init_weights(self.net_Dp)
            # state_dict 是一个简单的python的字典对象,将每一层与它的对应参数建立映射关系.(如model的每一层的weights及偏置等等)
            # 只有那些参数可以训练的layer才会被保存到模型的state_dict中,如卷积层,线性层
            # state_dict 是一个python的字典格式,以字典的格式存储,然后以字典的格式被加载,而且只加载key匹配的项
            
            # 设置权重
            state_dict = remove_module_key(torch.load(self.opt.netE_pretrain))
            
            # print('state_dict11')
            # print(state_dict)

            self.net_E.load_state_dict(state_dict)
            state_dict['embed_model.classifier.weight'] = state_dict['embed_model.classifier.weight'][1]
            state_dict['embed_model.classifier.bias'] = torch.FloatTensor([state_dict['embed_model.classifier.bias'][1]])
            
            # print('state_dict22')
            # print(state_dict['embed_model.classifier.weight'])

            self.net_Di.load_state_dict(state_dict)
        elif self.opt.stage==2:
            print('22222222222222')
            self._load_state_dict(self.net_E, self.opt.netE_pretrain)
            self._load_state_dict(self.net_G, self.opt.netG_pretrain)
            self._load_state_dict(self.net_Di, self.opt.netDi_pretrain)
            self._load_state_dict(self.net_Dp, self.opt.netDp_pretrain)
        else:
            assert('unknown training stage')

        # 模块的并行处理
        self.net_E = torch.nn.DataParallel(self.net_E).cuda()
        self.net_G = torch.nn.DataParallel(self.net_G).cuda()
        self.net_Di = torch.nn.DataParallel(self.net_Di).cuda()
        self.net_Dp = torch.nn.DataParallel(self.net_Dp).cuda()

    def reset_model_status(self):
        print('FDGANModel___reset_model_status_')
        if self.opt.stage==1:
            # train  eval 来切换模型的 训练测试阶段
            print('111111111111111')
            self.net_G.train()
            print('22222222222222222')
            self.net_Dp.train()
            print('3333333333333333')
            self.net_E.eval()
            print('444444444444444444')
            self.net_Di.train()
            print('555555555555555555')
            self.net_Di.apply(set_bn_fix)
            print('666666666')
        elif self.opt.stage==2:
            self.net_E.train()
            self.net_G.train()
            self.net_Di.train()
            self.net_Dp.train()
            self.net_E.apply(set_bn_fix)
            self.net_Di.apply(set_bn_fix)

    def _load_state_dict(self, net, path):
        print('FDGANModel___load_state_dict_')
        state_dict = remove_module_key(torch.load(path))
        net.load_state_dict(state_dict)

    # 初始化损失  criterionGAN_D  criterionGAN_G
    def _init_losses(self):
        print('FDGANModel___init_losses_')
        print('self.opt.smooth_label')
        print(self.opt.smooth_label)
        if self.opt.smooth_label:
            print('aaaaaa')
            self.criterionGAN_D = GANLoss(smooth=True).cuda()
            self.rand_list = [True] * 1 + [False] * 10000
        else:
            print('bbbbbb')
            self.criterionGAN_D = GANLoss(smooth=False).cuda()
            self.rand_list = [False]
        print('gggggggggggggggggggg')
        self.criterionGAN_G = GANLoss(smooth=False).cuda()

    # 初始化优化
    # optimizer_G  Adam net_G
    # optimizer_Di SGD net_Di
    # optimizer_Dp SGD net_Dp
    def _init_optimizers(self):
        print('FDGANModel___init_optimizers_')
        if self.opt.stage==1:
            print('GGGGGGGoooooo')
            self.optimizer_G = torch.optim.Adam(self.net_G.parameters(),
                                                lr=self.opt.lr*0.1, betas=(0.5, 0.999))
            print('DiDIdidididididoooo')
            self.optimizer_Di = torch.optim.SGD(self.net_Di.parameters(),
                                                lr=self.opt.lr*0.01, momentum=0.9, weight_decay=1e-4)
            print('DPDPDPDPooooo')
            self.optimizer_Dp = torch.optim.SGD(self.net_Dp.parameters(),
                                                lr=self.opt.lr, momentum=0.9, weight_decay=1e-4)
        elif self.opt.stage==2:
            param_groups = [{'params': self.net_E.module.base_model.parameters(), 'lr_mult': 0.1},
                            {'params': self.net_E.module.embed_model.parameters(), 'lr_mult': 1.0},
                            {'params': self.net_G.parameters(), 'lr_mult': 0.1}]
            self.optimizer_G = torch.optim.Adam(param_groups,
                                                lr=self.opt.lr*0.1, betas=(0.5, 0.999))
            self.optimizer_Di = torch.optim.SGD(self.net_Di.parameters(),
                                                lr=self.opt.lr, momentum=0.9, weight_decay=1e-4)
            self.optimizer_Dp = torch.optim.SGD(self.net_Dp.parameters(),
                                                lr=self.opt.lr, momentum=0.9, weight_decay=1e-4)

        self.schedulers = []
        self.optimizers = []
        self.optimizers.append(self.optimizer_G) 
        self.optimizers.append(self.optimizer_Di)
        self.optimizers.append(self.optimizer_Dp)
        for optimizer in self.optimizers:
            self.schedulers.append(get_scheduler(optimizer, self.opt))

    def set_input(self, input):
        print('FDGANModel___set_input_')
        # pid   torch.LongTensor of size 1x1
        # origin torch.FloatTensor of size 1x3x256x128
        # posemap  torch.FloatTensor of size 1x18x256x128
        # target torch.FloatTensor of size 1x3x256x128
        input1, input2 = input

        # print('input1')
        # # print(type(input1['origin']))
        # # print(input1['origin'].size)
        # print(input1)

        # print('input2')
        # # print(type(input1['origin']))
        # # print(input1['origin'].size)
        # print(input2)
        
        # print('input1[origin')
        # print(type(input2['origin']))
        # print(input2['origin'].size)
        # print(input1['origin'])

        # print('input1[posemap')
        # print(type(input2['posemap']))
        # print(input2['posemap'].size)
        # print(input1['posemap'])

        # print('input1[target')
        # print(type(input2['target']))
        # print(input2['target'].size)
        # print(input1['target'])

        # print('input2[origin')
        # print(type(input2['origin']))
        # print(input2['origin'].size)
        # print(input2['origin'])

        # print('input2[posemap')
        # print(type(input2['posemap']))
        # print(input2['posemap'].size)
        # print(input2['posemap'])

        # print('input2[target')
        # print(type(input2['target']))
        # print(input2['target'].size)
        # print(input2['target'])

        # print('input1[pid')
        # print(type(input2['pid']))
        # print(input2['pid'].size)
        # print(input1['pid'])

        # print('input2[pid')
        # print(type(input2['pid']))
        # print(input2['pid'].size)
        # print(input2['pid'])

        # labels  0 不一样
        # labels  1 一样
        labels = (input1['pid']==input2['pid']).long()

        print('labels')
        # # print(type(input1['origin']))
        # # print(input1['origin'].size)
        # print(labels)
        # print(labels.view(-1,1,1,1))

        # labels.size(0)  1
        # 返回一个张量，包含了从标准正态分布(均值为0，方差为 1，即高斯白噪声)中抽取一组随机数，形状由可变参数sizes定义
        noise = torch.randn(labels.size(0), self.opt.noise_feature_size)

        # keep the same pose map for persons with the same identity
        # 保持相同的给人带来映射用同样的身份
        #  a.view()创建了一个新的对象
        # view函数相当于numpy的reshape
        mask = labels.view(-1,1,1,1).expand_as(input1['posemap']) 
        input2['posemap'] = input1['posemap']*mask.float() + input2['posemap']*(1-mask.float())

        # print(mask)
        # print('input2[posemap')
        # print(type(input2['posemap']))
        # print(input2['posemap'].size)
        # print(input2['posemap'])

        mask = labels.view(-1,1,1,1).expand_as(input1['target'])
        input2['target'] = input1['target']*mask.float() + input2['target']*(1-mask.float())
        # print(mask)
        # print('input2[target')
        # print(type(input2['target']))
        # print(input2['target'].size)
        # print(input2['target'])

        # 在给定维度上对输入的张量序列seq 进行连接操作
        origin = torch.cat([input1['origin'], input2['origin']])
        target = torch.cat([input1['target'], input2['target']])
        posemap = torch.cat([input1['posemap'], input2['posemap']])
        noise = torch.cat((noise, noise))
        # print('input1[origin]')
        # # print(type(input1['origin']))
        # # print(input1['origin'].size)
        # print(input1['origin'])

        # print('input2[origin')
        # print(type(input2['origin']))
        # print(input2['origin'].size)
        # print(input2['origin'])
        # oprint('input1[origin]')
        # # print(type(input1['origin']))
        # # print(input1['origin'].size)
        # print(input1['origin'])rigin  torch.FloatTensor of size 2x3x256x128\
        # target torch.FloatTensor of size 2x3x256x128
        # posemap torch.FloatTensor of size 2x18x256x128
        # noise torch.FloatTensor of size 2x256
        # print('origin')
        # print(type(origin))
        # print(origin.size)
        # print(origin)

        # print('target')
        # print(type(target))
        # print(target.size)
        # print(target)

        # print('posemap')
        # print(type(posemap))
        # print(posemap.size)
        # print(posemap)

        # print('noise')
        # print(type(noise))
        # print(noise.size)
        # print(noise)

        self.origin = origin.cuda()
        self.target = target.cuda()
        self.posemap = posemap.cuda()
        self.labels = labels.cuda()
        self.noise = noise.cuda()

    def forward(self):
        print('FDGANModel_forward')
        # A torch.cuda.FloatTensor of size 2x3x256x128 (GPU 0)  type(A) class 'torch.autograd.variable.Variable'
        A = Variable(self.origin)
        B_map = Variable(self.posemap)
        z = Variable(self.noise)
        bs = A.size(0)
        # print(A)
        # print(type(A))
        print(len(A))
        print('bs')
        print(bs)
        # A_id1 = A[:bs//2] [torch.cuda.FloatTensor of size 1x2048 (GPU 0)]
        # A_id2 =  [torch.cuda.FloatTensor of size 1x2048 (GPU 0)]
        # id_score分类的分数
        A_id1, A_id2, self.id_score = self.net_E(A[:bs//2], A[bs//2:])
        # print('A_id1')
        # # print(type(input1['origin']))
        # # print(input1['origin'].size)
        # print(A_id1)

        # print('A_id2')
        # # print(type(input1['origin']))
        # # print(input1['origin'].size)
        # print(A_id2)

        # print('self.id_score')
        # # print(type(input1['origin']))
        # # print(input1['origin'].size)
        # print(self.id_score)

        A_id = torch.cat((A_id1, A_id2))

        # print('A_id.size(0)')
        # # print(type(input1['origin']))
        # # print(input1['origin'].size)
        # print(A_id.size(0))

        # print('A_id.size(1)')
        # # print(type(input1['origin']))
        # # print(input1['origin'].size)
        # print(A_id.size(1))
        
        
        # self.fake torch.cuda.FloatTensor of size 2x3x256x128 (GPU 0)
        # A_id.size(0)  2 A_id.size(1)    2048

        self.fake = self.net_G(B_map, A_id.view(A_id.size(0), A_id.size(1), 1, 1), z.view(z.size(0), z.size(1), 1, 1))
        # print('self.fake ')
        # print(self.fake)

    def backward_Dp(self):
        print('FDGANModel_backward_Dp')
        # detach就是截断反向传播的梯度流
        real_pose = torch.cat((Variable(self.posemap), Variable(self.target)),dim=1)
        fake_pose = torch.cat((Variable(self.posemap), self.fake.detach()),dim=1)
        # pred_real  torch.cuda.FloatTensor of size 2x1x30x14 (GPU 0)
        # pred_fake  torch.cuda.FloatTensor of size 2x1x30x14 (GPU 0)
        pred_real = self.net_Dp(real_pose)
        pred_fake = self.net_Dp(fake_pose)

        # print('pred_real')
        # # print(type(input1['origin']))
        # # print(input1['origin'].size)
        # print(pred_real)

        # print('pred_fake')
        # # print(type(input1['origin']))
        # # print(input1['origin'].size)
        # print(pred_fake)

        # print('rand_listppp')
        # print(self.rand_list)

        if random.choice(self.rand_list):
            print('ifififififif')
            loss_D_real = self.criterionGAN_D(pred_fake, True)
            loss_D_fake = self.criterionGAN_D(pred_real, False)
        else:
            loss_D_real = self.criterionGAN_D(pred_real, True)
            loss_D_fake = self.criterionGAN_D(pred_fake, False)
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        print('loss_Dp')
        print(loss_D)
        loss_D.backward()
        self.loss_Dp = loss_D.data[0]

    def backward_Di(self):
        print('FDGANModel_backward_Di')
        # self.fake.detach() 返回一个新的 从当前图中分离的 Variable。
        _, _, pred_real = self.net_Di(Variable(self.origin), Variable(self.target))
        # print('pred_real')
        # print(pred_real)
        _, _, pred_fake = self.net_Di(Variable(self.origin), self.fake.detach())
        # print('pred_fake')
        # print(pred_fake)

        # print('rand_list')
        # print(self.rand_list)
        # random.choice()，生成指定size的随机数
        if random.choice(self.rand_list):
            print('111111111111111')
            loss_D_real = self.criterionGAN_D(pred_fake, True)
            loss_D_fake = self.criterionGAN_D(pred_real, False)
        else:
            print('22222222222222222')
            loss_D_real = self.criterionGAN_D(pred_real, True)
            print('hihihi')
            loss_D_fake = self.criterionGAN_D(pred_fake, False)
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        print('loss_Di')
        print(loss_D)
        # 损失回传
        loss_D.backward()
        print('jijiji')
        self.loss_Di = loss_D.data[0]

    def backward_G(self):
        print('FDGANModel_backward_G')
        # cross_entropy交叉熵
        loss_v = F.cross_entropy(self.id_score, Variable(self.labels).view(-1))
        loss_r = F.l1_loss(self.fake, Variable(self.target))
        fake_1 = self.fake[:self.fake.size(0)//2]        
        fake_2 = self.fake[self.fake.size(0)//2:]
        # print('loss_v')
        # # print(type(input1['origin']))
        # # print(input1['origin'].size)
        # print(loss_v)


        # print('loss_r')
        # # print(type(input1['origin']))
        # # print(input1['origin'].size)
        # print(loss_r)

        # print('fake.size(0)')
        # # print(type(input1['origin']))
        # # print(input1['origin'].size)
        # print(self.fake.size(0))

        # print('fake_1')
        # # print(type(input1['origin']))
        # # print(input1['origin'].size)
        # print(fake_1)

        # print('fake_2')
        # # print(type(input1['origin']))
        # # print(input1['origin'].size)
        # print(fake_2)

        # fake_2  torch.cuda.FloatTensor of size 1x3x256x128 (GPU 0)    
        # fake_1[self.labels.view(self.labels.size(0),1,1,1).expand_as(fake_1)==1 [torch.cuda.FloatTensor of size 98304 (GPU 0)]    
        print('1111111111111111111111111111')

        # print(fake_1[self.labels.view(self.labels.size(0),1,1,1).expand_as(fake_1)==1])
        # loss_sp = F.l1_loss(fake_1[self.labels.view(self.labels.size(0),1,1,1).expand_as(fake_1)==1], 
        #                     fake_2[self.labels.view(self.labels.size(0),1,1,1).expand_as(fake_1)==1])

        _, _, pred_fake_Di = self.net_Di(Variable(self.origin), self.fake)
        pred_fake_Dp = self.net_Dp(torch.cat((Variable(self.posemap),self.fake),dim=1))
        loss_G_GAN_Di = self.criterionGAN_G(pred_fake_Di, True)
        loss_G_GAN_Dp = self.criterionGAN_G(pred_fake_Dp, True)

        loss_G = loss_G_GAN_Di + loss_G_GAN_Dp + \
                loss_r * self.opt.lambda_recon + \
                loss_v * self.opt.lambda_veri 
                # loss_sp * self.opt.lambda_sp
        print('loss_G')
        print(loss_G)
        loss_G.backward()

        del self.id_score
        self.loss_G = loss_G.data[0]
        self.loss_v = loss_v.data[0]
        # self.loss_sp = loss_sp.data[0]
        self.loss_r = loss_r.data[0]
        self.loss_G_GAN_Di = loss_G_GAN_Di.data[0]
        self.loss_G_GAN_Dp = loss_G_GAN_Dp.data[0]
        self.fake = self.fake.data

    def optimize_parameters(self):
        print('FDGANModel_optimize_parameters')
        self.forward()

        print('aaaaaa')
        self.optimizer_Di.zero_grad()       #梯度清零 进行反向传播
        self.backward_Di()
        self.optimizer_Di.step()                        #更新权重

        self.optimizer_Dp.zero_grad()
        self.backward_Dp()
        self.optimizer_Dp.step()            #更新权重

        self.optimizer_G.zero_grad()
        self.backward_G()
        print('lalalal')    
        self.optimizer_G.step()             #更新权重

    def get_current_errors(self):
        print('FDGANModel_get_current_errors')
        return OrderedDict([('G_v', self.loss_v),
                            ('G_r', self.loss_r),
                            # ('G_sp', self.loss_sp),
                            ('G_gan_Di', self.loss_G_GAN_Di),
                            ('G_gan_Dp', self.loss_G_GAN_Dp),
                            ('D_i', self.loss_Di),
                            ('D_p', self.loss_Dp)
                            ])

    def get_current_visuals(self):
        print('FDGANModel_get_current_visuals')
        input = util.tensor2im(self.origin)
        target = util.tensor2im(self.target)
        fake = util.tensor2im(self.fake)
        map = self.posemap.sum(1)
        map[map>1]=1
        map = util.tensor2im(torch.unsqueeze(map,1))
        return OrderedDict([('input', input), ('posemap', map), ('fake', fake), ('target', target)])

    def save(self, epoch):
        print('FDGANModel_save')
        self.save_network(self.net_E, 'E', epoch)
        self.save_network(self.net_G, 'G', epoch)
        self.save_network(self.net_Di, 'Di', epoch)
        self.save_network(self.net_Dp, 'Dp', epoch)

    def save_network(self, network, network_label, epoch_label):
        print('FDGANModel_save_network')
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(network.state_dict(), save_path)

    def update_learning_rate(self):
        print('FDGANModel_update_learning_rate')
        for scheduler in self.schedulers:
            scheduler.step()
        lr = self.optimizers[0].param_groups[0]['lr']