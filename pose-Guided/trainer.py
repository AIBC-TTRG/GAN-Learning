#-*- coding: utf-8 -*-


from __future__ import print_function

import matplotlib as mpl
mpl.use('Agg')

import matplotlib.pyplot as plt 
import scipy.misc

import os, pdb
import StringIO
import scipy.misc
import numpy as np
import glob
from tqdm import trange
from itertools import chain
from collections import deque
import pickle, shutil
from tqdm import tqdm
# from PIL import img
import matplotlib.pyplot as plt
import tensorflow as tf
# tf.reset_default_graph()
import numpy as np

from tensorflow.python.ops import control_flow_ops

from tensorflow.python import pywrap_tensorflow

from models import *
from utils import save_image, _getPoseMask, _getSparsePose, _sparse2dense, _get_valid_peaks
import tflib as lib
from wgan_gp import *

# tf.reset_default_graph()
tf.Graph().as_default()

def next(loader):
    return loader.next()[0].data.numpy()

def to_nhwc(image, data_format):
    if data_format == 'NCHW':
        new_image = nchw_to_nhwc(image)
    else:
        new_image = image
    return new_image

def to_nchw_numpy(image):
    if image.shape[3] in [1, 3]:
        new_image = image.transpose([0, 3, 1, 2])
    else:
        new_image = image
    return new_image

def norm_img(image, data_format=None):
    image = image/127.5 - 1.
    if data_format:
        image = to_nhwc(image, data_format)
    return image

def denorm_img(norm, data_format):
#  clip_by_value  功能：可以将一个张量中的数值限制在一个范围之内。（可以避免一些运算错误）
# 参数：（1）v：input数据（2）a、b是对数据的限制。
# 当v小于a时，输出a；
# 当v大于a小于b时，输出原值；
# 当v大于b时，输出b； 
    return tf.clip_by_value(to_nhwc((norm + 1)*127.5, data_format), 0, 255)

def slerp(val, low, high):
    """Code from https://github.com/soumith/dcgan.torch/issues/14"""
    omega = np.arccos(np.clip(np.dot(low/np.linalg.norm(low), high/np.linalg.norm(high)), -1, 1))
    so = np.sin(omega)
    if so == 0:
        return (1.0-val) * low + val * high # L'Hopital's rule/LERP
    return np.sin((1.0-val)*omega) / so * low + np.sin(val*omega) / so * high

from datasets import market1501, dataset_utils
import utils_wgan
from skimage.measure import compare_ssim as ssim
from skimage.color import rgb2gray
from PIL import Image
from tensorflow.python.ops import sparse_ops

class PG2(object):
    def _common_init(self, config):
        # 一般的初始化
        print('PG2_common_init')
        self.config = config
        self.data_loader = None
        self.dataset = config.dataset

        # Adam  optimizer with β1 β2
        self.beta1 = config.beta1   #0.5
        self.beta2 = config.beta2   #0.999
        self.optimizer = config.optimizer   #adam
        self.batch_size = config.batch_size

        # 定义图变量tf.Variable(initializer,name),参数initializer是初始化参数，name是可自定义的变量名称
        # trainable如果为True，会把它加入到GraphKeys.TRAINABLE_VARIABLES，才能对它使用Optimizer
        self.step = tf.Variable(config.start_step, name='step', trainable=False)

        self.g_lr = tf.Variable(config.g_lr, name='g_lr')   #0.00008
        self.d_lr = tf.Variable(config.d_lr, name='d_lr')   #0.00008

        # 这个函数的功能主要是把g_lr的值变为g_lr * 0.5 
        self.g_lr_update = tf.assign(self.g_lr, self.g_lr * 0.5, name='g_lr_update')
        self.d_lr_update = tf.assign(self.d_lr, self.d_lr * 0.5, name='d_lr_update')

        self.gamma = config.gamma   #0.5
        self.lambda_k = config.lambda_k #0.001

        self.z_num = config.z_num   #64
        self.conv_hidden_num = config.conv_hidden_num   #128
        self.img_H, self.img_W = config.img_H, config.img_W     #128*64

        self.model_dir = config.model_dir
        self.load_path = config.load_path

        self.use_gpu = config.use_gpu
        self.data_format = config.data_format   #NHWC

        _, self.height, self.width, self.channel = self._get_conv_shape()   #128*64  3
        self.repeat_num = int(np.log2(self.height)) - 2         #计算各元素的以2为底的对数

        self.data_path = config.data_path
        self.pretrained_path = config.pretrained_path   #None

        # self.checkpoint_path = config.checkpoint_path   #None

        self.ckpt_path = config.ckpt_path       #NONE
        self.start_step = config.start_step     #0
        self.log_step = config.log_step     #200
        self.max_step = config.max_step     #500000
        # self.save_model_secs = config.save_model_secs
        self.lr_update_step = config.lr_update_step     #100000

        self.is_train = config.is_train
        if self.is_train:
            self.num_threads = 4
            self.capacityCoff = 2
        else: # during testing to keep the order of the input data
            # 在测试过程中保持输入数据的顺序
            self.num_threads = 1
            self.capacityCoff = 1

    def __init__(self, config):
        # Trainer.__init__(self, config, data_loader=None)
        print('PG2__init__')

        self._common_init(config)

        self.keypoint_num = 18      #18个关键点
        self.D_arch = config.D_arch     #DCGAN
        # Datasets：一种为 TensorFlow 模型创建输入管道的新方式。
        # lower（）字符串中所有大写字符转换为小写后生成的字符串。
        if 'market' in config.dataset.lower():
            if config.is_train:
                # 数据集元组阅读Market1501指示。
                self.dataset_obj = market1501.get_split('train', config.data_path)
            else:
                self.dataset_obj = market1501.get_split('test', config.data_path)

        if config.test_one_by_one:      #  test_one_by_one False
            # tf.placeholder 此函数可以理解为形参，用于定义过程，在执行的时候再赋具体的值
                # dtype：数据类型。常用的是tf.float32,tf.float64等数值类型
                # shape：数据形状。默认是None，就是一维值，也可以是多维，比如[2,3], [None, 3]表示列是3，行不定
                # name：名称。
            self.x = tf.placeholder(tf.float32, shape=(None, self.img_H, self.img_W, 3))
            self.x_target = tf.placeholder(tf.float32, shape=(None, self.img_H, self.img_W, 3))
            self.pose = tf.placeholder(tf.float32, shape=(None, self.img_H, self.img_W, 18))
            self.pose_target = tf.placeholder(tf.float32, shape=(None, self.img_H, self.img_W, 18))
            self.mask = tf.placeholder(tf.float32, shape=(None, self.img_H, self.img_W, 1))
            self.mask_target = tf.placeholder(tf.float32, shape=(None, self.img_H, self.img_W, 1))
        else:
            # images_0, images_1, poses_0, poses_1, masks_0, masks_1
            self.x, self.x_target, self.pose, self.pose_target, self.mask, self.mask_target = self._load_batch_pair_pose(self.dataset_obj)

    # 初始化网络
    def init_net(self):
        print('PG2 init_net')
        self.build_model()

        print('aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa')

        if self.pretrained_path is not None:
            # 从一个结合中取出全部变量，是一个列表
            # get_collection集合中具有给定 name 的值的列表，或者如果没有值已添加到该集合中，则为空列表。该列表包含按其收集顺序排列的值
            # var = tf.get_collection(tf.GraphKeys.VARIABLES, scope='Pose_AE')+tf.get_collection(tf.GraphKeys.VARIABLES, scope='UAEnoFC') + tf.get_collection(tf.GraphKeys.VARIABLES, scope='UAEnoFC1')
            var = tf.get_collection(tf.GraphKeys.VARIABLES, scope='Pose_AE')+tf.get_collection(tf.GraphKeys.VARIABLES, scope='UAEnoFC') 
            # 模型的保存与恢复(Saver)
            # 模型保存，先要创建一个Saver对象
            #  max_to_keep 参数，这个是用来设置保存模型的个数
            # tf.reset_default_graph()
            self.saverPart = tf.train.Saver(var, max_to_keep=30)
            
        # 分布式执行 有关如何配置分布式 TensorFlow 程序的详细信息，
        self.saver = tf.train.Saver(max_to_keep=30)
        # 用来方便的将字符数据写入文件的类
        self.summary_writer = tf.summary.FileWriter(self.model_dir)
        print('lllllllllllllllllllllllllllllll')
    #     tf.train.Supervisor()可以帮我们简化一些事情，可以保存模型参数和Summary，它有以下的作用：
    # 　　1）自动去checkpoint加载数据或初始化数据 ，因此我们就不需要手动初始化或者从checkpoint中加载数据
    # 　　2）自身有一个Saver，可以用来保存checkpoint，因此不需要创建Saver，直接使用Supervisor里的Saver即可
    # 　　3）有一个summary_computed用来保存Summary，因此不需要创建summary_writer

        sv = tf.train.Supervisor(logdir=self.model_dir,
                                is_chief=True,
                                saver=None,
                                summary_op=None,
                                summary_writer=self.summary_writer,
                                global_step=self.step,
                                save_model_secs=0,
                                ready_for_local_init_op=None)

        # 在服务器上用多GPU做训练时，由于想只用其中的一个GPU设备做训练，
        # 可使用深度学习代码运行时往往出现多个GPU显存被占满清理。出现该现象主要是tensorflow训练时默认占用所有GPU的显存。
        # 使用allow_growth option，刚一开始分配少量的GPU容量，然后按需慢慢的增加，由于不会释放
        #内存，所以会导致碎片
        gpu_options = tf.GPUOptions(allow_growth=True)
        print('vvvvvvvvvvvvvvvvvvvvvvvvvvvvvv')
        # tf.ConfigProto一般用在创建session的时候。用来对session进行参数配置
        # allow_soft_placement=True ： 如果你指定的设备不存在，允许TF自动分配设备
        sess_config = tf.ConfigProto(allow_soft_placement=True,
                                    gpu_options=gpu_options)
        # 使用 sv.prepare_or_wait_for_session创建sess的时候,一定不要使用with block
        # prepare_or_wait_for_session在同步模式，不但参数初始化完成，还得主节点也准备好了，其他节点才开始干活
        # self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        self.sess = sv.prepare_or_wait_for_session(config=sess_config)
        # self.sess.run(tf.global_variables_initializer())
        print('ssssssssssssssssssssssssssssss')
        # with tf.Session() as sess:
            # sess.run(tf.global_variables_initializer())
        
        # checkpoint_path = os.path.join(self.model_dir, "model.ckpt")
        # self.reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
        # var_to_shape_map = self.reader.get_variable_to_shape_map()
        # for key in var_to_shape_map:
        #     print("tensor_name: ", key)
        #     print(self.reader.get_tensor(key)) # Remove this is you want to print only variable names
        
        if self.pretrained_path is not None:
            print('qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq')
            # tf.reset_default_graph()
            print('wwwwwwwwwwwwwwwwwwwwwwwwwwww')
            self.saverPart.restore(self.sess, self.pretrained_path)
            print('eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee')
            print('restored from pretrained_path:', self.pretrained_path)
        elif self.ckpt_path is not None:
            self.saver.restore(self.sess, self.ckpt_path)
            print('restored from ckpt_path:', self.ckpt_path)
        
        with tf.Session() as sess:  
            print('qqqqqqqqqqqqqqqqqqqqqq')
            print(self.G1.shape)
            # print(self.x[0].shape)
            ccc = self.sess.run(self.G1)
            print('jjjjjjjjjjjjjjjjj')
            print(ccc) 
            print(type(ccc))

    def _get_conv_shape(self):
        print('PG2 _get_conv_shape')
        shape = [self.batch_size, self.img_H, self.img_W, 3]
        return shape

    # 优化器
    def _getOptimizer(self, wgan_gp, gen_cost1, gen_cost2, disc_cost, G_var1, G_var2, D_var):
        print('PG2 _getOptimizer')
        clip_disc_weights = None
        # 判断模式
        # tf.train 提供了一组帮助训练模型的类和函数

        if wgan_gp.MODE == 'wgan':
            # 使用的是指数加权平均，旨在消除梯度下降中的摆动，与Momentum的效果一样，某一维度的导数比较大，则指数加权平均就大，
            # 某一维度的导数比较小，则其指数加权平均就小，这样就保证了各维度导数都在一个量级，进而减少了摆动。允许使用一个更大的学习率η
            # RMSprop 是一种自适应学习率方法。
            print('MODE == wgan')
            gen_train_op1 = tf.train.RMSPropOptimizer(learning_rate=5e-5).minimize(gen_cost1,
                                                 var_list=G_var1, colocate_gradients_with_ops=True)
            gen_train_op2 = tf.train.RMSPropOptimizer(learning_rate=5e-5).minimize(gen_cost2,
                                                 var_list=G_var2, colocate_gradients_with_ops=True)
            disc_train_op = tf.train.RMSPropOptimizer(learning_rate=5e-5).minimize(disc_cost,
                                                 var_list=D_var, colocate_gradients_with_ops=True)

            clip_ops = []
            for var in lib.params_with_name('Discriminator'):
                clip_bounds = [-.01, .01]
                clip_ops.append(tf.assign(var, tf.clip_by_value(var, clip_bounds[0], clip_bounds[1])))
            # 正则表达式中，group（）用来提出分组截获的字符串，（）用来分组
            clip_disc_weights = tf.group(*clip_ops)

        elif wgan_gp.MODE == 'wgan-gp':
            print('MODE == wgan-gp')
            gen_train_op1 = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(gen_cost1,
                                              var_list=G_var1, colocate_gradients_with_ops=True)
            gen_train_op2 = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(gen_cost2,
                                              var_list=G_var2, colocate_gradients_with_ops=True)
            disc_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(disc_cost,
                                               var_list=D_var, colocate_gradients_with_ops=True)

        elif wgan_gp.MODE == 'dcgan':
            print('MODE == dcgan')
            # 优化器基类提供了计算渐变的方法，并将渐变应用于变量。子类的集合实现了经典的Adam优化算法，
            # 此函数是Adam优化算法：是一个寻找全局最优点的优化算法，引入了二次方梯度校正。
            # 相比于基础SGD算法，1.不容易陷于局部优点。2.速度更快
            gen_train_op1 = tf.train.AdamOptimizer(learning_rate=2e-5, beta1=0.5).minimize(gen_cost1,
                                              var_list=G_var1, colocate_gradients_with_ops=True)
            gen_train_op2 = tf.train.AdamOptimizer(learning_rate=2e-5, beta1=0.5).minimize(gen_cost2,
                                              var_list=G_var2, colocate_gradients_with_ops=True)
            disc_train_op = tf.train.AdamOptimizer(learning_rate=2e-5, beta1=0.5).minimize(disc_cost,
                                               var_list=D_var, colocate_gradients_with_ops=True)

        elif wgan_gp.MODE == 'lsgan':
            print('MODE == lsgan')
            gen_train_op1 = tf.train.RMSPropOptimizer(learning_rate=1e-4).minimize(gen_cost1,
                                                 var_list=G_var1, colocate_gradients_with_ops=True)
            gen_train_op2 = tf.train.RMSPropOptimizer(learning_rate=1e-4).minimize(gen_cost2,
                                                 var_list=G_var2, colocate_gradients_with_ops=True)
            disc_train_op = tf.train.RMSPropOptimizer(learning_rate=1e-4).minimize(disc_cost,
                                                  var_list=D_var, colocate_gradients_with_ops=True)
        else:
            raise Exception()
        return gen_train_op1, gen_train_op2, disc_train_op, clip_disc_weights

    # 获得辨别器
    def _getDiscriminator(self, wgan_gp, arch='DCGAN'):
        print('PG2 _getDiscriminator')
        """
        Choose which generator and discriminator architecture to use by
        uncommenting one of these lines.    选择要使用的发电机和鉴别器体系结构中的其中一个。
        """        
        if 'DCGAN'==arch:
            # Baseline (G: DCGAN, D: DCGAN)
            return wgan_gp.DCGANDiscriminator
        raise Exception('You must choose an architecture!')

    # def build_test_model(self):
    #     G1, DiffMap, self.G_var1, self.G_var2  = GeneratorCNN_Pose_UAEAfterResidual_UAEnoFCAfter2Noise(
    #             self.x, self.pose_target, 
    #             self.channel, self.z_num, self.repeat_num, self.conv_hidden_num, self.data_format, activation_fn=tf.nn.relu, noise_dim=0, reuse=False)

    #     G2 = G1 + DiffMap
    #     self.G1 = denorm_img(G1, self.data_format)
    #     self.G2 = denorm_img(G2, self.data_format)
    #     self.G = self.G2
    #     self.DiffMap = denorm_img(DiffMap, self.data_format)

    #     self.wgan_gp = WGAN_GP(DATA_DIR='', MODE='dcgan', DIM=64, BATCH_SIZE=self.batch_size, ITERS=200000, LAMBDA=10, G_OUTPUT_DIM=128*64*3)
    #     Dis = self._getDiscriminator(self.wgan_gp, arch=self.D_arch)

    # 建立模型
    def build_model(self):
        print('PG2 build_model')
        # out1, out2, var1, var2
        G1, DiffMap, self.G_var1, self.G_var2  = GeneratorCNN_Pose_UAEAfterResidual_UAEnoFCAfter2Noise(
                self.x, self.pose_target, 
                self.channel, self.z_num, self.repeat_num, self.conv_hidden_num, self.data_format, activation_fn=tf.nn.relu, noise_dim=0, reuse=False)

        # out1+ out2
        print('G1=====')
        print(G1)
        print('DiffMap=====')
        print(DiffMap)
        G2 = G1 + DiffMap 
        print('G2=====')
        print(G2)
        # print('G3=====')
        # print(G3)

        # print(tf.Session().run(G2))
        # with tf.Session() as sess:
        #     print (G2)
        #     for x in G2.eval():      #b.eval()就得到tensor的数组形式
        #         print (G2)

        # with tf.Session() as sess:
        #     sess.run(tf.global_variables_initializer()) 
        #     print (G2.eval())

        # sess=tf.Session() 
        # sess.run(tf.global_variables_initializer()) 
        # #转化为numpy数组 
        # img_numpy_G1=img.eval() 
        # print("out2=",type(img_numpy_G1)) 
        # img_numpy_DiffMap=DiffMap.eval() 
        # print("out2=",type(img_numpy_DiffMap)) 
        # img_numpy_G2=G2.eval() 
        # print("out2=",type(img_numpy_G2)) 
        #转化为tensor img_tensor= tf.convert_to_tensor(img_numpy) print("out2=",type(img_tensor))
        self.G1 = denorm_img(G1, self.data_format)
        self.G2 = denorm_img(G2, self.data_format)
        # self.G3 = denorm_img(G3, self.data_format)
        self.G = self.G2
        self.DiffMap = denorm_img(DiffMap, self.data_format)

        
        print('self.G1=====')
        print(self.G1)
        print('self.DiffMap=====')
        print(self.DiffMap)
        print('self.G2=====')
        print(self.G2)

        # with tf.Session() as sess:  
        #     print('qqqqqqqqqqqqqqqqqqqqqq')
        #     print(self.G1.shape)
        #     # print(self.x[0].shape)
        #     ccc = self.sess.run(self.G1)
        #     print('jjjjjjjjjjjjjjjjj')
        #     print(ccc) 
        #     print(type(ccc))

        self.wgan_gp = WGAN_GP(DATA_DIR='', MODE='dcgan', DIM=64, BATCH_SIZE=self.batch_size, ITERS=200000, LAMBDA=10, G_OUTPUT_DIM=128*64*3)
        # print('wgan_gp=====')
        # print(wgan_gp)
        Dis = self._getDiscriminator(self.wgan_gp, arch=self.D_arch)
        print('Dis=====')
        print(Dis)
        triplet = tf.concat([self.x_target, self.x, G1, G2], 0)
        # print('x_target=====')
        # print(self.x_target)
        # print('x=====')
        # print(self.x)
        # print('triplet=====')
        # print(triplet)

        ## WGAN-GP code uses NCHW
        #transpose 作用是改变序列
        # print('in=====')
        print(tf.transpose( triplet, [0,3,1,2] ))
        self.D_z = Dis(tf.transpose( triplet, [0,3,1,2] ), input_dim=3)

        print('D_z=====')
        print(self.D_z)

        self.D_var = lib.params_with_name('Discriminator.')
        print('self.G_var1=====')
        print(self.G_var1)
        print('self.G_var2=====')
        print(self.G_var2)
        print('self.D_var=====')
        print(self.D_var)

        # split() 通过指定分隔符对字符串进行切片，如果参数 num 有指定值，则仅分隔 num 个子字符串
        D_z_pos_x_target, D_z_neg_x, D_z_neg_g1, D_z_neg_g2 = tf.split(self.D_z, 4)
        print('D_z_pos_x_target=====')
        print(D_z_pos_x_target)
        print('D_z_neg_x=====')
        print(D_z_neg_x)
        print('D_z_neg_g1=====')
        print(D_z_neg_g1)
        print('D_z_neg_g2=====')
        print(D_z_neg_g2)

        print('self.wgan_gp=====')
        print(self.wgan_gp)
        print('Dis=====')
        print(Dis)
        print('self.D_arch=====')
        print(self.D_arch)



        # reduce_mean某个维度求平均值  abs绝对值
        self.PoseMaskLoss1 = tf.reduce_mean(tf.abs(G1 - self.x_target) * (self.mask_target))
        self.g_loss1 = tf.reduce_mean(tf.abs(G1-self.x_target)) + self.PoseMaskLoss1
        # print('self.PoseMaskLoss1=====')
        # print(self.PoseMaskLoss1)
        # print('self.g_loss1=====')
        # print(self.g_loss1)
    # 计算损失
        # gen_cost, disc_cost, g2_g1_cost
        self.g_loss2, self.d_loss, self.g2_g1_loss = self._gan_loss(self.wgan_gp, Dis, D_z_pos_x_target, D_z_neg_x, D_z_neg_g1, D_z_neg_g2, arch=self.D_arch)
        # print('self.g_loss2=====')
        # print(self.g_loss2)
        # print('self.d_loss=====')
        # print(self.d_loss)
        # print('self.g2_g1_loss=====')
        # print(self.g2_g1_loss)
        self.PoseMaskLoss2 = tf.reduce_mean(tf.abs(G2 - self.x_target) * (self.mask_target))
        self.L1Loss2 = tf.reduce_mean(tf.abs(G2 - self.x_target)) + self.PoseMaskLoss2
        self.g_loss2 += self.L1Loss2 * 10

        # print('self.g_loss2=====')
        # print(self.g_loss2)

        # 获得优化算法
        # return gen_train_op1, gen_train_op2, disc_train_op, clip_disc_weights
        self.g_optim1, self.g_optim2, self.d_optim, self.clip_disc_weights = self._getOptimizer(self.wgan_gp, 
                                self.g_loss1, self.g_loss2, self.d_loss, self.G_var1,self.G_var2, self.D_var)
        print('self.g_optim1=====')
        print(self.g_optim1)
        print('self.g_optim2=====')
        print(self.g_optim2)
        print('self.d_optim=====')
        print(self.d_optim)
        print('self.clip_disc_weights=====')
        print(self.clip_disc_weights)
        #  数据合并函数merge( )
        # 合并summaries        该op创建了一个summary协议缓冲区，它包含了输入的summaries的所有value的union
        self.summary_op = tf.summary.merge([
            # tf.summary.image输出带有图像的Summary协议缓冲区
            tf.summary.image("G1", self.G1),
            tf.summary.image("G2", self.G2),
            # tf.summary.image("G3", self.G3),
            tf.summary.image("DiffMap", self.DiffMap),
            #  输出仅有一个标量值的Summary协议缓冲区
            tf.summary.scalar("loss/PoseMaskLoss1", self.PoseMaskLoss1),
            tf.summary.scalar("loss/PoseMaskLoss2", self.PoseMaskLoss2),
            tf.summary.scalar("loss/L1Loss2", self.L1Loss2),
            tf.summary.scalar("loss/g_loss1", self.g_loss1),
            tf.summary.scalar("loss/g_loss2", self.g_loss2),
            tf.summary.scalar("loss/d_loss", self.d_loss),
            tf.summary.scalar("loss/g2_g1_loss", self.g2_g1_loss),
            tf.summary.scalar("misc/d_lr", self.d_lr),
            tf.summary.scalar("misc/g_lr", self.g_lr),
        ])

    # self._gan_loss(self.wgan_gp, Dis, D_z_pos_x_target, D_z_neg_x, D_z_neg_g1, D_z_neg_g2, arch=self.D_arch)
    def _gan_loss(self, wgan_gp, Discriminator, disc_real, disc_fake_x, disc_fake_g1, disc_fake_g2, arch='DCGAN'):
        print('PG2 _gan_loss')
        if wgan_gp.MODE == 'dcgan':
            if 'DCGAN'==arch:
                # reduce_mean某个维度求平均值  sigmoid_cross_entropy_with_logits对于给定的logits计算sigmoid的交叉熵
                # 第一个参数logits：就是神经网络最后一层的输出，如果有batch的话，       一个张量的类型float32或float64。
                # 它的大小就是[batchsize，num_classes]，单样本的话，大小就是num_classes
                # 第二个参数labels：实际的标签，大小同上        相同类型的一个张量和形状分对数。

                # logits就是神经网络模型中的 W * X矩阵，注意不需要经过sigmoid，而targets的shape和logits相同，就是正确的label值
                # 创建一个tensor，左右的元素都设置为1。
                # 给定一个tensor（tensor 参数），该操作返回一个具有和给定tensor相同形状（shape）和相同数据类型（dtype），但是所有的元素都被设置为1的tensor。
                g2_g1_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake_g2-disc_fake_g1, labels=tf.ones_like(disc_fake_g2-disc_fake_g1)))
                
                gen_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake_g2, labels=tf.ones_like(disc_fake_g2)))
                # gen_cost = 0.5*tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake_g2, labels=tf.ones_like(disc_fake_g2))) \
                #             + 0.5*g2_g1_cost*10
                # gen_cost = g2_g1_cost
                            
                disc_cost = 0.25*tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake_x, labels=tf.zeros_like(disc_fake_x))) \
                            + 0.25*tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake_g2, labels=tf.zeros_like(disc_fake_g2)))
                disc_cost += 0.5*tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_real, labels=tf.ones_like(disc_real)))

        return gen_cost, disc_cost, g2_g1_cost

    def train(self):
        print('PG2 train')
        x_fixed, x_target_fixed, pose_fixed, pose_target_fixed, mask_fixed, mask_target_fixed = self.get_image_from_loader()
        save_image(x_fixed, '{}/x_fixed.png'.format(self.model_dir))
        save_image(x_target_fixed, '{}/x_target_fixed.png'.format(self.model_dir))
        save_image((np.amax(pose_fixed, axis=-1, keepdims=True)+1.0)*127.5, '{}/pose_fixed.png'.format(self.model_dir))
        save_image((np.amax(pose_target_fixed, axis=-1, keepdims=True)+1.0)*127.5, '{}/pose_target_fixed.png'.format(self.model_dir))
        save_image(mask_fixed, '{}/mask_fixed.png'.format(self.model_dir))
        save_image(mask_target_fixed, '{}/mask_target_fixed.png'.format(self.model_dir))

        

        # 0-500000
        for step in trange(self.start_step, self.max_step):
            if step < 22000:
                self.sess.run(self.g_optim1)
            else:
                # Train generator
                if step > 0:
                    self.sess.run(self.g_optim2)

                # Train critic
                if (self.wgan_gp.MODE == 'dcgan') or (self.wgan_gp.MODE == 'lsgan'):
                    disc_ITERS = 1
                else:
                    disc_ITERS = self.wgan_gp.CRITIC_ITERS
                for i in xrange(disc_ITERS):
                    self.sess.run(self.d_optim)
                    if self.wgan_gp.MODE == 'wgan':
                        self.sess.run(self.clip_disc_weights)

            fetch_dict = {}
            if step % self.log_step == self.log_step-1:
                fetch_dict.update({
                    "summary": self.summary_op
                })
                    # "k_t": self.k_t,
            result = self.sess.run(fetch_dict)

            if step % self.log_step == self.log_step-1:
                self.summary_writer.add_summary(result['summary'], step)
                self.summary_writer.flush()

            if step % (self.log_step * 3) == (self.log_step * 3)-1:
                # if self.data_format == 'NCHW':
                #     x = x_fixed.transpose([0, 3, 1, 2])
                # else:
                #     x = x_fixed
                x = utils_wgan.process_image(x_fixed, 127.5, 127.5)
                x_target = utils_wgan.process_image(x_target_fixed, 127.5, 127.5)
                self.generate(x, x_target, pose_target_fixed, self.model_dir, idx=step)

            if step % self.lr_update_step == self.lr_update_step - 1:
                self.sess.run([self.g_lr_update, self.d_lr_update])

            if step % (self.log_step * 30) == (self.log_step * 30)-1:
                self.saver.save(self.sess, os.path.join(self.model_dir, 'model.ckpt'), global_step=step)

    def test(self): 
        print('PG2 test')

        # print('self.PoseMaskLoss1=====')
        # print(self.sess.run(self.PoseMaskLoss1))
        # print('self.PoseMaskLoss2=====')
        # print(self.sess.run(self.PoseMaskLoss2))
        # print('self.L1Loss2=====')
        # print(self.sess.run(self.L1Loss2))
        # print('self.g_loss1=====')
        # print(self.sess.run(self.g_loss1))

        # print('self.g_loss2=====')
        # print(self.sess.run(self.g_loss2))
        # print('self.d_loss=====')
        # print(self.sess.run(self.d_loss))
        # print('self.L1Loss2=====')
        # print(self.sess.run(self.L1Loss2))
        # print('self.g2_g1_loss=====')
        # print(self.sess.run(self.g2_g1_loss))

        # print('self.d_lr=====')
        # print(self.sess.run(self.d_lr))
        # print('self.g_lr=====')
        # print(self.sess.run(self.g_lr))

        # print('self.g_optim1=====')
        # print(self.sess.run(self.g_optim1))
        # print('self.g_optim2=====')
        # print(self.sess.run(self.g_optim2))
        # print('self.d_optim=====')
        # print(self.sess.run(self.d_optim))
        # print('self.clip_disc_weights=====')
        # print(self.sess.run(self.clip_disc_weights))


        test_result_dir = os.path.join(self.model_dir, 'test_result')
        test_result_dir_x = os.path.join(test_result_dir, 'x')
        test_result_dir_x_target = os.path.join(test_result_dir, 'x_target')
        test_result_dir_G = os.path.join(test_result_dir, 'G')
        test_result_dir_G1 = os.path.join(test_result_dir, 'G1')
        test_result_dir_G2 = os.path.join(test_result_dir, 'G2')
        test_result_dir_G3 = os.path.join(test_result_dir, 'G3')
        # test_result_dir_G4 = os.path.join(test_result_dir, 'G4')
        test_result_dir_pose = os.path.join(test_result_dir, 'pose')
        test_result_dir_pose_target = os.path.join(test_result_dir, 'pose_target')
        test_result_dir_mask = os.path.join(test_result_dir, 'mask')
        test_result_dir_mask_target = os.path.join(test_result_dir, 'mask_target')
        if not os.path.exists(test_result_dir):
            os.makedirs(test_result_dir)
        if not os.path.exists(test_result_dir_x):
            os.makedirs(test_result_dir_x)
        if not os.path.exists(test_result_dir_x_target):
            os.makedirs(test_result_dir_x_target)
        if not os.path.exists(test_result_dir_G):
            os.makedirs(test_result_dir_G)
        if not os.path.exists(test_result_dir_G1):
            os.makedirs(test_result_dir_G1)
        if not os.path.exists(test_result_dir_G2):
            os.makedirs(test_result_dir_G2)
        if not os.path.exists(test_result_dir_G3):
            os.makedirs(test_result_dir_G3)
        # if not os.path.exists(test_result_dir_G4):
        #     os.makedirs(test_result_dir_G4)
        if not os.path. exists(test_result_dir_pose):
            os.makedirs(test_result_dir_pose)
        if not os.path.exists(test_result_dir_pose_target):
            os.makedirs(test_result_dir_pose_target)
        if not os.path.exists(test_result_dir_mask):
            os.makedirs(test_result_dir_mask)
        if not os.path.exists(test_result_dir_mask_target):
            os.makedirs(test_result_dir_mask_target)

        for i in xrange(5):
            # x, x_target, pose, pose_target, mask, mask_target
            # x_fixed, x_target_fixed, pose_fixed, pose_target_fixed, mask_fixed, mask_target_fixed,G1_fixed,G2_fixed,DiffMap_fixed ,G3_fixed = self.get_image_from_loader()
            x_fixed, x_target_fixed, pose_fixed, pose_target_fixed, mask_fixed, mask_target_fixed,G1_fixed,G2_fixed,DiffMap_fixed = self.get_image_from_loader()
            x = utils_wgan.process_image(x_fixed, 127.5, 127.5)
            x_target = utils_wgan.process_image(x_target_fixed, 127.5, 127.5)
            # print(type(x))
            if 0==i:
                x_fake = self.generate(x, x_target, pose_target_fixed, test_result_dir, idx=self.start_step, save=True)
            else:
                x_fake = self.generate(x, x_target, pose_target_fixed, test_result_dir, idx=self.start_step, save=False)
            # np.amax这个的用法是，把每个元素都取最大值，最后组成一个数组
            p = (np.amax(pose_fixed, axis=-1, keepdims=False)+1.0)*127.5
            pt = (np.amax(pose_target_fixed, axis=-1, keepdims=False)+1.0)*127.5
            for j in xrange(self.batch_size):
                idx = i*self.batch_size+j
                # # 这里将narray转为Image类，Image转narray：a=np.array(img)
                # 保存图片`
                im = Image.fromarray(x_fixed[j,:].astype(np.uint8))
                im.save('%s/%05d.png'%(test_result_dir_x, idx))
                # print(type(im))
                im = Image.fromarray(x_target_fixed[j,:].astype(np.uint8))
                im.save('%s/%05d.png'%(test_result_dir_x_target, idx))
                im = Image.fromarray(x_fake[j,:].astype(np.uint8))
                im.save('%s/%05d.png'%(test_result_dir_G, idx))
                im = Image.fromarray(G1_fixed[j,:].astype(np.uint8))
                im.save('%s/%05d.png'%(test_result_dir_G1, idx))
                im = Image.fromarray(DiffMap_fixed[j,:].astype(np.uint8))
                im.save('%s/%05d.png'%(test_result_dir_G2, idx))
                im = Image.fromarray(G2_fixed[j,:].astype(np.uint8))
                im.save('%s/%05d.png'%(test_result_dir_G3, idx))
                # im = Image.fromarray(G3_fixed[j,:].astype(np.uint8))
                # im.save('%s/%05d.png'%(test_result_dir_G4, idx))
                # im = Image.fromarray(p[j,:].astype(np.uint8))
                im.save('%s/%05d.png'%(test_result_dir_pose, idx))
                im = Image.fromarray(pt[j,:].astype(np.uint8))
                im.save('%s/%05d.png'%(test_result_dir_pose_target, idx))
                im = Image.fromarray(mask_fixed[j,:].squeeze().astype(np.uint8))
                im.save('%s/%05d.png'%(test_result_dir_mask, idx))
                im = Image.fromarray(mask_target_fixed[j,:].squeeze().astype(np.uint8))
                im.save('%s/%05d.png'%(test_result_dir_mask_target, idx))
            if 0==i:
                save_image(x_fixed, '{}/x_fixed.png'.format(test_result_dir))
                save_image(x_target_fixed, '{}/x_target_fixed.png'.format(test_result_dir))
                save_image(mask_fixed, '{}/mask_fixed.png'.format(test_result_dir))
                save_image(mask_target_fixed, '{}/mask_target_fixed.png'.format(test_result_dir))
                save_image((np.amax(pose_fixed, axis=-1, keepdims=True)+1.0)*127.5, '{}/pose_fixed.png'.format(test_result_dir))
                save_image((np.amax(pose_target_fixed, axis=-1, keepdims=True)+1.0)*127.5, '{}/pose_target_fixed.png'.format(test_result_dir))

    # generate(x, x_target, pose_target_fixed, test_result_dir, idx=self.start_step, save=True)
    def generate(self, x_fixed, x_target_fixed, pose_target_fixed, root_path=None, path=None, idx=None, save=True):
        print('PG2 generate')
        G = self.sess.run(self.G, {self.x: x_fixed, self.pose_target: pose_target_fixed})
        ssim_G_x_list = []
        # x_0_255 = utils_wgan.unprocess_image(x_target_fixed, 127.5, 127.5)
        for i in xrange(G.shape[0]):
            # G_gray = rgb2gray((G[i,:]/127.5-1).clip(min=-1,max=1))
            # x_target_gray = rgb2gray((x_target_fixed[i,:]).clip(min=-1,max=1))
            # Clip（limit）.这个方法会给出一个区间，在区间之外的数字将被剪除到区间的边缘，
            # 例如给定一个区间[0,1]，则小于0的将变成0，大于1则变成1. 
            G_gray = rgb2gray((G[i,:]).clip(min=0,max=255).astype(np.uint8))
            x_target_gray = rgb2gray(((x_target_fixed[i,:]+1)*127.5).clip(min=0,max=255).astype(np.uint8))
            # SSIM（结构相似性评价）
            ssim_G_x_list.append(ssim(G_gray, x_target_gray, data_range=x_target_gray.max() - x_target_gray.min(), multichannel=False))
        # 求取均值
        ssim_G_x_mean = np.mean(ssim_G_x_list)
        if path is None and save:
            path = os.path.join(root_path, '{}_G_ssim{}.png'.format(idx,ssim_G_x_mean))
            save_image(G, path)
            print("[*] Samples saved: {}".format(path))
        return G

    # 加载批次一对姿势
    def _load_batch_pair_pose(self, dataset):
        print('PG2 _load_batch_pair_pose')
        # provider对象根据dataset信息读取数据
        # common_queue_capacity 公共队列容量  32
        data_provider = slim.dataset_data_provider.DatasetDataProvider(dataset, common_queue_capacity=32, common_queue_min=8)
        # 获取数据，获取到的数据是单个数据，还需要对数据进行预处理，组合数据    pose_0
        image_raw_0, image_raw_1, label, pose_0, pose_1, mask_0, mask_1  = data_provider.get([
            'image_raw_0', 'image_raw_1', 'label', 'pose_sparse_r4_0', 'pose_sparse_r4_1', 'pose_mask_r4_0', 'pose_mask_r4_1'])
        
        # print(image_raw_0.initializer())
        print(tf.size(image_raw_0))
        print('aaaaaaaaaaaaaaaaaaaa')
        print(image_raw_0)
        print(image_raw_0[0])
        print(image_raw_0[1])
        print(image_raw_0[2])
        print(type(image_raw_0))
        
        # sparse_tensor_to_dense  将SparseTensor 转换为稠密张量  validate_indices 一个布尔值
        pose_0 = sparse_ops.sparse_tensor_to_dense(pose_0, default_value=0, validate_indices=False)
        pose_1 = sparse_ops.sparse_tensor_to_dense(pose_1, default_value=0, validate_indices=False)

        # 重塑张量。给定tensor，这个操作返回一个张量，它与带有形状shape的tensor具有相同的值。
        # shape：一个Tensor；必须是以下类型之一：int32，int64；用于定义输出张量的形状
        image_raw_0 = tf.reshape(image_raw_0, [128, 64, 3])        
        image_raw_1 = tf.reshape(image_raw_1, [128, 64, 3]) 

        print('imimimimimimimimimiimimimmimi')
        print(image_raw_0.get_shape())
        print(tf.size(image_raw_0))
        print(image_raw_0)
        a = image_raw_0[0]
        print(a)
        print(type(image_raw_0))

        # print(image_raw_0[0])
        # print(image_raw_0[0][1])
        # print(image_raw_0[1][2])
        # print(image_raw_0[0][1][2])
        print('bbbbbbbbbbbbbbbbbbb')
        b = image_raw_0[0][1]
        print (b)

        # print('**********************************')
        # sess=tf.Session()
        # print('fffffffffffffffffffffffff') 
        # sess.run(tf.global_variables_initializer())
        # print('gggggggggggggggggggggggg') 
        # #转化为numpy数组 
        # img_numpy=a.eval(session=sess) 
        # print("out2=",type(img_numpy))
        # print(img_numpy)
        # image_raw_0 = tf.image.decode_jpeg(image_raw_0)
        # image_raw_0 = tf.cast(image_raw_0, tf.float32)
        # cc = 500
        # print('===============================')
        # tf.reset_default_graph()
        # sv = tf.train.Supervisor()
        # # with sv.managed_session() as sess:
        # self.sess = sv.prepare_or_wait_for_session(config=sess_config)
        # with tf.Session() as sess:  
        #     print('qqqqqqqqqqqqqqqqqqqqqq')
        #     init_op = tf.global_variables_initializer()
        #     print(image_raw_0.shape)
        #     sess.run(init_op)
        #     print(image_raw_0[0].shape)
        #     ccc = self.sess.run(image_raw_0)
        #     print('jjjjjjjjjjjjjjjjj')
        #     print(ccc[0]) 


        # sess=tf.Session()
        # print('fffffffffffffffffffffffff') 
        # with sess.as_default():
        #     print('gggggggggggggggggg')
        #     print(image_raw_0.eval(()))
        # sess.run(tf.global_variables_initializer())
        # print('gggggggggggggggggggggggg') 
        # #转化为numpy数组 
        # img_numpy=image_raw_0.eval(session=sess) 
        # print("out2=",type(img_numpy))

        # cast类型转换函数
        pose_0 = tf.cast(tf.reshape(pose_0, [128, 64, self.keypoint_num]), tf.float32)
        pose_1 = tf.cast(tf.reshape(pose_1, [128, 64, self.keypoint_num]), tf.float32)
        mask_0 = tf.cast(tf.reshape(mask_0, [128, 64, 1]), tf.float32)
        mask_1 = tf.cast(tf.reshape(mask_1, [128, 64, 1]), tf.float32)

        # tf.train.batch是按顺序读取数据，队列中的数据始终是一个有序的队列
        # 图像预处理
        # 图像生成batch序列。
        images_0, images_1, poses_0, poses_1, masks_0, masks_1 = tf.train.batch([image_raw_0, image_raw_1, pose_0, pose_1, mask_0, mask_1], 
                    batch_size=self.batch_size, num_threads=self.num_threads, capacity=self.capacityCoff * self.batch_size)

        images_0 = utils_wgan.process_image(tf.to_float(images_0), 127.5, 127.5)
        images_1 = utils_wgan.process_image(tf.to_float(images_1), 127.5, 127.5)
        poses_0 = poses_0*2-1
        poses_1 = poses_1*2-1
        print('imimimimimimimimimiimimimmimi')
        print(images_0)
        print(images_0[0])
        print(images_0[1])
        print(images_0[2])

        # tf.reset_default_graph()
        print(images_0)
        # sv = tf.train.Supervisor()
        # # with sv.managed_session() as sess:
        # with tf.Session() as sess:  
        #     print('qqqqqqqqqqqqqqqqqqqqqq')
        #     print(images_0.shape)
        #     # print(cc * 1000000)
        #     print(images_0[0].shape)
        #     ccc = sess.run(images_0)
        #     print('jjjjjjjjjjjjjjjjj')
        #     print(ccc[0]) 

        return images_0, images_1, poses_0, poses_1, masks_0, masks_1

    def get_image_from_loader(self):
        print('PG2 get_image_from_loader')
        # 这里 self.sess.run(）函数是执行一个会话，第一个参数是图的输出节点，第二个参数图的输入节点
        # x,  pose,  mask,  G1,G2, DiffMap ,G3= self.sess.run([self.x,  self.pose,  self.mask, self.G1, self.G2, self.DiffMap,self.G3])
        # x,  pose,  mask,  G1,G2, DiffMap = self.sess.run([self.x,  self.pose,  self.mask, self.G1, self.G2, self.DiffMap])
        # x_target, pose_target, mask_target  = self.sess.run([self.x_target,self.pose_target, self.mask_target])
        print(type(self.x))
        print(self.x)

        
        # tf.reset_default_graph()
        # sv = tf.train.Supervisor()
        # with sv.managed_session() as sess:
        # with tf.Session() as sess:  
        #     print('qqqqqqqqqqqqqqqqqqqqqq')
        #     print(self.G1.shape)
        #     # print(self.x[0].shape)
        #     ccc = self.sess.run(self.G1)
        #     print('jjjjjjjjjjjjjjjjj')
        #     print(ccc) 
        #     print(type(ccc))

        # sess=tf.Session()
        # print('fffffffffffffffffffffffff') 
        # with sess.as_default():
        #     print('gggggggggggggggggg')
        #     print(self.x.eval(()))

        

        x, x_target, pose,pose_target,mask, mask_target ,  G1,G2, DiffMap = self.sess.run([self.x, self.x_target,self.pose,self.pose_target, self.mask,self.mask_target, self.G1, self.G2, self.DiffMap])
        # G1,G2, DiffMap = self.sess.run([self.G1, self.G2, self.DiffMap])
        
        print('===============================')
        # print(x)
        # print(type(x))
        # tf.reset_default_graph()
        # sv = tf.train.Supervisor()
        # # with sv.managed_session() as sess:
        # with tf.Session() as sess:  
        #     print('qqqqqqqqqqqqqqqqqqqqqq')
        #     print(x.shape)
        #     print(x[0].shape)
        #     ccc = sess.run(x)
        #     print('jjjjjjjjjjjjjjjjj')
        #     print(ccc[0]) 
        
        x = utils_wgan.unprocess_image(x, 127.5, 127.5)
        # print(x)
        # print(type(x))
        x_target = utils_wgan.unprocess_image(x_target, 127.5, 127.5)
        mask = mask*255
        mask_target = mask_target*255
        return x, x_target, pose, pose_target, mask, mask_target,G1,G2,DiffMap
        # return x, x_target, pose, pose_target, mask, mask_target
