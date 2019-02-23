#-*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
slim = tf.contrib.slim
import pdb
import utils_wgan

def LeakyReLU(x, alpha=0.3):
    return tf.maximum(alpha*x, x)

def LeakyReLU2(x, alpha=0.2):
    return tf.maximum(alpha*x, x)

def Batchnorm(inputs, is_training, name=None, data_format='NHWC'):
    bn = tf.contrib.layers.batch_norm(inputs, 
                                      center=True, scale=True, 
                                      is_training=is_training,
                                      scope=name, 
                                      data_format=data_format)
    return bn

## Ref code: https://github.com/tensorflow/models/blob/master/slim/nets/resnet_v2.py
def ResBottleNeckBlock(x, n1, n2, n3, data_format, activation_fn=LeakyReLU):
    if n1 != n3:
        shortcut = slim.conv2d(x, n3, 1, 1, activation_fn=None, data_format=data_format)
    else:
        shortcut = x
    x = slim.conv2d(x, n2, 1, 1, activation_fn=activation_fn, data_format=data_format)
    x = slim.conv2d(x, n2, 3, 1, activation_fn=activation_fn, data_format=data_format)
    x = slim.conv2d(x, n3, 1, 1, activation_fn=None, data_format=data_format)
    out = activation_fn(shortcut + x)
    return out

def ResBlock(x, n1, n2, n3, data_format, activation_fn=LeakyReLU):
    if n1 != n3:
        shortcut = slim.conv2d(x, n3, 1, 1, activation_fn=None, data_format=data_format)
    else:
        shortcut = x
    x = slim.conv2d(x, n2, 3, 1, activation_fn=activation_fn, data_format=data_format)
    x = slim.conv2d(x, n3, 3, 1, activation_fn=None, data_format=data_format)
    out = activation_fn(shortcut + x)
    return out

def LuNet(x, input_H, input_W, is_train_tensor, data_format='NHWC', activation_fn=LeakyReLU, reuse=False):
    input_shape = x.get_shape().as_list()
    if len(input_shape) != 4:
        raise ValueError('Invalid input tensor rank, expected 4, was: %d' % len(input_shape))

    with tf.variable_scope("CNN",reuse=reuse) as vs:
        #
        x = slim.conv2d(x, 128, 7, 1, data_format=data_format, activation_fn=activation_fn)
        x = ResBottleNeckBlock(x, 128, 32, 128, data_format=data_format, activation_fn=activation_fn)
        x = tf.contrib.layers.max_pool2d(x, [3, 3], [2, 2], padding='SAME')
        input_H = input_H/2
        input_W = input_W/2
        #
        x = ResBottleNeckBlock(x, 128, 32, 128, data_format=data_format, activation_fn=activation_fn)
        x = ResBottleNeckBlock(x, 128, 32, 128, data_format=data_format, activation_fn=activation_fn)
        x = ResBottleNeckBlock(x, 128, 64, 256, data_format=data_format, activation_fn=activation_fn)
        x = tf.contrib.layers.max_pool2d(x, [3, 3], [2, 2], padding='SAME')
        input_H = input_H/2
        input_W = input_W/2
        #
        x = ResBottleNeckBlock(x, 256, 64, 256, data_format=data_format, activation_fn=activation_fn)
        x = ResBottleNeckBlock(x, 256, 64, 256, data_format=data_format, activation_fn=activation_fn)
        x = tf.contrib.layers.max_pool2d(x, [3, 3], [2, 2], padding='SAME')
        input_H = input_H/2
        input_W = input_W/2
        #
        x = ResBottleNeckBlock(x, 256, 64, 256, data_format=data_format, activation_fn=activation_fn)
        x = ResBottleNeckBlock(x, 256, 64, 256, data_format=data_format, activation_fn=activation_fn)
        x = ResBottleNeckBlock(x, 256, 128, 512, data_format=data_format, activation_fn=activation_fn)
        x = tf.contrib.layers.max_pool2d(x, [3, 3], [2, 2], padding='SAME')
        input_H = input_H/2
        input_W = input_W/2
        #
        x = ResBottleNeckBlock(x, 512, 128, 512, data_format=data_format, activation_fn=activation_fn)
        x = ResBottleNeckBlock(x, 512, 128, 512, data_format=data_format, activation_fn=activation_fn)
        x = tf.contrib.layers.max_pool2d(x, [3, 3], [2, 2], padding='SAME')
        input_H = input_H/2
        input_W = input_W/2
        #
        # x = slim.dropout(x, keep_prob=0.6)
        x = ResBlock(x, 512, 512, 128, data_format=data_format, activation_fn=activation_fn)
        x = tf.reshape(x, [-1, 128*input_H*input_W])
        print('dim:%d'%(128*input_H*input_W))
        x = slim.fully_connected(x, 512, activation_fn=None)
        x = Batchnorm(x, is_train_tensor, 'LuNet.BN', data_format='NHWC')
        # x = tf.nn.relu(x)
        x = activation_fn(x)
        # x = selu(x) ## Relpace BN+ReLU
        out = slim.fully_connected(x, 128, activation_fn=None)

    variables = tf.contrib.framework.get_variables(vs)
    return out, variables

def weight_variable(shape):
    initial=tf.truncated_normal(shape,stddev=0.1)
    return initial

################################################################
#######################       GAN       ########################
def GeneratorCNN_Pose_UAEAfterResidual(x, pose_target, input_channel, z_num, repeat_num, hidden_num, data_format, activation_fn=tf.nn.elu, min_fea_map_H=8, noise_dim=0, reuse=False):
    #     返回一个用于定义创建variable（层）的op的上下文管理器。
# 通过tf.variable_scope()指定作用域进行区分
# 指定了第一个卷积层作用域为G
    print('GeneratorCNN_Pose_UAEAfterResidual')
    with tf.variable_scope("G") as vs:
        if pose_target is not None:
            if data_format == 'NCHW':
                # concat 将张量沿一个维度串联
                # values：张量对象或单个张量列表。
                # axis：0 维 int32 张量，要连接的维度。
                print('NCHW')
                print(x)
                print(pose_target)
                x = tf.concat([x, pose_target], 1)
            elif data_format == 'NHWC':
                print('NHWC')
                print(x)
                print(pose_target)
                x = tf.concat([x, pose_target], 3)

        # Encoder  编码器
        encoder_layer_list = []

#         inputs同样是指需要做卷积的输入图像    x
            # num_outputs指定卷积核的个数（就是filter的个数）hidden_num 128
            # kernel_size用于指定卷积核的维度（卷积核的宽度，卷积核的高度）3
            # stride为卷积时在图像每一维的步长  1
            # padding为padding的方式选择，VALID或者SAME
            # data_format是用于指定输入的input的格式
            # rate这个参数不是太理解，而且tf.nn.conv2d中也没有，对于使用atrous convolution的膨胀率（不是太懂这个atrous convolution）
            # activation_fn用于激活函数的指定，默认的为ReLU函数
            # normalizer_fn用于指定正则化函数
            # normalizer_params用于指定正则化函数的参数
            # weights_initializer用于指定权重的初始化程序
            # weights_regularizer为权重可选的正则化程序
            # biases_initializer用于指定biase的初始化程序
            # biases_regularizer: biases可选的正则化程序
            # reuse指定是否共享层或者和变量
            # variable_collections指定所有变量的集合列表或者字典
            #outputs_collections指定输出被添加的集合
            # trainable:卷积层的参数是否可被训练
            # scope:共享变量所指的variable_scope
        print(x)
        print('\n')
        x = slim.conv2d(x, hidden_num, 3, 1, activation_fn=activation_fn, data_format=data_format)
        print(x)
        print('\n')
        print('ffffffffffffffffff')

        # 从 0 开始到 repeat_num    = 5
        for idx in range(repeat_num):
            channel_num = hidden_num * (idx + 1)
            # channel_num = x.get_shape()[-1]
            # 两个卷积
            print('idx = %d'%idx)
            print('channel_num = %d'%channel_num)
            print('\n')
            
            res = x
            x = slim.conv2d(x, channel_num, 3, 1, activation_fn=activation_fn, data_format=data_format)
            print(x)
            x = slim.conv2d(x, channel_num, 3, 1, activation_fn=activation_fn, data_format=data_format)
            print(x)
            x = x + res
            print(x)
            # 在序列encoder_layer_list的尾部追加x
            encoder_layer_list.append(x)
            print('ififififififififiifififiiifi')
            if idx < repeat_num - 1:
                x = slim.conv2d(x, hidden_num * (idx + 2), 3, 2, activation_fn=activation_fn, data_format=data_format)
                print(x)
                print('num = %d'%(hidden_num * (idx + 2)))
                #x = tf.contrib.layers.max_pool2d(x, [2, 2], [2, 2], padding='VALID')

        # reshape函数的作用是将tensor变换为参数shape的形式。 
        # 其中shape为一个列表形式，特殊的一点是列表中可以存在-1。-1代表的含义是不用我们自己指定这一维的大小，函数会自动计算，
        # 但列表中只能存在一个-1。（当然如果存在多个-1，就是一个存在多解的方程了）
        # 重塑张量。
        # 给定tensor，这个操作返回一个张量，它与带有形状shape的tensor具有相同的值。
        # 如果shape的一个分量是特殊值-1，则计算该维度的大小，以使总大小保持不变。
        # 特别地情况为，一个[-1]维的shape变平成1维。至多能有一个shape的分量可以是-1。

        # 如果shape是1-D或更高，则操作返回形状为shape的张量，其填充为tensor的值。
        # 在这种情况下，隐含的shape元素数量必须与tensor元素数量相同。

        # np.prod 定轴上的数组元素的乘积
        x = tf.reshape(x, [-1, np.prod([min_fea_map_H, min_fea_map_H/2, channel_num])])
        # FullyConnected        丰富连接
        print(x)
        # z_num 输出神经元的数目
        z = x = slim.fully_connected(x, z_num, activation_fn=None)
        print(x)
        print(z)

        # config = tf.ConfigProto(allow_soft_placement=True)

        # #最多占gpu资源的70%
        # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)

        #开始不会给tensorflow全部gpu资源 而是按需增加
        # config.gpu_options.allow_growth = True
        # sess = tf.Session(config=config)

        
        # with tf.Session() as sess:
        #     sess.run(tf.global_variables_initializer()) 
        #     print('zzzzzzzzzzzzzzzzzzzzzzzzz')
        #     print (z.eval())

        # W_conv1 = weight_variable(z)
        # with tf.Session() as sess:
        #     print (sess.run(W_conv1))

        # with tf.Session() as sess:
        #     sess.run(tf.global_variables_initializer())
        #     print('zzzzzzzzzzzzzzzzzzzzzzzzz')
        #     print (sess.run(z))

        # print('zzzzzzzzzzzzzzzzzzzzzzzzz')
        # print(tf.size(z))
        # sess=tf.Session()
        # #W_conv1 = weight_variable([5, 5, 1, 32])
        # a=z.eval(session=sess)
        # print (a)

        # 随机噪声
        if noise_dim>0:
            print(noise_dim)
            noise = tf.random_uniform(
                (tf.shape(z)[0], noise_dim), minval=-1.0, maxval=1.0)
            print(noise)
            # tf.concat是连接两个矩阵的操作
            # 如果concat_dim是1，那么在某一个shape的第二个维度上连
            # 第二个参数values：就是两个或者一组待连接的tensor了
            z = tf.concat([z, noise], 1)
            print(z)

        # Decoder   解码器
        x = slim.fully_connected(z, np.prod([min_fea_map_H, min_fea_map_H/2, hidden_num]), activation_fn=None)
        print(x)
        # reshape是一种函数，函数可以重新调整矩阵的行数、列数、维数。
        x = reshape(x, min_fea_map_H, min_fea_map_H/2, hidden_num, data_format)
        print(x)
        
        # 从 0 开始到 repeat_num    5
        for idx in range(repeat_num):
            # pdb.set_trace()
            print(idx)
            print(encoder_layer_list)
            # tf.concat是连接两个矩阵的操作
            # 如果concat_dim是1，那么在某一个shape的第二个维度上连
            # 第二个参数values：就是两个或者一组待连接的tensor了
            x = tf.concat([x, encoder_layer_list[repeat_num-1-idx]], axis=-1)
            print(x)
            res = x
            # channel_num = hidden_num * (repeat_num-idx)
            # a.get_shape()中a的数据类型只能是tensor,且返回的是一个元组（tuple）
            channel_num = x.get_shape()[-1]
            print(channel_num)
            x = slim.conv2d(x, channel_num, 3, 1, activation_fn=activation_fn, data_format=data_format)
            print(x)
            x = slim.conv2d(x, channel_num, 3, 1, activation_fn=activation_fn, data_format=data_format)
            print(x)
            x = x + res
            print(x)
            print('ififififififififififififi')
            if idx < repeat_num - 1:
                # x = slim.layers.conv2d_transpose(x, hidden_num * (repeat_num-idx-1), 3, 2, activation_fn=activation_fn, data_format=data_format)
                x = upscale(x, 2, data_format)
                print(x)
                x = slim.conv2d(x, hidden_num * (repeat_num-idx-1), 1, 1, activation_fn=activation_fn, data_format=data_format)
                print(x)
                print('num = %d'%(hidden_num * (repeat_num-idx-1)))

        out = slim.conv2d(x, input_channel, 3, 1, activation_fn=None, data_format=data_format)
        print(out)

    variables = tf.contrib.framework.get_variables(vs)
    # print(variables)
    return out, z, variables

def UAE_noFC_After2Noise(x, input_channel, z_num, repeat_num, hidden_num, data_format, activation_fn=tf.nn.elu, noise_dim=64, reuse=False):
    with tf.variable_scope("G") as vs:
        # Encoder   编码器
        # print('noise_dim = %d'%noise_dim)
        encoder_layer_list = []
        # print(x)
        x = slim.conv2d(x, hidden_num, 3, 1, activation_fn=activation_fn, data_format=data_format)
        # print(x)

        prev_channel_num = hidden_num
        # repeat_num = 3
        for idx in range(repeat_num):
            channel_num = hidden_num * (idx + 1)
            # print(idx)
            # print(channel_num)
            x = slim.conv2d(x, channel_num, 3, 1, activation_fn=activation_fn, data_format=data_format)
            # print(x)
            x = slim.conv2d(x, channel_num, 3, 1, activation_fn=activation_fn, data_format=data_format)
            # print(x)
            if idx > 0:
                encoder_layer_list.append(x)
            if idx < repeat_num - 1:
                x = slim.conv2d(x, channel_num, 3, 2, activation_fn=activation_fn, data_format=data_format)
                # print(x)
                #x = tf.contrib.layers.max_pool2d(x, [2, 2], [2, 2], padding='VALID')
        
        # 是否加入噪声
        print('noise_dim = %d'%noise_dim)
        if noise_dim>0:
            # pdb.set_trace()
            noise = tf.random_uniform(
                (tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], noise_dim), minval=-1.0, maxval=1.0)
            # tf.concat是连接两个矩阵的操作
            # 如果concat_dim是1，那么在某一个shape的第二个维度上连
            # 第二个参数values：就是两个或者一组待连接的tensor了
            # print(noise)
            x = tf.concat([x, noise], -1)
            # print(x)
# 
        for idx in range(repeat_num):
            # print(idx)
            if idx < repeat_num - 1:
                # print('<<<<<<<<<<<<<<<<<<<<<<<,')
                x = tf.concat([x,encoder_layer_list[-1-idx]], axis=-1)
                # print(x)
            x = slim.conv2d(x, hidden_num, 3, 1, activation_fn=activation_fn, data_format=data_format)
            # print(x)
            x = slim.conv2d(x, hidden_num, 3, 1, activation_fn=activation_fn, data_format=data_format)
            # print(x)
            if idx < repeat_num - 1:
                # 高精度
                # print('<<<<<<<<<<<<<<<<<<<<<<<,')
                x = upscale(x, 2, data_format)
                # print(x)

        out = slim.conv2d(x, input_channel, 3, 1, activation_fn=None, data_format=data_format)
        # print('oooooooooooooooooooooooooo')
        # print(out)

    variables = tf.contrib.framework.get_variables(vs)
    # print(variables)
    return out, variables


def UAE_noFC_AfterNoise(x, input_channel, z_num, repeat_num, hidden_num, data_format, activation_fn=tf.nn.elu, noise_dim=64, reuse=False):
    with tf.variable_scope("G") as vs:
        # Encoder
        encoder_layer_list = []
        x = slim.conv2d(x, hidden_num, 3, 1, activation_fn=activation_fn, data_format=data_format)

        prev_channel_num = hidden_num
        for idx in range(repeat_num):
            channel_num = hidden_num * (idx + 1)
            x = slim.conv2d(x, channel_num, 3, 1, activation_fn=activation_fn, data_format=data_format)
            x = slim.conv2d(x, channel_num, 3, 1, activation_fn=activation_fn, data_format=data_format)
            encoder_layer_list.append(x)
            if idx < repeat_num - 1:
                x = slim.conv2d(x, channel_num, 3, 2, activation_fn=activation_fn, data_format=data_format)
                #x = tf.contrib.layers.max_pool2d(x, [2, 2], [2, 2], padding='VALID')
        
        if noise_dim>0:
            # pdb.set_trace()
            noise = tf.random_uniform(
                (tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], noise_dim), minval=-1.0, maxval=1.0)
            x = tf.concat([x, noise], -1)

        for idx in range(repeat_num):
            x = tf.concat([x,encoder_layer_list[-1-idx]], axis=-1)
            x = slim.conv2d(x, hidden_num, 3, 1, activation_fn=activation_fn, data_format=data_format)
            x = slim.conv2d(x, hidden_num, 3, 1, activation_fn=activation_fn, data_format=data_format)
            if idx < repeat_num - 1:
                x = upscale(x, 2, data_format)

        out = slim.conv2d(x, input_channel, 3, 1, activation_fn=None, data_format=data_format)

    variables = tf.contrib.framework.get_variables(vs)
    return out, variables

def UAE_noFC_AfterNoise1(x, input_channel, z_num, repeat_num, hidden_num, data_format, activation_fn=tf.nn.elu, noise_dim=64, reuse=False):
    with tf.variable_scope("G") as vs:
        # Encoder
        encoder_layer_list = []
        x = slim.conv2d(x, hidden_num, 3, 1, activation_fn=activation_fn, data_format=data_format)

        prev_channel_num = hidden_num
        for idx in range(repeat_num):
            channel_num = hidden_num * (idx + 1)
            x = slim.conv2d(x, channel_num, 3, 1, activation_fn=activation_fn, data_format=data_format)
            x = slim.conv2d(x, channel_num, 3, 1, activation_fn=activation_fn, data_format=data_format)
            encoder_layer_list.append(x)
            if idx < repeat_num - 1:
                x = slim.conv2d(x, channel_num, 3, 2, activation_fn=activation_fn, data_format=data_format)
                #x = tf.contrib.layers.max_pool2d(x, [2, 2], [2, 2], padding='VALID')
        
        if noise_dim>0:
            # pdb.set_trace()
            noise = tf.random_uniform(
                (tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], noise_dim), minval=-1.0, maxval=1.0)
            x = tf.concat([x, noise], -1)

        for idx in range(repeat_num):
            x = tf.concat([x,encoder_layer_list[-1-idx]], axis=-1)
            x = slim.conv2d(x, hidden_num, 3, 1, activation_fn=activation_fn, data_format=data_format)
            x = slim.conv2d(x, hidden_num, 3, 1, activation_fn=activation_fn, data_format=data_format)
            if idx < repeat_num - 1:
                x = upscale(x, 2, data_format)

        out = slim.conv2d(x, input_channel, 3, 1, activation_fn=None, data_format=data_format)

    variables = tf.contrib.framework.get_variables(vs)
    return out, variables
    
def GeneratorCNN_Pose_UAEAfterResidual_UAEnoFCAfter2Noise(x, pose_target, input_channel, z_num, repeat_num, hidden_num, data_format, activation_fn=tf.nn.elu, noise_dim=64, reuse=False):
    # 有一些任务，可能事先需要设置，事后做清理工作。对于这种场景，
    # Python的with语句提供了一种非常方便的处理方式。一个很好的例子是文件处理，
    # 你需要获取一个文件句柄，从文件中读取数据，然后关闭文件句柄。

#     返回一个用于定义创建variable（层）的op的上下文管理器。
# 通过tf.variable_scope()指定作用域进行区分
# 指定了第一个作用域为Pose_AE
    with tf.variable_scope("Pose_AE") as vs1:
        # out, z, variables
        out1, _, var1 = GeneratorCNN_Pose_UAEAfterResidual(x, pose_target, input_channel, z_num, repeat_num, hidden_num, data_format, activation_fn=activation_fn, noise_dim=0, reuse=False)
    with tf.variable_scope("UAEnoFC") as vs2:
        # out, variables
        # print('noise_dim = %d'%noise_dim)
        out2, var2 = UAE_noFC_After2Noise(tf.concat([out1,x],axis=-1), input_channel, z_num, repeat_num-2, hidden_num, data_format, noise_dim=noise_dim, activation_fn=activation_fn, reuse=False)
    return out1, out2, var1, var2


def int_shape(tensor):
    shape = tensor.get_shape().as_list()
    return [num if num is not None else -1 for num in shape]

def get_conv_shape(tensor, data_format):
    shape = int_shape(tensor)
    # always return [N, H, W, C]
    if data_format == 'NCHW':
        return [shape[0], shape[2], shape[3], shape[1]]
    elif data_format == 'NHWC':
        return shape

def nchw_to_nhwc(x):
    return tf.transpose(x, [0, 2, 3, 1])

def nhwc_to_nchw(x):
    return tf.transpose(x, [0, 3, 1, 2])

def reshape(x, h, w, c, data_format):
    if data_format == 'NCHW':
        x = tf.reshape(x, [-1, c, h, w])
    else:
        x = tf.reshape(x, [-1, h, w, c])
    return x

def resize_nearest_neighbor(x, new_size, data_format):
    if data_format == 'NCHW':
        x = nchw_to_nhwc(x)
        x = tf.image.resize_nearest_neighbor(x, new_size)
        x = nhwc_to_nchw(x)
    else:
        x = tf.image.resize_nearest_neighbor(x, new_size)
    return x

def upscale(x, scale, data_format):
    _, h, w, _ = get_conv_shape(x, data_format)
    return resize_nearest_neighbor(x, (h*scale, w*scale), data_format)