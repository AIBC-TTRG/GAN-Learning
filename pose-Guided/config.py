#-*- coding: utf-8 -*-
# 命令行解析
import argparse

print(' config')
def str2bool(v):
    return v.lower() in ('true', '1')

arg_lists = []
# 创建解析器对象ArgumentParser
parser = argparse.ArgumentParser()

# add_argument()方法，用来指定程序需要接受的命令参数
def add_argument_group(name):
    arg = parser.add_argument_group(name)
    # append() 方法用于在列表末尾添加新的对象
    arg_lists.append(arg)
    return arg

# Network
net_arg = add_argument_group('Network')
# net_arg.add_argument('--input_scale_size', type=int, default=64,
#                      help='input image will be resized with the given value as width and height')
# 可选参数：
# dest：如果提供dest，例如dest="a"，那么可以通过args.a访问该参数
# default：设置参数的默认值
# action：参数出发的动作
# store：保存参数，默认
# store_const：保存一个被定义为参数规格一部分的值（常量），而不是一个来自参数解析而来的值。
# store_ture/store_false：保存相应的布尔值
# append：将值保存在一个列表中。
# append_const：将一个定义在参数规格中的值（常量）保存在一个列表中。
# count：参数出现的次数
# parser.add_argument("-v", "--verbosity", action="count", default=0, help="increase output verbosity")
# version：打印程序版本信息
# type：把从命令行输入的结果转成设置的类型
# choice：允许的参数值
# parser.add_argument("-v", "--verbosity", type=int, choices=[0, 1, 2], help="increase output verbosity")
# help：参数命令的介绍
net_arg.add_argument('--img_H', type=int, default=128,
                     help='input image height')
net_arg.add_argument('--img_W', type=int, default=64,
                     help='input image width')
net_arg.add_argument('--conv_hidden_num', type=int, default=128,
                     choices=[64, 128],help='n in the paper')
# net_arg.add_argument('--z_num', type=int, default=64, choices=[64, 128])
net_arg.add_argument('--z_num', type=int, default=64)
# net_arg.add_argument('--noise_dim', type=int, default=10, choices=[10, 128])

# Data
data_arg = add_argument_group('Data')
data_arg.add_argument('--dataset', type=str, default='CelebA')
data_arg.add_argument('--split', type=str, default='train')
data_arg.add_argument('--batch_size', type=int, default=2)
data_arg.add_argument('--grayscale', type=str2bool, default=False)
data_arg.add_argument('--num_worker', type=int, default=4)

# Training / test parameters       测试参数
train_arg = add_argument_group('Training')
train_arg.add_argument('--is_train', type=str2bool, default=True)
train_arg.add_argument('--test_one_by_one', type=str2bool, default=False)
train_arg.add_argument('--optimizer', type=str, default='adam')
train_arg.add_argument('--start_step', type=int, default=0)
data_arg.add_argument('--ckpt_path', type=str, default=None)
data_arg.add_argument('--pretrained_path', type=str, default=None)
# data_arg.add_argument('--checkpoint_path', type=str, default=None)
train_arg.add_argument('--max_step', type=int, default=500000)
# train_arg.add_argument('--lr_update_step', type=int, default=100000, choices=[100000, 75000])
train_arg.add_argument('--lr_update_step', type=int, default=100000, choices=[50000, 100000])
train_arg.add_argument('--d_lr', type=float, default=0.00008)
train_arg.add_argument('--g_lr', type=float, default=0.00008)
train_arg.add_argument('--beta1', type=float, default=0.5)
train_arg.add_argument('--beta2', type=float, default=0.999)
train_arg.add_argument('--gamma', type=float, default=0.5)
train_arg.add_argument('--lambda_k', type=float, default=0.001)
train_arg.add_argument('--use_gpu', type=str2bool, default=True)
train_arg.add_argument('--gpu', type=int, default=-1)
train_arg.add_argument('--model', type=int, default=0)
train_arg.add_argument('--D_arch', type=str, default='DCGAN')  # 'DCGAN'  'noNormDCGAN'  'MultiplicativeDCGAN'  'tanhNonlinearDCGAN'  'resnet101'

# Misc      混合
misc_arg = add_argument_group('Misc')
misc_arg.add_argument('--load_path', type=str, default='')
misc_arg.add_argument('--log_step', type=int, default=200)
misc_arg.add_argument('--save_model_secs', type=int, default=1000)
misc_arg.add_argument('--num_log_samples', type=int, default=3)
misc_arg.add_argument('--log_level', type=str, default='INFO', choices=['INFO', 'DEBUG', 'WARN'])
misc_arg.add_argument('--log_dir', type=str, default='logs')
misc_arg.add_argument('--model_dir', type=str, default=None)
misc_arg.add_argument('--data_dir', type=str, default='data')
misc_arg.add_argument('--test_data_path', type=str, default=None,
                      help='directory with images which will be used in test sample generation')
misc_arg.add_argument('--sample_per_image', type=int, default=64,
                      help='# of sample per image during test sample generation')
misc_arg.add_argument('--random_seed', type=int, default=123)

def get_config():
    print(' get_config')
    # 有时间一个脚本只需要解析所有命令行参数中的一小部分，剩下的命令行参数给两一个脚本或者程序。
    config, unparsed = parser.parse_known_args()
    # 是否使用gpu
    # data_format表示一组彩色图片的问题
    # 第0个维度是样本维，代表样本的数目，第1个维度是通道维，代表颜色通道数。后面两个就是高和宽了。
    # 这种theano风格的数据组织方法，称为“channels_first”，即通道维靠前。
    #  N 表示这批图像有几张，
    # H 表示图像在竖直方向有多少像素，W 表示水平方向像素数，
    # C 表示通道数（例如黑白图像的通道数 C = 1，而 RGB 彩色图像的通道数 C = 3）
    if config.use_gpu:
        data_format = 'NCHW'            #[batch, in_channels, in_height, in_width]  

    else:
        data_format = 'NHWC'            #[batch, in_height, in_width, in_channels]
    # 用于设置属性值，该属性必须存在。
    # object -- 对象。
        # name -- 字符串，对象属性。
        # value -- 属性值
    setattr(config, 'data_format', data_format)
    return config, unparsed
