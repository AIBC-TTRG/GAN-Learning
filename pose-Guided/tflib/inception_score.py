#-*- coding: utf-8 -*-
# From https://github.com/openai/improved-gan/blob/master/inception_score/model.py
# Code derived from tensorflow/tensorflow/models/image/imagenet/classify_image.py
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import sys
import tarfile

import numpy as np
from six.moves import urllib
import tensorflow as tf
import glob
import scipy.misc
import math
import sys

MODEL_DIR = '/home/aibc/Documents/pose/Pose-Guided-Person-Image-Generation/test_result'
DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
softmax = None

# Call this function with list of images. Each of elements should be a 
# numpy array with values ranging from 0 to 255.
# 调用这个函数与图像列表。每个元素应该numpy数组值从0到255不等
def get_inception_score(images, splits=10):
  # 用以检查某一条件是否为True，若该条件为False则会给出一个AssertionError。
  print('get_inception_score')
  assert(type(images) == list)
  assert(type(images[0]) == np.ndarray)
  assert(len(images[0].shape) == 3)
  assert(np.max(images[0]) > 10)
  assert(np.min(images[0]) >= 0.0)
  inps = []
  for img in images:
    img = img.astype(np.float32)
    inps.append(np.expand_dims(img, 0))
  bs = 100
  # 在服务器上用多GPU做训练时，由于想只用其中的一个GPU设备做训练，
        # 可使用深度学习代码运行时往往出现多个GPU显存被占满清理。出现该现象主要是tensorflow训练时默认占用所有GPU的显存。
        # 使用allow_growth option，刚一开始分配少量的GPU容量，然后按需慢慢的增加，由于不会释放
        #内存，所以会导致碎片
  gpu_options = tf.GPUOptions(allow_growth=True)
  # tf.ConfigProto一般用在创建session的时候。用来对session进行参数配置
        # allow_soft_placement=True ： 如果你指定的设备不存在，允许TF自动分配设备
  sess_config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
  # 启动图, 运行 op
  with tf.Session(config=sess_config) as sess:
    preds = []
    # ceil() 函数返回数字的上入整数
    n_batches = int(math.ceil(float(len(inps)) / float(bs)))
    for i in range(n_batches):
        # sys.stdout.write(".")
        # sys.stdout.flush()
        inp = inps[(i * bs):min((i + 1) * bs, len(inps))]
        # 这个函数用于将多个数组进行连接，这与stack函数很容易混淆，
        # 他们之间的区别是concatenate会把当前要匹配的元素降一维，即去掉最外面那一层括号
        inp = np.concatenate(inp, 0)
        pred = sess.run(softmax, {'ExpandDims:0': inp})
        preds.append(pred)
    preds = np.concatenate(preds, 0)
    scores = []
    # splits = 10
    for i in range(splits):
      part = preds[(i * preds.shape[0] // splits):((i + 1) * preds.shape[0] // splits), :]
      kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
      kl = np.mean(np.sum(kl, 1))
      scores.append(np.exp(kl))
    return np.mean(scores), np.std(scores)

# This function is called automatically.
def _init_inception():
  print('_init_inception')
  global softmax
  if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
  filename = DATA_URL.split('/')[-1]
  filepath = os.path.join(MODEL_DIR, filename)
  if not os.path.exists(filepath):
    def _progress(count, block_size, total_size):
      sys.stdout.write('\r>> Downloading %s %.1f%%' % (
          filename, float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()
    filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
    print()
    statinfo = os.stat(filepath)
    print('Succesfully downloaded', filename, statinfo.st_size, 'bytes.')
  tarfile.open(filepath, 'r:gz').extractall(MODEL_DIR)
  with tf.gfile.FastGFile(os.path.join(
      MODEL_DIR, 'classify_image_graph_def.pb'), 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')
  # Works with an arbitrary minibatch size.models
  gpu_options = tf.GPUOptions(allow_growth=True)
  sess_config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
  with tf.Session(config=sess_config) as sess:
    pool3 = sess.graph.get_tensor_by_name('pool_3:0')
    ops = pool3.graph.get_operations()
    for op_idx, op in enumerate(ops):
        for o in op.outputs:
            shape = o.get_shape()
            shape = [s.value for s in shape]
            new_shape = []
            for j, s in enumerate(shape):
                if s == 1 and j == 0:
                    new_shape.append(None)
                else:
                    new_shape.append(s)
            o._shape = tf.TensorShape(new_shape)
    w = sess.graph.get_operation_by_name("softmax/logits/MatMul").inputs[1]
    logits = tf.matmul(tf.squeeze(pool3), w)
    softmax = tf.nn.softmax(logits)

if softmax is None:
  _init_inception()
