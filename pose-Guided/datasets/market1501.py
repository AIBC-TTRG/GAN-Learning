#-*- coding: utf-8 -*-
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Provides data for the Market1501 dataset.

The dataset scripts used to create the dataset is modified from:
tensorflow/models/slim/datasets/download_and_convert_flowers.py
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf

from datasets import dataset_utils
import pickle
import pdb

# slim是一个使构建，训练，评估神经网络变得简单的库
# slim作为一种轻量级的tensorflow库，使得模型的构建，训练，测试都变得更加简单。
slim = tf.contrib.slim

_FILE_PATTERN = '%s_%s_*.tfrecord'

SPLITS_TO_SIZES = {'train': None, 'test': None}

_NUM_CLASSES = 2

_ITEMS_TO_DESCRIPTIONS = {
    'image_raw_0': 'A color image of varying size.',
    'image_raw_1': 'A color image of varying size.',
    'label': 'A single integer between 0 and 1',
    'id_0': 'A single integer',
    'id_1': 'A single integer',
}

# 通过slim来读取生成的tfrecord
# 读取TFRecord文件的本质，就是通过队列的方式依次将数据解码，并按需要进行数据随机化、图像随机化的过程
from tensorflow.python.ops import parsing_ops
def get_split(split_name, dataset_dir, data_name='Market1501', file_pattern=None, reader=None):
  """Gets a dataset tuple with instructions for reading Market1501.
  数据集元组阅读Market1501指示

  Args:
    split_name: A train/validation split name.
    dataset_dir: The base directory of the dataset sources.
    file_pattern: The file pattern to use when matching the dataset sources.
      It is assumed that the pattern contains a '%s' string so that the split
      name can be inserted.
    reader: The TensorFlow reader type.

    split_name:训练/验证分割的名字。
    dataset_dir:数据集的基本目录来源。
    file_pattern:文件模式匹配时使用数据来源。
    假设模式包含了一个“%s这分割的字符串名称可以插入。
    读者:TensorFlow读者类型。

  Returns: 
    A `Dataset` namedtuple. 数据集元组

  Raises:
    ValueError: if `split_name` is not a valid train/validation split.
    ValueError:如果“split_name”不是一个有效的训练/验证分裂。
  """
  if split_name not in SPLITS_TO_SIZES:
    raise ValueError('split name %s was not recognized.' % split_name)

  if not file_pattern:
    file_pattern = _FILE_PATTERN
  file_pattern = os.path.join(dataset_dir, file_pattern % (data_name, split_name))
  # tfrecord数据文件是一种将图像数据和标签统一存储的二进制文件，能更好的利用内存，
  # 在tensorflow中快速的复制，移动，读取，存储等。
  # Allowing None in the signature so that dataset_factory can use the default.
  if reader is None:
    # 读取tfrecord数据
    reader = tf.TFRecordReader

# 原始图像经过处理后，生成5个文件。flowers_train_00000-of-00005.
# tfrecord到flowers_train_00004-of-00005.tfrecord。 
# 训练时，就要通过slim从这5个文件file_pattern中读取数据，然后组合成batch。
  
  # 第一步
  # 将example反序列化成存储之前的格式。由tf完成

  keys_to_features = {
    # tf.FixedLenFeature 返回的是一个定长的tensor 定长特征解析
    # shape：可当 reshape 来用，如 vector 的 shape 从 (3,) 改动成了 (1,3)。
    # 注：如果写入的 feature 使用了. tostring() 其 shape 就是 ()
    # dtype：必须是 tf.float32， tf.int64， tf.string 中的一种。
    # default_value：feature 值缺失时所指定的值
     'image_raw_0' : tf.FixedLenFeature([], tf.string),
     'image_raw_1' : tf.FixedLenFeature([], tf.string),
     'label': tf.FixedLenFeature([], tf.int64), # For FixedLenFeature, [] means scalar  ,[]意味着标量
     'id_0': tf.FixedLenFeature([], tf.int64),
     'id_1': tf.FixedLenFeature([], tf.int64),
     'cam_0': tf.FixedLenFeature([], tf.int64),
     'cam_1': tf.FixedLenFeature([], tf.int64),
     'image_format': tf.FixedLenFeature([], tf.string, default_value='jpg'),
     'image_height': tf.FixedLenFeature([], tf.int64, default_value=128),
     'image_width': tf.FixedLenFeature([], tf.int64, default_value=64),
     'real_data': tf.FixedLenFeature([], tf.int64, default_value=1),
     'pose_peaks_0': tf.FixedLenFeature([16*8*18], tf.float32),
     'pose_peaks_1': tf.FixedLenFeature([16*8*18], tf.float32),
     'pose_mask_r4_0': tf.FixedLenFeature([128*64*1], tf.int64),
     'pose_mask_r4_1': tf.FixedLenFeature([128*64*1], tf.int64),
     
     'shape': tf.FixedLenFeature([1], tf.int64),
    #  VarLenFeature 不定长特征解析
      'indices_r4_0': tf.VarLenFeature(dtype=tf.int64),
      'values_r4_0': tf.VarLenFeature(dtype=tf.float32),
      'indices_r4_1': tf.VarLenFeature(dtype=tf.int64),
      'values_r4_1': tf.VarLenFeature(dtype=tf.float32),
     'pose_subs_0': tf.FixedLenFeature([20], tf.float32),
     'pose_subs_1': tf.FixedLenFeature([20], tf.float32),
  }

# 将反序列化的数据组装成更高级的格式。由slim完成
  items_to_handlers = {
      'image_raw_0': slim.tfexample_decoder.Image(image_key='image_raw_0', format_key='image_format'),
      'image_raw_1': slim.tfexample_decoder.Image(image_key='image_raw_1', format_key='image_format'),
      'label': slim.tfexample_decoder.Tensor('label'),
      'id_0': slim.tfexample_decoder.Tensor('id_0'),
      'id_1': slim.tfexample_decoder.Tensor('id_1'),
      'pose_peaks_0': slim.tfexample_decoder.Tensor('pose_peaks_0',shape=[16*8*18]),
      'pose_peaks_1': slim.tfexample_decoder.Tensor('pose_peaks_1',shape=[16*8*18]),
      'pose_mask_r4_0': slim.tfexample_decoder.Tensor('pose_mask_r4_0',shape=[128*64*1]),
      'pose_mask_r4_1': slim.tfexample_decoder.Tensor('pose_mask_r4_1',shape=[128*64*1]),

      'pose_sparse_r4_0': slim.tfexample_decoder.SparseTensor(indices_key='indices_r4_0', values_key='values_r4_0', shape_key='shape', densify=False),
      'pose_sparse_r4_1': slim.tfexample_decoder.SparseTensor(indices_key='indices_r4_1', values_key='values_r4_1', shape_key='shape', densify=False),
      
      'pose_subs_0': slim.tfexample_decoder.Tensor('pose_subs_0',shape=[20]),
      'pose_subs_1': slim.tfexample_decoder.Tensor('pose_subs_1',shape=[20]),
  }

# 解码器，进行解码
# keys_to_features,和items_to_handlers两个字典参数。
# key_to_features这个字典需要和TFrecord文件中定义的字典项匹配。
# items_to_handlers中的关键字可以是任意值，但是它的handler的初始化参数必须要来自于keys_to_features中的关键字。

  decoder = slim.tfexample_decoder.TFExampleDecoder(
      keys_to_features, items_to_handlers)

  labels_to_names = None
  if dataset_utils.has_labels(dataset_dir):
    labels_to_names = dataset_utils.read_label_file(dataset_dir)

  print('load pn_pairs_num ......')
  # 用os.path.join()连接两个文件名地址的时候
  fpath = os.path.join(dataset_dir, 'pn_pairs_num_'+split_name+'.p')
  with open(fpath,'r') as f:
    pn_pairs_num = pickle.load(f)

# dataset对象定义了数据集的文件位置，解码方式等元信息
# 定义数据集的数据提供者类
  return slim.dataset.Dataset(
      data_sources=file_pattern,
      reader=reader,
      decoder=decoder,
      num_samples=pn_pairs_num,   #数据的总数   # 手动生成了pn_pairs_num个文件， 每个文件里只包含一个example  I287516  I 把整数或布尔值;十进制字符串参数
      items_to_descriptions=_ITEMS_TO_DESCRIPTIONS,
      num_classes=_NUM_CLASSES,     #2
      labels_to_names=labels_to_names #字典形式，格式为：id:class_call,   #None
      )    





