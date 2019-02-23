#-*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from trainer import *
from trainer import *
from trainer256 import *
from config import get_config
from utils import prepare_dirs_and_logger, save_config

import pdb, os
print(' main111')

def main(config):
    
    print(' main')
    prepare_dirs_and_logger(config)

    if config.gpu>-1:
        # 设置系统环境变量 
        # 1、os.environ['环境变量名称']='环境变量值' #其中key和value均为string类型
        # print('config.gpu = %d',config.gpu)
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"]=str(config.gpu)

    config.data_format = 'NHWC'
    
    if 1==config.model: 
        trainer = PG2(config)
        print(' jieshula ------------')
        trainer.init_net()
    elif 11==config.model:
        trainer = PG2_256(config)
        trainer.init_net()
        
    if config.is_train:
        save_config(config)
        trainer.train()
    else:
        # if not config.load_path:
        #     raise Exception("[!] You should specify `load_path` to load a pretrained model")
        # 如果没有配置。load_path:提高异常(“[!)你应该指定load_path负载pretrained模型”
        trainer.test()

if __name__ == "__main__":
    config, unparsed = get_config()
    print(' ...........')
    main(config)

