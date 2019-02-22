#-*- coding: utf-8 -*-
from __future__ import print_function, absolute_import
import os.path as osp

from ..utils.data import Dataset
from ..utils.osutils import mkdir_if_missing
from ..utils.serialization import write_json


class Market1501(Dataset):

    print('Market1501=========')

    def __init__(self, root, split_id=0, num_val=100):
        # super() 函数是用于调用父类(超类)的一个方法
        print('Market1501__init__')
        print(root)
        super(Market1501, self).__init__(root, split_id=split_id)

        print(self._check_integrity())    
        if not self._check_integrity():
            print('nonoonononono')
            raise RuntimeError("Dataset not found or corrupted. " +
                               "Please follow README.md to prepare Market1501 dataset.")

        self.load(num_val)