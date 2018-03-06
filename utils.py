import pickle as pkl
import time
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat    # 加载mat文件
import tensorflow as tf

extra_class = 0

from urllib.request import urlretrieve
from os.path import isfile, isdir
from tqdm import tqdm    # 用包装好的方法实现进度条

data_dir = 'data'

if not isdir( data_dir ):
    os.makedirs( data_dir )

class DLProgress( tqdm ):
    last_block = 0

    def hook( self, block_num = 1, block_size = 1, total_size = None ):
        self.total = total_size
        self.updata( ( block_num - self.last_block ) * block_size )
        self.last_block = block_num


if not isfile( data_dir + 'train_32x32.mat' ):
    with DLProgress( unit = 'B', unit_scale = True, miniters = 1, desc = 'SVHN Training Set' ) as pbar:    # ????
        urlretrieve(
            'http://ufldl.stanford.edu/housenumbers/train_32x32.amt',
            data_dir + 'train_32x32.mat',
            pbar.hook
        )