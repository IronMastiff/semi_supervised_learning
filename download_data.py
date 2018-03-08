import pickle as pkl
import time
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat    # 加载mat文件
import tensorflow as tf

from urllib.request import urlretrieve    # 通过url下载数据
from os.path import isfile, isdir
from tqdm import tqdm    # 用包装好的方法实现进度条

data_dir = './data/'

'''--------Download Training/Test data--------'''
def download_data():

    if not isdir( data_dir ):
        os.makedirs( data_dir )

    class DLProgress( tqdm ):
        last_block = 0

        def hook( self, block_num = 1, block_size = 1, total_size = None ):
            self.total = total_size
            self.update( ( block_num - self.last_block ) * block_size )
            self.last_block = block_num


    if not isfile( data_dir + 'train_32x32.mat' ):
        with DLProgress( unit = 'B', unit_scale = True, miniters = 1, desc = 'SVHN Training Set' ) as pbar:
            urlretrieve(
                'http://ufldl.stanford.edu/housenumbers/train_32x32.mat',
                data_dir + 'train_32x32.mat',
                pbar.hook
            )

    if not isflie( data_dir + 'test_32x32.mat' ):
        with DLProgress( unit = 'B', unit_scale = True, miniters = 1, desc = 'SVHN Testing Set' ) as pbar:
            urlretrieve(
                'http://ufldl.stanford.edu/housenumbers/test_32x32.mat',
                data_dir + 'test_32x32.mat',
                pbar.hook
            )