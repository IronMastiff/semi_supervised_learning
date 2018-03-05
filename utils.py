import pickle as pkl
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat    # 加载mat文件
import tensorflow as tf

extra_class = 0

from urllib.request import urlretrieve
from os.path import isfile, isdir
from tqdm import tqdm    # 用包装好的方法实现进度条

data_dir = 'data'

# if not isdir( data_dir ):
