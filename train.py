import pickle as pkl
import time
import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as loadmat
import tensorflow as tf

import download_data

data_dir = './data/'

if not( os.path.exists( data_dir ) ):
    download_data.download_data()

trainset = loadmat( data_dir + 'train_32x32.mat' )
testset = loadmat( data_dir + 'test_32x32.mat' )
