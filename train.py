import pickle as pkl
import time
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
import tensorflow as tf

import download_data
import data_preprocessor
import net

'''--------Confing Data--------'''
data_dir = './data/'
checkpoint = './checkpoint'
image = './image'
real_size = ( 32, 32, 3 )
z_size = 100
learning_rate = 0.0003

batch_size = 128
epochs = 25


if not( os.path.exists( data_dir ) ):
    download_data.download_data()

if not( os.path.exists( checkpoint ) ):
    os.makedirs( checkpoint )

if not ( os.path.exists( image ) ):
    os.makedirs( image )

trainset = loadmat( data_dir + 'train_32x32.mat' )
testset = loadmat( data_dir + 'test_32x32.mat' )

net = net.GAN( real_size, z_size, learning_rate )

dataset = data_preprocessor.Dataset( trainset, testset )

train_accuracies, test_accuracies, samples = net.train( net, dataset, epoches, batch_size, figsize = ( 10, 5 ) )

fig, ax = plt.subplots()
plt.plot( train_accuracies, label = 'Train', alpha = 0.5 )
plt.plot( test_accuracies, label = 'Test', alpha = 0.5 )
plt.title( 'Accuracy' )
plt.legend()

for i in range( len( samples ) ):
    fig.ax = net.view_samples( i, samples, 5, 10, figsize = ( 10, 5 ) )
    fig.savefig( 'image/smaples_{:03d}.png'.format( i ) )
    plt.close

