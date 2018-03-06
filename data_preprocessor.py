import pickle as pkl
import time
import os

import matplotlib.pyplot as plot
import numpy as np
from scipy.io import loadmat
import tensorflow as tf

import utils

'''--------Scale data to target size--------'''
def scale( x, feature_range = ( -1, 1 ) ):
    # Scale to ( 0 ,1 )
    x = ( ( x - x.min() ) / ( 255 - x.min() ) )

    # Scale to feature_range
    min, max = feature_range
    x = x * ( max - min ) + min

    return x



'''--------Preprocessing data--------'''
class Dataset:
    def __init__( self, train, test, val_frac = 0.5, shuffle =  True, scale_func = None ):
        split_idx = int( len( test['y'] ) * ( 1 - val_frac ) )
        self.test_x, self.valid_x = test['X'][:, :, :, : split_idx], test['X'][:, :, :, split_idx :]
        self.test_y, self.valid_y = test['y'][:, :, :, : split_idx], test['y'][:, :, :, split_idx :]
        self.train_x, self.train_y = train['X'], train['y']
        # The SVHM dataset comes with lot of labels, but for the purpose of this exercise,
        # we will pretend that there are only 1000.
        # We use this mask to say which labels we will allow ourselves to use.
        self.label_mask = np.zeros_like( self.train_y )
        self.label_mask[0 : 1000] = 1

        self.train_x = np.rollaxis( self.train_x, 3 )    # np.rollaxis( a, axis, start = 0 )数组换轴， 把axis轴换到start上
        self.valid_x = np.rollaxis( self.valid_x, 3 )
        self.test_x = np.rollaxis( self.test_x, 3 )

        if scale_func is None:
            self.scaler =scale     # 一种骚操作
        else:
            self.scaler = scale_func
        self.train_x = self.scaler( self.train_x )
        self.valid_x = self.scaler( self.valid_x )
        self.test_x = self.scaler( self.test_x )
        self.shuffle = shuffle

    def batches( self, batch_size, which_set = 'train' ):
        x_name = which_set + '_x'
        y_name = which_set + '_y'

        num_examples = len( getattr( dataset, y_name ) )
        if self.shuffle:
            idx = np.arange( num_examples )    # np.arange( start, stop, step, dtype )按顺序输出数字
            np.random.shuffle( idx )    # np.random.shuffle( x )打乱x的顺序并返回
            setattr( dataset, x_name, getattr( dataset, x_name )[idx] )     # setattr( x, 'y', v ) 令x.y = v
            setattr( dataset, y_name, getattr( dataset, y_name )[idx] )
            if which_set == 'train':
                dataset.label_mask = dataset.label_mask[idx]

            dataset_x = getattr( dataset, x_name )
            dataset_y = getattr( dataset, y_name )
            for i in range( 0, num_examples, batch_size ):
                x = dataset_x[i : i + batch_size]
                y = dataset_y[i : i + batch_size]

                if which_set == 'train':
                    # When we use the data for traing, we need to include
                    # the label mask, so we can pretend we don't have access
                    # to some of the labels, as an exercise of our semi-supervised
                    # learning ability
                    yield x, y, self.label_mask[ i: i + batch_size]
                else:
                    yield x, y