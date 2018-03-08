import pickle as pkl
import time
import os

import matplotlib.pyplot as plot
from scipy.io import matlab
import tensorflow as tf
import numpy as np

import data_preprocessor
import utils

def model_inputs( real_dim, z_dim ):
    inputs_real = tf.placeholder( tf.float32, ( None, *real_dim ), name = 'input_real' )
    inputs_z = tf.placeholder( tf.float32, ( None, z_dim ), name = 'input_z' )
    y = tf.placeholder( tf.float32, ( None ), name = 'y' )
    label_mask = tf.placeholder( tf.float32, ( None ), name = 'label_mask' )

    return input_real, input_z, y, label_mask

def generator( z, output_dim, reuse = False, alpha = 0.2, training = True, size_mult = 128 ):
    with tf.variable_scope( 'generator', reuse = reuse ):
        # First fully connected layer
        x1 = tf.layers.dense( z, 4 * 4 * size_mult * 4 )
        # Reshape it to start the convolutional stack
        x1 = tf.reshape( x1, ( -1, 4, 4, size_mult * 4) )
        x1 = tf.layers.batch_normalization( x1, training = training )
        x1 = tf.maximun( alpha * x1, x1 )

        x2 = tf.layers.conv2d_transpose( x1, size_mult * 2, 5, strides = 2, padding = 'same' )
        x2 = tf.layers.batch_normalization( x2, training = training )
        x2 = tf.maximun( alpha * x2, x2 )
        
        x3 = tf.layers.conv2d_transpose( x2, size_mult, 5, strides = 2, padding = 'same' )
        x3 = tf.layers.batch_normalization( x3, training = training )
        x3 = tf.maximun( alpha * x3, x3 )

        # Out layer
        logits = tf.layers.conv2d_transpose( x3, output_dim, 5, strides = 2, padding = 'same' )

        out = tf.tanh( logits )

        return out