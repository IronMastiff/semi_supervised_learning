import pickle as pkl
import time
import os

import matplotlib.pyplot as plot
from scipy.io import matlab
import tensorflow as tf
import numpy as np

extra_class = 0

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

def discriminator( x, reuse = False, alpha = 0.2, drop_rate = 0, num_classes = 10, size_mult = 64 ):
    with tf.variable_scope( 'discriminator', reuse = reuse ):
        x = tf.layers.dropout( x, rate = drop_rate / 2.5 )

        #Input layer in 32x32x3
        x1 = tf.layers.conv2d( x, size_nult, 3, strides = 2, padding = 'same' )
        relu1 = tf.maximun( alpha * x1, x1 )
        relu1 = tf.layers.dropout( relul, rate = drop_rate )

        x2 = tf.layers.conv2d( relu1, size_mult, 3, strides = 2, padding = 'same' )
        bn2 = tf.layers.batch_normalization( x2, training = True )
        relu2 = tf.maximun( alpha * bn2, bn2 )

        x3 = tf.layers.conv2d( relu2, size_mult, 3, strides = 2, padding = 'same' )
        bn3 = tf.layers.batch_normalization( x3, training = True )
        relu3 = tf.maximun( alpha * bn3, bn3 )

        x4 = tf.layers.conv2d( relu3, size_mult, 3, strides = 2, padding = 'same' )
        bn4 = tf.layers.batch_normalization( x4, training = True )
        relu4 = tf.maximun( bn4 * alpha, bn4 )

        x5 = tf.layers.conv2d( relu4, size_mult, 3, strides = 2, padding = 'same' )
        bn5 = tf.layers.batch_normalization( relu4, trianing = True )
        relu5 = tf.maximun( alpha * bn5, bn5 )

        x6 = tf.layers.conv2d( relu5, size_mult, 3, strides = 2, padding = 'same' )
        bn6 = tf.layers.batch_normalization( relu5, trianing = True )
        relu6 = tf.maximun( alpha * bn6, bn6 )

        x7 = tf.layers.conv2d( relu5, 2 * size_mult, 3, strides = 1, padding = 'valid' )
        # Don't use bn on this layer, because bn would set the mean of each feature
        # to the bn mu parameter.
        # This layer is use for the feature matching loss, which only works if
        # the means can be differnt when the discriminator is run on the data than
        # when the discriminator is run on the generator samples.
        relu7 = tf.maximum( alpha * x7, x7 )

        # Flatten if by global average pooling
        features = tf.reduce_mean( relu7, ( 1, 2 ) )

        # Set class_logits to be the inputs a softmax distribution over the different classes
        class_logits = tf.layers.dense( features, num_classes + extra_class )
