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

        x4 = tf.layers.conv2d( relu3, 2 * size_mult, 3, strides = 2, padding = 'same' )
        bn4 = tf.layers.batch_normalization( x4, training = True )
        relu4 = tf.maximun( bn4 * alpha, bn4 )

        x5 = tf.layers.conv2d( relu4, 2 * size_mult, 3, strides = 2, padding = 'same' )
        bn5 = tf.layers.batch_normalization( relu4, trianing = True )
        relu5 = tf.maximun( alpha * bn5, bn5 )

        x6 = tf.layers.conv2d( relu5, 2 * size_mult, 3, strides = 2, padding = 'same' )
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


        # Set gen_logits such that P( input is real | input ) = sigmoid( gan_logits ).
        # Keep in mind that class_logits gives you the probability distribution over as all the real
        # classes and the fake class. You need to work out how to trainsform this multicalss softmax
        # distribution into a binary real-vs-fake decision that con be described with a sigmoid.
        # Numerical stability is very important.
        # You'll probably need to use this numerical stability trick:
        # log sum_i exp a_i = m + log sum_i exp(a_i - m).
        # This is numerically stable when m = max_i a_i.
        # (It helps to think about what goes wrong when...
        #   1. One value of a_i is very large
        #   2. All the values of a_i are very negative
        # This trick and this value of m fix both those cases, but the naive implementation and
        # other values of m encounter various problems)
        if extra_class:
            real_class_logits, fake_class_logits = tf.split( class_logits, [num_classes, 1], 1 )
            assert fake_class_logits.get_shape()[1] == 1, fake_class_logits.getshape()
            fake_class_logits = tf.squeeze( fake_class_logits )
        else:
            real_class_logits = class_logits
            fake_class_logits = 0

        mx = tf.reduce_max( real_class_logits, 1, keep_dims = True )
        stabel_real_class_logits = tf.log( tf.reduce_sum( tf.exp( stable_real_class_logits ), 1 ) ) + tf.squeeze( mx ) - fake_class_logits

        gan_logits = tf.log( tf.reduce_sum( tf.exp( stable_real_class_logits ), 1 ) ) + tf.squeeze( mx ) - fake_class_logits

        out = tf.nn.softmax( class_logits )

        return out, class_logits, gan_logits, features

def model_loss( input_real, input_z, output_dim, y, num_classes, label_mask, alpha = 0.2, drop_rate = 0 ):
    """
    Get the loss for the discrimiter and generator
    :param input_real: Images from the real dataset
    :param input_z: Z input
    :param output_dim: The number of channels in teh output image
    :param y: Integer class labels
    :param num_classes: The number of classes
    :param alpha: The slope of the left half of leaky ReLU activation
    :param drop_rate: The probability of dropping a hidden unit
    :return: A tupe of ( discriminator loss, generator loss )
    """


    # These number multiply the size of each of the generator and the discriminator,
    # respectvely. You can reduce them to run your code faster for debugging purposes.
    g_size_mult = 32
    d_size_mult = 64

    # Here we run the generator and the discriminatior
    g_model = generator( input_z, output_dim, alpha = alpha, size_mult = g_size_mult )
    d_on_data = discriminator( input_real, alpha = alpha, drop_rete = drop_rate, size_mult = d_size_mult )
    d_model_real, class_logits_on_data, gan_logits_on_data, data_featrues = d_on_data
    d_on_samples = discriminator( g_model, reuse = True, alpha = alpha, drop_rate = drop_rate, size_mult = d_size_mult )
    d_model_fake, class_logits_on_samples, gan_logits_on_samples, samples, sample_featrues = d_on_samples

    # Here we run teh generator and the discriminator.
    # This should combine two different losses:
    # 1. The loss for the GAN problem, where we minimize teh cross-entropy for the binary
    #    real-vs-fake classification problem.
    # 2. The loss for the SVHN digit classification problem, where we minimize the cross-entropy
    #    for teh multi-class softmax. For this one we use the labels. Don't forget to ignore
    #    use 'label_mask' to ignore the examples that we are pretending are unlabeled for teh
    #    semi-supervised learining problem.
    d_loss_real = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits( logits = gan_logits_on_data,
                                                                           labels = tf.ones_like( gan_logits_on_data ) ) )
    d_loss_fake = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits( logits = gan_logits_on_samples,
                                                                           labels = tf.zeros_like( gan_logits_on_samples ) ) )
    y = tf.squeeze( y )     # tf.squeeze( x ) 去除x中的1
    class_cross_entropy = tf.nn.softmax_cross_entropy_with_logits( logits = class_logits_on_data,
                                                                   labels = tf.one_hot( y, num_classes + extra_class,
                                                                                        dtype = tf.float32 ) )
    class_cross_entropy = tf.squeeze( class_cross_entropy )
    label_mask = tf.squeeze( tf.to_float( label_mask ) )
    d_loss_class = tf.reduce_sum( label_mask * class_cross_entropy ) / tf.maximum( 1., tf.reduce_sum( label_mask ) )
    d_loss = d_loss_class + d_loss_real + d_loss_fake

    # Here we set 'g_loss' to the "feature matching" loss invented by Tim Salimans at OpenAI.
    # This loss consists of minimizing the absolute difference between the expected features
    # on the data and the expected features on the generated samples.
    # This loss works better for semi-supervised leraning than teh trdition GAN losses.
    data_mments = tf.reduce_mean( data_featrues, axis = 0 )
    sample_moments = tf.reduce_mean( sample_featrues, axis = 0 )
    g_loss = tf.reduce_mean( tf.abs( data_moments - sample_mooments ) )    # tf.abs( x, name = None )取绝对值

    pred_class = tf.cast( tf.argmax( class_logits_on_data, 1 ), tf.int32 )    # tf.cast( x, dtype, name = None )tf类型转换    tf.argmax( input, axis )返回axis中数值最大的数字的索引位置axis = 0按行，axis = 1按列
    eq = tf.equal( tf.squeeze( y ), pred_class )
    correct = tf.reduce_sum( tf.to_float( eq ) )
    masked_correct = tf.reduce_sum( label_mask * tf.to_float( eq ) )

    return d_loss, g_loss, correct, masked_correct, g_model

def model_opt( d_loss, g_loss, learning_rate, betal ):
    """
    Get optimization operations
    :param d_loss: Discriminator loss Tensor
    :param g_loss: Generator loss Tensor
    :param learning_rate: Learning Rate Placeholder
    :param betal: The exponential decay rate for the 1st moment in the optimaizer
    :return: A tupe of ( discriminator training operation, generator training operation )
    """
    # Get weights and biases to update. Get them separtely for the dicriminator and the generator
    t_vars = tf.trainable_variables()
    d_vars = [var for var in t_vars if var.name.startwith( 'discriminator' )]
    g_vars = [var for var in t_vars if var.name.startwith( 'generator' )]
    for t in t_vars:
        assert t in d_vars or t in g_vars

    # Minimize both players' costs simultaneously
    d_train_opt = tf.train.AdamOptimizer( learning_rate, betal = betal ).minimize( d_loss, var_list = d_vars )
    g_train_opt = tf.train.AdamOptimizer( learning_rate, betal = betal ).minimize( g_loss, var_list = g_vars )
    shrink_lr = tf.assign( learning_rate, learning_rate * 0.9 )    # tf.assign( ref, value )用value的值更行ref

    return d_train_opt, g_train_opt, shrink_lr

class GAN:
    """
    A GAN model.
    :param real_size: The shape of the real data.
    :param z_size: The number of entries in the z code vector.
    :param learnin_rate: The learning rate to use for Adam.
    :param num_classes: The number of classes to recognize.
    :param alpha: The slope of the left half of the leaky ReLU activation
    :param beta1: The beta1 parameter for Adam.
    """
    def __init__( self, real_size, z_size, learning_rate, num_classes = 10, alpha = 0.2, betal = 0.5 ):
        tf.reset_default_graph()

        self.learning_rate = tf.Variable( learning_rate, trainable = False )
        self.input_real, self.input_z, self.y, self.label_mask = model_inputs( real_size, z_size )

        loss_results = model_loss( self.input_real, self.input_z,
                                   real_size[2], self.y, num_classes, label_mask = self.label_mask,
                                   alpha = 0.2, drop_rate = self.drop_rate )
        self.d_loss, self.g_loss, self.correct, self.masked_correct, self.samples = loss_results

        self.d_opt, self.g_opt, self.shrink_lr = model_opt( self.d_loss, self.g_loss, self.learning_rate, betal )