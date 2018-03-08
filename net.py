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