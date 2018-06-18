"""Image embedding ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import tensorflow as tf

slim = tf.contrib.slim


def crnn_base(inputs,
              is_training=True,
              dropout_keep_prob=0.5,
              scope='cnn'):

    with tf.variable_scope(scope, 'cnn', [inputs]) as scope:
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            activation_fn=tf.nn.relu,
                            padding='SAME',
                            weights_initializer=tf.truncated_normal_initializer(
                                0.0, 0.01),
                            weights_regularizer=slim.l2_regularizer(0.0005)):

            net = slim.conv2d(inputs, 64, [3, 3], 1, scope='conv1')
            net = slim.max_pool2d(net, [2, 2], 2, scope='pool1')

            net = slim.conv2d(net, 128, [3, 3], 1, scope='conv2')
            net = slim.max_pool2d(net, [2, 2], 2, scope='pool2')

            net = slim.conv2d(net, 256, [3, 3], 1, activation_fn=None, scope='conv3')
            net = slim.batch_norm(net, is_training=is_training, scope='Bn1')
            net = slim.conv2d(net, 256, [3, 3], 1, scope='conv4')

            net = slim.max_pool2d(net, [2, 1], stride=[2, 1], scope='pool3')
            

            net = slim.conv2d(net, 512, [3, 3], 1, activation_fn=None, scope='conv5')
            net = slim.batch_norm(net, is_training=is_training, scope='Bn2')

            net = slim.conv2d(net, 512, [3, 3], 1, scope='conv6')

            net = slim.max_pool2d(net, [2, 1], stride=[2, 1], scope='pool4')
            
            net = slim.conv2d(net, 512, [2, 2], [2, 1], activation_fn=None, scope='conv7')
            net = slim.batch_norm(net, is_training=is_training, scope='Bn3')


    return net
