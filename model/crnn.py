from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import tensorflow as tf
from model.image_embedding import crnn_base



slim = tf.contrib.slim


class CRNN(object):


    def __init__(self, lstm_size, num_classes, mode):

        assert mode in ['train', 'eval', 'inference']
        self.mode = mode
        self.num_classes = num_classes
        # Reader for the input data.
        self.lstm_size = lstm_size

        #self.input = tf.place



    def is_training(self):
        return self.mode == 'train'

    def build(self, images):

        with tf.variable_scope('CRNN', reuse=tf.AUTO_REUSE):
             # image_embedding
            with tf.variable_scope("image_embedding"):

                image_embeddings = crnn_base(
                    images,
                    is_training=self.is_training(),
                    dropout_keep_prob=0.7,
                )#[batch, 1, 25, 512]

                image_embeddings = tf.squeeze(
                    input=image_embeddings,
                    axis=1,
                )#[batch, 25, 512]
                
            #LSTM layers
            with tf.variable_scope('LSTM'):

                def lstm_cell(lstm_size):
                    return tf.contrib.rnn.BasicLSTMCell(lstm_size)

                fw_cell_list = [lstm_cell(nh) 
                            for nh in [self.lstm_size, self.lstm_size]]
                bw_cell_list = [lstm_cell(nh) 
                            for nh in [self.lstm_size, self.lstm_size]]


                lstm_outputs, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
                    cells_fw=fw_cell_list,
                    cells_bw=bw_cell_list,
                    inputs=image_embeddings,
                    dtype=tf.float32
                )#[batch,25, 512*2]

                lstm_outputs = tf.layers.dropout(
                    inputs=lstm_outputs,
                    rate=0.5,
                    training=self.is_training(),
                    name='dropout'
                )

                [_, frame, cell_ouput_size] = lstm_outputs.get_shape().as_list()
                lstm_outputs = tf.reshape(lstm_outputs, [-1, cell_ouput_size])


                lstm_outputs = tf.layers.dense(
                    inputs=lstm_outputs,
                    units=self.num_classes,
                    kernel_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                    use_bias=False,
                    name='affine',
                )
            
            lstm_outputs = tf.reshape(lstm_outputs, [-1, frame, self.num_classes])

            raw_pred = tf.argmax(tf.nn.softmax(lstm_outputs), axis=2, name='raw_prediction')

            logits = tf.transpose(lstm_outputs, (1, 0, 2), name='logits')

            tf.add_to_collection('logits', logits)
            tf.add_to_collection('raw_pred', raw_pred)

            if not slim.get_model_variables('CRNN/LSTM'):
                lstm_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='CRNN/LSTM')
                for var in lstm_variables:
                    tf.add_to_collection(tf.GraphKeys.MODEL_VARIABLES, var)

        return logits









                




