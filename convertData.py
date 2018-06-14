from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import sys
import numpy as np
import tensorflow as tf

from data_utils.vocabulary import Vocabulary
from data_utils.DataIO import DataWriter

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'dataset_dir', None,
    '数据集所在路径')
tf.app.flags.DEFINE_string(
    'name', 'ocr',
    '输出文件的前缀名字')
tf.app.flags.DEFINE_string(
    'output_dir', './',
    '输出路径')
tf.app.flags.DEFINE_integer(
    'str_size', 10,
    '数据集的字符串size.')
tf.app.flags.DEFINE_boolean(
    'split', True,
    '训练集的百分比'
)



def main(_):
    if not FLAGS.dataset_dir:
        raise ValueError('You must supply the dataset directory with --dataset_dir')

    if not FLAGS.output_dir:
        raise ValueError('You must supply the output directory with --output_dir')
    print('Dataset directory:', FLAGS.dataset_dir)
    print('Output directory:', FLAGS.output_dir)

    vocab = Vocabulary()
    
    writer = DataWriter(vocab, FLAGS.dataset_dir, FLAGS.output_dir, 
                        FLAGS.str_size, FLAGS.name, FLAGS.split)

    writer.build_data()

if __name__ == '__main__':
    tf.app.run()
    


    