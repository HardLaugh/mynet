from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import sys
import tensorflow as tf
import numpy as np

from model import configuration, crnn, CRNNestimator
from data_utils.DataIO import DataReader
from data_utils.vocabulary import Vocabulary, compute_acuracy

slim = tf.contrib.slim


tf.flags.DEFINE_string(
    "dataset_dir", "",
    "dataset dir."
)

tf.flags.DEFINE_string(
    "file_pattern", "",
    "File pattern."
)

tf.flags.DEFINE_string(
    "checkpoints", "",
    "从指定的路径恢复模型"
)
tf.flags.DEFINE_string(
    "train_checkpoints", "",
    "训练存放路径"
)
tf.flags.DEFINE_string(
    "summaries_dir", "",
    "训练存放路径"
)
tf.flags.DEFINE_integer(
    "batch_size", 32,
    "batch size."
)
tf.flags.DEFINE_integer(
    "number_of_steps", 1000,
    "Number of training steps."
)
tf.flags.DEFINE_integer(
    "log_every_n_steps", 100,
    "Frequency at which loss and global step are logged."
)
tf.flags.DEFINE_float(
    'gpu_memory_fraction', 0.7,
    'GPU memory fraction to use.'
)
tf.flags.DEFINE_float(
    "learning_rate", 0.1,
    "learning rate."
)

FLAGS = tf.app.flags.FLAGS


def input_fn(dataset_dir, file_pattern, model_config, batch_size, mode='train'):
    """训练集输入函数
    """
    queue = DataReader(dataset_dir, file_pattern,
                       model_config, batch_size=batch_size)
    with tf.name_scope(None, 'input_queue_'+mode):
        return queue.input_fn()


def main(_):

    assert FLAGS.file_pattern, "--file_pattern is required"
    assert FLAGS.train_checkpoints, "--train_checkpoints is required"
    # assert FLAGS.summaries_dir, "--summaries_dir is required"

    vocab = Vocabulary()

    model_config = configuration.ModelConfig()

    training_config = configuration.TrainingConfig()
    
    estimator = CRNNestimator.CRNN_test(
        model_dir=FLAGS.train_checkpoints,
        model_config=model_config,
        train_config=training_config,
        FLAGS=FLAGS,
    )
    # 数据队列初始化
    train_queue = input_fn(FLAGS.dataset_dir, FLAGS.file_pattern,
                                model_config, batch_size=FLAGS.batch_size)

    eval_queue = input_fn(FLAGS.dataset_dir, 'ocr_val_*.tfrecord',
                                model_config, batch_size=64)

    train_spec = tf.estimator.TrainSpec(
        input_fn=lambda:train_queue,
        max_steps=FLAGS.number_of_steps,
        hooks=None,
    )

    eval_spec = tf.estimator.EvalSpec(
        input_fn=lambda:eval_queue,
        steps=30,
        throttle_secs=2,
    )

    tf.estimator.train_and_evaluate(
        estimator,
        train_spec,
        eval_spec)


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()