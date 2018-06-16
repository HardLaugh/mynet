from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import sys
import tensorflow as tf
import numpy as np

from model import configuration, crnn
from data_utils.DataIO import DataReader
from data_utils.vocabulary import Vocabulary

slim = tf.contrib.slim



tf.flags.DEFINE_string(
    "dataset_dir", "",
    "dataset dir."
)

tf.flags.DEFINE_string(
    "file_pattern", "",
    "File pattern of sharded TFRecord input files."
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
tf.logging.set_verbosity(tf.logging.INFO)


def main(_):

    assert FLAGS.file_pattern, "--file_pattern is required"
    assert FLAGS.train_checkpoints, "--train_checkpoints is required"
    assert FLAGS.summaries_dir, "--summaries_dir is required"

    vocab = Vocabulary()

    model_config = configuration.ModelConfig()

    training_config = configuration.TrainingConfig()
    print(FLAGS.learning_rate)
    training_config.initial_learning_rate=FLAGS.learning_rate


    sequence_length = model_config.sequence_length
    batch_size = FLAGS.batch_size
    

    summaries_dir = FLAGS.summaries_dir
    if not tf.gfile.IsDirectory(summaries_dir):
        tf.logging.info("Creating training directory: %s", summaries_dir)
        tf.gfile.MakeDirs(summaries_dir)

    train_checkpoints = FLAGS.train_checkpoints
    if not tf.gfile.IsDirectory(train_checkpoints):
        tf.logging.info("Creating training directory: %s", train_checkpoints)
        tf.gfile.MakeDirs(train_checkpoints)

    # 数据队列初始化
    input_queue = DataReader(FLAGS.dataset_dir, FLAGS.file_pattern,
                             model_config, batch_size=batch_size)

    g = tf.Graph()
    with g.as_default():
        # 数据队列
        with tf.name_scope(None, 'input_queue'):
            input_images, input_labels = input_queue.read()

        # 模型建立
        model = crnn.CRNN(256, model_config.num_classes, 'train')
        logits = model.build(input_images)

        with tf.name_scope(None, 'loss'):

            loss = tf.reduce_mean(
                tf.nn.ctc_loss(
                    labels=input_labels,
                    inputs=logits,
                    sequence_length=sequence_length *
                    tf.ones(batch_size, dtype=tf.int32)
                ),
                name='compute_loss',
            )
            tf.losses.add_loss(loss)
            total_loss = tf.losses.get_total_loss(False)

        with tf.name_scope(None, 'decoder'):
            decoded, _ = tf.nn.ctc_beam_search_decoder(
                logits,
                sequence_length*tf.ones(batch_size, dtype=tf.int32),
                merge_repeated=False,
            )
            with tf.name_scope(None, 'acurracy'):
                sequence_dist = tf.reduce_mean(
                    tf.edit_distance(
                        tf.cast(decoded[0], tf.int32),
                        input_labels
                    ),
                    name='seq_dist',
                )
            preds = tf.sparse_tensor_to_dense(decoded[0], name='prediction')
            gt_labels = tf.sparse_tensor_to_dense(
                input_labels, name='Ground_Truth')

        global_step = tf.Variable(
            initial_value=0,
            name="global_step",
            trainable=False,
            collections=[tf.GraphKeys.GLOBAL_STEP,
                         tf.GraphKeys.GLOBAL_VARIABLES]
        )
        
        # 训练时需要logging的hook
        tensors_print = {
            'global_step': global_step,
            #'loss': loss,
        }
        loghook = tf.train.LoggingTensorHook(
            tensors=tensors_print,
            every_n_iter=FLAGS.log_every_n_steps,
        )
        # 停止hook
        stophook = tf.train.StopAtStepHook(last_step=FLAGS.number_of_steps)

        gpu_options = tf.GPUOptions(
            per_process_gpu_memory_fraction=FLAGS.gpu_memory_fraction)
        session_config = tf.ConfigProto(log_device_placement=False,
                                        gpu_options=gpu_options)

        
        train_op = tf.assign_add(global_step, tf.constant(1))
        session = tf.train.ChiefSessionCreator(
            config=session_config,
            checkpoint_dir=FLAGS.train_checkpoints,
        )

        labels_shape = input_labels.dense_shape
        with tf.train.MonitoredSession(
            session,
            hooks=[loghook, stophook]) as sess:

            while not sess.should_stop():
                test_logits, test_images, test_shape, _ = \
                        sess.run([logits, input_images, labels_shape, input_labels])
                if test_logits.shape[1] != FLAGS.batch_size or test_images.shape[0] != FLAGS.batch_size or test_shape[0] != FLAGS.batch_size:
                    print("get it!!!!!")
                test_loss = sess.run([loss])
                sess.run(train_op)
                


if __name__ == "__main__":
    tf.app.run()
