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
from data_utils.vocabulary import Vocabulary, compute_acuracy

slim = tf.contrib.slim  # pylint: disable=E1101


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
tf.logging.set_verbosity(tf.logging.INFO)


def main(_):

    assert FLAGS.file_pattern, "--file_pattern is required"
    assert FLAGS.train_checkpoints, "--train_checkpoints is required"
    # assert FLAGS.summaries_dir, "--summaries_dir is required"

    vocab = Vocabulary()

    model_config = configuration.ModelConfig()

    training_config = configuration.TrainingConfig()
    print(FLAGS.learning_rate)
    training_config.initial_learning_rate = FLAGS.learning_rate

    sequence_length = model_config.sequence_length
    batch_size = FLAGS.batch_size

    # summaries_dir = FLAGS.summaries_dir
    # if not tf.gfile.IsDirectory(summaries_dir):
    #     tf.logging.info("Creating training directory: %s", summaries_dir)
    #     tf.gfile.MakeDirs(summaries_dir)

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
        # 为避免Dataset.batch的不确定提供，改为动态获取batch_size
        dyn_batch_size = tf.shape(logits)[1]

        with tf.name_scope(None, 'loss'):
            loss = tf.reduce_mean(
                tf.nn.ctc_loss(
                    labels=input_labels,
                    inputs=logits,
                    sequence_length=sequence_length *
                    tf.ones(dyn_batch_size, dtype=tf.int32)
                ),
                name='compute_loss',
            )
            tf.losses.add_loss(loss)
            total_loss = tf.losses.get_total_loss(False)

        with tf.name_scope(None, 'decoder'):
            decoded, _ = tf.nn.ctc_beam_search_decoder(
                logits,
                sequence_length*tf.ones(dyn_batch_size, dtype=tf.int32),
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

        # print(len(slim.get_model_variables()))
        # print('>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        # print(len(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)))
        # sys.exit()
        global_step = tf.Variable(
            initial_value=0,
            name="global_step",
            trainable=False,
            collections=[tf.GraphKeys.GLOBAL_STEP,
                         tf.GraphKeys.GLOBAL_VARIABLES]
        )

        start_learning_rate = training_config.initial_learning_rate
        learning_rate = tf.train.exponential_decay(
            start_learning_rate,
            global_step,
            decay_steps=training_config.learning_decay_steps,
            decay_rate=training_config.learning_rate_decay_factor,
            staircase=True,
        )

        # summary
        # Add summaries for variables.
        for variable in slim.get_model_variables():
            tf.summary.histogram(variable.op.name, variable)
        tf.summary.scalar(name='Seq_Dist', tensor=sequence_dist)
        tf.summary.scalar(name='global_step', tensor=global_step)
        tf.summary.scalar(name='learning_rate', tensor=learning_rate)
        tf.summary.scalar(name='total_loss', tensor=total_loss)

        # global/secs hook
        globalhook = tf.train.StepCounterHook(
            every_n_steps=FLAGS.log_every_n_steps,
        )
        # 保存chekpoints的hook
        # saver = tf.train.Saver(max_to_keep=training_config.max_checkpoints_to_keep)
        # saverhook = tf.train.CheckpointSaverHook(
        #     checkpoint_dir=FLAGS.train_checkpoints,
        #     save_steps=2000,
        #     saver=saver,
        # )
        # #保存summaries的hook
        # merge_summary_op = tf.summary.merge_all()
        # summaryhook = tf.train.SummarySaverHook(
        #     save_steps=200,
        #     output_dir=FLAGS.summaries_dir,
        #     summary_op=merge_summary_op,
        # )
        # 训练时需要logging的hook
        tensors_print = {
            'global_step': global_step,
            'loss': loss,
            'Seq_Dist': sequence_dist,
            'LR': learning_rate,
            # 'accurays':accurays,
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

        # extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        # with tf.control_dependencies(extra_update_ops):
        #     optimizer = tf.train.AdadeltaOptimizer(
        #         learning_rate=learning_rate).minimize(loss=total_loss, global_step=global_step)

        optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate)
        train_op = tf.contrib.training.create_train_op(  # pylint: disable=E1101
            total_loss=total_loss,
            optimizer=optimizer,
            global_step=global_step
        )
        # train_op = tf.group([optimizer, total_loss, sequence_dist])
        with tf.train.MonitoredTrainingSession(
                checkpoint_dir=FLAGS.train_checkpoints,
                hooks=[globalhook, loghook, stophook],
                save_checkpoint_secs=180,
                save_summaries_steps=100,
                config=session_config) as sess:
            while not sess.should_stop():
                oloss, opreds, ogt_labels = sess.run(
                    [train_op, preds, gt_labels])
                accuray = compute_acuracy(opreds, ogt_labels)
                print("accuracy: %9f" % (accuray))

        # tf.contrib.training.train(
        #     train_op,
        #     logdir=FLAGS.train_checkpoints,
        #     hooks=[loghook, stophook],
        #     save_checkpoint_secs=180,
        #     save_summaries_steps=100,
        #     config=session_config,
        # )


if __name__ == "__main__":
    tf.app.run()
