from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import tensorflow as tf
from model.image_embedding import crnn_base
from tensorflow.python.estimator import estimator  # pylint: disable=E0611

slim = tf.contrib.slim  # pylint: disable=E1101


def embedding_and_squeeze(images, is_training):
    """CNN层
    """
    image_embeddings = crnn_base(
        images,
        is_training=is_training,
        dropout_keep_prob=0.7,
    )  # [batch, 1, 25, 512]
    image_embeddings = tf.squeeze(
        input=image_embeddings,
        axis=1,
    )  # [batch, 25, 512]

    return image_embeddings


def Bilstm(image_embeddings, lstm_size, num_classes, is_training):
    """bi_lstm层
    """
    def lstm_cell(lstm_size):
        return tf.contrib.rnn.BasicLSTMCell(lstm_size)  # pylint: disable=E1101

    fw_cell_list = [lstm_cell(nh)
                    for nh in [lstm_size, lstm_size]]

    bw_cell_list = [lstm_cell(nh)
                    for nh in [lstm_size, lstm_size]]
    lstm_outputs, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(  # pylint: disable=E1101
        cells_fw=fw_cell_list,
        cells_bw=bw_cell_list,
        inputs=image_embeddings,
        dtype=tf.float32
    )

    lstm_outputs = tf.layers.dropout(
        inputs=lstm_outputs,
        rate=0.5,
        training=is_training,
        name='dropout'
    )

    [batch_s, frame, cell_ouput_size] = lstm_outputs.get_shape().as_list()

    lstm_outputs = tf.layers.dense(
        inputs=lstm_outputs,
        units=num_classes,
        kernel_initializer=tf.truncated_normal_initializer(0.0, 0.01),
        use_bias=False,
        name='affine',
    )
    if frame:
        lstm_outputs = tf.reshape(
            lstm_outputs, [-1, frame, num_classes])
    else:
        lstm_outputs = tf.reshape(
            lstm_outputs, [batch_s, -1, num_classes])

    logits = tf.transpose(lstm_outputs, (1, 0, 2), name='logits')

    tf.add_to_collection('logits', logits)

    return logits


def crnn(features, lstm_size, num_classes, is_training=True):
    """CRNN model
    """
    with tf.variable_scope('CRNN', reuse=tf.AUTO_REUSE):

        with tf.variable_scope("image_embedding"):
            image_embeddings = embedding_and_squeeze(features, is_training)

        with tf.variable_scope('LSTM'):
            logits = Bilstm(image_embeddings,
                            lstm_size=lstm_size, num_classes=num_classes,
                            is_training=is_training)
        # 由于lstm不是slim库添加的，需要手动把lstm的模型变量添加到 GraphKeys.GLOBAL_VARIABLES
        if not slim.get_model_variables('CRNN/LSTM'):
            lstm_variables = tf.get_collection(
                tf.GraphKeys.GLOBAL_VARIABLES, scope='CRNN/LSTM')
            for var in lstm_variables:
                tf.add_to_collection(tf.GraphKeys.MODEL_VARIABLES, var)

    return logits


def compute_sparse_tensor_accuracy(preds, labels):
    """compute batch accuracy
        args: preds, labels both are sparse tensor
    """
    # reset shape to max ,preds.dense_shape[1] and labels.dense_shape[1]
    # default_value set to -1
    with tf.name_scope(None, 'compute_accuracy'):

        batch_size = preds.dense_shape[0]
        preds_length = preds.dense_shape[1]
        labels_length = labels.dense_shape[1]
        # [batch_p, preds_length] = preds.get_shape()
        # [batch_l, labels_length] = labels.get_shape()

        max_length = tf.maximum(preds_length, labels_length)
        # reset 为统一尺寸的矩阵，方便后续比较
        pad_preds = tf.sparse_reset_shape(preds, [batch_size, max_length])
        pad_labels = tf.sparse_reset_shape(labels, [batch_size, max_length])

        # 函数默认置为0，故手动default value set to negative integer, 为了区分label 0
        pad_preds = tf.sparse_tensor_to_dense(pad_preds, default_value=-1)
        pad_preds = tf.to_int32(pad_preds)
        pad_labels = tf.sparse_tensor_to_dense(pad_labels, default_value=-1)

        # sparse value mask，也即有效的位置掩膜
        preds_mask = pad_preds > -1
        labels_mask = pad_labels > -1
        # gt length ，用于统计单个字母正确率
        # gt_length = tf.reduce_sum(tf.cast(labels_mask, dtype=tf.int32), axis=1)

        # 每一个样本需要比较的掩膜
        mask = tf.logical_or(preds_mask, labels_mask)
        # 每一个样本需要比较的有效长度。每个样本的每个time都为正确，则判断这个样本为正确
        compare_length = tf.reduce_sum(tf.cast(mask, dtype=tf.int32), axis=1)

        accuracy = tf.logical_and(
            tf.equal(pad_preds, pad_labels),
            mask,
        )
        accuracy = tf.reduce_sum(tf.cast(accuracy, dtype=tf.int32), axis=1)

        # 全部字符匹配的正确率
        acc1 = tf.reduce_mean(
            tf.to_float(tf.equal(accuracy, compare_length)),
            name='acc1',
        )

        # 单个字符的匹配所在位置的正确率
        acc2 = tf.reduce_mean(
            tf.div(
                tf.to_float(accuracy),
                tf.to_float(compare_length)
            ),
            name='acc2',
        )
    tf.summary.scalar(name='acc1', tensor=acc1)
    tf.summary.scalar(name='acc2', tensor=acc2)

    return acc1, acc2


def model_fn(features, labels, mode, params):
    """model function build
        args: features, labels, mode, params 
    """
    def is_training():
        return mode == tf.estimator.ModeKeys.TRAIN

    num_classes = params['ModelConfig'].num_classes
    lstm_size = params['ModelConfig'].num_lstm_units
    # sequence_length = params['ModelConfig'].sequence_length
    log_every_n_steps = params['log_every_n_steps']

    # 模型建立
    logits = crnn(features,
                  lstm_size,
                  num_classes,
                  is_training=is_training())

    dyn_batch_size = tf.shape(logits)[1]
    sequence_length = tf.shape(logits)[0]
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
                    labels
                ),
                name='seq_dist',
            )

    if mode == tf.estimator.ModeKeys.PREDICT:
        """Prediction
        """
        preds = tf.sparse_tensor_to_dense(decoded[0], default_value=-1)
        predictions = {
            'preds': preds,
            'preds_seq_dist': sequence_dist,
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # compute loss
    with tf.name_scope(None, 'loss'):
        loss = tf.reduce_mean(
            tf.nn.ctc_loss(
                labels=labels,
                inputs=logits,
                sequence_length=sequence_length *
                tf.ones(dyn_batch_size, dtype=tf.int32)
            ),
            name='compute_loss',
        )

    tf.summary.scalar(name='Seq_Dist', tensor=sequence_dist)

    acc1, acc2 = compute_sparse_tensor_accuracy(decoded[0], labels)
    if mode == tf.estimator.ModeKeys.EVAL:
        """Evaluation
        """
        # metrics = {
        #     'acc1': (acc1, None),
        #     'acc2': (acc2, None),
        # }
        eval_loghook = tf.train.LoggingTensorHook(
            tensors={
                'loss': loss,
                'Seq_Dist': sequence_dist,
                'acc1': acc1,
                'acc2': acc2,
            },
            every_n_iter=log_every_n_steps,
        )
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=None, evaluation_hooks=[eval_loghook])

    # Create training op.
    assert mode == tf.estimator.ModeKeys.TRAIN
    # note:: 不再需要手动创建global_step
    global_step = tf.train.get_or_create_global_step()
    assert global_step.graph == tf.get_default_graph()

    start_learning_rate = params['TrainingConfig'].initial_learning_rate
    learning_rate = tf.train.exponential_decay(
        start_learning_rate,
        global_step,
        decay_steps=params['TrainingConfig'].learning_decay_steps,
        decay_rate=params['TrainingConfig'].learning_rate_decay_factor,
        staircase=True,
    )
    # add summaries
    for variable in slim.get_model_variables():
        tf.summary.histogram(variable.op.name, variable)
    tf.summary.scalar(name='global_step', tensor=global_step)
    tf.summary.scalar(name='learning_rate', tensor=learning_rate)

    optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate)
    train_op = tf.contrib.training.create_train_op(  # pylint: disable=E1101
        total_loss=loss,
        optimizer=optimizer,
        global_step=global_step
    )

    # globalhook = tf.train.StepCounterHook(
    #     every_n_steps=log_every_n_steps,
    # )
    loghook = tf.train.LoggingTensorHook(
        tensors={
            'Seq_Dist': sequence_dist,
            'LR': learning_rate,
            'acc1': acc1,
            'acc2': acc2,
        },
        every_n_iter=log_every_n_steps,
    )
    return tf.estimator.EstimatorSpec(mode,
                                      loss=loss,
                                      train_op=train_op,
                                      training_hooks=[loghook])


# CRNN estimator
class CRNNestimator(estimator.Estimator):
    """CRNN estimator, for train, eval and predict
    """

    def __init__(self, model_config, train_config, FLAGS, config=None):

        params = {'ModelConfig': model_config,
                  'TrainingConfig': train_config,
                  'log_every_n_steps': FLAGS.log_every_n_steps,
                  }

        def _model_fn(features, labels, mode, params):

            return model_fn(features, labels, mode, params)

        super(CRNNestimator, self).__init__(
            model_fn=_model_fn, model_dir=None, config=config,
            params=params, warm_start_from=None)
