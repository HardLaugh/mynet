from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import tensorflow as tf
from tensorflow.python.estimator import estimator
from model.image_embedding import crnn_base

slim = tf.contrib.slim


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
        return tf.contrib.rnn.BasicLSTMCell(lstm_size)

    fw_cell_list = [lstm_cell(nh)
                    for nh in [lstm_size, lstm_size]]

    bw_cell_list = [lstm_cell(nh)
                    for nh in [lstm_size, lstm_size]]
    lstm_outputs, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
        cells_fw=fw_cell_list,
        cells_bw=bw_cell_list,
        inputs=image_embeddings,
        dtype=tf.float32
    )  # [batch,25, 256*2]

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


def crnn_fn(features, labels, mode, params):

    def is_training():
        return mode == tf.estimator.ModeKeys.TRAIN

    num_classes = params['ModelConfig'].num_classes
    lstm_size = params['ModelConfig'].num_lstm_units
    sequence_length = params['ModelConfig'].sequence_length
    log_every_n_steps = params['log_every_n_steps']

    # 模型建立
    with tf.variable_scope('CRNN', reuse=tf.AUTO_REUSE):

        with tf.variable_scope("image_embedding"):
            image_embeddings = embedding_and_squeeze(features, is_training())
        print(image_embeddings)
        print(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
        sys.exit()
        # print(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
        # sys.exit()
        with tf.variable_scope('LSTM'):
            logits = Bilstm(image_embeddings,
                            lstm_size=lstm_size, num_classes=num_classes,
                            is_training=is_training())
        # 把lstm的模型变量添加到 GraphKeys.GLOBAL_VARIABLES
        if not slim.get_model_variables('CRNN/LSTM'):
            lstm_variables = tf.get_collection(
                tf.GraphKeys.GLOBAL_VARIABLES, scope='CRNN/LSTM')
            for var in lstm_variables:
                tf.add_to_collection(tf.GraphKeys.MODEL_VARIABLES, var)

    dyn_batch_size = tf.shape(logits)[1]
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
        preds = tf.sparse_tensor_to_dense(decoded[0], name='prediction')
        gt_labels = tf.sparse_tensor_to_dense(
            labels, name='Ground_Truth')

    if mode == tf.estimator.ModeKeys.PREDICT:
        """Prediction
        """
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
        tf.losses.add_loss(loss)
        total_loss = tf.losses.get_total_loss(False)

    tf.summary.scalar(name='Seq_Dist', tensor=sequence_dist)

    if mode == tf.estimator.ModeKeys.EVAL:
        """Evaluation
        """
        eval_tensors_print = {
            'loss': loss,
            'Seq_Dist': sequence_dist,
        }
        eval_loghook = tf.train.LoggingTensorHook(
            tensors=eval_tensors_print,
            every_n_iter=log_every_n_steps,
        )
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=None, evaluation_hooks=[eval_loghook])

    # Create training op.
    assert mode == tf.estimator.ModeKeys.TRAIN
    global_step = tf.Variable(
        initial_value=0,
        name="global_step",
        trainable=False,
        collections=[tf.GraphKeys.GLOBAL_STEP,
                     tf.GraphKeys.GLOBAL_VARIABLES]
    )

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
    tf.summary.scalar(name='total_loss', tensor=total_loss)

    optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate)
    train_op = tf.contrib.training.create_train_op(
        total_loss=total_loss,
        optimizer=optimizer,
        global_step=global_step
    )

    globalhook = tf.train.StepCounterHook(
        every_n_steps=log_every_n_steps,
    )
    tensors_print = {
        'global_step': global_step,
        'loss': loss,
        'Seq_Dist': sequence_dist,
        'LR': learning_rate,
    }
    loghook = tf.train.LoggingTensorHook(
        tensors=tensors_print,
        every_n_iter=log_every_n_steps,
    )
    return tf.estimator.EstimatorSpec(mode,
                                      loss=loss,
                                      train_op=train_op,
                                      training_hooks=[globalhook, loghook])


# CRNN estimator
class CRNN_test(estimator.Estimator):
    """CRNN estimator, for train, eval and predict
    """

    def __init__(self, model_dir, model_config, train_config, FLAGS, config=None):

        params = {'ModelConfig': model_config,
                  'TrainingConfig': train_config,
                  'log_every_n_steps': FLAGS.log_every_n_steps,
                  }

        def _model_fn(features, labels, mode, params):

            return crnn_fn(features, labels, mode, params)

        super(CRNN_test, self).__init__(
            model_fn=_model_fn, model_dir=model_dir, config=config,
            params=params, warm_start_from=None)
