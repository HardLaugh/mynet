from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import tensorflow as tf
from model import image_embedding
from data_utils import inputs_ops, image_processing


class ocr_inception_v3(object):
    def __init__(self, vocab, config, mode, train_inception=False):
        assert mode in ["train", "eval", "inference"]
        self.config = config
        self.mode = mode
        self.train_inception = train_inception
        self.vocab = vocab
        # Reader for the input data.
        self.reader = tf.TFRecordReader()

        # To match the "Show and Tell" paper we initialize all variables with a
        # random uniform initializer.
        self.initializer = tf.random_uniform_initializer(
            minval=-self.config.initializer_scale,
            maxval=self.config.initializer_scale)

        # A float32 Tensor with shape [batch_size, height, width, channels].
        self.images = None

        # An int32 Tensor with shape [batch_size, padded_length].
        self.input_seqs = None

        # An int32 Tensor with shape [batch_size, padded_length].
        self.target_seqs = None

        # An int32 0/1 Tensor with shape [batch_size, padded_length].
        self.input_mask = None

        # A float32 Tensor with shape [batch_size, embedding_size].
        self.image_embeddings = None

        # A float32 Tensor with shape [batch_size, padded_length, embedding_size].
        self.seq_embeddings = None

        # A float32 scalar Tensor; the total loss for the trainer to optimize.
        self.total_loss = None

        # A float32 Tensor with shape [batch_size * padded_length].
        self.target_cross_entropy_losses = None

        # A float32 Tensor with shape [batch_size * padded_length].
        self.target_cross_entropy_loss_weights = None

        # Collection of variables from the inception submodel.
        self.inception_variables = []

        # Function to restore the inception submodel from checkpoint.
        self.init_fn = None

        # Global step Tensor.
        self.global_step = None

    def is_training(self):
        return self.mode == "train"

    def process_image(self, encoded_image):
        return image_processing._process_image(encoded_image,
                                                self.is_training(),
                                                Height=self.config.image_height,
                                                Width=self.config.image_width,
                                                image_format=self.config.image_format)
    def batch_and_pad_captions(self, image, caption):
        return inputs_ops._batch_and_pad_captions(
            image,
            caption, 
            start=self.vocab.start,
            stop=self.vocab.stop,
            batch_size=self.config.batch_size
        )

    def build_inputs(self):
        
        encoded_image, _, caption = inputs_ops.queue_iterator(
            dataset_dir=self.config.dataset_dir,
            file_pattern=self.config.input_file_pattern,
            nums_thread=self.config.num_preprocess_threads
        )
        image = self.process_image(encoded_image)

        images, input_seqs, target_seqs, input_mask = (
            self.batch_and_pad_captions(image, caption)
        )

        self.images = images
        self.input_seqs = input_seqs
        self.target_seqs = target_seqs
        self.input_mask = input_mask

    def build_image_embeddings(self):

        inception_output = image_embedding.inception_v3(
            self.images,
            trainable=self.train_inception,
            is_training=self.is_training()
        )

        self.inception_variables = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, scope="InceptionV3")

        # Map inception output into embedding space.
        with tf.variable_scope("image_embedding") as scope:
            # image_embeddings = tf.contrib.layers.fully_connected(
            #     inputs=inception_output,
            #     num_outputs=self.config.embedding_size,
            #     activation_fn=None,
            #     weights_initializer=self.initializer,
            #     biases_initializer=None,
            #     scope=scope)
            image_embeddings = tf.layers.dense(
                inputs=inception_output,
                units=self.config.embedding_size,
                activation=None,
                kernel_initializer=self.initializer,
                bias_initializer=None,
            )

        # Save the embedding size in the graph.
        tf.constant(self.config.embedding_size, name="embedding_size")

        self.image_embeddings = image_embeddings

    def build_seq_embeddings(self):

        with tf.variable_scope("seq_embedding"), tf.device("/cpu:0"):
            embedding_map = tf.get_variable(
                name="map",
                shape=[self.config.vocab_size, self.config.embedding_size],
                initializer=self.initializer
            )
            seq_embeddings = tf.nn.embedding_lookup(embedding_map, self.input_seqs)

        self.seq_embeddings = seq_embeddings

    def build_model(self):

        lstm_cell = tf.contrib.rnn.BasicLSTMCell(
            num_units=self.config.num_lstm_units, 
            state_is_tuple=True
        )
        if self.mode == "train":
            lstm_cell = tf.contrib.rnn.DropoutWrapper(
                lstm_cell,
                input_keep_prob=self.config.lstm_dropout_keep_prob,
                output_keep_prob=self.config.lstm_dropout_keep_prob
            )
        
        with tf.variable_scope("lstm", initializer=self.initializer) as lstm_scope:
            zero_state = lstm_cell.zero_state(
                batch_size=self.image_embeddings.get_shape()[0], 
                dtype=tf.float32
            )
            _, initial_state = lstm_cell(self.image_embeddings, zero_state)

            lstm_scope.reuse_variables()

            if self.mode == "inference":
                pass
            else:
                sequence_length = tf.reduce_sum(self.input_mask, 1)
                lstm_outputs, _ = tf.nn.dynamic_rnn(
                    cell=lstm_cell,
                    inputs=self.seq_embeddings,
                    sequence_length=sequence_length,
                    initial_state=initial_state,
                    dtype=tf.float32,
                    scope=lstm_scope
                )

        lstm_outputs = tf.reshape(lstm_outputs, [-1, lstm_cell.output_size])

        with tf.variable_scope("logits") as logits_scope:
            # logits = tf.contrib.layers.fully_connected(
            #     inputs=lstm_outputs,
            #     num_outputs=self.config.vocab_size,
            #     activation_fn=None,
            #     weights_initializer=self.initializer,
            #     scope=logits_scope
            # )
            logits = tf.layers.dense(
                inputs=lstm_outputs,
                units=self.config.vocab_size,
                activation=None,
                kernel_initializer=self.initializer,
                bias_initializer=None,
            )

        if self.mode == "inference":
            pass
        else:
            targets = tf.reshape(self.target_seqs, [-1])
            weights = tf.to_float(tf.reshape(self.input_mask, [-1]))

            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=targets,
                logits=logits
            )

            batch_loss = tf.div(
                tf.reduce_sum(tf.multiply(losses, weights)),
                tf.reduce_sum(weights),
                name="batch_loss"
            )

            tf.losses.add_loss(batch_loss)
            total_loss = tf.losses.get_total_loss()

            # Add summaries.
            tf.summary.scalar("losses/batch_loss", batch_loss)
            tf.summary.scalar("losses/total_loss", total_loss)
            for var in tf.trainable_variables():
                tf.summary.histogram("parameters/" + var.op.name, var)

            self.total_loss = total_loss
            self.target_cross_entropy_losses = losses  # Used in evaluation.
            self.target_cross_entropy_loss_weights = weights  # Used in evaluation.

    def setup_inception_initializer(self):

        if self.mode != "inference":
            saver = tf.train.Saver(self.inception_variables)

        def restore_fn(sess):
            tf.logging.info("Restoring Inception variables from checkpoint file %s",
                            self.config.inception_checkpoint_file)
            saver.restore(sess, self.config.inception_checkpoint_file)

        self.init_fn = restore_fn

    def setup_global_step(self):

        global_step = tf.Variable(
            initial_value=0,
            name="global_step",
            trainable=False,
            collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES]
        )

        self.global_step = global_step

    def build(self):

        self.build_inputs()
        self.build_image_embeddings()
        self.build_seq_embeddings()
        self.build_model()
        self.setup_inception_initializer()
        self.setup_global_step()



            







