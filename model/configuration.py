from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class ModelConfig(object):

    def __init__(self):

        self.num_classes = 37
        self.sequence_length = 53

        self.num_preprocess_threads = 4

        self.image_height = 32
        self.image_width = 300

        self.initializer_scale = 0.08

        self.num_lstm_units = 256
        self.layers_lstm = 2

        self.lstm_dropout_keep_prob = 0.7
        self.cnn_dropout_keep_prob = 0.7


class TrainingConfig(object):

    def __init__(self):

        self.num_examples_per_epoch = 20000

        self.optimizer = "adadelta"

        self.initial_learning_rate = 0.1
        self.learning_rate_decay_factor = 0.1
        self.learning_decay_steps = 40002

        self.clip_gradients = None

        self.max_checkpoints_to_keep = 5
