# %%
import sys
import os
import math
import random

import numpy as np
import tensorflow as tf
import cv2

slim = tf.contrib.slim
%matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from data_utils.vocabulary import Vocabulary
from data_utils.DataIO import DataReader
from model import configuration, crnn
tf.logging.set_verbosity(tf.logging.INFO)

gpu_options = tf.GPUOptions(allow_growth=True)
session_config = tf.ConfigProto(
    log_device_placement=False, gpu_options=gpu_options)

DATASET_DIR = './data/output/'
FP = 'ocr_train_*.tfrecord'
model_config = configuration.ModelConfig()

input_queue = DataReader(DATASET_DIR, FP,
                         model_config, batch_size=2)

with tf.name_scope(None, 'input_queue'):
    input_images, input_labels = input_queue.read()
input_labels = tf.sparse_tensor_to_dense(input_labels)
vocab = Vocabulary()
#%%
with tf.train.MonitoredTrainingSession(config=session_config) as sess:

    for i in range(1):

        images, labels = sess.run([input_images, input_labels])
        print(vocab._to_string(labels[0]))
        plt.figure(1)
        plt.imshow(np.uint8(images[0, :, :, :]))
        plt.figure(2)
        plt.imshow(np.uint8(images[1, :, :, :]))
        plt.show()
