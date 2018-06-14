# %%
import sys
import os
import math
import random

import numpy as np
import tensorflow as tf
import cv2

slim = tf.contrib.slim

# %%
%matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# %%
from data_utils.vocabulary import Vocabulary
from data_utils.DataIO import DataReader
from model import configuration, crnn
tf.logging.set_verbosity(tf.logging.INFO)

# %%
gpu_options = tf.GPUOptions(allow_growth=True)
config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
sess = tf.InteractiveSession(config=config)

# %% 模型建立
DATASET_DIR='./data/output/'
FP = 'ocr_train_*.tfrecord'
model_config = configuration.ModelConfig()

input_queue = DataReader(DATASET_DIR, FP,
                            model_config, batch_size=2)

with tf.name_scope(None, 'input_queue'):
    input_images, input_labels = input_queue.read()

with tf.train.MonitoredTrainingSession(config=session_config) as sess:

    for i in range(1):

        images, input_labels = sess.run([input_images, input_labels])

        plt.figure(1)
        plt.imshow(images[0,:,:,:])
        plt.figure(2)
        plt.imshow(images[1,:,:,:])
        plt.show()


