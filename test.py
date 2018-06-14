# %%
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
from model import configuration, crnn
tf.logging.set_verbosity(tf.logging.INFO)

# %%
gpu_options = tf.GPUOptions(allow_growth=True)
config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
sess = tf.InteractiveSession(config=config)

# %% 模型建立
vocab = Vocabulary()

with tf.name_scope(None, 'input_image'):
    img_input = tf.placeholder(tf.uint8, shape=(32, 100, 3))
    image = tf.to_float(img_input)
    image = tf.expand_dims(image, 0)

model = crnn.CRNN(256, 37, 'train')
logit = model.build(image)

decodes, _ = tf.nn.ctc_beam_search_decoder(inputs=logit,
                                           sequence_length=25*np.ones(1),
                                           merge_repeated=False
                                           )
pred = tf.sparse_tensor_to_dense(decodes[0])

# %% 模型恢复
save_path = './log/checkpoints/model.ckpt-11002'
saver = tf.train.Saver()

saver.restore(sess, save_path=save_path)

# %% inference

image_path = './data/sim_sub_15w/0_0_27HJF30FRY.jpg'

with sess.as_default():

    inputdata = cv2.imread(image_path, cv2.IMREAD_COLOR)
    #scale = 32.0 / inputdata.shape[0]
    #width = max(0 , scale * inputdata.shape[1])
    inputdata = cv2.resize(inputdata, (100, 32))

    out = sess.run(pred, feed_dict={img_input: inputdata})

    #preds = decoder.writer.sparse_tensor_to_str(preds[0])
    print(vocab._to_string(out[0]))
    plt.figure(1)
    plt.imshow(cv2.imread(image_path, cv2.IMREAD_COLOR)[:, :, (2, 1, 0)])
    plt.figure(2)
    plt.imshow(inputdata)
    plt.show()
