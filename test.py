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

from data_utils.vocabulary import Vocabulary
from model import configuration, crnn
tf.logging.set_verbosity(tf.logging.INFO)


gpu_options = tf.GPUOptions(allow_growth=True)
config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
sess = tf.InteractiveSession(config=config)


#模型建立
vocab = Vocabulary()

with tf.name_scope(None, 'input_image'):
    img_input = tf.placeholder(tf.uint8, shape=(32, 300, 3))
    image = tf.to_float(img_input)
    image = tf.expand_dims(image, 0)

model = crnn.CRNN(256, 37, 'inference')
logit = model.build(image)

# print(logit.get_shape().as_list())
# print(tf.shape(logit)[0])
# sys.exit()

decodes, _ = tf.nn.ctc_beam_search_decoder(inputs=logit,
                                           sequence_length=tf.shape(
                                               logit)[0]*np.ones(1),
                                           merge_repeated=False
                                           )
pred = tf.sparse_tensor_to_dense(decodes[0])


# 模型恢复
# model_variables = slim.get_model_variables('CRNN/LSTM')
save_path = '../log/checkpoints/model.ckpt-18500'
saver = tf.train.Saver()

saver.restore(sess, save_path=save_path)

# %% inference
# path = './data/sim_sub_15w/'
path = '../data/allSubUni/'
imagefile = os.listdir(path)[23 ]
imagefile = os.path.join(path, imagefile)

with sess.as_default():

    inputdata = cv2.imread(imagefile, cv2.IMREAD_COLOR)
    scale = 32.0 / inputdata.shape[0]
    width = int(max(0, scale * inputdata.shape[1]))
    print(width)
    inputdata = cv2.resize(inputdata, (300, 32))

    out = sess.run(pred, feed_dict={img_input: inputdata})

    #preds = decoder.writer.sparse_tensor_to_str(preds[0])
    print(vocab._to_string(out[0]))
    plt.figure(1)
    plt.imshow(cv2.imread(imagefile, cv2.IMREAD_COLOR)[:, :, (2, 1, 0)])
    plt.figure(2)
    plt.imshow(inputdata)
    plt.show()
