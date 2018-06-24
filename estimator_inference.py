# %%
import sys
import os
import math
import random

import numpy as np
import tensorflow as tf
import cv2

slim = tf.contrib.slim  # pylint: disable=E1101

%matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from model import configuration, crnn, CRNNestimator  # pylint: disable=protected-access
from model.CRNNestimator import CRNNestimator  # pylint: disable=protected-access
from data_utils.DataIO import DataReader
from data_utils.vocabulary import Vocabulary, compute_acuracy


# %% inference

def preprocess(imagefile):

    inputdata = cv2.imread(imagefile, cv2.IMREAD_COLOR)
    # inputdata = inputdata[:,0:,:]
    scale = 64.0 / inputdata.shape[0]
    width = int(max(430, scale * inputdata.shape[1]))
    if inputdata.shape[1] > 300:
        width = inputdata.shape[1]
    else:
        width = 300
    width = 800
    print(width)
    # inputdata = inputdata[10:44, :, :]
    inputdata = cv2.resize(inputdata, (width, 64))
    inputdata = inputdata[:, :, :]

    return inputdata


def input_fn(inputdata):
    """输入函数
    """
    image = tf.convert_to_tensor(inputdata, name='input')
    # img_input = tf.placeholder(tf.uint8, shape=(batch_size, 32, None, 3))
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.subtract(image, 0.5)
    image = tf.multiply(image, 2.0)

    return image, None


# %% model build
vocab = Vocabulary()
model_config = configuration.ModelConfig()
checkpointsPath = './log/checkpoints/'

estimator = CRNNestimator(
    model_dir=checkpointsPath,
    model_config=model_config,
    train_config=None,
    log_every_n_steps=None,
)

#%%
# path = './data/sim_sub_15w/'
path = './data/allSubUni/'
imagefiles = os.listdir(path)
for i in range(len(imagefiles)):
    imagefiles[i] = os.path.join(path, imagefiles[i])


# %%
# 随机抽样batch_size数量的样本
batch_size = 2
sample = random.sample([imagefiles[v]
                        for v in range(len(imagefiles))], batch_size)
inputdata = np.array(
    [preprocess(file) for file in sample],
)


stophook = tf.train.StopAtStepHook(last_step=1)
predictions = estimator.predict(
    input_fn=lambda: input_fn(inputdata),
    hooks=[stophook],
)

for pred_dict, imagefile in zip(predictions, sample):
    gt = imagefile.split('/')[-1].split('.')[0]
    predict = vocab._to_string(pred_dict['preds'])
    print(pred_dict['soft'])
    print("gt: %s" %(gt))
    print("predcit: %s" %(predict))
    if gt == predict:
        print("true")
    else:
        print("false")
    plt.figure(1)
    ori = cv2.imread(imagefile, cv2.IMREAD_COLOR)
    plt.imshow(ori)
    plt.figure(2)
    # ori = ori[10:44, :, :]
    plt.imshow(cv2.resize(ori, (800, 64)))
    plt.show()
