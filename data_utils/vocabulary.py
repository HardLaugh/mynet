from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import sys
import numpy as np


class Vocabulary(object):
    """字典构建
    """

    def __init__(self):
        self.base_num = 2
        self.vocab_base = {
            0: '0',
            # 1:'a',
            1: 'A',
        }

        self.vocab = {}
        self.reverse_vocab = {}

        print("Initializing vocabulary:....")
        id = 0
        for i in range(self.base_num):
            vb = self.vocab_base[i]
            if vb == '0':
                num = 10
            else:
                num = 26
            for i in range(num):
                self.vocab[chr(ord(vb)+i)] = id
                self.reverse_vocab[id] = chr(ord(vb)+i)
                id += 1

        print("finish build vocabulary:....")

    def letter_to_id(self, l):

        return self.vocab[l]

    def id_to_letter(self, id):
        return self.reverse_vocab[id]

    def string_to_id(self, s):
        id = []
        for v in s:
            id.append(self.letter_to_id(v))
        return id

    def _to_string(self, v):
        strs = ''
        for s in v:
            strs += self.id_to_letter(s)
        return strs


def compute_acuracy(preds, gt_labels):
    """ 计算字符匹配准确度
    """
    accuracy = []
    length = preds.shape[0]

    for index, gt_label in enumerate(gt_labels):
        pred = preds[index]
        totol_count = len(gt_label)
        correct_count = 0
        try:
            for i, tmp in enumerate(gt_label):
                if tmp == pred[i]:
                    correct_count += 1
        except IndexError:
            continue
        finally:
            try:
                accuracy.append(correct_count / totol_count)
            except ZeroDivisionError:
                if len(pred) == 0:
                    accuracy.append(1)
                else:
                    accuracy.append(0)
    accuracy = np.mean(np.array(accuracy).astype(np.float32), axis=0)

    return accuracy
