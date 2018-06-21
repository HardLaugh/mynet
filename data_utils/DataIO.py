from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import numpy as np
import random
import tensorflow as tf

from tensorflow.python.ops import control_flow_ops  # pylint: disable=E0611


def _int64_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def float_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _int64_feature_list(values):
    return tf.train.FeatureList(feature=[_int64_feature(v) for v in values])


def _bytes_feature_list(values):
    return tf.train.FeatureList(feature=[_bytes_feature(v) for v in values])


def apply_with_random_selector(x, func, num_cases):
    """Computes func(x, sel), with sel sampled from [0...num_cases-1].

    Args:
        x: input Tensor.
        func: Python function to apply.
        num_cases: Python int32, number of cases to sample sel from.

    Returns:
        The result of func(x, sel), where func receives the value of the
        selector as a python integer, but sel is sampled dynamically.
    """
    sel = tf.random_uniform([], maxval=num_cases, dtype=tf.int32)
    # Pass the real x only to one of the func calls.
    return control_flow_ops.merge([
        func(control_flow_ops.switch(x, tf.equal(sel, case))[1], case)
        for case in range(num_cases)])[0]


class DataReader(object):

    def __init__(self, dataset_dir, file_pattern, modelconifg, batch_size=32, mode='train'):

        self.modelconfig = modelconifg

        self.dataset_dir = dataset_dir
        self.file_pattern = file_pattern
        self.image_format = 'jpeg'
        self.batch_size = batch_size
        self.mode = mode

    def is_training(self):
        return self.mode == 'train'

    def read(self):

        batch_size = self.batch_size

        def queue_iterator():

            Height = self.modelconfig.image_height
            Width = self.modelconfig.image_width

            data_files = []
            for pattern in self.file_pattern.split(","):
                data_files.extend(tf.gfile.Glob(self.dataset_dir + pattern))
            if not data_files:
                tf.logging.fatal(
                    "Found no input files matching %s", self.file_pattern)
            else:
                tf.logging.info("Prefetching values from %d files matching %s",
                                len(data_files), self.file_pattern)

            dataset = tf.data.TFRecordDataset(data_files)

            # 解析单个样本数据
            def _parse_single_example(example):
                """解析单个样本数据.

                Args:
                    param:example

                """

                features = {"image/data": tf.FixedLenFeature((), tf.string, default_value=''),
                            "image/format": tf.FixedLenFeature([1], tf.string, default_value='jpeg'),
                            "image/caption": tf.VarLenFeature(tf.int64), }
                parsed_example = tf.parse_single_example(example, features)

                #image_format = parsed_example["image/format"]

                encoded_image = parsed_example["image/data"]

                caption = parsed_example["image/caption"]

                #caption = tf.sparse_tensor_to_dense(caption)
                #caption = tf.to_int32(caption)

                return encoded_image, caption

            def _distort_image(image, color_ordering):

                random_uniform_width = random.randint(0, 30)
                random_uniform_height = random.randint(0, 30)
                random_tw = random.randint(0, random_uniform_width)
                random_th = random.randint(0, random_uniform_height)

                random_constant = random.uniform(0, 0.99)
                random_angle = random.uniform(-0.05, 0.05)

                image = tf.pad(image,
                               [[0, random_uniform_height], [
                                   0, random_uniform_width], [0, 0]],
                               mode='CONSTANT',
                               name=None,
                               constant_values=random_constant,
                               )

                with tf.name_scope("distort_color", values=[image]):
                    if color_ordering == 0:
                        image = tf.image.random_brightness(
                            image, max_delta=32. / 255.)
                        image = tf.image.random_saturation(
                            image, lower=0.5, upper=1.5)
                        image = tf.image.random_hue(image, max_delta=0.032)
                        image = tf.image.random_contrast(
                            image, lower=0.5, upper=1.5)
                        image = tf.clip_by_value(image, 0.0, 1.0)

                        angle = tf.contrib.image.angles_to_projective_transforms(  # pylint: disable=E1101
                            random_angle,
                            50, 500)
                        image = tf.contrib.image.transform(  # pylint: disable=E1101
                            image, angle, "BILINEAR")

                    elif color_ordering == 1:
                        # random_tw = tf.random_uniform([], 0, random_uniform_width, tf.int32)
                        # random_th = tf.random_uniform([], 0, random_uniform_height, tf.int32)

                        image = tf.image.random_brightness(
                            image, max_delta=32. / 255.)
                        image = tf.image.random_contrast(
                            image, lower=0.5, upper=1.5)
                        image = tf.image.random_saturation(
                            image, lower=0.5, upper=1.5)
                        image = tf.image.random_hue(image, max_delta=0.032)
                        image = tf.clip_by_value(image, 0.0, 1.0)

                        image = tf.contrib.image.translate(  # pylint: disable=E1101
                            image, [random_tw, random_th], "BILINEAR")

                    else:
                        raise ValueError('color_ordering must be in [0, 1]')

                # The random_* ops do not necessarily clamp.
                # image = tf.clip_by_value(image, 0.0, 1.0)

                return image

            def _process_image(encoded_image, caption):
                """图片预处理.

                Args:
                    param:encoded_image 图片信息
                    param:caption 标注信息

                """

                if self.image_format == 'jpeg':
                    image = tf.image.decode_jpeg(encoded_image, channels=3)
                elif self.image_format == 'png':
                    image = tf.image.decode_png(encoded_image, channels=3)
                else:
                    raise ValueError("Invalid image format: %s" %
                                     self.image_format)
                image = tf.image.convert_image_dtype(image, dtype=tf.float32)

                # image = tf.image.resize_images(image,
                #                                size=[Height, Width],
                #                                method=tf.image.ResizeMethod.BILINEAR)

                if self.is_training():
                    image = apply_with_random_selector(
                        image,
                        lambda x, ordering: _distort_image(x, ordering),
                        num_cases=2
                    )
                image = tf.image.resize_images(image,
                                               size=[Height, Width],
                                               method=tf.image.ResizeMethod.BILINEAR)
                # image = tf.image.convert_image_dtype(image, dtype=tf.uint8, saturate=True)

                # Rescale to [-1,1] instead of [0, 1]
                image = tf.subtract(image, 0.5)
                image = tf.multiply(image, 2.0)

                caption = tf.to_int32(caption)

                return image, caption

            dataset = dataset.map(_parse_single_example)
            dataset = dataset.map(_process_image)

            # 如果数据集N不是batch_size的倍数，最后的批次将会是N%batch_size
            dataset = dataset.shuffle(256).repeat().batch(batch_size)

            return dataset.make_one_shot_iterator().get_next()

        images, input_labels = queue_iterator()

        return images, input_labels

    def input_fn(self):
        """输入函数，适配estimator
        """
        return self.read()


class DataWriter(object):

    def __init__(self, vocab, dataset_dir, output_dir,
                 size, name='ocr', split=False):
        """转换数据的接口.

        Args:


        """

        self.dataset_dir = dataset_dir
        self.output_dir = output_dir
        self.size = size  # 图片名字属于标注的长度
        self.per_samples = 5000
        self.vocab = vocab

        self.name = name

        if split:
            self.split_size = (0.7, 0.2, 0.1)
        else:
            self.split_size = None

        self.image_metadata = None

        self.train = None
        self.val = None
        self.test = None

    def load_and_process_metadata(self):

        print("starting load metadata!")

        image_metadata = []
        for file in os.listdir(self.dataset_dir):
            caption = file.split('.')[0][-self.size:]
            filename = os.path.join(self.dataset_dir, file)
            image_format = file.split('.')[1]
            image_metadata.append([filename, caption, image_format])

        print("finish load metadata!")

        random.seed(407)
        random.shuffle(image_metadata)

        data_nums = len(image_metadata)

        if self.split_size is None:
            self.image_metadata = image_metadata
        else:

            train_nums = int(data_nums * self.split_size[0])

            val_nums = int(data_nums * self.split_size[1])

            split_data = np.split(
                image_metadata, [train_nums, train_nums+val_nums])
            self.train = split_data[0]
            self.val = split_data[1]
            self.test = split_data[2]
            print('finish split data: train--%d, val--%d, test--%d'
                  % (len(self.train), len(self.val), len(self.test)))

    def write(self, metadata, split_name):

        def _get_output_filename(output_dir, name, split_name, idx):
            return '%s/%s_%s_%03d.tfrecord' % (output_dir, name, split_name, idx)

        def _to_example(image_meta): 

            with tf.gfile.FastGFile(image_meta[0], "rb") as f:
                encoded_image = f.read()

            example = tf.train.Example(features=tf.train.Features(feature={
                "image/data": _bytes_feature(encoded_image),
                "image/format": _bytes_feature(image_meta[2].encode("utf8")),
                "image/caption": _int64_feature(self.vocab.string_to_id(image_meta[1]))}))

            return example

        i = 0
        fidx = 0

        data_nums = len(metadata)

        while i < data_nums:
            tf_filename = _get_output_filename(
                self.output_dir, self.name, split_name, fidx)
            with tf.python_io.TFRecordWriter(tf_filename) as writer:
                j = 0
                while i < data_nums and j < self.per_samples:
                    sys.stdout.write('\r>> Converting %s image %d/%d'
                                     % (split_name, i+1, data_nums))
                    sys.stdout.flush()

                    one_meta = metadata[i]
                    example = _to_example(one_meta)

                    writer.write(example.SerializeToString())

                    i += 1
                    j += 1
                fidx += 1
        print(">>>>>> Great! you have finished converting %s dataset!" %
              (split_name))

    def build_data(self):

        self.load_and_process_metadata()

        if self.image_metadata is not None:
            print("start converting>>>>>>>:")
            self.write(self.image_metadata, 'train')
            print('\nFinished converting the ocr dataset!')
        else:
            print("start converting>>>>>>>:")
            self.write(self.train, 'train')
            self.write(self.val, 'val')
            self.write(self.test, 'test')
            print('\nFinally, you have Finished spliting and converting the ocr dataset!')
