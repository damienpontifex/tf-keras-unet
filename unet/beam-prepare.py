#!/usr/bin/env python

"""Prepare inria image dataset using apache beam"""

import tensorflow as tf
import apache_beam as beam

from argparse import ArgumentParser
import os
from random import shuffle
import numpy as np

tf.enable_eager_execution()

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def rgb2gray(rgb) -> np.ndarray:
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])

def np_from_img_path(path: str, gray=False):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img)
    if gray:
        img = tf.image.rgb_to_grayscale(img)
    return img.numpy()

def split_samples(element):
    """Take the image and label path then read and split into segments for smaller
        training image sizes
    """
    frame_path, label_path = element
    np_frame_image = np_from_img_path(frame_path)
    np_label_image = np_from_img_path(label_path, gray=True)
    # Make buildings 1.0 and everything else 0.0
    np_label_image = np.round(np_label_image / 255.).astype(np.float32)
    
    width, height, _ = np_frame_image.shape
    
    # Split the 5000x5000 image into 25 subgrids
    for x in range(5):
        for y in range(5):
            
            sub_width = width // 5
            sub_height = height // 5
            
            x_lower, x_upper = x * sub_width, (x + 1) * sub_width
            y_lower, y_upper = y * sub_height, (y + 1) * sub_height
            
            np_frame_sub_image = np_frame_image[x_lower:x_upper, y_lower:y_upper]
            np_label_sub_image = np_label_image[x_lower:x_upper, y_lower:y_upper]
            
            assert np_frame_sub_image.shape == (1000, 1000, 3)
            yield np_frame_sub_image, np_label_sub_image, sub_width, sub_height


def to_tf_example(element):
    """Take elements from splitting samples and transform into tf Example objects"""

    np_frame_sub_image, np_label_sub_image, sub_width, sub_height = element

    image_raw = np_frame_sub_image.tobytes()
    label_raw = np_label_sub_image.tobytes()

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': _int64_feature(sub_height),
        'image/width': _int64_feature(sub_width),
        'image/label': _bytes_feature(label_raw),
        'image/bytes': _bytes_feature(image_raw)
    }))
    yield example


def convert_to_tfrecords(data_set, name, data_directory, num_shards=1):
    """Convert the dataset to a sharded set of TFRecord files"""

    output_dir = os.path.join(data_directory, name)

    with beam.Pipeline(options=PipelineOptions()) as p:
        data_collection = p | beam.Create(data_set)
        _ = (
            data_collection 
            | 'SplitImages' >> beam.FlatMap(split_samples)
            | 'MakeTfExample' >> beam.Map(to_tf_example)
            | 'SaveToTfRecords' >> beam.io.WriteToTfRecord(
                output_dir, file_name_suffix='.tfrecord', num_shards=num_shards,
                shard_name_template='-SS', coder=beam.coders.ProtoCoder(tf.train.Example))
        )

def prepare(data_dir: str, output_dir: str):

    labels_path = os.path.join(data_dir, 'AerialImageDataset/train/gt')
    frames_path = os.path.join(data_dir, 'AerialImageDataset/train/images')

    frame_files = tf.gfile.Glob(os.path.join(frames_path, '*.tif'))
    label_files = [os.path.join(labels_path, os.path.basename(fn)) for fn in frame_files]
    files = list(zip(frame_files, label_files))
    shuffle(files)

    # 5 validation files when split into subgrid gives 125 validation patches
    val_files = files[:5] 
    train_files = files[5:]

    convert_to_tfrecords(train_files, 'train', output_dir, num_shards=10)
    convert_to_tfrecords(val_files, 'validation', output_dir)

if __name__ == '__main__':
    parser = ArgumentParser(description='Prepare data for unet model')
    parser.add_argument('--data-dir', help='Directory where image dataset is')
    parser.add_argument('--output-dir', help='Directory where TFRecord files are to be written')
    args = parser.parse_args()

    prepare(**vars(args))

