import os
import tensorflow as tf

def _map_batch(record_batch):
    img_batch = tf.decode_raw(record_batch['image/bytes'], tf.uint8)
    img_batch = tf.reshape(img_batch, (-1, 1000, 1000, 3))
    img_batch = tf.image.resize_images(img_batch, (512, 512))

    img_batch = tf.image.convert_image_dtype(img_batch, dtype=tf.float32)

    label_batch = tf.decode_raw(record_batch['image/label'], tf.float32)
    #         label_batch = tf.expand_dims(label_batch, axis=-1)
    label_batch = tf.reshape(label_batch, (-1, 1000, 1000))
    label_batch = tf.expand_dims(label_batch, axis=-1)

    label_batch = tf.image.resize_images(label_batch, (512, 512))
    # Convert the labels where 255 is building to 1 is building and zero is not-building for sigmoid
#         label_batch = tf.round(label_batch / 255.)

    # Need to make sure same transformation is done to image and label...maybe use seed?
    # Or something like https://stackoverflow.com/a/38403715/1602729
#         if training:
#             # Do some random image augmentation - need to make sure it's random but applied to both image and label
#             image_raw = tf.image.random_flip_left_right(image_raw)
#             label_raw = tf.image.random_flip_left_right(label_raw)

#         img_shape = tf.pack([parsed_record['height'], parsed_record['width'], parsed_record['depth']])

    return { 'image': img_batch }, label_batch

def make_dataset(file_pattern, num_epochs=None, batch_size=32, shuffle=True):

    features = {
        'image/height': tf.FixedLenFeature([], tf.int64),
        'image/width': tf.FixedLenFeature([], tf.int64),
        'image/label': tf.FixedLenFeature([], tf.string),
        'image/bytes': tf.FixedLenFeature([], tf.string)
    }

    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern, batch_size, features,
        num_epochs=num_epochs, shuffle=shuffle,
        shuffle_buffer_size=4*batch_size, sloppy_ordering=True,
        reader_num_threads=os.cpu_count(), parser_num_threads=os.cpu_count(),
        prefetch_buffer_size=4)

    dataset = dataset.map(_map_batch)

    return dataset
