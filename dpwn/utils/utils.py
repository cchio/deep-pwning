from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import errno
import gzip
import random
import tarfile

from six.moves import urllib
from six.moves import xrange

import numpy as np
import tensorflow as tf
import matplotlib
# Circumvent error when X11 forwarding is not available
if 'DISPLAY' not in os.environ: matplotlib.use('Pdf')
import matplotlib.pyplot as plt

def random_string(n):
    """Generates a random alphanumeric (lower case only) string of length n."""
    if n < 0:
        return ''
    return ''.join(random.choice('0123456789abcdefghijklmnopqrstuvwxyz') 
        for i in range(n))

def maybe_download(config, filename, extract=False):
    """Download the data from the source url, unless it's already here."""
    work_directory = config.get('data', 'work_directory')
    source_url = config.get('data', 'source_url')
    if not tf.gfile.Exists(work_directory):
        tf.gfile.MakeDirs(work_directory)
    filepath = os.path.join(work_directory, filename)
    if not tf.gfile.Exists(filepath):
        def _progress(count, block_size, total_size):
          sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
              float(count * block_size) / float(total_size) * 100.0))
          sys.stdout.flush()
        filepath, _ = urllib.request.urlretrieve(source_url + filename, 
            filepath, _progress)
        print()
        with tf.gfile.GFile(filepath) as f:
            size = f.Size()
        print('Successfully downloaded', filename, size, 'bytes.')
    if extract:
        tarfile.open(filepath, 'r:gz').extractall(work_directory)
    return filepath

def _generate_image_and_label_batch(image, label, min_queue_examples,
                                    batch_size, shuffle):
    """Construct a queued batch of images and labels.

    Args:
        image: 3-D Tensor of [height, width, 3] of type.float32.
        label: 1-D Tensor of type.int32
        min_queue_examples: int32, minimum number of samples to retain
          in the queue that provides of batches of examples.
        batch_size: Number of images per batch.
        shuffle: boolean indicating whether to use a shuffling queue.

    Returns:
        images: Images. 4D tensor of [batch_size, height, width, 3] size.
        labels: Labels. 1D tensor of [batch_size] size.
    """
    # Create a queue that shuffles the examples, and then
    # read 'batch_size' images + labels from the example queue.
    num_preprocess_threads = 16
    if shuffle:
        images, label_batch = tf.train.shuffle_batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size,
            min_after_dequeue=int(min_queue_examples))
    else:
        images, label_batch = tf.train.batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size)
    return images, tf.reshape(label_batch, [batch_size])

def read_cifar10(config, filename_queue):
    """Reads and parses examples from CIFAR10 data files.

    Recommendation: if you want N-way read parallelism, call this function
    N times.  This will give you N independent Readers reading different
    files & positions within those files, which will give better mixing of
    examples.

    Args:
        filename_queue: A queue of strings with the filenames to read from.

    Returns:
        An object representing a single example, with the following fields:
            height: number of rows in the result (32)
            width: number of columns in the result (32)
            depth: number of color channels in the result (3)
            key: a scalar string Tensor describing the filename & record number
                 for this example.
            label: an int32 Tensor with the label in the range 0..9.
            uint8image: a [height, width, depth] uint8 Tensor with the image data
    """

    class CIFAR10Record(object):
        pass
    result = CIFAR10Record()

    # Dimensions of the images in the CIFAR-10 dataset.
    # See http://www.cs.toronto.edu/~kriz/cifar.html for a description of the
    # input format.
    label_bytes = 1  # 2 for CIFAR-100
    result.height = int(config.get('main', 'image_size'))
    result.width = int(config.get('main', 'image_size'))
    result.depth = int(config.get('main', 'num_channels'))
    image_bytes = result.height * result.width * result.depth
    # Every record consists of a label followed by the image, with a
    # fixed number of bytes for each.
    record_bytes = label_bytes + image_bytes

    # Read a record, getting filenames from the filename_queue.  No
    # header or footer in the CIFAR-10 format, so we leave header_bytes
    # and footer_bytes at their default of 0.
    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    result.key, value = reader.read(filename_queue)

    # Convert from a string to a vector of uint8 that is record_bytes long.
    record_bytes = tf.decode_raw(value, tf.uint8)

    # The first bytes represent the label, which we convert from uint8->int32.
    result.label = tf.cast(
        tf.slice(record_bytes, [0], [label_bytes]), tf.int32)

    # The remaining bytes after the label represent the image, which we reshape
    # from [depth * height * width] to [depth, height, width].
    depth_major = tf.reshape(tf.slice(record_bytes, [label_bytes], [image_bytes]),
                             [result.depth, result.height, result.width])
    # Convert from [depth, height, width] to [height, width, depth].
    result.uint8image = tf.transpose(depth_major, [1, 2, 0])
    return result

def cifar10_inputs(config, distort=False, whiten=True, for_eval=False, shuffle=False):
    """Construct input for CIFAR evaluation using the Reader ops.

    Args:
        eval_data: bool, indicating if one should use the train or eval data set.

    Returns:
        images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
        labels: Labels. 1D tensor of [batch_size] size.

    Raises:
        ValueError: If no data_dir
    """

    data_dir = os.path.join(config.get('data', 'work_directory'), 'cifar-10-batches-bin')
    batch_size = int(config.get('main', 'batch_size'))

    if not for_eval:
        filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i) for i in xrange(1, 6)]
        num_examples_per_epoch = int(config.get('main', 'num_examples_per_epoch_train'))
    else:
        filenames = [os.path.join(data_dir, 'test_batch.bin')]
        num_examples_per_epoch = int(config.get('main', 'num_examples_per_epoch_eval'))

    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)

    # Create a queue that produces the filenames to read.
    filename_queue = tf.train.string_input_producer(filenames)

    # Read examples from files in the filename queue.
    read_input = read_cifar10(config, filename_queue)
    reshaped_image = tf.cast(read_input.uint8image, tf.float32)

    height = int(config.get('main', 'subsection_image_size'))
    width = int(config.get('main', 'subsection_image_size'))
    num_channels = int(config.get('main', 'num_channels'))

    # Image processing for evaluation.
    # Crop the central [height, width] of the image.
    resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image,
                                                           width, height)

    if distort:
        # Image processing for training the network. Note the many random
        # distortions applied to the image.

        # Randomly crop a [height, width] section of the image.
        distorted_image = tf.random_crop(reshaped_image, [height, width, num_channels])

        # Randomly flip the image horizontally.
        distorted_image = tf.image.random_flip_left_right(distorted_image)

        # Because these operations are not commutative, consider randomizing
        # the order their operation.
        distorted_image = tf.image.random_brightness(distorted_image,
                                                     max_delta=63)
        resulting_image = tf.image.random_contrast(distorted_image,
                                                   lower=0.2, 
                                                   upper=1.8)
    else:
        resulting_image = resized_image

    if whiten:
        float_image = tf.image.per_image_whitening(resulting_image)
    else:
        float_image = resulting_image

    # Ensure that the random shuffling has good mixing properties.
    min_queue_examples = num_examples_per_epoch * \
        float(config.get('main', 'min_fraction_of_examples_in_queue'))

    if not for_eval:
        print ('Filling queue with %d CIFAR images before starting to train. '
           'This will take a few minutes.' % min_queue_examples)

    # Generate a batch of images and labels by building up a queue of examples.
    return _generate_image_and_label_batch(float_image, read_input.label,
                                         min_queue_examples, 
                                         batch_size,
                                         shuffle=shuffle)

def extract_data(config, filename, num_images):
    """Extract the images into a 4D tensor [image index, y, x, channels].
       Values are rescaled from [0, 255] down to [-0.5, 0.5]."""
    print('Extracting', filename)
    image_size = config.getint('main', 'image_size')
    pixel_depth = config.getint('main', 'pixel_depth')
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(image_size * image_size * num_images)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = (data - (pixel_depth / 2.0)) / pixel_depth
        data = data.reshape(num_images, image_size, image_size, 1)
    return data

def extract_labels(filename, num_images):
    """Extract the labels into a vector of int64 label IDs."""
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * num_images)
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
    return labels

def error_rate(predictions, labels, onehot_labels=False):
    """Return the error rate based on dense predictions and sparse labels."""
    if onehot_labels:
        norm_labels = np.argmax(labels, axis=1)
    else:
        norm_labels = labels
    return 100.0 - (
        100.0 *
        np.sum(np.argmax(predictions, 1) == norm_labels) /
        predictions.shape[0])

def ensure_dir(d):
    if os.path.exists(d) and os.path.isdir(d):
        return
    else:
        os.makedirs(d)

# http://stackoverflow.com/questions/20038011/trying-to-find-majority-element-in-a-list
def find_majority(k):
    m = {}
    max_val = ('', 0) # (occurring element, occurrences)
    for n in k:
        if n in m: m[n] += 1
        else: m[n] = 1
        if m[n] > max_val[1]: max_val = (n,m[n])
    return max_val[0]

def display_image_sync(im):
    plt.imshow(im)
    plt.show()

def compare_mnist_digits(im1, im2, 
    im1_label, im2_label, idx, perturbation=.0, out_dir='', method=''):
    """ Plot 2 MNIST images side by side."""
    ensure_dir('{0}/fooled/{1}/'.format(out_dir, perturbation))
    ensure_dir('{0}/not-fooled/{1}/'.format(out_dir, perturbation))

    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)
    ax.matshow(im1, cmap='Greys')
    plt.title('im1 predicted label: ' + str(im1_label))
    plt.xticks(np.array([]))
    plt.yticks(np.array([]))
    ax = fig.add_subplot(1, 2, 2)
    ax.matshow(im2, cmap='Greys')
    plt.title('im2 predicted label: ' + str(im2_label))
    plt.xticks(np.array([]))
    plt.yticks(np.array([]))

    if not out_dir:
        plt.show()
    else:
        if im1_label != im2_label:
            out_path = '{0}/fooled/{1}/{2}-{3}.png'.format(
                out_dir, perturbation, method, idx)
        else:
            out_path = '{0}/not-fooled/{1}/{2}-{3}.png'.format(
                out_dir, perturbation, method, idx)
        fig.savefig(out_path)
        plt.close(fig)
