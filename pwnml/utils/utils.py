from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import errno
import gzip

from six.moves import urllib
from six.moves import xrange

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# FIXME: DECIDE WHICH CONFIG TO USE BASED ON ARGUMENT? FLAG? 
# OR JUST MAKE CONFIG AN OBJECT PASSED INTO TO EVALUATOR INIT 
# import config.mnist_config as config
import config.semantic_config as config

def maybe_download(filename):
    """Download the data from Yann's website, unless it's already here."""
    if not tf.gfile.Exists(config.work_directory):
        tf.gfile.MakeDirs(config.work_directory)
    filepath = os.path.join(config.work_directory, filename)
    if not tf.gfile.Exists(filepath):
        filepath, _ = urllib.request.urlretrieve(config.source_url + filename, filepath)
        with tf.gfile.GFile(filepath) as f:
            size = f.Size()
        print('Successfully downloaded', filename, size, 'bytes.')
    return filepath

def extract_data(filename, num_images):
    """Extract the images into a 4D tensor [image index, y, x, channels].
       Values are rescaled from [0, 255] down to [-0.5, 0.5]."""
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(config.image_size * config.image_size * num_images)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = (data - (config.pixel_depth / 2.0)) / config.pixel_depth
        data = data.reshape(num_images, config.image_size, config.image_size, 1)
    return data

def extract_labels(filename, num_images):
    """Extract the labels into a vector of int64 label IDs."""
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * num_images)
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
    return labels

def fake_data(num_images):
    """Generate a fake dataset that matches the dimensions of MNIST."""
    data = np.ndarray(
        shape=(num_images, config.image_size, config.image_size, config.num_channels),
        dtype=np.float32)
    labels = np.zeros(shape=(num_images,), dtype=np.int64)
    for image in xrange(num_images):
        label = image % 2
        data[image, :, :, 0] = label - 0.5
        labels[image] = label
    return data, labels

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
            out_path = '{0}/fooled/{1}/{2}-{3}.png'.format(out_dir, perturbation, method, idx)
        else:
            out_path = '{0}/not-fooled/{1}/{2}-{3}.png'.format(out_dir, perturbation, method, idx)
        fig.savefig(out_path)
        plt.close(fig)
