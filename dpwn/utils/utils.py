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

def maybe_download(config, filename):
    """Download the data from Yann's website, unless it's already here."""
    work_directory = config.get('data', 'work_directory')
    source_url = config.get('data', 'source_url')
    if not tf.gfile.Exists(work_directory):
        tf.gfile.MakeDirs(work_directory)
    filepath = os.path.join(work_directory, filename)
    if not tf.gfile.Exists(filepath):
        filepath, _ = urllib.request.urlretrieve(source_url + filename, filepath)
        with tf.gfile.GFile(filepath) as f:
            size = f.Size()
        print('Successfully downloaded', filename, size, 'bytes.')
    return filepath

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
