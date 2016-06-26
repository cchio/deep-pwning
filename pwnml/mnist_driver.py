"""Simple, end-to-end, LeNet-5-like convolutional MNIST model example.

This should achieve a test error of 0.7%. Please keep this model as simple and
linear as possible, it is meant as a tutorial for simple convolutional models.
Run with --self_test on the command line to execute a short self-test.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import sys
import time
import pickle

import numpy as np
from six.moves import urllib
from six.moves import xrange
import tensorflow as tf
import matplotlib.pyplot as plt

import utils.utils as utils
from models.lenet5 import LeNet5
from evaluator import Evaluator
import config.mnist_config as config
from adversarial.fast_gradient_sign import FGS_AdversarialGenerator

tf.app.flags.DEFINE_boolean("self_test", False, "True if running a self test.")
tf.app.flags.DEFINE_string('checkpoint', '', 'Tensorflow session checkpoint file path.')
FLAGS = tf.app.flags.FLAGS

def main(argv=None):

    if FLAGS.checkpoint:
        print('Loading model checkpoint from: ', FLAGS.checkpoint)
    if FLAGS.self_test:
        print('Running self-test.')
        train_data, train_labels = fake_data(256)
        validation_data, validation_labels = fake_data(config.batch_size)
        test_data, test_labels = fake_data(config.batch_size)
        num_epochs = 1
    else:
        # Get the data.
        train_data_filename = utils.maybe_download(config.train_data_filename)
        train_labels_filename = utils.maybe_download(config.train_labels_filename)
        test_data_filename = utils.maybe_download(config.test_data_filename)
        test_labels_filename = utils.maybe_download(config.test_labels_filename)

        # Extract it into np arrays.
        train_data = utils.extract_data(train_data_filename, 60000)
        train_labels = utils.extract_labels(train_labels_filename, 60000)
        test_data = utils.extract_data(test_data_filename, 10000)
        test_labels = utils.extract_labels(test_labels_filename, 10000)

        # Generate a validation set.
        validation_data = train_data[:config.validation_size, ...]
        validation_labels = train_labels[:config.validation_size]
        train_data = train_data[config.validation_size:, ...]
        train_labels = train_labels[config.validation_size:]
        num_epochs = config.num_epochs
    train_size = train_labels.shape[0]

    lenet5 = LeNet5()

    x, y_ = lenet5.train_input_placeholders()
    y_conv, logits, keep_prob, param_dict = lenet5.model(x)

    loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(logits, y_))

    # L2 regularization for the fully connected parameters.
    regularizers = (tf.nn.l2_loss(param_dict['fc1_W']) 
                  + tf.nn.l2_loss(param_dict['fc1_b']) 
                  + tf.nn.l2_loss(param_dict['fc2_W']) 
                  + tf.nn.l2_loss(param_dict['fc2_b']))
    # Add the regularization term to the loss.
    loss += 5e-4 * regularizers

    # Optimizer: set up a variable that's incremented once 
    # per batch and controls the learning rate decay.
    batch = tf.Variable(0, dtype=config.data_type)

    # Decay once per epoch, using an exponential schedule starting at 0.01.
    learning_rate = tf.train.exponential_decay(
        0.01,                # Base learning rate.
        batch * config.batch_size,  # Current index into the dataset.
        train_size,          # Decay step.
        0.95,                # Decay rate.
        staircase=True)

    optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9) \
        .minimize(loss, global_step=batch)

    input_dict = {
        "x": x,
        "y_": y_,
        "y_conv": y_conv,
        "keep_prob": keep_prob,
        "train_data": train_data,
        "train_labels": train_labels,
        "test_data": test_data,
        "test_labels": test_labels,
        "validation_data": validation_data,
        "validation_labels": validation_labels,
        "num_epochs": num_epochs,
        "train_size": train_size
    }

    saver = tf.train.Saver(tf.all_variables())

    evaluator = Evaluator(FLAGS, optimizer, learning_rate, loss, saver)
    evaluator.run(input_dict)

    fgs_adversarial_generator = FGS_AdversarialGenerator([1, 28, 28, 1], saver)
    adversarial_output_df = fgs_adversarial_generator.run(input_dict)
    # CHECK IF IMAGE OUTPUT PATH DEFINED THEN OUTPUT IMAGE, IF PICKLE FILE PATH DEFINED THEN SAVE PICKLE?
    utils.ensure_dir(os.path.dirname(config.pickle_filepath))
    with open(config.pickle_filepath, "wb") as pkl:
        pickle.dump(adversarial_output_df, pkl)

if __name__ == '__main__':
    tf.app.run()
