from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import sys
import time
import pickle
from ConfigParser import SafeConfigParser

import numpy as np
from six.moves import urllib
from six.moves import xrange
import tensorflow as tf

import utils.utils as utils
from models.lenet5 import LeNet5
from evaluator import Evaluator
from adversarial.fastgradientsign_advgen import FastGradientSign_AdvGen

tf.app.flags.DEFINE_string('config_path', './config/mnist.conf', 'Application configuration file.')
tf.app.flags.DEFINE_boolean('restore_checkpoint', False, 'Skip training, restore from checkpoint.')
tf.app.flags.DEFINE_boolean('test', False, 'Test run with a fraction of the data.')
cmd_args = tf.app.flags.FLAGS

def main(argv=None):
    config = SafeConfigParser()
    config.read(cmd_args.config_path)
    if cmd_args.restore_checkpoint:
        print('Skipping training phase, loading model checkpoint from: ', 
            config.get('main', 'checkpoint_path'))

    # Get the data.
    train_data_filename = utils.maybe_download(config, 
        config.get('data', 'train_data_filename'))
    train_labels_filename = utils.maybe_download(config, 
        config.get('data', 'train_labels_filename'))
    test_data_filename = utils.maybe_download(config, 
        config.get('data', 'test_data_filename'))
    test_labels_filename = utils.maybe_download(config, 
        config.get('data', 'test_labels_filename'))

    # Extract it into np arrays.
    train_data = utils.extract_data(config, train_data_filename, 60000)
    train_labels = utils.extract_labels(train_labels_filename, 60000)
    test_data = utils.extract_data(config, test_data_filename, 10000)
    test_labels = utils.extract_labels(test_labels_filename, 10000)

    validation_size = config.getint('main', 'validation_size')
    num_epochs = config.getint('main', 'num_epochs')

    # Generate a validation set.
    validation_data = train_data[:validation_size, ...]
    validation_labels = train_labels[:validation_size]
    train_data = train_data[validation_size:, ...]
    train_labels = train_labels[validation_size:]
    num_epochs = num_epochs
    train_size = train_labels.shape[0]

    lenet5 = LeNet5(config)

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
    batch = tf.Variable(0, dtype=tf.float32)

    # Decay once per epoch, using an exponential schedule starting at 0.01.
    learning_rate = tf.train.exponential_decay(
        0.01,
        batch * config.getint('main', 'batch_size'),
        train_size,
        0.95,
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

    evaluator = Evaluator(cmd_args, config, optimizer, 
        learning_rate, loss, saver)
    evaluator.run(input_dict)

    fastgradientsign_advgen = FastGradientSign_AdvGen(cmd_args, [1, 28, 28, 1], saver, config)
    adv_out_df = fastgradientsign_advgen.run(input_dict)

    pkl_path = config.get('main', 'pickle_filepath')
    utils.ensure_dir(os.path.dirname(pkl_path))
    with open(pkl_path, "wb") as pkl:
        pickle.dump(adv_out_df, pkl)

if __name__ == '__main__':
    tf.app.run()
