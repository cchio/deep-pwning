from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import sys
from datetime import datetime
import time
import pickle
import math
from ConfigParser import SafeConfigParser

import numpy as np
from six.moves import urllib
from six.moves import xrange
import tensorflow as tf

import matplotlib.pyplot as plt

import utils.utils as utils
from models.cifar10_cnn import Cifar10CNN
from evaluator import Evaluator
from adversarial.fastgradientsign_advgen import FastGradientSign_AdvGen

tf.app.flags.DEFINE_string('config_path', './config/cifar10.conf', 'Application configuration file.')
tf.app.flags.DEFINE_boolean('restore_checkpoint', False, 'Skip training, restore from checkpoint.')
tf.app.flags.DEFINE_boolean('test', False, 'Test run with a fraction of the data.')
cmd_args = tf.app.flags.FLAGS

def calculate_loss(logits, labels):
    # Calculate the average cross entropy loss across the batch.
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits, labels, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)

    # The total loss is defined as the cross entropy loss plus all of the weight
    # decay terms (L2 loss).
    return tf.add_n(tf.get_collection('losses'), name='total_loss')

def train(config, total_loss, global_step):
    """Train CIFAR-10 model.

        Create an optimizer and apply to all trainable variables. Add moving
        average for all trainable variables.

    Args:
        total_loss: Total loss from loss().
        global_step: Integer Variable counting the number of training steps
            processed.
    Returns:
        train_op: op for training.
    """
    # Variables that affect learning rate.
    num_batches_per_epoch = \
        int(config.get('main', 'num_examples_per_epoch_train')) / \
        int(config.get('main', 'batch_size'))
    decay_steps = int(num_batches_per_epoch * 
        float(config.get('main', 'num_epochs_per_decay')))

    # Decay the learning rate exponentially based on the number of steps.
    lr = tf.train.exponential_decay(float(config.get('main', 'initial_learning_rate')),
                                    global_step,
                                    decay_steps,
                                    float(config.get('main', 'learning_rate_decay_factor')),
                                    staircase=True)

    # Compute gradients.
    opt = tf.train.GradientDescentOptimizer(lr)
    grads = opt.compute_gradients(total_loss)

    # Apply gradients.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(
        float(config.get('main', 'moving_average_decay')), 
        global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        train_op = tf.no_op(name='train')

    return train_op

def evaluate(config, cifar10_cnn, input_dict):
    graph = input_dict["graph"]
    images = input_dict["x"]
    labels = input_dict["y_"]
    logits = input_dict["y_conv"]

    print("Starting evaluation of trained CIFAR-10 model...")

    with graph.as_default():
        # Calculate predictions.
        top_k_op = tf.nn.in_top_k(logits, labels, 1)

        # Restore the moving average version of the learned variables for eval.
        variable_averages = tf.train.ExponentialMovingAverage(
            float(config.get('main', 'moving_average_decay')))
        variables_to_restore = variable_averages.variables_to_restore()
        del variables_to_restore['Variable']
        saver = tf.train.Saver(variables_to_restore)

        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(config.get('main', 'checkpoint_dir'))
            if ckpt and ckpt.model_checkpoint_path:
                # Restores from checkpoint
                saver.restore(sess, ckpt.model_checkpoint_path)
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            else:
                print('No checkpoint file found')
                return

            # Start the queue runners.
            coord = tf.train.Coordinator()
            try:
                threads = []
                for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                    threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                                     start=True))

                num_iter = int(math.ceil( \
                    int(config.get('main', 'num_examples_per_epoch_eval')) / \
                    int(config.get('main', 'batch_size'))))
                true_count = 0  # Counts the number of correct predictions.
                total_sample_count = num_iter * int(config.get('main', 'batch_size'))
                step = 0
                while step < num_iter and not coord.should_stop():
                    predictions = sess.run([top_k_op])
                    true_count += np.sum(predictions)
                    step += 1

                # Compute precision @ 1.
                precision = true_count / total_sample_count
                print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))

            except Exception as e:
                coord.request_stop(e)
            coord.request_stop()
            coord.join(threads, stop_grace_period_secs=10)

def main(argv=None):
    config = SafeConfigParser()
    config.read(cmd_args.config_path)

    # Get the data.
    utils.maybe_download(config, 
                         config.get('data', 'data_filename'), 
                         extract=True)
        
    with tf.Graph().as_default() as g:
        global_step = tf.Variable(0, trainable=False)

        # Get images and labels for CIFAR-10.
        images_train, labels_train = utils.cifar10_inputs(config, distort=True, shuffle=True)

        # Build a Graph that computes the logits predictions from the
        # inference model.
        cifar10_cnn = Cifar10CNN(config)
        logits_train = cifar10_cnn.model(images_train)

        # Calculate loss.
        loss = calculate_loss(logits_train, labels_train)

        # Build a Graph that trains the model with one batch of examples and
        # updates the model parameters.
        train_op = train(config, loss, global_step)

        # Create a saver.
        saver = tf.train.Saver(tf.all_variables())

        # Build an initialization operation to run below.
        init = tf.initialize_all_variables()

        if not cmd_args.restore_checkpoint:
            # Start running operations on the Graph.
            sess = tf.Session(config=tf.ConfigProto())
            sess.run(init)

            # Start the queue runners.
            tf.train.start_queue_runners(sess=sess)

            num_batches_to_run = int(config.get('main', 'num_batches_to_run'))

            for step in xrange(num_batches_to_run):
                start_time = time.time()
                _, loss_value = sess.run([train_op, loss])
                duration = time.time() - start_time

                assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

                if step % 10 == 0:
                    num_examples_per_step = int(config.get('main', 'batch_size'))
                    examples_per_sec = num_examples_per_step / duration
                    sec_per_batch = float(duration)

                    format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                                  'sec/batch)')
                    print (format_str % (datetime.now(), step, loss_value,
                                         examples_per_sec, sec_per_batch))

                # Save the model checkpoint periodically.
                if step % 1000 == 0 or (step + 1) == num_batches_to_run:
                    checkpoint_path = config.get('main', 'checkpoint_path')
                    saver.save(sess, checkpoint_path, global_step=step)
        else:
            print('Skipping training phase, loading model checkpoint from:', 
                config.get('main', 'checkpoint_dir'))
    

        x = tf.placeholder(tf.float32,
                           shape=[None,
                                  config.getint('main', 'subsection_image_size'),
                                  config.getint('main', 'subsection_image_size'),
                                  config.getint('main', 'num_channels')],
                           name="x")
        images_raw, _ = utils.cifar10_inputs(config, whiten=False, for_eval=True)
        images_eval, labels_eval = utils.cifar10_inputs(config, for_eval=True)
        logits_eval = cifar10_cnn.model(images_eval, eval=True)
        logits_single = cifar10_cnn.model(images_eval, eval=True, image_placeholder=x)

    input_dict = {
        "graph": g,
        "x_raw": images_raw,
        "x": images_eval,
        "y_": labels_eval,
        "y_conv": logits_eval,
        "y_conv_single": logits_single,
        "adv_image_placeholder": x,
        "keep_prob": None,
    }

    evaluate(config, cifar10_cnn, input_dict)

    print("Starting generation of adversarial examples for CIFAR-10...")
    fastgradientsign_advgen = FastGradientSign_AdvGen(cmd_args, [1, 24, 24, 3], saver, config)
    fastgradientsign_advgen.run_queue(input_dict)

if __name__ == '__main__':
    tf.app.run()
