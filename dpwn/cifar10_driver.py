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
# from models.lenet5 import LeNet5
from evaluator import Evaluator
from adversarial.fastgradientsign_advgen import FastGradientSign_AdvGen

tf.app.flags.DEFINE_string('config_path', './config/cifar10.conf', 'Application configuration file.')
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
    utils.maybe_download(config, 
                         config.get('data', 'data_filename'), 
                         extract=True)

    with tf.Graph().as_default():
        global_step = tf.Variable(0, trainable=False)

        # Get images and labels for CIFAR-10.
        images, labels = utils.cifar10_distorted_inputs(config)

        # # Build a Graph that computes the logits predictions from the
        # # inference model.
        # logits = cifar10.inference(images)

        # # Calculate loss.
        # loss = cifar10.loss(logits, labels)

        # # Build a Graph that trains the model with one batch of examples and
        # # updates the model parameters.
        # train_op = cifar10.train(loss, global_step)

        # # Create a saver.
        # saver = tf.train.Saver(tf.all_variables())

        # Build an initialization operation to run below.
        init = tf.initialize_all_variables()

        # Start running operations on the Graph.
        sess = tf.Session(config=tf.ConfigProto())
        sess.run(init)

        # Start the queue runners.
        tf.train.start_queue_runners(sess=sess)

        # for step in xrange(FLAGS.max_steps):
        #     start_time = time.time()
        #     _, loss_value = sess.run([train_op, loss])
        #     duration = time.time() - start_time

        #     assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

        #     if step % 10 == 0:
        #         num_examples_per_step = FLAGS.batch_size
        #         examples_per_sec = num_examples_per_step / duration
        #         sec_per_batch = float(duration)

        #         format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
        #                       'sec/batch)')
        #         print (format_str % (datetime.now(), step, loss_value,
        #                              examples_per_sec, sec_per_batch))

        #     # Save the model checkpoint periodically.
        #     if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
        #         checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
        #         saver.save(sess, checkpoint_path, global_step=step)


if __name__ == '__main__':
    tf.app.run()
