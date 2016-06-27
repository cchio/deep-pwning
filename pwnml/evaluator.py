from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time

import numpy as np
import tensorflow as tf

import utils.utils as utils
# FIXME: DECIDE WHICH CONFIG TO USE BASED ON ARGUMENT? FLAG? 
# OR JUST MAKE CONFIG AN OBJECT PASSED INTO TO EVALUATOR INIT 
# import config.mnist_config as config
import config.semantic_config as config

class Evaluator:

    def __init__(self, 
        cmd_args, optimizer, learning_rate, loss, saver, onehot_labels=False):
        self.cmd_args = cmd_args
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.loss = loss
        self.onehot_labels = onehot_labels
        self.saver = saver

    def eval_in_batches(self, y_conv, x, keep_prob, data, sess):
        """Get all predictions for a dataset by running it in small batches."""
        size = data.shape[0]
        if size < config.batch_size:
            raise ValueError("batch size for evals larger than dataset: %d" % size)
        predictions = np.ndarray(shape=(size, config.num_classes), dtype=np.float32)
        for begin in xrange(0, size, config.batch_size):
            end = begin + config.batch_size
            if end <= size:
                predictions[begin:end, :] = sess.run(
                    y_conv,
                    feed_dict={x: data[begin:end, ...], keep_prob: 1.0})
            else:
                batch_predictions = sess.run(
                    y_conv,
                    feed_dict={x: data[-config.batch_size:, ...], keep_prob: 1.0})
                predictions[begin:, :] = batch_predictions[begin - size:, :]
        return predictions

    def run(self, input_dict):
        x = input_dict["x"]
        y_ = input_dict["y_"]
        y_conv = input_dict["y_conv"]
        keep_prob = input_dict["keep_prob"]
        train_data = input_dict["train_data"]
        train_labels = input_dict["train_labels"]
        test_data = input_dict["test_data"]
        test_labels = input_dict["test_labels"]
        validation_data = input_dict["validation_data"]
        validation_labels = input_dict["validation_labels"]
        num_epochs = input_dict["num_epochs"]
        train_size = input_dict["train_size"]

        utils.ensure_dir(os.path.dirname(config.checkpoint_path))

        start_time = time.time()

        with tf.Session() as sess:
            tf.initialize_all_variables().run()
            print('Initialized!')

            if not self.cmd_args.checkpoint:
                print('No checkpoint to load, training model from scratch...')

                for step in xrange(int(num_epochs * train_size) // config.batch_size):
                    offset = (step * config.batch_size) % (train_size - config.batch_size)
                    batch_data = train_data[offset:(offset + config.batch_size), ...]
                    batch_labels = train_labels[offset:(offset + config.batch_size)]

                    feed_dict = {
                        x: batch_data, 
                        y_: batch_labels,
                        keep_prob: 0.5
                    }

                    _, l, lr, predictions = sess.run(
                        [self.optimizer, self.loss, self.learning_rate, y_conv], feed_dict=feed_dict)

                    if step % config.eval_frequency == 0:
                        path = self.saver.save(sess, config.checkpoint_path)
                        print("Saved model checkpoint to {}\n".format(path))
                        elapsed_time = time.time() - start_time
                        start_time = time.time()
                        print('Step %d (epoch %.2f), %.1f ms' %
                            (step, float(step) * config.batch_size / train_size,
                            1000 * elapsed_time / config.eval_frequency))
                        print('Minibatch loss: %.3f, learning rate: %.6f' % (l, lr))
                        print('Minibatch error: %.1f%%' % utils.error_rate(predictions, batch_labels, self.onehot_labels))
                        print('Validation error: %.1f%%' % utils.error_rate(
                            self.eval_in_batches(y_conv, x, keep_prob, validation_data, sess), validation_labels, self.onehot_labels))
                        sys.stdout.flush()
        
                # Finally print the result!
                test_error = utils.error_rate(self.eval_in_batches(y_conv, x, keep_prob, test_data, sess), test_labels, self.onehot_labels)
                print('Test error: %.1f%%' % test_error)
