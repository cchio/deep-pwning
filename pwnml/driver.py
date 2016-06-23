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

import numpy as np
from six.moves import urllib
from six.moves import xrange
import tensorflow as tf
import matplotlib.pyplot as plt

import utils
from lenet5 import LeNet5

# FIXME: Move to config class
SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'
WORK_DIRECTORY = 'data'
IMAGE_SIZE = 28
NUM_CHANNELS = 1
PIXEL_DEPTH = 255
NUM_LABELS = 10
VALIDATION_SIZE = 5000  # Size of the validation set.
SEED = 66478  # Set to None for random seed.
BATCH_SIZE = 64
NUM_EPOCHS = 10
EVAL_BATCH_SIZE = 64
EVAL_FREQUENCY = 100  # Number of steps between evaluations.
CHECKPOINT_PATH = '/Users/cchio/Desktop/lenet/fucking.ckpt'
DATA_TYPE = tf.float32

tf.app.flags.DEFINE_boolean("self_test", False, "True if running a self test.")
tf.app.flags.DEFINE_string('checkpoint', '', 'Tensorflow session checkpoint file path.')
FLAGS = tf.app.flags.FLAGS


def main(argv=None):

  # FIXME: PUT INTO 'LOADER' MODULE
  if FLAGS.checkpoint:
    print('Loading model checkpoint from: ', FLAGS.checkpoint)
  if FLAGS.self_test:
    print('Running self-test.')
    train_data, train_labels = fake_data(256)
    validation_data, validation_labels = fake_data(EVAL_BATCH_SIZE)
    test_data, test_labels = fake_data(EVAL_BATCH_SIZE)
    num_epochs = 1
  else:
    # Get the data.
    train_data_filename = utils.maybe_download('train-images-idx3-ubyte.gz')
    train_labels_filename = utils.maybe_download('train-labels-idx1-ubyte.gz')
    test_data_filename = utils.maybe_download('t10k-images-idx3-ubyte.gz')
    test_labels_filename = utils.maybe_download('t10k-labels-idx1-ubyte.gz')

    # Extract it into np arrays.
    train_data = utils.extract_data(train_data_filename, 60000)
    train_labels = utils.extract_labels(train_labels_filename, 60000)
    test_data = utils.extract_data(test_data_filename, 10000)
    test_labels = utils.extract_labels(test_labels_filename, 10000)

    # Generate a validation set.
    validation_data = train_data[:VALIDATION_SIZE, ...]
    validation_labels = train_labels[:VALIDATION_SIZE]
    train_data = train_data[VALIDATION_SIZE:, ...]
    train_labels = train_labels[VALIDATION_SIZE:]
    num_epochs = NUM_EPOCHS
  train_size = train_labels.shape[0]


  lenet5 = LeNet5()

  x, y_ = lenet5.train_input_placeholders()
  y_conv, logits, keep_prob, param_dict = lenet5.model(x)

  eval_data = tf.placeholder(
      DATA_TYPE,
      shape=(EVAL_BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))

  loss = tf.reduce_mean(
    tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits, y_))

  # L2 regularization for the fully connected parameters.
  regularizers = (tf.nn.l2_loss(param_dict['fc1_W']) 
                + tf.nn.l2_loss(param_dict['fc1_b']) 
                + tf.nn.l2_loss(param_dict['fc2_W']) 
                + tf.nn.l2_loss(param_dict['fc2_b']))
  # Add the regularization term to the loss.
  loss += 5e-4 * regularizers

  # Optimizer: set up a variable that's incremented once 
  # per batch and controls the learning rate decay.
  batch = tf.Variable(0, dtype=DATA_TYPE)

  # Decay once per epoch, using an exponential schedule starting at 0.01.
  learning_rate = tf.train.exponential_decay(
      0.01,                # Base learning rate.
      batch * BATCH_SIZE,  # Current index into the dataset.
      train_size,          # Decay step.
      0.95,                # Decay rate.
      staircase=True)

  # Use simple momentum for the optimization.
  optimizer = tf.train.MomentumOptimizer(learning_rate,
                                         0.9).minimize(loss,
                                                       global_step=batch)

  # FIXME: MOVE EVAL TO SEPARATE MODULE/CLASS
  def eval_in_batches(data, sess):
    """Get all predictions for a dataset by running it in small batches."""
    size = data.shape[0]
    if size < EVAL_BATCH_SIZE:
      raise ValueError("batch size for evals larger than dataset: %d" % size)
    predictions = np.ndarray(shape=(size, NUM_LABELS), dtype=np.float32)
    for begin in xrange(0, size, EVAL_BATCH_SIZE):
      end = begin + EVAL_BATCH_SIZE
      if end <= size:
        predictions[begin:end, :] = sess.run(
            y_conv,
            feed_dict={x: data[begin:end, ...], keep_prob: 1.0})
      else:
        batch_predictions = sess.run(
            y_conv,
            feed_dict={x: data[-EVAL_BATCH_SIZE:, ...], keep_prob: 1.0})
        predictions[begin:, :] = batch_predictions[begin - size:, :]
    return predictions

  # Create a local session to run the training.
  start_time = time.time()
  saver = tf.train.Saver()

  with tf.Session() as sess:
    tf.initialize_all_variables().run()
    print('Initialized!')

    if not FLAGS.checkpoint:
      print('No checkpoint to load, training model from scratch...')

      for step in xrange(int(num_epochs * train_size) // BATCH_SIZE):
        offset = (step * BATCH_SIZE) % (train_size - BATCH_SIZE)
        batch_data = train_data[offset:(offset + BATCH_SIZE), ...]
        batch_labels = train_labels[offset:(offset + BATCH_SIZE)]
        feed_dict = {
          x: batch_data, 
          y_: batch_labels,
          keep_prob: 0.5
        }

        _, l, lr, predictions = sess.run(
            [optimizer, loss, learning_rate, y_conv],
            feed_dict=feed_dict)
        if step % EVAL_FREQUENCY == 0:
          saver.save(sess, CHECKPOINT_PATH)
          elapsed_time = time.time() - start_time
          start_time = time.time()
          print('Step %d (epoch %.2f), %.1f ms' %
                (step, float(step) * BATCH_SIZE / train_size,
                 1000 * elapsed_time / EVAL_FREQUENCY))
          print('Minibatch loss: %.3f, learning rate: %.6f' % (l, lr))
          print('Minibatch error: %.1f%%' % utils.error_rate(predictions, batch_labels))
          print('Validation error: %.1f%%' % utils.error_rate(
              eval_in_batches(validation_data, sess), validation_labels))
          sys.stdout.flush()
    
    # Finally print the result!
    test_error = utils.error_rate(eval_in_batches(test_data, sess), test_labels)
    print('Test error: %.1f%%' % test_error)
    if FLAGS.self_test:
      print('test_error', test_error)
      assert test_error == 0.0, 'expected 0.0 test_error, got %.2f' % (
          test_error,)

  # FIXME: MOVE ADVERSARIAL GENERATION TO SEPARATE INPUT

  # # Generating adversarial input - Fast Gradient Descent Method
  # x = tf.placeholder("float", shape=[1, 28, 28, 1])
  # y_ = tf.placeholder("float", shape=[10])
  # y_conv = model(x)

  # not_fooled = .0
  # fooled = .0

  # correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
  # accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
  # cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
  # grad = tf.gradients(cross_entropy, x)

  # sess = tf.Session()
  # saver.restore(sess, CHECKPOINT_PATH)
  # for idx in xrange(len(test_data)):
  #   if idx % 100 == 0:
  #     print(idx)
  #   image = test_data[idx]
  #   label = test_labels[idx]
  #   y_onehot = np.eye(10)[label]

  #   pred = sess.run(y_conv, feed_dict={x:np.reshape(image, [1, 28, 28, 1])})
  #   label = np.argmax(pred)
  #   grad_val = sess.run(grad, feed_dict={x:np.reshape(image, [1, 28, 28, 1]), y_:y_onehot})
  #   grad_sign = np.sign(grad_val[0])
  #   grad_norm = sum([np.abs(W) for W in grad_val[0]])
  #   grad_sign = np.sign(grad_val[0])
  #   grad_norm = sum([np.abs(W) for W in grad_val[0]])
  #   adv_image = .1 * grad_sign + image
  #   adv_pred = sess.run(y_conv, feed_dict={x:adv_image})
  #   adv_label = np.argmax(adv_pred)

  #   if (adv_label != label):
  #     fooled = fooled + 1
  #     # Plotting original and adversarial image side-by-side
  #     compare_mnist_digits(
  #       np.reshape(image, [28,28]), np.reshape(adv_image, [28,28]),
  #       label, adv_label)
  #   else:
  #     not_fooled = not_fooled + 1

  # # Generating adversarial input - Jacobian Method
  # x = tf.placeholder("float", shape=[1, 28, 28, 1])
  # y_ = tf.placeholder("float", shape=[10])
  # y_conv = model(x)

  # not_fooled = .0
  # fooled = .0

  # correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
  # accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
  # cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
  # grad = tf.gradients(cross_entropy, x)

  # sess = tf.Session()
  # saver.restore(sess, CHECKPOINT_PATH)
  # for idx in xrange(len(test_data)):
  #   if idx % 100 == 0:
  #     print(idx)
  #   image = test_data[idx]
  #   label = test_labels[idx]
  #   y_onehot = np.eye(10)[label]

  #   pred = sess.run(y_conv, feed_dict={x:np.reshape(image, [1, 28, 28, 1])})
  #   label = np.argmax(pred)
  #   grad_val = sess.run(grad, feed_dict={x:np.reshape(image, [1, 28, 28, 1]), y_:y_onehot})
  #   grad_sign = np.sign(grad_val[0])
  #   grad_norm = sum([np.abs(W) for W in grad_val[0]])
  #   grad_sign = np.sign(grad_val[0])
  #   grad_norm = sum([np.abs(W) for W in grad_val[0]])
  #   adv_image = .1 * grad_sign + image
  #   adv_pred = sess.run(y_conv, feed_dict={x:adv_image})
  #   adv_label = np.argmax(adv_pred)

  #   if (adv_label != label):
  #     fooled = fooled + 1
  #     # Plotting original and adversarial image side-by-side
  #     compare_mnist_digits(
  #       np.reshape(image, [28,28]), np.reshape(adv_image, [28,28]),
  #       label, adv_label)
  #   else:
  #     not_fooled = not_fooled + 1

  # print("Adversarial sample yield: ", fooled/(fooled+not_fooled))
  # print("Adversarial samples fooled: ", fooled)
  # print("Adversarial samples not fooled: ", not_fooled)

if __name__ == '__main__':
  tf.app.run()

