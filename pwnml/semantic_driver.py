from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import sys
import re
import time
import datetime
import pickle

import numpy as np
from six.moves import urllib
from six.moves import xrange
import tensorflow as tf
from tensorflow.contrib import learn
import matplotlib.pyplot as plt

import utils.utils as utils
from models.semantic_cnn import SemanticCNN
from evaluator import Evaluator
import config.semantic_config as config
from adversarial.fast_gradient_sign import FGS_AdversarialGenerator

tf.app.flags.DEFINE_boolean("self_test", False, "True if running a self test.")
tf.app.flags.DEFINE_string('checkpoint', '', 'Tensorflow session checkpoint file path.')
FLAGS = tf.app.flags.FLAGS

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def load_data_and_labels():
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    positive_examples = list(open("./data/rt-polaritydata/rt-polarity.pos", "r").readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open("./data/rt-polaritydata/rt-polarity.neg", "r").readlines())
    negative_examples = [s.strip() for s in negative_examples]
    # Split by words
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]
    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]

def main(argv=None):

    x_text, y = load_data_and_labels()

    # Build vocabulary
    # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/learn/python/learn/preprocessing/text.py
    max_document_length = max([len(x.split(" ")) for x in x_text])
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
    x = np.array(list(vocab_processor.fit_transform(x_text)))

    # Randomly shuffle data
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]

    # Split train/test set
    x_train, x_eval, x_test = x_shuffled[:-2000], x_shuffled[-2000:-1000], x_shuffled[-1000:]
    y_train, y_eval, y_test = y_shuffled[:-1000], y_shuffled[-2000:-1000], y_shuffled[-1000:]
    print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
    print("Train/Dev/Test split: {:d}/{:d}/{:d}".format(len(y_train), len(y_eval), len(y_test)))

    semantic_cnn = SemanticCNN(x_train.shape[1],
                               len(vocab_processor.vocabulary_),
                               128,
                               128)

    x, y_ = semantic_cnn.train_input_placeholders()
    y_conv, logits, keep_prob, l2_loss, embedded_words, embed_W = semantic_cnn.model(x)

    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits, y_))

    # Add the regularization term to the loss.
    loss += 5e-4 * l2_loss

    # Optimizer: set up a variable that's incremented once 
    # per batch and controls the learning rate decay.
    global_step = tf.Variable(0, name="global_step", trainable=False)

    learning_rate = tf.Variable(tf.constant(1e-3), dtype=config.data_type)
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)

    pred_labels = tf.argmax(y_conv, 1, name="pred_labels")
    correct_predictions = tf.equal(pred_labels, tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
    cross_entropy = -tf.reduce_sum(tf.cast(y_, "float") * tf.log(y_conv))
    grad = tf.gradients(cross_entropy, embedded_words)

    # http://stackoverflow.com/questions/20038011/trying-to-find-majority-element-in-a-list
    def find_majority(k):
        myMap = {}
        maximum = ('', 0) # (occurring element, occurrences)
        for n in k:
            if n in myMap: myMap[n] += 1
            else: myMap[n] = 1
            if myMap[n] > maximum[1]: maximum = (n,myMap[n])
        return maximum[0]

    def train_step(x_batch, y_batch):
        feed_dict = {
          x: x_batch,
          y_: y_batch,
          keep_prob: 0.5
        }
        _ = sess.run(optimizer, feed_dict)
        step = sess.run(global_step, feed_dict)
        cur_loss = sess.run(loss, feed_dict)
        cur_accuracy = sess.run(accuracy, feed_dict)
        time_str = datetime.datetime.now().isoformat()
        print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, cur_loss, cur_accuracy))

    def eval_step(x_batch, y_batch, writer=None, generate_adversarial=False):
        feed_dict = {
          x: x_batch,
          y_: y_batch,
          keep_prob: 1.0
        }
        step, cur_loss, cur_accuracy = sess.run([global_step, loss, accuracy], feed_dict)
        time_str = datetime.datetime.now().isoformat()
        print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, cur_loss, cur_accuracy))

        embed_W_val = embed_W.eval()

        print("")
        if generate_adversarial:
            num_differs = 0
            for idx in xrange(len(x_batch)):
                x_sample = [x_batch[idx]]
                y_sample = [y_batch[idx]]
                pred = y_conv.eval(feed_dict={x:x_sample, keep_prob:1.0})
                pred_label = np.argmax(pred)
                correct_label = np.argmax(y_sample)

                target_label = [np.eye(config.num_classes)[np.abs(correct_label-1)]]
                cross_entropy_val = sess.run(cross_entropy, feed_dict={x:x_sample, y_:target_label, keep_prob: 1.0})

                # Mess around with gradient
                grad_val = sess.run(grad, feed_dict={x:x_sample, y_:target_label, keep_prob: 1.0})
                grad_sign = np.sign(grad_val[0])
                x_sample_str = ''.join(vocab_processor.reverse(x_sample)).replace('<UNK>', '').strip()

                # Note:
                # longest string is 56 words, pick 56 vectors (size 128 each) from embed_W.shape
                #       embed_W.shape is (18758, 128)
                #       embedded_words is (1, 56, 128)
                # to form embedded_words representation of the string - now we got to reverse lookup..
                # note that embed_W is trained (changes) during the training phase
                # maybe the cnn is too deep for the amt of data... overfitting is definitely happening...

                adv_x_sample = []
                adv_embedded_words_val = embedded_words.eval(feed_dict={x:x_sample}) # + 0.1 * grad_sign
                for word_vec in adv_embedded_words_val[0]:
                    word_vec_dist_matrix = np.absolute((embed_W_val - word_vec))
                    closest_match_idx = np.sum(word_vec_dist_matrix, axis=1).argmax()
                    adv_x_sample.append(closest_match_idx)
                majority_val = find_majority(adv_x_sample)
                for index, item in enumerate(adv_x_sample):
                    if item == majority_val:
                        adv_x_sample[index] = 0
                adv_x_sample_str = ''.join(vocab_processor.reverse([adv_x_sample])).replace('<UNK>', '').strip()
                adv_pred = y_conv.eval(feed_dict={x:[adv_x_sample], keep_prob:1.0})
                adv_pred_label = np.argmax(pred)

                print("Original test string:    \"" + x_sample_str + "\"")
                print("Adversarial test string: \"" + adv_x_sample_str + "\"")
                print("Correct Label: {0}, Predicted Label: {1}, Adversarial Label: {2}"
                    .format(correct_label, pred_label, adv_pred_label))
                if correct_label == pred_label:
                    print("Correct Prediction.\n")
                else:
                    print("\n")
                if pred_label != adv_pred_label:
                    num_differs = num_differs + 1
                    print("Adversarial prediction differs.\n")
            print("Number of differing adversarial predictions: " + str(num_differs))

    def batch_iter(data, batch_size, num_epochs, shuffle=True):
        """
        Generates a batch iterator for a dataset.
        """
        data = np.array(data)
        data_size = len(data)
        num_batches_per_epoch = int(len(data)/batch_size) + 1
        for epoch in range(num_epochs):
            # Shuffle the data at each epoch
            if shuffle:
                shuffle_indices = np.random.permutation(np.arange(data_size))
                shuffled_data = data[shuffle_indices]
            else:
                shuffled_data = data
            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_size)
                yield shuffled_data[start_index:end_index]

    saver = tf.train.Saver(tf.all_variables())

    # Generate batches
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        if FLAGS.checkpoint:
            tf.initialize_all_variables().run(session=sess)
            saver.restore(sess, FLAGS.checkpoint)
        else:
            batches = batch_iter(
                list(zip(x_train, y_train)), config.batch_size, config.num_epochs)
            # Training loop. For each batch...
            for batch in batches:
                x_batch, y_batch = zip(*batch)
                train_step(x_batch, y_batch)
                current_step = tf.train.global_step(sess, global_step)
                if current_step % config.eval_frequency == 0:
                    print("\nEvaluation:")
                    eval_step(x_eval, y_eval, writer=None)
                    print("")
                    # FIXME: check if checkpoint path exists, create if doesn't
                    path = saver.save(sess, config.checkpoint_path)
                    print("Saved model checkpoint to {}\n".format(path))


        # Time to test
        print("\nTest Results:")
        eval_step(x_test, y_test, writer=None, generate_adversarial=True)
        print("")

    # input_dict = {
    #     "x": x,
    #     "y_": y_,
    #     "y_conv": y_conv,
    #     "keep_prob": keep_prob,
    #     "train_data": x_train,
    #     "train_labels": y_train,
    #     # "test_data": test_data,
    #     # "test_labels": test_labels,
    #     "validation_data": x_eval,
    #     "validation_labels": y_eval,
    #     "num_epochs": config.num_epochs,
    #     "train_size": len(y_train)
    # }

    # saver = tf.train.Saver()
    # evaluator = Evaluator(FLAGS, optimizer, learning_rate, loss, saver)
    # evaluator.run(input_dict)

    # fgs_adversarial_generator = FGS_AdversarialGenerator([1, 28, 28, 1], saver)
    # adversarial_output_df = fgs_adversarial_generator.run(input_dict)
    # # CHECK IF IMAGE OUTPUT PATH DEFINED THEN OUTPUT IMAGE, IF PICKLE FILE PATH DEFINED THEN SAVE PICKLE?
    # utils.ensure_dir(os.path.dirname(config.pickle_filepath))
    # with open(config.pickle_filepath, "wb") as pkl:
    #     pickle.dump(adversarial_output_df, pkl)

if __name__ == '__main__':
    tf.app.run()
