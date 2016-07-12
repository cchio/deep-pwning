from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import sys
import re
import time
import datetime
from ConfigParser import SafeConfigParser

import numpy as np
from six.moves import urllib
from six.moves import xrange
import tensorflow as tf
from tensorflow.contrib import learn

import utils.utils as utils
from models.semantic_cnn import SemanticCNN
from evaluator import Evaluator
from adversarial.wordvec_advgen import WordVec_AdvGen

tf.app.flags.DEFINE_string('config_path', './config/semantic.conf', 'Application configuration file.')
tf.app.flags.DEFINE_boolean('restore_checkpoint', False, 'Skip training, restore from checkpoint.')
tf.app.flags.DEFINE_boolean('test', False, 'Test run with a fraction of the data.')
cmd_args = tf.app.flags.FLAGS

# FIXME: Move to utils
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

# FIXME: Move to utils, just like we're doing with the mnist module
def load_data_and_labels():
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    positive_examples = list(open("./data/semantic/rt-polaritydata/rt-polarity.pos", "r").readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open("./data/semantic/rt-polaritydata/rt-polarity.neg", "r").readlines())
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
    config = SafeConfigParser()
    config.read(cmd_args.config_path)
    if cmd_args.restore_checkpoint:
        print('Skipping training phase, loading model checkpoint from: ', 
            config.get('main', 'checkpoint_path'))

    x_text, y = load_data_and_labels()

    # Build vocabulary
    # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/learn/python/learn/preprocessing/text.py
    max_document_length = max([len(x.split(" ")) for x in x_text])
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
    x = np.array(list(vocab_processor.fit_transform(x_text)))

    # Randomly shuffle data
    if config.get('main', 'seed') == 'None':
        seed = None
    else:
        seed = config.getint('main', 'seed')
    np.random.seed(seed)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]

    # Split train/test set
    x_train, x_eval, x_test = x_shuffled[:-2000], x_shuffled[-2000:-1000], x_shuffled[-1000:]
    y_train, y_eval, y_test = y_shuffled[:-2000], y_shuffled[-2000:-1000], y_shuffled[-1000:]
    print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
    print("Train/Dev/Test split: {:d}/{:d}/{:d}".format(len(y_train), len(y_eval), len(y_test)))

    semantic_cnn = SemanticCNN(config, x_train.shape[1],
                               len(vocab_processor.vocabulary_), 128, 128)

    x, y_ = semantic_cnn.train_input_placeholders()
    y_conv, logits, keep_prob, l2_loss, embedded_words, embed_W = semantic_cnn.model(x)

    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits, y_))

    # Add the regularization term to the loss.
    loss += 5e-4 * l2_loss

    learning_rate = tf.Variable(tf.constant(1e-3), dtype=tf.float32)
    global_step = tf.Variable(0, name="global_step", trainable=False)
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)

    pred_labels = tf.argmax(y_conv, 1, name="pred_labels")
    correct_predictions = tf.equal(pred_labels, tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
    cross_entropy = -tf.reduce_sum(tf.cast(y_, "float") * tf.log(y_conv))
    grad = tf.gradients(cross_entropy, embedded_words)

    input_dict = {
        "x": x,
        "y_": y_,
        "y_conv": y_conv,
        "keep_prob": keep_prob,
        "train_data": x_train,
        "train_labels": y_train,
        "test_data": x_test,
        "test_labels": y_test,
        "validation_data": x_eval,
        "validation_labels": y_eval,
        "num_epochs": config.getint('main', 'num_epochs'),
        "train_size": len(y_train),
        "embedded_words": embedded_words,
        "vocab_processor": vocab_processor,
        "embed_W": embed_W
    }

    saver = tf.train.Saver()
    evaluator = Evaluator(cmd_args, config, optimizer, 
        learning_rate, loss, saver, onehot_labels=True)
    evaluator.run(input_dict)

    wordvec_advgen = WordVec_AdvGen(cmd_args, saver, config)
    wordvec_advgen.run(input_dict)

if __name__ == '__main__':
    tf.app.run()
