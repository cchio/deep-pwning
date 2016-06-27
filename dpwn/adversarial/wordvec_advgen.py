from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import numpy as np
import pandas as pd
import tensorflow as tf

import utils.utils as utils

class WordVec_AdvGen:

    def __init__(self, cmd_args, saver, config):
        self.cmd_args = cmd_args
        self.saver = saver
        self.config = config

    def run(self, input_dict):
        x = input_dict["x"]
        y_ = input_dict["y_"]
        y_conv = input_dict["y_conv"]
        keep_prob = input_dict["keep_prob"]
        test_data = input_dict["test_data"]
        test_labels = input_dict["test_labels"]
        embedded_words = input_dict["embedded_words"]
        vocab_processor = input_dict["vocab_processor"]
        embed_W = input_dict["embed_W"]

        checkpoint_path = self.config.get('main', 'checkpoint_path')
        eval_frequency = self.config.getint('main', 'eval_frequency')
        num_classes = self.config.getint('main', 'num_classes')

        num_differs = .0
        correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        cross_entropy = -tf.reduce_sum(tf.cast(y_, "float") * tf.log(y_conv))
        grad = tf.gradients(cross_entropy, embedded_words)

        sess = tf.Session()
        tf.initialize_all_variables().run(session=sess)
        self.saver.restore(sess, checkpoint_path)
        embed_W_val = sess.run(embed_W)
        df = pd.DataFrame()

        start_time = time.time()

        if self.cmd_args.test:
            iter_range = xrange(1)
        else:
            iter_range = xrange(len(test_data))

        for idx in iter_range:
            if idx % eval_frequency == 0:
                elapsed_time = time.time() - start_time
                print('Adversarial text generation step %d of %d, (%.1fms/step)' %
                    (idx, len(test_data),
                    1000 * elapsed_time / eval_frequency))

            x_sample = [test_data[idx]]
            y_sample = [test_labels[idx]]
            pred = sess.run(y_conv, feed_dict={x:x_sample, keep_prob:1.0})
            pred_label = np.argmax(pred)
            correct_label = np.argmax(y_sample)

            target_label = [np.eye(num_classes)[np.abs(correct_label-1)]]
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
            adv_embedded_words_val = sess.run(embedded_words, feed_dict={x:x_sample}) # + 0.1 * grad_sign
            for word_vec in adv_embedded_words_val[0]:
                word_vec_dist_matrix = np.absolute((embed_W_val - word_vec))
                closest_match_idx = np.sum(word_vec_dist_matrix, axis=1).argmax()
                adv_x_sample.append(closest_match_idx)
            majority_val = utils.find_majority(adv_x_sample)
            for index, item in enumerate(adv_x_sample):
                if item == majority_val:
                    adv_x_sample[index] = 0
            adv_x_sample_str = ''.join(vocab_processor.reverse([adv_x_sample])).replace('<UNK>', '').strip()
            adv_pred = sess.run(y_conv, feed_dict={x:[adv_x_sample], keep_prob:1.0})
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
        print("Number of differing adversarial predictions: (smaller is better) " + str(num_differs))
