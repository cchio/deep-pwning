from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import numpy as np
import pandas as pd
import tensorflow as tf

import utils.utils as utils

class FastGradientSign_AdvGen:

    def __init__(self, cmd_args, input_x_shape, saver, config):
        self.cmd_args = cmd_args
        self.input_x_shape = input_x_shape
        self.saver = saver
        self.config = config
        self.config = config

    def run(self, input_dict):
        x = input_dict["x"]
        y_ = input_dict["y_"]
        y_conv = input_dict["y_conv"]
        keep_prob = input_dict["keep_prob"]
        test_data = input_dict["test_data"]
        test_labels = input_dict["test_labels"]

        checkpoint_path = self.config.get('main', 'checkpoint_path')
        eval_frequency = self.config.getint('main', 'eval_frequency')
        num_classes = self.config.getint('main', 'num_classes')
        image_output_path = self.config.get('main', 'image_output_path')
        adversarial_perturbation_min = self.config.getfloat(
            'main', 'adversarial_perturbation_min')
        adversarial_perturbation_max = self.config.getfloat(
            'main', 'adversarial_perturbation_max')
        adversarial_perturbation_steps = self.config.getfloat(
            'main', 'adversarial_perturbation_steps')

        not_fooled = .0
        fooled = .0
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        cross_entropy = -tf.reduce_sum(tf.cast(y_, "float") * tf.log(y_conv))
        grad = tf.gradients(cross_entropy, x)

        sess = tf.Session()
        tf.initialize_all_variables().run(session=sess)
        self.saver.restore(sess, checkpoint_path)
        df = pd.DataFrame()

        start_time = time.time()

        if self.cmd_args.test:
            iter_range = xrange(1)
            adversarial_perturbation_max = adversarial_perturbation_min
            adversarial_perturbation_steps = 1
        else:
            iter_range = xrange(len(test_data))

        for idx in iter_range:
            if idx % eval_frequency == 0:
                elapsed_time = time.time() - start_time
                print('Adversarial image generation step %d of %d, (%.1fms/step)' %
                    (idx, len(test_data),
                    1000 * elapsed_time / eval_frequency))

            image = test_data[idx]
            label = test_labels[idx]
            y_onehot = np.eye(num_classes)[label]

            pred = sess.run(y_conv, feed_dict={x: (np.reshape(image, self.input_x_shape)), keep_prob: 1.0})
            pred_label = np.argmax(pred)
            grad_val = sess.run(grad, feed_dict={x:np.reshape(image, self.input_x_shape), y_:y_onehot, keep_prob: 1.0})
            grad_sign = np.sign(grad_val[0])
            grad_norm = sum([np.abs(W) for W in grad_val[0]])

            for perturbation in np.linspace(adversarial_perturbation_min, 
                                            adversarial_perturbation_max, 
                                            adversarial_perturbation_steps):
                adv_image = perturbation * grad_sign + image
                adv_pred = sess.run(y_conv, feed_dict={x:adv_image, keep_prob: 1.0})
                adv_label = np.argmax(adv_pred)

                if (adv_label != label): fooled = fooled + 1
                else: not_fooled = not_fooled + 1

                series = pd.Series([idx, label, pred_label, adv_label, grad_norm, pred, adv_pred, image, adv_image, 
                            perturbation, grad_val],
                            index = ["Idx", "True Label", "Predicted Label", "Predicted Label Adversarial", \
                                    "Gradient Norm", "Predicted Prob", "Predicted Prob Adversarial", "Image", \
                                    "Adversarial Image", "Gradient Step", "Gradient"])
                df = df.append(series, ignore_index=True)

        print("Adversarial sample yield: ", fooled/(fooled+not_fooled))
        print("Adversarial samples fooled: ", fooled)
        print("Adversarial samples not fooled: ", not_fooled)
        return df

    def run_queue(self, input_dict):
        graph = input_dict["graph"]
        images = input_dict["x"]
        raw_images = input_dict["x_raw"]
        labels = input_dict["y_"]
        logits = input_dict["y_conv"]
        logits_single = input_dict["y_conv_single"]
        x = input_dict["adv_image_placeholder"]

        adversarial_perturbation_min = self.config.getfloat(
            'main', 'adversarial_perturbation_min')
        adversarial_perturbation_max = self.config.getfloat(
            'main', 'adversarial_perturbation_max')
        adversarial_perturbation_steps = self.config.getfloat(
            'main', 'adversarial_perturbation_steps')

        with graph.as_default():
            # Restore the moving average version of the learned variables for eval.
            variable_averages = tf.train.ExponentialMovingAverage(
                float(self.config.get('main', 'moving_average_decay')))
            variables_to_restore = variable_averages.variables_to_restore()
            del variables_to_restore['Variable']
            saver = tf.train.Saver(variables_to_restore)

            y_ = tf.one_hot(indices=tf.cast(labels, "int64"), 
                depth=int(self.config.get('main', 'num_classes')), 
                on_value=1.0, 
                off_value=0.0)

            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, y_)
            grad = tf.gradients(cross_entropy, images)

            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(self.config.get('main', 'checkpoint_dir'))
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

                    sample_count = int(self.config.get('main', 'num_examples_per_epoch_eval'))
                    true_count = 0  # Counts the number of correct predictions.
                    step = 0
                    pred_correct = 0
                    adv_correct = 0
                    adv_diff = 0
                    while step < sample_count and not coord.should_stop():
                        raw_images_val, images_val, labels_val, cross_entropy_val, grad_val = sess.run([raw_images, images, labels, cross_entropy, grad[0]])
                        step += 1
                        for i in range(len(images_val)):
                            image = raw_images_val[i]
                            true_label = labels_val[i]

                            grad_sign = np.sign(grad_val[i])
                            grad_norm = sum([np.abs(W) for W in grad_val[i]])

                            one_adv_correct = False
                            one_pred_correct = False
                            one_adv_diff = False
                            for perturbation in np.linspace(adversarial_perturbation_min, 
                                                            adversarial_perturbation_max, 
                                                            adversarial_perturbation_steps):
                                adv_image = perturbation * grad_sign + image
                                adv_image_reshaped = np.reshape(adv_image, np.insert(adv_image.shape, 0 , 1))
                                raw_image_reshaped = np.reshape(image, np.insert(image.shape, 0 , 1))

                                pred_logit = sess.run(logits_single, feed_dict={x:raw_image_reshaped})
                                pred_label = np.argmax(pred_logit)

                                adv_pred = sess.run(logits_single, feed_dict={x:adv_image_reshaped})
                                adv_label = np.argmax(adv_pred)

                                if pred_label == true_label:
                                    one_pred_correct = True
                                if adv_label == true_label:
                                    one_adv_correct = True
                                if adv_label != pred_label:
                                    one_adv_diff = True
                            if one_pred_correct:
                                pred_correct = pred_correct + 1
                            if one_adv_correct:
                                adv_correct = adv_correct + 1
                            if one_adv_diff:
                                adv_diff = adv_diff + 1

                    print("PRED CORRECT: " + str(pred_correct))
                    print("ADV CORRECT: " + str(adv_correct))
                    print("ADV DIFF: " + str(adv_diff))

                except Exception as e:
                    coord.request_stop(e)
                coord.request_stop()
                coord.join(threads, stop_grace_period_secs=10)
