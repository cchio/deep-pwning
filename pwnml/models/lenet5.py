import tensorflow as tf

import utils.utils as utils
import config.mnist_config as config

class LeNet5:

    def conv2d(self, data, weight):
        return tf.nn.conv2d(data,
                            weight,
                            strides=[1, 1, 1, 1],
                            padding='SAME')

    def max_pool(self, data):
        return tf.nn.max_pool(data,
                              ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1],
                              padding='SAME')

    def variable(self, flavor, shape):
        if flavor == 'W':
            return tf.Variable(
                tf.truncated_normal(shape,
                                    stddev=0.1,
                                    seed=config.seed, 
                                    dtype=config.data_type))      
        elif flavor == 'b':
            return tf.Variable(tf.constant(0.1, shape=shape), 
                               dtype=config.data_type)
        else:
            return None

    def train_input_placeholders(self):
        x = tf.placeholder(config.data_type,
                           shape=[None, config.image_size, config.image_size, config.num_channels],
                           name="x")
        y_ = tf.placeholder(tf.int64, shape=[None,], name="y_")
        return x, y_

    def model(self, data):
        conv1_W = self.variable('W', [5, 5, config.num_channels, 32])
        conv1_b = self.variable('b', [32])
        conv1 = tf.nn.relu(tf.nn.bias_add(self.conv2d(data, conv1_W), conv1_b))

        pool1 = self.max_pool(conv1)

        conv2_W = self.variable('W', [5, 5, 32, 64])
        conv2_b = self.variable('b', [64])
        conv2 = tf.nn.relu(tf.nn.bias_add(self.conv2d(pool1, conv2_W), conv2_b))

        pool2 = self.max_pool(conv2)

        pool2_shape = pool2.get_shape().as_list()
        pool2_reshaped = tf.reshape(pool2,
            [-1, pool2_shape[1] * pool2_shape[2] * pool2_shape[3]])

        fc1_W = self.variable('W', [config.image_size // 4 * config.image_size // 4 * 64, 512])
        fc1_b = self.variable('b', [512])
        fc1 = tf.nn.relu(tf.matmul(pool2_reshaped, fc1_W) + fc1_b)

        keep_prob = tf.placeholder("float", name="keep_prob")
        fc1_dropout = tf.nn.dropout(fc1, keep_prob, seed=config.seed)

        fc2_W = self.variable('W', [512, config.num_classes])
        fc2_b = self.variable('b', [config.num_classes])

        logits = tf.matmul(fc1_dropout, fc2_W) + fc2_b
        y_conv = tf.nn.softmax(logits)
        param_dict = {
            'conv1_W': conv1_W,
            'conv1_b': conv1_b,
            'conv2_W': conv2_W,
            'conv2_b': conv2_b,
            'fc1_W': fc1_W,
            'fc1_b': fc1_b,
            'fc2_W': fc2_W,
            'fc2_b': fc2_b
        }
        return y_conv, logits, keep_prob, param_dict
