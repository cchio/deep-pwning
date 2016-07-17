import tensorflow as tf

import utils.utils as utils

class Cifar10CNN:

    def __init__(self, config):
        self.config = config

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

    def variable_on_cpu(self, name, shape, initializer):
        """Helper to create a Variable stored on CPU memory.

        Args:
            name: name of the variable
            shape: list of ints
            initializer: initializer for Variable

        Returns:
            Variable Tensor
        """
        with tf.device('/cpu:0'):
            var = tf.get_variable(name, shape, initializer=initializer, dtype=tf.float32)
        return var

    def variable_with_weight_decay(self, name, shape, stddev, wd):
        """Helper to create an initialized Variable with weight decay.

            Note that the Variable is initialized with a truncated normal distribution.
            A weight decay is added only if one is specified.

        Args:
            name: name of the variable
            shape: list of ints
            stddev: standard deviation of a truncated Gaussian
            wd: add L2Loss weight decay multiplied by this float. If None, weight
                decay is not added for this Variable.

        Returns:
            Variable Tensor
        """
        var = self.variable_on_cpu(name,
                               shape,
                               tf.truncated_normal_initializer(stddev=stddev, dtype=tf.float32))
        if wd is not None:
            weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
            tf.add_to_collection('losses', weight_decay)
        return var

    def model(self, images, eval=False, image_placeholder=None):
        num_classes = int(self.config.get('main', 'num_classes'))
        image_size = int(self.config.get('main', 'subsection_image_size'))
        num_channels = int(self.config.get('main', 'num_channels'))

        with tf.variable_scope('conv1', reuse=eval) as scope:
            kernel = self.variable_with_weight_decay('weights',
                                                     shape=[5, 5, 3, 64],
                                                     stddev=5e-2,
                                                     wd=0.0)
            if image_placeholder is None:
                conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
            else:
                whitened_image = tf.image.per_image_whitening(tf.reshape(image_placeholder, [image_size, image_size, num_channels]))
                whitened_image_reshaped = tf.reshape(whitened_image, [1, image_size, image_size, num_channels])
                conv = tf.nn.conv2d(whitened_image_reshaped, kernel, [1, 1, 1, 1], padding='SAME')
            biases = self.variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
            bias = tf.nn.bias_add(conv, biases)
            conv1 = tf.nn.relu(bias, name=scope.name)

        # pool1
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], 
                               strides=[1, 2, 2, 1],
                               padding='SAME', name='pool1')
        # norm1
        norm1 = tf.nn.lrn(pool1, 4, bias=1.0, 
                          alpha=0.001 / 9.0, beta=0.75, name='norm1')

        with tf.variable_scope('conv2', reuse=eval) as scope:
            kernel = self.variable_with_weight_decay('weights',
                                                 shape=[5, 5, 64, 64],
                                                 stddev=5e-2,
                                                 wd=0.0)
            conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = self.variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
            bias = tf.nn.bias_add(conv, biases)
            conv2 = tf.nn.relu(bias, name=scope.name)

        # norm2
        norm2 = tf.nn.lrn(conv2, 4, bias=1.0, 
                          alpha=0.001 / 9.0, beta=0.75, name='norm2')
        # pool2
        pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], 
                               strides=[1, 2, 2, 1],
                               padding='SAME', name='pool2')

        # local3
        with tf.variable_scope('local3', reuse=eval) as scope:
            # Move everything into depth so we can perform a single matrix multiply.
            if image_placeholder is None:
                reshape = tf.reshape(pool2, 
                    [int(self.config.get('main', 'batch_size')), -1])
            else:
                reshape = tf.reshape(pool2, [1, -1])
            dim = reshape.get_shape()[1].value
            weights = self.variable_with_weight_decay('weights', shape=[dim, 384],
                                                  stddev=0.04, wd=0.004)
            biases = self.variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
            local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)

        # local4
        with tf.variable_scope('local4', reuse=eval) as scope:
            weights = self.variable_with_weight_decay('weights', shape=[384, 192],
                                                  stddev=0.04, wd=0.004)
            biases = self.variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
            local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)

        # softmax, i.e. softmax(WX + b)
        with tf.variable_scope('softmax_linear', reuse=eval) as scope:
            weights = self.variable_with_weight_decay('weights', [192, num_classes],
                                                  stddev=1/192.0, wd=0.0)
            biases = self.variable_on_cpu('biases', [num_classes],
                                      tf.constant_initializer(0.0))
            softmax_linear = tf.add(tf.matmul(local4, weights), biases, name=scope.name)

        return softmax_linear
