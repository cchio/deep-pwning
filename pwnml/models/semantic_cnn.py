import tensorflow as tf

import utils.utils as utils

class SemanticCNN:

    def __init__(self, config,
        sequence_length, vocab_size, embedding_size, num_filters):
        self.config = config
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.num_filters = num_filters
        if config.get('main', 'seed') == 'None':
            self.seed = None
        else:
            self.seed = config.getint('main', 'seed')

    def conv2d(self, data, weight):
        return tf.nn.conv2d(data,
                            weight,
                            strides=[1, 1, 1, 1],
                            padding='VALID')

    def max_pool(self, data, filter_size):
        return tf.nn.max_pool(data,
                              ksize=[1, self.sequence_length - filter_size + 1, 1, 1],
                              strides=[1, 1, 1, 1],
                              padding='VALID')

    def variable(self, flavor, shape):
        if flavor == 'W_truncated_normal':
            return tf.Variable(
                tf.truncated_normal(shape,
                                    stddev=0.1,
                                    seed=self.seed, 
                                    dtype=tf.float32))
        elif flavor == 'W_random_uniform':
            return tf.Variable(
                tf.random_uniform(shape,
                                  minval=-1.0,
                                  maxval=1.0))
        elif flavor == 'b':
            return tf.Variable(tf.constant(0.1, shape=shape), 
                               dtype=tf.float32)
        else:
            return None

    def train_input_placeholders(self):
        x = tf.placeholder(tf.float32,
                           shape=[None, self.sequence_length],
                           name="x")
        y_ = tf.placeholder(tf.float32, 
            [None, self.config.getint('main', 'num_classes')], name="y_")
        return x, y_

    def model(self, data):
        l2_loss = tf.constant(0.0)
        keep_prob = tf.placeholder(tf.float32, name="keep_prob")

        embed_W = self.variable('W_random_uniform', [self.vocab_size, self.embedding_size])
        embedded_words = tf.nn.embedding_lookup(embed_W, tf.cast(data, tf.int32))
        embedded_words_expanded = tf.expand_dims(embedded_words, -1)

        filter3_shape = [3, self.embedding_size, 1, self.num_filters]
        pool_filter3_W = self.variable('W_truncated_normal', filter3_shape)
        pool_filter3_b = self.variable('b', [self.num_filters])

        conv1 = tf.nn.relu(tf.nn.bias_add(
            self.conv2d(embedded_words_expanded, pool_filter3_W), pool_filter3_b))
        pool_filter3 = self.max_pool(conv1, 3)

        filter4_shape = [4, self.embedding_size, 1, self.num_filters]
        pool_filter4_W = self.variable('W_truncated_normal', filter4_shape)
        pool_filter4_b = self.variable('b', [self.num_filters])

        conv2 = tf.nn.relu(tf.nn.bias_add(
            self.conv2d(embedded_words_expanded, pool_filter4_W), pool_filter4_b))
        pool_filter4 = self.max_pool(conv2, 4)

        filter5_shape = [5, self.embedding_size, 1, self.num_filters]
        pool_filter5_W = self.variable('W_truncated_normal', filter5_shape)
        pool_filter5_b = self.variable('b', [self.num_filters])

        conv3 = tf.nn.relu(tf.nn.bias_add(
            self.conv2d(embedded_words_expanded, pool_filter5_W), pool_filter5_b))
        pool_filter5 = self.max_pool(conv3, 5)

        pool_combined = tf.concat(3, [pool_filter3, pool_filter4, pool_filter5])
        pool_final = tf.reshape(pool_combined, [-1, self.num_filters * 3])

        dropout = tf.nn.dropout(pool_final, keep_prob)

        final_W = tf.get_variable("W", shape=[self.num_filters * 3, 
            self.config.getint('main', 'num_classes')],
            initializer=tf.contrib.layers.xavier_initializer())
        final_b = tf.Variable(tf.constant(0.1, 
            shape=[self.config.getint('main', 'num_classes')]), name="b")

        logits = tf.matmul(dropout, final_W) + final_b
        y_conv = tf.nn.softmax(logits)

        l2_loss += tf.nn.l2_loss(final_W) + tf.nn.l2_loss(final_b)

        return y_conv, logits, keep_prob, l2_loss, embedded_words, embed_W
