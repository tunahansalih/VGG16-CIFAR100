import tensorflow as tf
import numpy as np
from CIFAR100 import CIFAR100
from tensorflow import keras


class VGG16:
    NUM_CLASSES = 100
    BATCH_SIZE = 50
    LEARNING_RATE = 0.001
    MAX_STEPS = 100000
    CONV_FILTER_NUMS = [[64, 64],
                        [128, 128],
                        [256, 256, 256],
                        [512, 512, 512],
                        [512, 512, 512]]

    FC_LAYER_SIZE = [4096,
                     4096,
                     1000]

    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    @staticmethod
    def conv_layer(input, ksize, scope):
        kernel = tf.Variable(initial_value=tf.truncated_normal(ksize, dtype=tf.float32, stddev=0.1),
                             name='weights')
        bias = tf.Variable(initial_value=tf.truncated_normal(ksize[3:], dtype=tf.float32, stddev=0.1),
                           name='biases')
        conv = tf.nn.conv2d(input=input,
                            filter=kernel,
                            strides=[1, 1, 1, 1],
                            padding='SAME')
        out = tf.nn.bias_add(conv, bias)
        return tf.nn.relu(out,
                          name=scope)

    @staticmethod
    def fc_layer(input, input_size, layer_size, scope):
        weights = tf.Variable(initial_value=tf.truncated_normal([input_size, layer_size], dtype=tf.float32, stddev=0.1),
                             name='weights')
        bias = tf.Variable(initial_value=tf.truncated_normal([layer_size], dtype=tf.float32, stddev=0.1),
                           name='biases')
        out = tf.nn.bias_add(tf.matmul(input, weights), bias)
        return tf.nn.relu(out, name=scope)

    def create_architecture(self, input_placeholder):

        with tf.name_scope('conv1_1') as scope:
            conv1_1 = self.conv_layer(input=input_placeholder,
                                      ksize=[3, 3, 3, 64],
                                      scope=scope)

        with tf.name_scope('conv1_2') as scope:
            conv1_2 = self.conv_layer(input=conv1_1,
                                      ksize=[3, 3, 64, 64],
                                      scope=scope)
        with tf.name_scope('maxpool1') as scope:
            maxpool1 = tf.nn.max_pool(value=conv1_2,
                                      ksize=[1, 2, 2, 1],
                                      strides=[1, 2, 2, 1],
                                      padding='SAME',
                                      name=scope)

        with tf.name_scope('conv2_1') as scope:
            conv2_1 = self.conv_layer(input=maxpool1,
                                      ksize=[3, 3, 64, 128],
                                      scope=scope)

        with tf.name_scope('conv2_2') as scope:
            conv2_2 = self.conv_layer(input=conv2_1,
                                      ksize=[3, 3, 128, 128],
                                      scope=scope)

        with tf.name_scope('maxpool2') as scope:
            maxpool2 = tf.nn.max_pool(value=conv2_2,
                                      ksize=[1, 2, 2, 1],
                                      strides=[1, 2, 2, 1],
                                      padding='SAME',
                                      name=scope)

        with tf.name_scope('conv3_1') as scope:
            conv3_1 = self.conv_layer(input=maxpool2,
                                      ksize=[3, 3, 128, 256],
                                      scope=scope)

        with tf.name_scope('conv3_2') as scope:
            conv3_2 = self.conv_layer(input=conv3_1,
                                      ksize=[3, 3, 256, 256],
                                      scope=scope)

        with tf.name_scope('conv3_3') as scope:
            conv3_3 = self.conv_layer(input=conv3_2,
                                      ksize=[3, 3, 256, 256],
                                      scope=scope)

        with tf.name_scope('maxpool3') as scope:
            maxpool3 = tf.nn.max_pool(value=conv3_3,
                                      ksize=[1, 2, 2, 1],
                                      strides=[1, 2, 2, 1],
                                      padding='SAME',
                                      name=scope)

        with tf.name_scope('conv4_1') as scope:
            conv4_1 = self.conv_layer(input=maxpool3,
                                      ksize=[3, 3, 256, 512],
                                      scope=scope)

        with tf.name_scope('conv4_2') as scope:
            conv4_2 = self.conv_layer(input=conv4_1,
                                      ksize=[3, 3, 512, 512],
                                      scope=scope)

        with tf.name_scope('conv4_3') as scope:
            conv4_3 = self.conv_layer(input=conv4_2,
                                      ksize=[3, 3, 512, 512],
                                      scope=scope)

        with tf.name_scope('maxpool4') as scope:
            maxpool4 = tf.nn.max_pool(value=conv4_3,
                                      ksize=[1, 2, 2, 1],
                                      strides=[1, 2, 2, 1],
                                      padding='SAME',
                                      name=scope)

        with tf.name_scope('conv5_1') as scope:
            conv5_1 = self.conv_layer(input=maxpool4,
                                      ksize=[3, 3, 512, 512],
                                      scope=scope)

        with tf.name_scope('conv5_2') as scope:
            conv5_2 = self.conv_layer(input=conv5_1,
                                      ksize=[3, 3, 512, 512],
                                      scope=scope)

        with tf.name_scope('conv5_3') as scope:
            conv5_3 = self.conv_layer(input=conv5_2,
                                      ksize=[3, 3, 512, 512],
                                      scope=scope)

        with tf.name_scope('maxpool5') as scope:
            maxpool5 = tf.nn.max_pool(value=conv5_3,
                                      ksize=[1, 2, 2, 1],
                                      strides=[1, 2, 2, 1],
                                      padding='SAME',
                                      name=scope)

        with tf.name_scope('fc1') as scope:
            input_size = np.prod(maxpool5.shape[1:])
            flattened_input = tf.reshape(maxpool5, [-1, input_size])
            tf1 = self.fc_layer(flattened_input,
                                input_size=input_size,
                                layer_size=4096,
                                scope=scope)

        with tf.name_scope('fc2') as scope:
            tf2 = self.fc_layer(tf1,
                                input_size=4096,
                                layer_size=4096,
                                scope=scope)

        with tf.name_scope('fc3') as scope:
            tf3 = self.fc_layer(tf1,
                                input_size=4096,
                                layer_size=100,
                                scope=scope)

        with tf.name_scope('softmax-100') as scope:
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels,
                                                           logits=tf3,
                                                           name='softmax')


cifar100 = CIFAR100()

train_data, train_fine_labels, train_coarse_labels = cifar100.get_train_data()
test_data, test_fine_labels, test_coarse_labels = cifar100.get_test_data()

cifar100.show_random_images(train_data.astype(int), train_fine_labels, size=(5, 5))

img_placeholder = tf.placeholder(tf.float32, shape=[None, 32, 32, 3], name='images')

vgg16 = VGG16(train_data, train_fine_labels)
vgg16.create_architecture(img_placeholder)


