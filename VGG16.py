import tensorflow as tf
import numpy as np
from CIFAR100 import CIFAR100
from tensorflow import keras


class VGG16:
    NUM_CLASSES = 100
    BATCH_SIZE = 100
    LEARNING_RATE = 0.001
    MOMENTUM = 0.9
    MAX_STEPS = 100000
    CONV_FILTER_NUMS = [[64, 64],
                        [128, 128],
                        [256, 256, 256],
                        [512, 512, 512],
                        [512, 512, 512]]

    FC_LAYER_SIZE = [4096,
                     4096,
                     1000]

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

    def create_architecture(self, input):

        with tf.name_scope('conv1_1') as scope:
            conv1_1 = self.conv_layer(input=input,
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
            input_size = int(np.prod(maxpool5.shape[1:]))
            flattened_input = tf.reshape(maxpool5, [-1, input_size])
            fc1 = self.fc_layer(flattened_input,
                                input_size=input_size,
                                layer_size=4096,
                                scope=scope)

        with tf.name_scope('dropout1') as scope:
            do1 = tf.nn.dropout(fc1,
                                keep_prob=0.6,
                                name=scope)

        with tf.name_scope('fc2') as scope:
            fc2 = self.fc_layer(do1,
                                input_size=4096,
                                layer_size=4096,
                                scope=scope)

        with tf.name_scope('dropout2') as scope:
            do2 = tf.nn.dropout(fc2,
                                keep_prob=0.6,
                                name=scope)

        with tf.name_scope('fc3') as scope:
            weights = tf.Variable(initial_value=tf.truncated_normal([4096, 100], dtype=tf.float32, stddev=0.1),
                                  name='weights')
            bias = tf.Variable(initial_value=tf.truncated_normal([100], dtype=tf.float32, stddev=0.1),
                               name='biases')
            y = tf.nn.bias_add(tf.matmul(do2, weights),
                               bias,
                               name=scope)

        return y

    def classification_loss(self, logits, labels):

        return tf.losses.sparse_softmax_cross_entropy(labels=labels,
                                                      logits=logits)


    def train_step(self, loss, global_step):
        with tf.name_scope('optimizer') as scope:
            train_step = tf.train.MomentumOptimizer(self.LEARNING_RATE, momentum=self.MOMENTUM).minimize(loss, global_step=global_step, name=scope)

        return train_step

    def run_tf_session(self, image_placeholder, label_placeholder):
        pass


cifar100 = CIFAR100()

train_data, train_fine_labels, train_coarse_labels = cifar100.get_train_data()
test_data, test_fine_labels, test_coarse_labels = cifar100.get_test_data()

#cifar100.show_random_images(train_data.astype(int), train_fine_labels, size=(5, 5))

images = tf.placeholder(tf.float32, shape=[None, 32, 32, 3], name='images')
labels = tf.placeholder(tf.int32, name='labels')
global_step = tf.Variable(initial_value=0, trainable=False)

vgg16 = VGG16()

y_predicted = vgg16.create_architecture(images)

predicted_labels = tf.argmax(y_predicted, axis=1, output_type=tf.int32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted_labels, labels), tf.float32))

loss = vgg16.classification_loss(y_predicted, labels)

train_op = vgg16.train_step(loss, global_step)
n_batches = int(train_data.shape[0]/vgg16.BATCH_SIZE)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(vgg16.MAX_STEPS):
        for j in range(n_batches):
            print(i, j)
            _, accuracy_val, loss_val = sess.run([train_op, accuracy, loss],
                                         feed_dict={images: train_data[(j*100):(j*100+100)],
                                                    labels: train_fine_labels[(j*100):(j*100+100)]})
            print("Iter: {}, Loss: {:.4f}, Accuracy: {:.4f}".format(j, loss_val, accuracy_val))

        print("Iter: {}, Loss: {:.4f}, Accuracy: {:.4f}".format(i, loss_val, accuracy_val))


