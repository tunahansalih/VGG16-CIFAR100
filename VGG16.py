import tensorflow as tf
from CIFAR100 import CIFAR100
from tensorflow import keras


class VGG16:
    NUM_CLASSES = 100
    BATCH_SIZE = 50
    LEARNING_RATE = 0.001
    MAX_STEPS = 100000


    def __init__(self, images):
        self.images = images


    def conv_layers(self):
        img_placeholder = tf.placeholder(tf.float32, shape=[None, 32, 32, 3], name='images')





cifar100 = CIFAR100()

train_data, train_fine_labels, train_coarse_labels = cifar100.get_train_data()
test_data, test_fine_labels, test_coarse_labels = cifar100.get_test_data()

cifar100.show_random_images(train_data, train_fine_labels, size=(10, 10))





