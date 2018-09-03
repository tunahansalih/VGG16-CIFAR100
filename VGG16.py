import tensorflow as tf
from CIFAR100 import CIFAR100
from tensorflow import keras

cifar100 = CIFAR100()

train_data, train_fine_labels, train_coarse_labels = cifar100.get_train_data()
test_data, test_fine_labels, test_coarse_labels = cifar100.get_test_data()

cifar100.show_random_images(train_data, train_fine_labels, size=(10, 10))

data_gen = keras.preprocessing.image.ImageDataGenerator(featurewise_center=True,
                                                        featurewise_std_normalization=True,
                                                        zca_whitening=True,
                                                        horizontal_flip=True)

data_gen.fit(train_data)