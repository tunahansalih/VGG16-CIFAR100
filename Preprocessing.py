import numpy as np
import matplotlib.pyplot as plt
from CIFAR100 import CIFAR100

class Preprocessing:

    def global_contrast_normalization(self, image, epsilon, lambda_reg):
        mean_of_image = np.mean(image)
        std_of_image = max(epsilon,
                           np.sqrt(lambda_reg + (1/(image.shape[0]*image.shape[1]*image.shape[2])) * np.sum(np.square(image - mean_of_image))))

        return (image - mean_of_image) / std_of_image

    def show_image(self, image):
        plt.figure()
        plt.imshow(image)
        plt.show()

    def ZCA_Whitening(self, images):
        sigma = np.zeros((images.shape[0], images.shape[1], images.shape[2], images.shape[2]))
        for i in range(len(images)):
            images[i] = images[i] - np.mean(images[i])

        for i in range(len(images)):
            for j in range(3):
                sigma[i, :, :, j] = images[i, :, :, j].T.dot(images[i, :, :, j]) / images.shape[0]


cifar100 = CIFAR100()
train_data, train_fine_labels, train_coarse_labels = cifar100.get_train_data()

pre = Preprocessing()

train_batch = train_data[:10]

pre.ZCA_Whitening(train_batch)
