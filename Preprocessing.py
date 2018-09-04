import numpy as np
import matplotlib.pyplot as plt
from CIFAR100 import CIFAR100

class Preprocessing:

    @staticmethod
    def global_contrast_normalization(image, epsilon, lambda_reg):
        mean_of_image = np.mean(image)
        std_of_image = max(epsilon,
                           np.sqrt(lambda_reg + (1/(image.shape[0]*image.shape[1]*image.shape[2])) * np.sum(np.square(image - mean_of_image))))

        return (image - mean_of_image) / std_of_image

    @staticmethod
    def show_image(image):
        plt.figure()
        plt.imshow(image)
        plt.show()

    @staticmethod
    def ZCA_Whitening(images, epsilon=0.00001):

        images = np.reshape(images, [images.shape[0], images.shape[1] * images.shape[2] * images.shape[3]])
        images = images - np.mean(images, axis=1).reshape([-1, 1])
        print(np.mean(images, axis=1))
        sigma = (images.T @ images) / images.shape[0]

        U, S, V = np.linalg.svd(sigma)
        print(U)
        # whitened = np.dot(images, np.dot(np.dot(, ), ))
        whiten = U @ np.diag(1.0/np.sqrt(S+epsilon)) @ U.T
        # whitened = images @ whiten_
        return (images @ whiten).reshape([-1, 32, 32, 3])


    @staticmethod
    def ZCA_Whitening_v2(images, epsilon=0.00001):
        images = images.reshape([-1, images.shape[1] * images.shape[2] * images.shape[3]])
        featurewise_mean = np.mean(images, axis=0)
        images_centered = images - featurewise_mean
        sigma = np.dot(images_centered.T, images_centered) / (images_centered.shape[0]-1)
        U, S, _ = np.linalg.svd(sigma)
        S_inv = np.diag(1.0/np.sqrt(S.clip(epsilon)))
        zca = U @ S_inv @ U.T
        return (images_centered @ zca).reshape([-1, 32, 32, 3])


cifar100 = CIFAR100()
train_data, train_fine_labels, train_coarse_labels = cifar100.get_train_data()


images = train_data
zca = Preprocessing.ZCA_Whitening(images)
cifar100.show_random_images((zca-np.min(zca))/(np.max(zca)-np.min(zca)), train_fine_labels, size=(5,5))
