import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle


class CIFAR100:

    TRAIN_FILE = 'cifar-100-python/train'
    TEST_FILE = 'cifar-100-python/test'

    CIFAR100_LABELS_LIST = [
        'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
        'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
        'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
        'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
        'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
        'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
        'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
        'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
        'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
        'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
        'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
        'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
        'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
        'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
        'worm'
    ]

    def unpickle(self, file):
        '''Unpickle the pickled CIFAR-100 files'''
        with open(file, 'rb') as fo:
            dic = pickle.load(fo, encoding='bytes')
            for k in dic:
                print(k, type(dic[k]))
        return dic

    def get_data_from_dict(self, dic):
        '''Get data from the dictionary'''
        return dic[b'data'], dic[b'fine_labels'], dic[b'coarse_labels']

    def show_random_images(self, data, labels, size=(5, 5)):
        size_x = size[0]
        size_y = size[1]
        f, axarr = plt.subplots(size_x, size_y)
        plt.subplots_adjust(hspace=0.5)
        indices = np.random.choice(range(data.shape[0]), size=size_x*size_y, replace=False)

        for i, index in enumerate(indices):
            print(i, index)
            ax = axarr[i // size_x, i % size_y]
            ax.axis('off')
            ax.imshow(data[index])
            ax.set_title(self.CIFAR100_LABELS_LIST[labels[index]])
        plt.show()

    def get_train_data(self):
        train_dict = self.unpickle(self.TRAIN_FILE)
        train_data, train_fine_labels, train_coarse_labels = self.get_data_from_dict(train_dict)
        train_data = np.transpose(np.reshape(train_data, (-1, 3, 32, 32)), (0, 2, 3, 1))
        return train_data.astype(np.float32), train_fine_labels, train_coarse_labels

    def get_test_data(self):
        test_dict = self.unpickle(self.TEST_FILE)
        test_data, test_fine_labels, test_coarse_labels = self.get_data_from_dict(test_dict)
        test_data = np.transpose(np.reshape(test_data, (-1, 3, 32, 32)), (0, 2, 3, 1))
        return test_data.astype(np.float32), test_fine_labels, test_coarse_labels


