# Author: Wuli Zuo, a1785343
# Date: 2021-09-02


import matplotlib.pyplot as plt
import numpy as np
import pickle
import torchvision
import ssl
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical


# function to unpack a file
def unpack(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


# function to load CIFAR10 and process data
def load(path):
    # load data
    ssl._create_default_https_context = ssl._create_unverified_context
    '''
    train_set = torchvision.datasets.CIFAR10(root=path, train=True,
                                            download=True, transform=None)
    test_set = torchvision.datasets.CIFAR10(root=path, train=False,
                                           download=True, transform=None)

    train_set_size = len(train_set)
    print('## Train set size: ', train_set_size)
    test_set_size = len(test_set)
    print('## Test set size: ', test_set_size)
    # print('## Train set type: ', type(train_set[0]))
    train_set_labels = np.zeros(train_set_size)
    print('## Train set label shape: ', train_set_labels.shape)
    test_set_labels = np.zeros(test_set_size)
    print('## Test set label shape: ', test_set_labels.shape)
    '''

    file0 = unpack(path + "cifar-10-batches-py/test_batch")
    file1 = unpack(path + "cifar-10-batches-py/data_batch_1")
    file2 = unpack(path + "cifar-10-batches-py/data_batch_2")
    file3 = unpack(path + "cifar-10-batches-py/data_batch_3")
    file4 = unpack(path + "cifar-10-batches-py/data_batch_4")
    file5 = unpack(path + "cifar-10-batches-py/data_batch_5")

    # data processing: normalization, label encoding and separating

    data_train_all = np.r_[file1[b'data'], file2[b'data'], file3[b'data'], file4[b'data'], file5[b'data']]
    label_train_all = np.r_[file1[b'labels'], file2[b'labels'], file3[b'labels'], file4[b'labels'], file5[b'labels']]
    data_test = file0[b'data']
    label_test = np.r_[file0[b'labels']]
    '''
    plt.title("Training data distribution")
    plt.xticks(range(10))
    plt.hist(label_train_all, bins=10, width=0.8)
    plt.show()
    
    plt.title("Test data distribution")
    plt.xticks(range(10))
    plt.hist(label_test, bins=10, width=0.8)
    plt.show()
    '''
    # normalise
    data_train_all = np.transpose(data_train_all.reshape(-1, 3, 32, 32), (0, 2, 3, 1))
    data_test = np.transpose(data_test.reshape(-1, 3, 32, 32), (0, 2, 3, 1))
    data_train_all = data_train_all / 255
    data_test = data_test / 255
    # label encoding
    label_train_encode = to_categorical(label_train_all, 10)
    label_test_encode = to_categorical(label_test, 10)
    print("## Encoded label sample", label_train_encode[0])
    # separate: training set & validation set
    data_train, data_validate, label_train, label_validate = train_test_split(data_train_all, label_train_encode, train_size=0.7,
                                                                    random_state=1, stratify=label_train_encode)
    data_test, label_test = data_test, label_test_encode
    print("## Training data shape: ", data_train.shape, label_train.shape)
    print("## Validation data shape: ", data_validate.shape, label_validate.shape)
    print("## Test data shape: ", data_test.shape, label_test.shape)

    return data_train, label_train, data_validate, label_validate, data_test, label_test
