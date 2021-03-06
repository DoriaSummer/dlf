# Author: Wuli Zuo, a1785343
# Date: 2020-08-12 16:36


import numpy as np
import perceptron
import nn
import svm
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split


# function of loading data set from a file
def load_data_set(filepath):
    data, label = load_svmlight_file(filepath)
    data = np.array(data.todense())
    data_train, data_test, label_train, label_test = train_test_split(data, label, test_size=0.4,
                                                                      random_state=1, stratify=label)
    '''
    print(data_train.shape)
    print(data_test.shape)
    print(label_train.shape)
    print(label_test.shape)
    '''

    return data_train, data_test, label_train, label_test


if __name__ == "__main__":
    # load data
    data_train, data_test, label_train, label_test = load_data_set('../../data/a1/diabetes_scale.txt')

    # train & predict with perceptron
    print('\n\n# 1. Perceptron')
    # tune parameter
    best_lr = 0
    best_acc_train = 0
    best_acc_test = 0
    best_model = 0
    for learning_rate in [0.1, 0.01, 0.001, 0.0001]:
        print('\n## learning rate = ', learning_rate)
        model = perceptron.train(data_train, label_train, data_test, label_test, learning_rate, 1000)
        acc_train = perceptron.getAccuracy(data_train, label_train, model)
        acc_test = perceptron.getAccuracy(data_test, label_test, model)
        if acc_test > best_acc_test:
            best_acc_test = acc_test
            best_acc_train = acc_train
            best_lr = learning_rate
            best_model = model
        print('## learning rate = %f, training accuracy = %f, test accuracy = %f'
              % (learning_rate, acc_train, acc_test))

    print('\n* Best result:  ')
    print('## learning rate = %f, training accuracy = %f, test accuracy = %f'
          % (best_lr, best_acc_train, best_acc_test))

    # sklearn implementation for comparison
    # train & predict with sklearn
    print('\n\n# 2. SVM')
    model_sk = svm.sk(data_train, label_train, data_test, label_test)

    # pytorch implementation for comparison
    # train & predict with sklearn
    print('\n\n# 3. Neural Networks')
    svm_model_torch = nn.train(data_train, label_train, data_test, label_test)



