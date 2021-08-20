# Author: Wuli Zuo, a1785343
# Date: 2021-08-12 16:17


import numpy as np
import plot


# function for perceptron learning
def train(X_train, y_train, X_test, y_test, learning_rate, epoch):
    W = np.random.randn(len(X_train[0]) + 1)
    X_b = np.c_[np.ones(len(X_train)), X_train]

    acc_train_list = []
    acc_test_list = []
    acc_index_list = []
    for e in range(epoch):
        step = np.zeros([len(X_b[0])])
        for i in range(len(X_b)):
            if y_train[i] * np.dot(X_b[i], W) <= 0:
                step += y_train[i] * X_b[i]
        W = W + learning_rate * step
        if e % 50 == 0:
            acc_train = getAccuracy(X_train, y_train, W)
            acc_test = getAccuracy(X_test, y_test, W)
            acc_train_list.append(acc_train)
            acc_test_list.append(acc_test)
            acc_index_list.append(e)
            print('\t%d, training accuracy = %f, test accuracy = %f'
                  % (e, acc_train, acc_test))
    plot.plot_acc(acc_train_list, acc_test_list, acc_index_list, 'training', 'test', 'Perceptron', learning_rate)
    return W


# function to predict
def predict(X, W):
    X = np.c_[np.ones(len(X)), X]
    g = np.dot(X, W)
    pred = np.sign(g)
    return pred


# function to get accuracy
def getAccuracy(X, y, W):
    pred = predict(X, W)
    acc = sum(y * pred == 1) / len(y)
    return acc
