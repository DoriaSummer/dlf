# Author: Wuli Zuo, a1785343
# Date: 2021-09-02


import LeNet5
import load
import os


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print('\n#1. Load and process data')
data_train, label_train, data_validate, label_validate, data_test, label_test = load.load('../../data/a2/')

print('\n#2. LeNet-5 model')
print('\n#2-1. LeNet-5 model: original form')
LeNet5.train(data_train, label_train, data_validate, label_validate, data_test, label_test)
print('\n#2-2. LeNet-5 model: tune parameters')
LeNet5.select(LeNet5.LeNet5, data_train, label_train, data_validate, label_validate)
