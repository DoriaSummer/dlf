# Author: Wuli Zuo, a1785343
# Date: 2021-09-02


import AlexNet
import LeNet5
import load
import os
import VGG
# from tensorflow.keras import backend as K



os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# K.set_image_data_format('channels_first')


print('\n#1. Load and process data')
data_train, label_train, data_validate, label_validate, data_test, label_test = load.load('../../data/a2/')


print('\n#2. LeNet-5 model')
print('\n#2-1. LeNet-5 model: original')
# LeNet5.train(data_train, label_train, data_validate, label_validate, data_test, label_test, 0)
print('\n#2-2. LeNet-5 model: tune parameters')
#LeNet5.select_param(LeNet5.LeNet5, data_train, label_train, data_validate, label_validate)
print('\n#2-3. LeNet-5 model: dropout')
#LeNet5.train(data_train, label_train, data_validate, label_validate, data_test, label_test, 1)
print('\n#2-4. LeNet-5 model: dropout & tune convolutional layer filters and dense layer units')
LeNet5.select_size(LeNet5.LeNet5, data_train, label_train, data_validate, label_validate)
print('\n#2-5. LeNet-5 model: dropout & more filters and dense & more cov layer and batch normalisation')
# LeNet5.train(data_train, label_train, data_validate, label_validate, data_test, label_test, 2)


print('\n#3. AlexNet model: modified')
#AlexNet.train(data_train, label_train, data_validate, label_validate, data_test, label_test)


print('\n#4. VGG model: modified')
# VGG.train(data_train, label_train, data_validate, label_validate, data_test, label_test)