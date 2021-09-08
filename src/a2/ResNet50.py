# Author: Wuli Zuo, a1785343
# Date: 2021-09-08


import numpy as np
from keras.applications.resnet import ResNet50
from keras.layers import GlobalMaxPooling2D

import plot
from tensorflow import keras
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.models import Model


# ResNet50
model = ResNet50(include_top=False,input_shape=(32,32,3))

x = model.output
x = GlobalMaxPooling2D()(x)
x = Dropout(0.5)(x)
x = Dense(100, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(10, activation="softmax")(x)
model = Model(model.input, x, name="model")
model.compile(loss="binary_crossentropy", metrics=['acc'], optimizer="adam")
model.summary()