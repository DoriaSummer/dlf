# Author: Wuli Zuo, a1785343
# Date: 2021-09-08


import numpy as np
import plot
from tensorflow import keras
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, BatchNormalization


# function to construct a AlexNet
def AlexNet():
    model = keras.Sequential()
    model.add(Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4), activation='relu', input_shape=(32, 32, 3)))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding="same"))
    model.add(BatchNormalization())
    # model.add(MaxPool2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))  # decreased size
    model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding="same"))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=384, kernel_size=(1, 1), strides=(1, 1), activation='relu', padding="same"))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1), activation='relu', padding="same"))
    model.add(BatchNormalization())
    # model.add(MaxPool2D(pool_size=(3, 3), strides=(2, 2)))  # removed
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model


# function to train a AlexNet
def train(data_train, label_train, data_validate, label_validate, data_test, label_test):
    model = AlexNet()
    early_stopping_cb = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
    lr_scheduler = keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
    n_epochs = 100
    batch_size = 128
    AlexNet_history = model.fit(data_train, label_train, epochs=n_epochs, validation_data=(data_validate, label_validate),
                                        batch_size=batch_size, callbacks=[early_stopping_cb, lr_scheduler])
    AlexNet_evaluation = model.evaluate(data_test, label_test)
    AlexNet_acc_test = AlexNet_evaluation[1]
    AlexNet_acc_train = np.max(AlexNet_history.history['accuracy'])
    AlexNet_acc_validation = np.max(AlexNet_history.history['val_accuracy'])
    print("Accuracy on training data: ", f'{AlexNet_acc_train * 100:.2f}%')
    print("Accuracy on validation data: ", f'{AlexNet_acc_validation * 100:.2f}%')
    print("Accuracy on test data: ", f'{AlexNet_acc_test * 100:.2f}%')
    plot.plot_history(AlexNet_history)