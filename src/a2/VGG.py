# Author: Wuli Zuo, a1785343
# Date: 2021-09-08


import numpy as np
import plot
from tensorflow import keras
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, BatchNormalization


# function to construct a VGG network
def VGG():
    model = keras.Sequential()
    model.add(BatchNormalization())
    model.add(Conv2D(filters=64, kernel_size=(3, 3), input_shape=(32, 32, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(keras.layers.Dropout(0.25))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(keras.layers.Dropout(0.25))
    '''
    model.add(Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    '''

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dense(4096, activation='relu'))
    '''
    model.add(Dense(512, activation='relu'))
    model.add(Dense(512, activation='relu'))
    '''
    model.add(Dense(10, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# function to train a VGG network
def train(data_train, label_train, data_validate, label_validate, data_test, label_test):
    model = VGG()
    early_stopping_cb = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
    lr_scheduler = keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
    n_epochs = 100
    batch_size = 128
    VGG_history = model.fit(data_train, label_train, epochs=n_epochs, validation_data=(data_validate, label_validate),
                                        batch_size=batch_size, callbacks=[early_stopping_cb, lr_scheduler])
    VGG_evaluation = model.evaluate(data_test, label_test)
    VGG_acc_test = VGG_evaluation[1]
    VGG_acc_train = np.max(VGG_history.history['accuracy'])
    VGG_acc_validation = np.max(VGG_history.history['val_accuracy'])
    print("Accuracy on training data: ", f'{VGG_acc_train * 100:.2f}%')
    print("Accuracy on validation data: ", f'{VGG_acc_validation * 100:.2f}%')
    print("Accuracy on test data: ", f'{VGG_acc_test * 100:.2f}%')
    plot.plot_history(VGG_history)