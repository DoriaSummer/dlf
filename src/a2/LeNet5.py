# Author: Wuli Zuo, a1785343
# Date: 2021-09-04


import numpy as np
import plot
from sklearn.model_selection import RandomizedSearchCV
from tensorflow import keras
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten


# function to construct a LeNet-5 network
def LeNet5(activation, optimizer, learning_rate):
    model = keras.Sequential()
    model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu', input_shape=(32, 32, 3)))
    model.add(MaxPool2D(strides=2))
    model.add(Conv2D(filters=48, kernel_size=(5, 5), padding='valid', activation='relu'))
    model.add(MaxPool2D(strides=2))
    model.add(Flatten())
    model.add(Dense(256, activation=activation))
    model.add(Dense(84, activation=activation))
    model.add(Dense(10, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=optimizer(lr=learning_rate), metrics=['accuracy'])
    return model


# function to train a LeNet-5 network
def train(data_train, label_train, data_validate, label_validate, data_test, label_test):
    model = LeNet5('relu', keras.optimizers.Adam, 0.001)
    early_stopping_cb = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
    lr_scheduler = keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
    epoch = 100
    batch_size = 128
    LeNet_history = model.fit(data_train, label_train, epochs=epoch, validation_data=(data_validate, label_validate),
              batch_size=batch_size, callbacks=[early_stopping_cb, lr_scheduler])
    LeNet_evaluation = model.evaluate(data_test, label_test)
    LeNet_acc_test = LeNet_evaluation[1]
    LeNet_acc_train = np.max(LeNet_history.history['accuracy'])
    LeNet_acc_validate = np.max(LeNet_history.history['val_accuracy'])
    print("Accuracy on training data: ", f'{LeNet_acc_train * 100:.2f}%')
    print("Accuracy on validation data: ", f'{LeNet_acc_validate * 100:.2f}%')
    print("Accuracy on test data: ", f'{LeNet_acc_test * 100:.2f}%')

    plot.plot_history(LeNet_history)
    plot.plot_lr(LeNet_history)

    print("\n## LeNet-5 architecture:")
    model.summary()


# function to tune parameters for a LeNet-5 network
def select(model, data_train, label_train, data_validate, label_validate):
    keras_cls = keras.wrappers.scikit_learn.KerasClassifier(model)
    param_dist = {
        'activation': ['elu', 'relu'],
        'optimizer': [keras.optimizers.Adam, keras.optimizers.SGD],
        'learning_rate': [0.1, 0.01, 0.001, 0.0001]
    }
    epoch = 100
    batch_size = 128
    early_stopping_cb = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
    lr_scheduler = keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
    random_searched_model = RandomizedSearchCV(keras_cls, param_dist, n_iter=10, cv=3)
    random_searched_model.fit(data_train, label_train, epochs=epoch, validation_data=(data_validate, label_validate),
                          batch_size=batch_size, callbacks=[early_stopping_cb, lr_scheduler])

    print("Best parameters: ", random_searched_model.best_params_)
    print("Best score: ", random_searched_model.best_score_)
    plot.plot_history(random_searched_model.best_estimator_.model.history)