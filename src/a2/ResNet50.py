# Author: Wuli Zuo, a1785343
# Date: 2021-09-08

import numpy as np
import plot
from tensorflow import keras
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import Dense, Dropout, GlobalMaxPooling2D, Activation, BatchNormalization
from tensorflow.keras.models import Model


# ResNet50
def ResNet():
    base_model = ResNet50(weights='imagenet', include_top=False, layers=keras.layers, input_shape=(32, 32, 3))
    x = base_model.output
    x = GlobalMaxPooling2D()(x)
    x = Dropout(0.5)(x)
    x = Dense(100)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    #x = Dense(100, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(10, activation="softmax")(x)
    base_model = Model(base_model.input, x, name="base_model")
    base_model.compile(loss='categorical_crossentropy', metrics=['acc'], optimizer='adam')
    base_model.summary()
    return base_model


def train(data_train, label_train, data_validate, label_validate, data_test, label_test):
    model = ResNet()

    early_stopping_cb = keras.callbacks.EarlyStopping(monitor='val_acc', patience=5, restore_best_weights=True)
    lr_scheduler = keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)

    ResNet_history = model.fit(data_train, label_train, epochs=100, validation_data=(data_validate, label_validate),
                               batch_size=128, callbacks=[early_stopping_cb, lr_scheduler])

    ResNet_evaluation = model.evaluate(data_test, label_test)
    ResNet_acc_train = np.max(ResNet_history.history['acc'])
    ResNet_acc_validation = np.max(ResNet_history.history['val_acc'])
    ResNet_acc_test = ResNet_evaluation[1]
    print("Accuracy on training data: ", f'{ResNet_acc_train * 100:.2f}%')
    print("Accuracy on validation data: ", f'{ResNet_acc_validation * 100:.2f}%')
    print("Accuracy on test data: ", f'{ResNet_acc_test * 100:.2f}%')
    plot.plot_history_resN(ResNet_history)
    print("\n## ResNet-50 architecture:")
    # model.summary()