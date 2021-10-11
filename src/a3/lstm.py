# Author: Wuli Zuo, a1785343
# Date: 2021-09-30


import math
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.metrics import mean_squared_error


def lstm(trainX, trainY, validX, validY, epochs, look_back, batch_size, n_neurons):
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    model = Sequential()
    model.add(LSTM(n_neurons, input_shape=(look_back, 5)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])
    history = model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size, verbose=0, callbacks=[callback],
                        validation_data=(validX, validY))
    return model, history


'''
def cal_rmse(Y, predict, name):
    score = math.sqrt(mean_squared_error(Y[:], predict[:]))
    print(name, 'Score: %.2f RMSE' % (score))
    return score
'''


def repeat_inverse_trans(scaler, Y, dim):
    Y = np.repeat(Y, 5, axis=1)
    Y = scaler.inverse_transform(Y)
    if dim == 0:
        Y = Y[:, 0].reshape(Y.shape[0], )
    else:
        Y = Y[:, 0]
    return Y


def train(scaler, trainX_org, trainY_org, validX_org, validY_org, epochs, look_back, batch_size, neurons_num):
    model, history = lstm(trainX_org, trainY_org, validX_org, validY_org, epochs, look_back, batch_size, neurons_num)
    # predict
    trainPredict = model.predict(trainX_org)
    validPredict = model.predict(validX_org)
    # invert predictions
    trainPredict = repeat_inverse_trans(scaler, trainPredict, 1)
    trainY = repeat_inverse_trans(scaler, trainY_org.reshape(trainY_org.shape[0], 1), 0)
    validPredict = repeat_inverse_trans(scaler, validPredict, 1)
    validY = repeat_inverse_trans(scaler, validY_org.reshape(validY_org.shape[0], 1), 0)
    mse = tf.keras.losses.MeanSquaredError()
    train_score_mse = np.sqrt(mse(trainY, trainPredict).numpy())
    valid_score_mse = np.sqrt(mse(validY, validPredict).numpy())
    print("### training score: ", train_score_mse)
    print("### validation score: ", valid_score_mse)
    return trainPredict, validPredict, train_score_mse, valid_score_mse, history
