# Author: Wuli Zuo, a1785343
# Date: 2021-09-30


import matplotlib.pyplot as plt
import load
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.layers import Bidirectional
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import Sequential


def lstm(trainX, trainY, validX, validY, epochs, time_step, batch_size, neurons_num, flag):
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    model = Sequential()
    if flag > 0:  # Bidirectional
        model.add(Bidirectional(LSTM(neurons_num, input_shape=(time_step, 5))))
    else:
        model.add(LSTM(neurons_num, input_shape=(time_step, 5)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])
    history = model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size, verbose=0,
                        callbacks=[callback], validation_data=(validX, validY))
    return model, history


def repeat_inverse_trans(scaler, Y, dim):
    Y = np.repeat(Y, 5, axis=1)
    Y = scaler.inverse_transform(Y)
    if dim == 0:
        Y = Y[:, 0].reshape(Y.shape[0], )
    else:
        Y = Y[:, 0]
    return Y


def select_param(repeat, epochs, train, valid, scaler_train, time_steps, batch_sizes, neuron_nums, flag):
    search_list = []
    param = ''
    if len(time_steps) > 1:
        search_list = time_steps
        param = 'time_steps'
    elif len(batch_sizes) > 1:
        search_list = batch_sizes
        param = 'batch_sizes'
    elif len(neuron_nums) > 1:
        search_list = neuron_nums
        param = 'neuron_nums'
    train_scores_list = []
    valid_scores_list = []

    for i in range(repeat):
        train_scores_list_i = []
        valid_scores_list_i = []
        for time_step in time_steps:
            for batch_size in batch_sizes:
                for neuron_num in neuron_nums:
                    print('## (%d, %d, %d, %d)' % (i, time_step, batch_size, neuron_num))
                    trainX, trainY = load.create_dataset(train, time_step)
                    validX, validY = load.create_dataset(valid, time_step)
                    trainPredict, validPredict, train_scores, valid_scores, history, model =\
                        model_train(scaler_train, trainX, trainY, validX, validY,
                              epochs, time_step, batch_size, neuron_num, flag)
                    train_scores_list_i.append(train_scores)
                    valid_scores_list_i.append(valid_scores)
        train_scores_list.append(train_scores_list_i)
        valid_scores_list.append(valid_scores_list_i)
    train_scores_arr = np.array(train_scores_list)
    train_scores_mean = np.mean(train_scores_arr, axis=0)
    valid_scores_arr = np.array(valid_scores_list)
    valid_scores_mean = np.mean(valid_scores_arr, axis=0)
    best_param_index = np.argmin(valid_scores_mean)
    print('## training RMSE:\n', train_scores_arr)
    print('## train RMSE mean:\n', train_scores_mean)
    print('## validation RMSE:\n', valid_scores_arr)
    print('## validation RMSE mean:\n', valid_scores_mean)
    # print('## best param: ', search_list[best_param_index])

    # plot box to compare params
    train_scores_df = pd.DataFrame(train_scores_list, columns=search_list)
    ax = train_scores_df.boxplot(column=search_list)
    ax.set_xlabel('params')
    ax.set_ylabel('RMSE')
    plt.title("Training RMSE with different %s" % param)
    plt.show()
    valid_scores_arr_df = pd.DataFrame(valid_scores_list, columns=search_list)
    ax = valid_scores_arr_df.boxplot(column=search_list)
    ax.set_xlabel('params')
    ax.set_ylabel('RMSE')
    plt.title("Validation RMSE with different %s" % param)
    plt.show()
    return best_param_index


def model_train(scaler, trainX_org, trainY_org, validX_org, validY_org, epochs, time_step, batch_size, neurons_num, flag):
    model, history = lstm(trainX_org, trainY_org, validX_org, validY_org, epochs, time_step, batch_size, neurons_num, flag)
    # predict
    trainPredict = model.predict(trainX_org)
    validPredict = model.predict(validX_org)
    # invert predictions
    trainPredict = repeat_inverse_trans(scaler, trainPredict, 1)
    trainY = repeat_inverse_trans(scaler, trainY_org.reshape(trainY_org.shape[0], 1), 0)
    validPredict = repeat_inverse_trans(scaler, validPredict, 1)
    validY = repeat_inverse_trans(scaler, validY_org.reshape(validY_org.shape[0], 1), 0)
    mse = tf.keras.losses.MeanSquaredError()
    train_score_rmse = np.sqrt(mse(trainY, trainPredict).numpy())
    valid_score_rmse = np.sqrt(mse(validY, validPredict).numpy())
    print("### training RMSE: ", train_score_rmse)
    print("### validation RMSE: ", valid_score_rmse)
    return trainPredict, validPredict, train_score_rmse, valid_score_rmse, history, model


def model_test(model, scaler_test, dataset_test, time_steps):
    testX, testY = load.create_dataset(dataset_test, time_steps[0])
    testPredict = model.predict(testX)
    testPredict = repeat_inverse_trans(scaler_test, testPredict, 1)
    mse = tf.keras.losses.MeanSquaredError()
    test_score_rmse = np.sqrt(mse(testY, testPredict).numpy())
    print("## test RMSE: ", test_score_rmse)
    return testPredict
