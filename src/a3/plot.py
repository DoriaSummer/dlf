# Author: Wuli Zuo, a1785343
# Date: 2021-09-30


import matplotlib.pyplot as plt
import numpy


def plot_loss(title, history):
    plt.figure()
    plt.title('Training vs validation loss\n%s' % title)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.plot(history.history['loss'][5:], color='b')
    plt.plot(history.history['val_loss'][5:], color='orange')
    plt.legend(['training loss', 'validation loss'])
    plt.show()

def plot_train_prediction(scaler, dataset, trainPredict, validPredict, time_step, title):
    # shift training predictions for plotting
    trainPredictPlot = numpy.empty_like(dataset[:, 0])
    trainPredictPlot[:] = numpy.nan
    trainPredictPlot[time_step:len(trainPredict) + time_step] = trainPredict
    # shift validation predictions for plotting
    validPredictPlot = numpy.empty_like(dataset[:, 0])
    validPredictPlot[:] = numpy.nan
    validPredictPlot[len(trainPredict) + (time_step * 2) + 1:len(dataset) - 1] = validPredict
    # plot baseline and predictions
    plt.figure(figsize=(8, 5))
    plt.title(title)
    plt.xlabel('days')
    plt.ylabel('y')
    start = len(trainPredict)
    # print(start)
    plt.plot(scaler.inverse_transform(dataset)[:, 0][start:], linewidth=1, label='baseline')
    # plt.plot(trainPredictPlot, linewidth=1, label='training predictions')
    plt.plot(validPredictPlot[start:], linewidth=1, label='validation predictions')
    plt.legend()
    plt.show()


def plot_test_prediction(scaler, test_dataset, testPredict_simple, testPredict_bidire, title):
    # plt.figure(figsize=(8, 5))
    plt.title(title)
    plt.xlabel('days')
    plt.ylabel('y')
    plt.plot(scaler.inverse_transform(test_dataset[:len(testPredict_simple)])[:, 0], linewidth=1, label='baseline')
    plt.plot(testPredict_simple, color='orange', linewidth=1, label='simple LSTM test predictions')
    plt.plot(testPredict_bidire, color='g', linewidth=1, label='bidirectional LSTM test predictions')
    plt.legend()
    plt.show()
