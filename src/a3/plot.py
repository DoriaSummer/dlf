# Author: Wuli Zuo, a1785343
# Date: 2021-09-30


import matplotlib.pyplot as plt
import numpy


def plot_train_prediction(scaler, dataset, trainPredict, validPredict, look_back, title):
    # shift training predictions for plotting
    trainPredictPlot = numpy.empty_like(dataset[:, 0])
    trainPredictPlot[:] = numpy.nan
    trainPredictPlot[look_back:len(trainPredict) + look_back] = trainPredict
    # shift validation predictions for plotting
    validPredictPlot = numpy.empty_like(dataset[:, 0])
    validPredictPlot[:] = numpy.nan
    validPredictPlot[len(trainPredict) + (look_back * 2) + 1:len(dataset) - 1] = validPredict
    # plot baseline and predictions
    plt.title(title)
    plt.plot(scaler.inverse_transform(dataset)[:, 0], linewidth=1, label='baseline')
    plt.plot(trainPredictPlot, linewidth=1, label='training predictions')
    plt.plot(validPredictPlot, linewidth=1, label='validation predictions')
    plt.legend()
    plt.show()


def plot_test_prediction(scaler, test_dataset, testPredict, title):
    plt.title(title)
    plt.plot(scaler.inverse_transform(test_dataset[:len(testPredict)])[:, 0], linewidth=1, label='baseline')
    plt.plot(testPredict, linewidth=1, label='test predictions')
    plt.legend()
    plt.show()
