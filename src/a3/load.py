# Author: Wuli Zuo, a1785343
# Date: 2021-09-30


import numpy
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler


def init_dataset(path):
    dataframe = read_csv(path, usecols=[1, 2, 3, 4, 5], engine='python', thousands=',')
    dataset = dataframe.values
    dataset = dataset.astype('float32')
    return dataset


def preprocessing(dataset):
    # normalise the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)
    return scaler, dataset


def separate_data(dataset):
    train_size = int(len(dataset) * 0.7)
    train, valid = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
    return train, valid


def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i + time_step), :5]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return numpy.array(dataX), numpy.array(dataY)
