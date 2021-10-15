# Author: Wuli Zuo, a1785343
# Date: 2021-09-30


import load
import lstm
import matplotlib.pyplot as plt
import numpy as np
import os
import plot
import time


def main(title, epochs, train, valid, scaler_train, dataset_train,
         time_steps, batch_sizes, neuron_nums, flag):
    trainX, trainY = load.create_dataset(train, time_steps[0])
    validX, validY = load.create_dataset(valid, time_steps[0])
    trainPredict, validPredict, train_scores, valid_scores, history, model =\
        lstm.model_train(scaler_train, trainX, trainY, validX, validY,
                         epochs, time_steps[0], batch_sizes[0], neuron_nums[0], flag)
    plot.plot_loss(title, history)
    plot.plot_train_prediction(scaler_train, dataset_train, trainPredict, validPredict, time_steps[0], title)
    return model


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
np.random.seed(1)

# load and process data
print('\n# 0 Load data')
dataset_train = load.init_dataset('../../data/a3/Google_Stock_Price_Train.csv')
scaler_train, dataset_train = load.preprocessing(dataset_train)
train, valid = load.separate_data(dataset_train)
dataset_test = load.init_dataset('../../data/a3/Google_Stock_Price_Test.csv')
scaler_test, dataset_test = load.preprocessing(dataset_test)


# plot data
plt.title('Scaled training data')
plt.xlabel('days')
plt.ylabel('y')
plt.plot(dataset_train[:, 0], linewidth=1, label='scaled training data')
plt.legend()
plt.show()

plt.title('Scaled test data')
plt.xlabel('days')
plt.ylabel('y')
plt.plot(dataset_test[:, 0], linewidth=1, label='scaled test data')
plt.legend()
plt.show()

# experiment setting
repeats = 10
epochs = 50

"""
# trial
time_steps = [1]
batch_sizes = [1]
neuron_nums = [8]
start = time.time()
# entry point
title = "Trial: time_step=" + str(time_steps[0]) + ", batch_size=" + str(batch_sizes[0]) + ", neurons=" + str(neuron_nums[0])
model0 = main(title, repeats, epochs, train, valid, scaler_train, dataset_train,
     time_steps, batch_sizes, neuron_nums, 0)
end = time.time()
print('## Running time: %.4f s' % (end - start))

start = time.time()
# entry point
title = "Trial Test: time_step=" + str(time_steps[0]) + ", batch_size=" + str(batch_sizes[0]) + ", neurons=" + str(neuron_nums[0])
testPredict = lstm.model_test(model0, scaler_test, dataset_test, time_steps)
plot.plot_test_prediction(scaler_test, dataset_test, testPredict, title)
end = time.time()
print('## Running time: %.4f s' % (end - start))
"""

print('\n# 1.1 Tune time_step')
time_steps = [1, 2, 4, 8]
batch_sizes = [1]
neuron_nums = [8]

start = time.time()
# entry point
best = lstm.select_param(repeats, epochs, train, valid, scaler_train, time_steps, batch_sizes, neuron_nums, 0)
end = time.time()
print('## Running time: %.4f s' % (end - start))
time_steps = [time_steps[best]]
print('\n# 1.2 Best time_steps = ', time_steps[0])
#time_steps = [4]
start = time.time()
# entry point
title = "Tune: time_step=" + str(time_steps[0]) + ", batch_size=" + str(batch_sizes[0]) + ", neurons=" + str(neuron_nums[0])
model1 = main(title, epochs, train, valid, scaler_train, dataset_train,
     time_steps, batch_sizes, neuron_nums, 0)
end = time.time()
print('## Running time: %.4f s' % (end - start))


print('\n# 2.1 Tune batch_size')

batch_sizes = [1, 4, 8, 16]
start = time.time()
# entry point
best = lstm.select_param(repeats, epochs, train, valid, scaler_train, time_steps, batch_sizes, neuron_nums, 0)
end = time.time()
print('## Running time: %.4f s' % (end - start))
batch_sizes = [batch_sizes[best]]
print('\n# 2.2 best batch_size = ', batch_sizes[0])
# batch_sizes = [1]
start = time.time()
# entry point
title = "Tune: time_step=" + str(time_steps[0]) + ", batch_size=" + str(batch_sizes[0]) + ", neurons=" + str(neuron_nums[0])
model2 = main(title, epochs, train, valid, scaler_train, dataset_train,
     time_steps, batch_sizes, neuron_nums, 0)
end = time.time()
print('## Running time: %.4f s' % (end - start))

print('\n# 3.1 Tune neurons_num')

neuron_nums = [1, 2, 4, 8]
start = time.time()
# entry point
best = lstm.select_param(repeats, epochs, train, valid, scaler_train, time_steps, batch_sizes, neuron_nums, 0)
end = time.time()
print('## Running time: %.4f s' % (end - start))
neuron_nums = [neuron_nums[best]]
print('\n# 3.2 Best neuron_num = ', neuron_nums[0])
# neuron_nums = [8]
start = time.time()
# entry point
title = "Tune: time_step=" + str(time_steps[0]) + ", batch_size=" + str(batch_sizes[0]) + ", neurons=" + str(neuron_nums[0])
model3 = main(title, epochs, train, valid, scaler_train, dataset_train,
     time_steps, batch_sizes, neuron_nums, 0)
end = time.time()
print('## Running time: %.4f s' % (end - start))

print('\n# 4 Bidirectional')
start = time.time()
# entry point
title = "Bidirectional: time_step=" + str(time_steps[0]) + ", batch_size=" + str(batch_sizes[0]) + ", neurons=" + str(neuron_nums[0])
model4 = main(title, epochs, train, valid, scaler_train, dataset_train,
     time_steps, batch_sizes, neuron_nums, 1)
end = time.time()
print('## Running time: %.4f s' % (end - start))


print('\n# 5 Test')
start = time.time()
# entry point
title = "Test: time_step=" + str(time_steps[0]) + ", batch_size=" + str(batch_sizes[0]) + ", neurons=" + str(neuron_nums[0])
testPredict_simple = lstm.model_test(model3, scaler_test, dataset_test, time_steps)
testPredict_bidire = lstm.model_test(model4, scaler_test, dataset_test, time_steps)
plot.plot_test_prediction(scaler_test, dataset_test, testPredict_simple, testPredict_bidire, title)
end = time.time()
print('## Running time: %.4f s' % (end - start))
