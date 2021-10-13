# Author: Wuli Zuo, a1785343
# Date: 2021-09-30


import load
import lstm
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import plot
import tensorflow as tf
import time


def main(title, signal, repeat, epochs, look_backs, batch_sizes, neuron_nums, flag):
    # load and process data
    dataset_train = load.init_dataset('../../data/a3/Google_Stock_Price_Train.csv')
    scaler, dataset_train = load.preprocessing(dataset_train)
    train, valid = load.separate_data(dataset_train)

    # compare and select
    if signal != 0:
        search_list = []
        param = ''
        if len(look_backs) > 1:
            search_list = look_backs
            param = 'look_backs'
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
            for look_back in look_backs:
                for batch_size in batch_sizes:
                    for neuron_num in neuron_nums:
                        trainX, trainY = load.create_dataset(train, look_back)
                        validX, validY = load.create_dataset(valid, look_back)
                        print(trainX.shape)
                        print(trainY.shape)
                        trainPredict, validPredict, train_scores, valid_scores, history, model = lstm.train(scaler,
                                                                                                     trainX, trainY,
                                                                                                     validX, validY,
                                                                                                     epochs, look_back,
                                                                                                     batch_size, neuron_num, flag)
                        train_scores_list_i.append(train_scores)
                        valid_scores_list_i.append(valid_scores)
            train_scores_list.append(train_scores_list_i)
            valid_scores_list.append(valid_scores_list_i)
        print("## train_scores:")
        print(np.array(train_scores_list))
        print("## valid_scores:")
        print(np.array(valid_scores_list))

        # plot
        train_scores_df = pd.DataFrame(train_scores_list, columns=search_list)
        train_scores_df.boxplot(column=search_list)
        plt.title("Training RMSE with different %s" % param)
        plt.show()
        valid_scores_arr_df = pd.DataFrame(valid_scores_list, columns=search_list)
        valid_scores_arr_df.boxplot(column=search_list)
        plt.title("Validation RMSE with different %s" % param)
        plt.show()

    # train with selected params
    else:
        trainX, trainY = load.create_dataset(train, look_backs[0])
        validX, validY = load.create_dataset(valid, look_backs[0])
        trainPredict, validPredict, train_scores, valid_scores, history, model = lstm.train(scaler, trainX, trainY, validX,
                                                                                     validY, epochs, look_backs[0],
                                                                                     batch_sizes[0], neuron_nums[0], flag)
        if flag < 2:
            plot.plot_train_prediction(scaler, dataset_train, trainPredict, validPredict, look_backs[0], title)
        else:  # test
            dataset_test = load.init_dataset('../../data/a3/Google_Stock_Price_Test.csv')
            test_scaler, test_dataset = load.preprocessing(dataset_test)
            testX, testY = load.create_dataset(test_dataset, look_backs[0])
            testPredict = model.predict(testX)
            testPredict = lstm.repeat_inverse_trans(scaler, testPredict, 1)
            mse = tf.keras.losses.MeanSquaredError()
            test_score_mse = np.sqrt(mse(testY, testPredict).numpy())
            print("## test score: ", test_score_mse)
            plot.plot_test_prediction(scaler, test_dataset, testPredict, title)


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
np.random.seed(42)

repeats = 10
epochs = 100

print('\n# 1.1 Tune look_back')
look_backs = [1, 2, 3]
batch_sizes = [1]
neuron_nums = [8]

start = time.time()
# entry point
main('title', 1, repeats, epochs, look_backs, batch_sizes, neuron_nums, 0)
end = time.time()
print('## Running time: %.4f s' % (end - start))

print('\n# 1.2 best look_backs = 1')
look_backs = [1]
start = time.time()
# entry point
title = "Tune: look_back=" + str(look_backs[0]) + ", batch_size=" + str(batch_sizes[0]) + ", neurons=" + str(neuron_nums[0])
main(title, 0, repeats, epochs, look_backs, batch_sizes, neuron_nums, 0)
end = time.time()
print('## Running time: %.4f s' % (end - start))


print('\n# 2.1 tune batch_size')

batch_sizes = [1, 2, 4, 8]
start = time.time()
# entry point
main('title', 2, repeats, epochs, look_backs, batch_sizes, neuron_nums, 0)
end = time.time()
print('## Running time: %.4f s' % (end - start))

print('\n# 2.2 best batch_size = 8')
batch_sizes = [8]
start = time.time()
# entry point
title = "Tune: look_back=" + str(look_backs[0]) + ", batch_size=" + str(batch_sizes[0]) + ", neurons=" + str(neuron_nums[0])
main(title, 0, repeats, epochs, look_backs, batch_sizes, neuron_nums, 0)
end = time.time()
print('## Running time: %.4f s' % (end - start))

print('\n# 3.1 tune neurons_num')

neuron_nums = [1, 2, 4, 8]
start = time.time()
# entry point
main('title', 3, repeats, epochs, look_backs, batch_sizes, neuron_nums, 0)
end = time.time()
print('## Running time: %.4f s' % (end - start))

print('\n# 3.2 best neuron_num = 8')
neuron_nums = [8]
start = time.time()
# entry point
title = "Tune: look_back=" + str(look_backs[0]) + ", batch_size=" + str(batch_sizes[0]) + ", neurons=" + str(neuron_nums[0])
main(title, 0, repeats, epochs, look_backs, batch_sizes, neuron_nums, 0)
end = time.time()
print('## Running time: %.4f s' % (end - start))


print('\n# 4 Bidirectional')
start = time.time()
# entry point
title = "Bidirectional: look_back=" + str(look_backs[0]) + ", batch_size=" + str(batch_sizes[0]) + ", neurons=" + str(neuron_nums[0])
main(title, 0, repeats, epochs, look_backs, batch_sizes, neuron_nums, 1)
end = time.time()
print('## Running time: %.4f s' % (end - start))


print('\n# 5 Test')
start = time.time()
# entry point
title = "Test: look_back=" + str(look_backs[0]) + ", batch_size=" + str(batch_sizes[0]) + ", neurons=" + str(neuron_nums[0])
main(title, 0, repeats, epochs, look_backs, batch_sizes, neuron_nums, 2)
end = time.time()
print('## Running time: %.4f s' % (end - start))
