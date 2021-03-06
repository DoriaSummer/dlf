# Author: Wuli Zuo, a1785343
# Date: 2021-05-20


import matplotlib.pyplot as plt


# plot training and test accuracy
def plot_acc(acc1, acc2, acc_index_list, class1, class2, algo_name, learning_rate):
    plt.title('%s: Accuracy against iteration, learning rate = %.4f' % (algo_name, learning_rate))
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.ylim([0, 1])
    y1 = acc1
    y2 = acc2

    plt.plot(acc_index_list, y1, color='b', label='%s accuracy' % class1)
    plt.plot(acc_index_list, y2, color='r', label='%s accuracy' % class2)
    plt.legend(loc='lower left')

    plt.savefig('../../output/accuracy_%.4f.png' % learning_rate)
    plt.show()
    plt.close()
