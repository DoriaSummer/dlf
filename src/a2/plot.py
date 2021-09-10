# Author: Wuli Zuo, a1785343
# Date: 2021-09-03


import matplotlib.pyplot as plt
import numpy as np


# function to plot results
def plot_history(history):
    plt.figure(figsize=(8, 4))
    plt.title('Training and validation accuracy & loss')
    plt.xlabel('Epoch')
    x = np.arange(0, len(history.history['accuracy']))
    plt.plot(x, history.history['accuracy'], color='r')
    plt.plot(x, history.history['loss'], color='g')
    plt.plot(x, history.history['val_accuracy'], color='b')
    plt.plot(x, history.history['val_loss'], color='orange')
    plt.legend(['Training accuracy', 'Training loss', 'Validation accuracy', 'Validation loss'])
    plt.show()


def plot_history_resN(history):
    # Plot the results (shifting validation curves appropriately)
    plt.figure(figsize=(8, 4))
    plt.title('Training and validation accuracy & loss')
    plt.xlabel('Epoch')
    x = np.arange(len(history.history['acc']))
    plt.plot(x, history.history['acc'], color='r')
    plt.plot(x, history.history['loss'], color='g')
    plt.plot(x, history.history['val_acc'], color='b')
    plt.plot(x, history.history['val_loss'], color='orange')
    plt.legend(['Training accuracy', 'Training loss', 'Validation accuracy', 'Validation loss'])
    plt.show()


# function to plot learning rate
def plot_lr(history):
    plt.figure(figsize=(8, 4))
    plt.title('Reduce LR on Plateau')
    plt.xlabel('Epoch')
    plt.ylabel('Learning rate', color='b')
    plt.plot(history.epoch, history.history["lr"], "bo-")
    plt.tick_params('y', colors='b')
    ax = plt.gca().twinx()
    ax.plot(history.epoch, history.history["val_loss"], "r^-")
    ax.set_ylabel('Validation loss', color='r')
    ax.tick_params('y', colors='r')
    plt.show()
