# Author: Wuli Zuo, a1785343
# Date: 2021-08-14 05:10


import torch
import torch.nn as nn
import torch.nn.functional as F


# define the network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hidden = nn.Linear(8, 8)
        self.out = nn.Linear(8, 1)

    def forward(self, x):
        x = self.hidden(x)
        x = F.relu(x)
        x = self.out(x)

        return x


def train(X_train, y_train, X_test, y_test):
    # Step 1: construct input tensor

    x = torch.tensor(X_train).float()
    y = torch.tensor(y_train).float()
    xt = torch.tensor(X_test).float()
    yt = torch.tensor(y_test).float()

    # Step 2: create network
    model = Net()

    # Step 3: set the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1, betas=(0.9, 0.999))  # use Adam optimal approach
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    # Step 4: train
    loss_func = nn.MSELoss()

    # train stage 0
    for epoch in range(1000):
        # forward pass
        trainPred = model(x)
        # compute loss
        loss = loss_func(trainPred.squeeze(), y)
        # update network
        optimizer.zero_grad()  # clear gradient
        loss.backward()  # propagate loss backward
        optimizer.step()  # get the optimizer started

    # predict
    trainPred = model(x)
    y = torch.reshape(y, (y.shape[0], 1))
    correct = y*trainPred
    mask = correct > 0
    acc_train = mask.sum().item() / correct.shape[0]

    testPred = model(xt)
    yt = torch.reshape(yt, (yt.shape[0], 1))
    correct = yt * testPred
    mask = correct > 0
    acc_test = mask.sum().item() / correct.shape[0]

    print('training accuracy = %f, test accuracy = %f' % (acc_train, acc_test))
