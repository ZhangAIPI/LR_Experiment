import torch
from torch.autograd import Variable
from torch.autograd import Function
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from abc import ABCMeta, abstractmethod
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from torch.utils.data import Dataset
import time
import pickle as pkl
from torch import nn
import torch.optim as optim
import torch.nn.init
import math
import  torchvision
from olds import dataLoader

device = torch.device('cuda:4')


def LoadMNIST(root, transform, batch_size, download=True):
    train_dataset = datasets.MNIST(root=root, train=True, transform=transform, download=download)
    test_dataset = datasets.MNIST(root=root, train=False, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 100)
        self.fc2 = nn.Linear(100,50)
        self.fc3 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == '__main__':
    model = Network().to(device)
    transform = transforms.Compose([transforms.ToTensor()])
    batch_size = 128
    epoches = 10
    loss = 0.
    data_dir = './'
    tranform = transforms.Compose([transforms.ToTensor()])
    model_path = 'BP.pkl'
    train_dataset = torchvision.datasets.FashionMNIST(data_dir, download=True, train=True, transform=tranform)
    val_dataset = torchvision.datasets.FashionMNIST(root=data_dir, download=True, train=False, transform=tranform)
    # train_dataloader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True, num_workers=0)
    # test_dataloader = DataLoader(dataset=val_dataset, batch_size=128, num_workers=0, shuffle=False)
    train_dataloader, test_dataloader = dataLoader.LoadMNIST('../data/MNIST', tranform, batch_size, False)
    print('数据准备完成!')
    trainLoss = 0.
    testLoss = 0.
    learning_rate = 1e-2
    start_epoch = 0
    SoftmaxWithXent = nn.CrossEntropyLoss()
    # define optimization algorithm
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-04)
    print('epoch to run:{} learning rate:{}'.format(epoches, learning_rate))
    for epoch in range(start_epoch, start_epoch + epoches):
        train_N = 0.
        train_n = 0.
        trainLoss = 0.
        model.train()
        for batch, [trainX, trainY] in enumerate(tqdm(train_dataloader, ncols=10)):
            train_n = len(trainX)
            train_N += train_n
            trainX = trainX.to(device)
            trainY = trainY.to(device).long()
            trainX = trainX.reshape(len(trainX), -1)
            optimizer.zero_grad()
            predY = model(trainX)
            loss = SoftmaxWithXent(predY, trainY)

            loss.backward()  # get gradients on params
            optimizer.step()  # SGD update
            trainLoss += loss.detach().cpu().numpy()
        trainLoss /= train_N
        test_N = 0.
        testLoss = 0.
        correct = 0.
        model.eval()
        for batch, [testX, testY] in enumerate(tqdm(test_dataloader, ncols=10)):
            test_n = len(testX)
            test_N += test_n
            testX = testX.to(device)
            testY = testY.to(device).long()
            testX = testX.reshape(len(testX), -1)
            predY = model(testX)
            loss = SoftmaxWithXent(predY, testY)
            testLoss += loss.detach().cpu().numpy()
            _, predicted = torch.max(predY.data, 1)
            correct += (predicted == testY).sum()
        testLoss /= test_N
        acc = correct / test_N
        print('epoch:{} train loss:{} testloss:{} acc:{}'.format(epoch, trainLoss, testLoss, acc))
    if not os.path.exists('./mnist_model'):
        os.mkdir('mnist_model')
    torch.save(model.state_dict(), './mnist_model/blackBP.pth')
    print('模型已经保存')
