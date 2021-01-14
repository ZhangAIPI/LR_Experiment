import numpy as np
from olds import dataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import os
import pickle as pkl
from torch.utils.data import DataLoader
import torch
import time

device = torch.device('cuda:2')

def evalHot(y, pred):
    """
    评估效果
    :param y:真实值的独热编码
    :param pred: 预测值的输出
    :return: 正确的个数
    """
    _y = torch.argmax(y, dim=-1)
    _pred = torch.argmax(pred, dim=-1)
    N = np.sum((_y == _pred).cpu().numpy())
    return N


def KMeansRepeatX(X, repeat, train=True):
    """
    :param X:Raw data \\in R^{batch_size X n_dim}
    :param repeat:重复的次数、采样数
    :return: 加了偏置项和重复数据的样本 维度[batch_size,repeat,n_dum+1]
    """
    if train:
        X = torch.reshape(X, [X.shape[0], -1]).to(device)
        repeatX = torch.stack([X] * repeat).permute((1, 0, 2)).to(device)
        one_shape = tuple(repeatX.shape[:-1]) + (1,)
        one = torch.ones(size=one_shape, dtype=torch.float).to(device)
        return torch.cat([repeatX, one], dim=-1)
    else:
        X = torch.reshape(X, [X.shape[0], -1]).to(device)
        one = torch.ones(tuple(X.shape[:-1]) + (1,), dtype=torch.float).to(device)
        return torch.cat([X, one], dim=-1)


def OneHotLabel(Y, n):
    """
    :param Y:序列型标签
    :param n: 标签数目
    :return: 标签的独热编码
    """
    y = torch.zeros([len(Y), n]).to(device)
    y[torch.arange(0, len(Y)), Y] = 1
    return y


def KMeansRepeatY(Y, repeat):
    # print(Y.shape)
    repeatY = torch.stack([Y] * repeat).permute((1, 0, 2))
    return repeatY


class Activation:
    """
    包含激活函数
    """

    @staticmethod
    def logistic(z):
        return 1 / (1 + torch.exp(-z))

    @staticmethod
    def softmax(z):
        stable_exps = torch.exp(z)
        return stable_exps / stable_exps.sum(dim=-1, keepdim=True)

    @staticmethod
    def threshold(z):
        z[z < 0] = 0
        return torch.sign(z)

    @staticmethod
    def relu(z):
        z[z < 0] = 0
        return z



def CELoss(Y, T):
    """
    :param Y:模型输出
    :param T: 样本标签
    :return: 交叉熵损失
    """
    return -(T*torch.log(Y)).sum(dim=-1)


class KMeansLRLinearLayer(object):
    """
    采用均值方法稳定梯度估计方差的线性层
    """

    def __init__(self, n_input, n_output, sigma, activation):
        """
        :param n_input:输入维度
        :param n_output: 输出维度
        :param seed: 随机种子
        :param sigma: 方差
        :param activation: 激活函数
        """
        self.w = torch.randn(size=[n_input, n_output]).to(device)  # 多出来的是bias
        self.w *= (2 / self.w.shape[0] ** 0.5)
        self.sigma = sigma
        self.n_input = n_input
        self.n_output = n_output
        self.input = None
        self.output = None
        self.noise = None
        self.activation = activation
        self.accuate_grad = None

    def get_params(self):
        return self.w

    def forward(self, repeatX, train=True, BP=False):
        """
        :param train:
        :param repeatX:已经经过重复化处理/加bias项的数据
        :return: 预测值Y
        """
        if BP:
            self.input = repeatX
            self.output = self.input.matmul(self.w)
            if self.activation:
                self.output = self.activation(self.output)
            return self.output
        if not train:
            self.noise = 0
        else:
            self.noise = torch.randn(*(repeatX.shape[:-1] + (self.w.shape[-1],))).to(device) * self.sigma

        self.input = repeatX
        # print(self.input.shape)
        # print(self.w.shape)
        # print(self.noise.shape)
        z = self.input.matmul(self.w) + self.noise
        inference_z = self.input.matmul(self.w)
        if self.activation:
            self.output = self.activation(z)
            return self.activation(z)
        else:
            self.output = z
            return z

    def get_grad(self, loss):
        term = self.input * loss[:, :, np.newaxis]
        batch_grad = torch.einsum('nki, nkj->nkij', term, self.noise)
        batch_grad /= self.sigma ** 2
        return batch_grad

    def update_params(self, loss, learning_rate):
        batch_grad = self.get_grad(loss)
        mean_grad = torch.mean(batch_grad, dim=(0, 1))
        self.w -= learning_rate * mean_grad

    def backward(self, eta, target):
        if self.activation == Activation.softmax:
            #说明是最后一层，此时eta无用
            eta = self.output - target
        elif self.activation == Activation.logistic:
            eta = self.output * (1 - self.output) * eta
        else:
            print('not loaded\n')
            exit()
        batch_size = len(self.input) * len(self.input[0])
        # print(self.input.shape)
        # print(eta.shape)
        # exit()
        grad = self.input.transpose(1, 2).matmul(eta)
        # print(grad.shape)
        # print(grad.shape)
        # exit()
        grad = torch.mean(grad, dim=(0, 1))
        self.accuate_grad = grad
        return eta.matmul(self.w.transpose(0, 1))

    def update_by_backward(self, learning_rate):
        self.w -= learning_rate * self.accuate_grad



    def load_weight(self, w):
        self.w = w


class KMeansGLRNetwork(object):
    def __init__(self, n_input, units_per_layers: list, activation_per_layers: list, sigma_list: list):
        assert len(units_per_layers) == len(activation_per_layers)
        assert len(units_per_layers) == len(sigma_list)
        self.n_layers = len(units_per_layers)
        self.params = [(n_input, units_per_layers[0], sigma_list[0], activation_per_layers[0])]
        for i in range(self.n_layers - 1):
            self.params.append(
                (units_per_layers[i], units_per_layers[i + 1], sigma_list[i + 1],
                 activation_per_layers[i + 1]))
        self.layers = [KMeansLRLinearLayer(*self.params[i]) for i in range(self.n_layers)]
        print('模型层数为:{}'.format(len(self.layers)))

    def forward(self, X, train=True, BP=False):
        z = X
        for layer in self.layers:
            z = layer.forward(z, train, BP)
        return z

    def update_params(self, loss, learning_rate):
        for layer in self.layers:
            layer.update_params(loss, learning_rate)


    def backward(self, target):
        eta = 0
        for layer_index in range(n_layers-1, -1, -1):
            eta = self.layers[layer_index].backward(eta, target)

    def update_params_BP(self, learning_rate):
        for layer in self.layers:
            layer.update_by_backward(learning_rate)

    def load_weights(self, w_list):
        for layer_index in range(len(self.layers)):
            self.layers[layer_index].load_weight(w_list[layer_index])
        print('模型加载参数成功！\n')

    def save_weights(self, path, epoch, learning_rate):
        w_list = []
        for layer in self.layers:
            w_list.append(layer.get_params())
        with open(path, 'wb') as file:
            pkl.dump([epoch, learning_rate, w_list], file)
        print('模型保存参数成功！\n')


if __name__ == "__main__":
    mnist = 'MNIST'
    cifar = 'CIFAR'
    task = mnist
    train_loss = []
    test_loss = []
    acc = []
    time_list = []
    if task == mnist:
        print('run mnist')
        batch_size = 128
        repeat_n = 1
        n_input = 28 * 28 + 1
        n_output = 10
        n_layers = 3
        sigma = 1.0
        seed = None
        epoches = 30
        learning_rate = 1e-1
        loss = 0.
        num_classes = 10
        reuse = False
        BP_train = True
        if BP_train:
            print('BP train on ')
        transform = transforms.Compose([transforms.ToTensor()])
        model_path = 'multiK.pth'
        logfile = 'log_multiK.txt'
        start_epoch = 0
        if not reuse or not os.path.exists(logfile):
            writeFile = open(logfile, 'w')
        else:
            writeFile = open(logfile, 'a')
        train_dataloader, test_dataloader = dataLoader.LoadMNIST('../../data/MNIST', transform, batch_size, False)
        net = KMeansGLRNetwork(n_input, [100, 50, 10], [Activation.logistic, Activation.logistic, Activation.softmax], [sigma] * n_layers)
        if not os.path.exists('./models_mnist'):
            os.mkdir('models_mnist')
        if os.path.exists(os.path.join('./models_mnist', model_path)):
            if reuse:
                with open(os.path.join('./models_mnist', model_path), 'rb') as file:
                    [start_epoch, learning_rate, w_list] = pkl.load(file)
                    net.load_weights(w_list)
            else:
                print('从零训练！\n')
        else:
            print('从零训练！\n')
        # train_img, train_label = loadMNIST_RAM(train_dataloader, repeat_n)
        print('数据准备完成!')
        trainLoss = 0.
        testLoss = 0.
        print('epoch to run:{} learning rate:{}'.format(epoches, learning_rate))
        start = time.time()
        for epoch in range(start_epoch, start_epoch + epoches):
            loss = 0.
            nbatch = 0.
            N = 0.
            n = 0.
            trainLoss = 0.
            for batch, [trainX, trainY] in enumerate(tqdm(train_dataloader, ncols=10)):
                # break
                nbatch += 1
                trainX = trainX.to(device)
                trainY = trainY.to(device)
                trainY = OneHotLabel(trainY, num_classes)
                batch_train_repeatX, batch_train_repeatY = KMeansRepeatX(trainX, repeat_n), KMeansRepeatY(trainY,
                                                                                                          repeat_n)
                pre = net.forward(batch_train_repeatX, BP_train)

                loss = CELoss(pre, batch_train_repeatY)
                trainLoss += torch.mean(loss).detach().cpu().numpy()
                if not BP_train:
                    net.update_params(loss, learning_rate)
                else:
                    net.backward(batch_train_repeatY)
                    net.update_params_BP(learning_rate)
            trainLoss /= nbatch
            train_loss.append(trainLoss)
            # trainAcc = n / N
            print('train epoch:{} loss:{}'.format(epoch, trainLoss))
            if ((epoch + 1) % 10 == 0):
                learning_rate *= 0.8
                print('学习率衰减至{}'.format(learning_rate))
            loss = 0.
            N = 0.
            n = 0.
            nbatch = 0.
            for batch, [testX, testY] in enumerate(tqdm(test_dataloader, ncols=10)):
                nbatch += 1
                testX = KMeansRepeatX(testX, 1, False)
                testY = OneHotLabel(testY, num_classes)

                pre = net.forward(testX, train=False)
                testLoss += torch.mean(CELoss(pre, testY)).detach().cpu().numpy()
                N += len(testX)
                n += evalHot(testY, pre)
            testLoss /= nbatch
            test_loss.append(testLoss)
            testAcc = n / N
            acc.append(testAcc)
            print('test epoch:{} loss:{} acc:{}'.format(epoch, testLoss, n / N))
            time_list.append(time.time()-start)
            # net.save_weights(os.path.join('./models_mnist', model_path), epoch, learning_rate)
    elif task == cifar:
        print('run cifar')
        batch_size = 128
        repeat_n = 10
        n_input = 3 * 32 * 32 + 1
        n_output = 10
        n_layers = 3
        sigma = 1.0
        seed = None
        epoches = 100
        learning_rate = 1e-1
        loss = 0.
        num_classes = 10
        reuse = False
        transform = transforms.Compose(
            [transforms.ToTensor()])
        model_path = 'multiK_cifar.pth'
        logfile = 'log_multiK_cifar.txt'
        start_epoch = 0
        if not reuse or not os.path.exists(logfile):
            writeFile = open(logfile, 'w')
        else:
            writeFile = open(logfile, 'a')
        train_dataset = datasets.CIFAR10(root='../../EBP/', train=True, transform=transform, download=True)
        test_dataset = datasets.CIFAR10(root='../../EBP/', train=False, transform=transform, download=True)
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
        net = KMeansGLRNetwork(n_input, [100, 50, 10], [Activation.logistic, Activation.logistic, Activation.softmax], [sigma] * n_layers)
        if not os.path.exists('./models_cifar'):
            os.mkdir('models_cifar')
        if os.path.exists(os.path.join('./models_cifar', model_path)):
            if reuse:
                with open(os.path.join('./models_cifar', model_path), 'rb') as file:
                    [start_epoch, learning_rate, w_list] = pkl.load(file)
                    net.load_weights(w_list)
            else:
                print('从零训练！\n')
        else:
            print('从零训练！\n')
        # train_img, train_label = loadMNIST_RAM(train_dataloader, repeat_n)
        print('数据准备完成!')
        trainLoss = 0.
        testLoss = 0.
        print('cifar epoch to run:{} learning rate:{}'.format(epoches, learning_rate))
        for epoch in range(start_epoch, start_epoch + epoches):
            loss = 0.
            nbatch = 0.
            N = 0.
            n = 0.
            trainLoss = 0.
            for batch, [trainX, trainY] in enumerate(tqdm(train_dataloader, ncols=10)):
                # break
                nbatch += 1
                trainY = OneHotLabel(trainY, num_classes)
                if n>1:
                    batch_train_repeatX, batch_train_repeatY = KMeansRepeatX(trainX, repeat_n), KMeansRepeatY(trainY,
                                                                                                          repeat_n)
                else:
                    batch_train_repeatY = KMeansRepeatY(trainY, repeat_n)
                    batch_train_repeatX = trainX.reshape(len(trainY), 1, 755).to(device)
                pre = net.forward(batch_train_repeatX)
                loss = CELoss(pre, batch_train_repeatY)
                trainLoss += torch.mean(loss)
                net.update_params(loss, learning_rate)
            trainLoss /= nbatch

            print('train epoch:{} loss:{}'.format(epoch, trainLoss))
            if ((epoch + 1) % 10 == 0):
                learning_rate *= 0.8
                print('学习率衰减至{}'.format(learning_rate))
            loss = 0.
            N = 0.
            n = 0.
            nbatch = 0.
            for batch, [testX, testY] in enumerate(tqdm(test_dataloader, ncols=10)):
                nbatch += 1
                testX = KMeansRepeatX(testX, 1, False)
                testY = OneHotLabel(testY, num_classes)

                pre = net.forward(testX, train=False)
                testLoss += torch.mean(CELoss(pre, testY))
                N += len(testX)
                n += evalHot(testY, pre)
            testLoss /= nbatch
            testAcc = n / N
            print('test epoch:{} loss:{} acc:{}'.format(epoch, testLoss, n / N))
            writeFile.write(
                'epoch:{} trainLoss:{} testLoss:{} testACC:{}\n'.format(epoch, trainLoss, testLoss,
                                                                        testAcc))
            net.save_weights(os.path.join('./models_mnist', model_path), epoch, learning_rate)
    else:
        pass
    print('train_loss:{}\n test_loss:{}\n acc:{}'.format(train_loss, test_loss, acc))
    print('time:{}'.format(time_list))