import numpy as np
from torchvision import datasets, transforms
from tqdm import tqdm
import os
import pickle as pkl
import torch
import pickle
import time
from torch.utils.data import DataLoader, Dataset
from PIL import Image


class FastDataSet(Dataset):
    def __init__(self, data_path, label_path, transform4img):
        [train_img, train_label] = loadTinyImageNet(data_path, label_path,
                                                    transform4img)
        self.img = train_img
        self.label = train_label
        self.len = len(self.label)

    def __getitem__(self, item):
        return self.img[item], self.label[item]

    def __len__(self):
        return self.len


def loadTinyImageNet(data_path, label_file, transform4img):
    label_list = []
    train_img = []
    train_label = []
    with open(label_file, 'r') as winds:
        for line in winds:
            label_list.append(line.strip('\n'))
    for index, label in enumerate(label_list):
        train_label_imgs_path = os.path.join(data_path, label)
        train_img_file_list = os.listdir(train_label_imgs_path)
        for path in train_img_file_list:
            path = os.path.join(train_label_imgs_path, path)
            pic = Image.open(path)
            pic = transform4img(pic)
            train_img.append(pic)
            train_label.append(index)
    train_img = [img.numpy() for img in train_img]  # torch.from_numpy(np.array(train_img))
    train_img = torch.Tensor(train_img)
    train_label = torch.Tensor(np.array(train_label)).long()
    return train_img, train_label


def TinyImageNetLoader(data_path, label_path, transform4img, batch_size, shuffle=True):
    dataset = FastDataSet(data_path, label_path, transform4img)
    return DataLoader(dataset, batch_size, shuffle)


device = torch.device('cuda:4')


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
    X = X.reshape(len(X), -1)
    if train:
        repeatX = torch.cat([X] * repeat, dim=0).to(device)
        one_shape = tuple(repeatX.shape[:-1]) + (1,)
        one = torch.ones(size=one_shape, dtype=torch.float).to(device)
        return torch.cat([repeatX, one], dim=-1)
    else:
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
    repeatY = torch.cat([Y] * repeat, dim=0)
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
    return -(T * torch.log(Y)).sum(dim=-1)


class Layer:
    def __init__(self, n_input, n_output, sigma, activation):
        """
        :param n_input:输入维度
        :param n_output: 输出维度
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
        self.z = None
        self.noise = None
        self.activation = activation
        print('该层输入:{} 输出:{}\n激活函数:{}'.format(self.n_input, self.n_output, self.activation))
        self.bp_grad = None
        self.lr_grad = None
        self.adam_beta1 = 0.9
        self.adam_beta2 = 0.999
        self.adam_last_mt = 0
        self.adam_last_vt = 0
        self.t = 0
        self.epsilon = 1e-8
        self.nadam_alpha = 0.9
        self.nadam_beta = 0.999
        self.nadam_s = 0
        self.nadam_r = 0
        self.momentum = 0
        self.momentum_beta = 0.9

    def get_params(self):
        return self.w

    def forward(self, x, train=False, BP=False):
        self.input = x
        if BP:
            # print(self.input.shape)
            # print(self.w.shape)
            self.output = self.input.matmul(self.w)
            self.z = self.output
            if self.activation:
                self.output = self.activation(self.output)
            return self.output
        else:
            if not train:
                self.output = self.input.matmul(self.w)
                self.z = self.output
                if self.activation:
                    self.output = self.activation(self.output)
                return self.output
            else:
                self.noise = torch.randn([len(self.input), self.n_output]) * self.sigma
                self.noise = self.noise.to(device)
                self.output = self.input.matmul(self.w) + self.noise
                self.z = self.output
                if self.activation:
                    self.output = self.activation(self.output)
                return self.output

    def backward(self, target, BP=True):
        """
        :param target: BP训练模式下，target是残差；LR训练模式下，target是损失
        :param BP: 是否为BP训练
        :return: BP训练模式下，返回残差；LR训练模式下，返回损失
        """
        if BP:
            eta = target
            if self.activation == Activation.softmax:
                eta = self.output - eta
            elif self.activation == Activation.logistic:
                eta = self.output * (1 - self.output) * eta
            elif self.activation == Activation.relu:
                eta[self.z < 0] = 0
            else:
                print('尚未注册！\n')
                exit()
            batch_size = len(self.input)
            grad = self.input.T.matmul(eta)
            self.bp_grad = grad / batch_size
            return torch.einsum('ij,kj->ik', eta, self.w)
        else:
            term = self.input * target[:, np.newaxis]
            batch_grad = torch.einsum('ni, nj->nij', term, self.noise)
            batch_grad /= self.sigma ** 2
            batch_grad = torch.mean(batch_grad, dim=0)
            self.lr_grad = batch_grad
            return target

    def update_params(self, learning_rate, BP=True, method='sgd', weight_decay=1e-4):
        if BP:
            grad = self.bp_grad
        else:
            grad = self.lr_grad
        grad += weight_decay * self.w
        if method == 'sgd':
            self.w -= learning_rate * grad
            return
        elif method == 'adam':
            self.t += 1
            self.adam_last_mt = self.adam_beta1 * self.adam_last_mt + (1 - self.adam_beta1) * grad
            self.adam_last_vt = self.adam_beta2 * self.adam_last_vt + (1 - self.adam_beta2) * grad ** 2
            adam_mt_cap = self.adam_last_mt / (1 - self.adam_beta1 ** self.t)
            adam_vt_cap = self.adam_last_vt / (1 - self.adam_beta2 ** self.t)
            self.w -= learning_rate * adam_mt_cap / (torch.sqrt(adam_vt_cap) + self.epsilon)
            return
        elif method == 'nadam':
            self.nadam_s = self.nadam_alpha * self.nadam_s + (1 - self.nadam_alpha) * grad
            self.nadam_r = self.nadam_beta * self.nadam_r + (1 - self.nadam_beta) * grad ** 2
            sqrt_term = 1 - self.nadam_beta ** self.t
            sqrt_term = sqrt_term ** 0.5
            lr = learning_rate * sqrt_term / (1 - self.nadam_alpha ** self.t)
            delta = lr * (self.nadam_alpha * self.nadam_s + (1 - self.nadam_alpha) * grad) / torch.sqrt(self.nadam_r)
            self.w -= delta
        elif method == 'momentum':
            self.t += 1
            if self.t < 100:
                self.w -= learning_rate * grad
                return
            self.momentum = self.momentum_beta * self.momentum - learning_rate * grad
            self.w += self.momentum
            return
        else:
            print('优化方法尚未注册')
            exit()


class Network(object):
    def __init__(self, n_input, units_per_layers: list, activation_per_layers: list, sigma):
        assert len(units_per_layers) == len(activation_per_layers)
        self.n_layers = len(units_per_layers)
        self.params = [(n_input, units_per_layers[0], sigma, activation_per_layers[0])]
        for i in range(self.n_layers - 1):
            self.params.append(
                (units_per_layers[i], units_per_layers[i + 1], sigma,
                 activation_per_layers[i + 1]))
        self.layers = [Layer(*self.params[i]) for i in range(self.n_layers)]
        print('模型层数为:{} 各层及对应的激活函数为:{}'.format(len(self.layers),
                                               [(units_per_layers[i], activation_per_layers[i]) for i in
                                                range(self.n_layers)]))

    def forward(self, X, train=True, BP=False):
        z = X
        for layer in self.layers:
            # print(BP)
            z = layer.forward(z, train, BP)
        return z

    def backward(self, target, BP=True):
        """
        :param target:BP训练方式下target是标签 LR训练方式下target是损失
        :param BP: 是否为BP模式
        :return: None
        """
        if BP:
            for i in range(self.n_layers - 1, -1, -1):
                target = self.layers[i].backward(target, BP)
        else:
            for layer in self.layers:
                layer.backward(target, BP)

    def update_params(self, learning_rate, BP=True, method='sgd', weight_decay=1e-4):
        for layer in self.layers:
            layer.update_params(learning_rate, BP, method, weight_decay)
    def saveModel(self, path):
        weights_list = []
        for layer in self.layers:
            weights_list.append(layer.w.cpu().detach().numpy())
        with open(path, 'wb') as file:
            pkl.dump(weights_list, file)

    def loadModel(self, path):
        with open(path, 'rb') as file:
            weight_list = pkl.load(file)
        for index, layer in enumerate(self.layers):
            layer.w = weight_list[index].to(device)

if __name__ == "__main__":
    label_file = '../data/IMagenet-master/tiny-imagenet-200/label.txt'
    train_path = '../data/IMagenet-master/tiny-imagenet-200/train_LR'
    test_path = '../data/IMagenet-master/tiny-imagenet-200/test_LR'
    h = 32
    transform = transforms.Compose([
        transforms.Resize((h, h)),
        transforms.Grayscale(),
        transforms.ToTensor(),
    ])
    train_loss = []
    test_loss = []
    acc = []
    time_list = []
    batch_size = 128
    n_input = h * h + 1
    n_output = 10
    sigma = 1.0
    epoches = 1000
    loss = 0.
    reuse = False
    BP_train = True
    method = 'adam'
    repeat_n = 1
    net_arc = [300, 100, 50, n_output]
    learning_rate = 1e-3
    start_epoch = 0
    trainLoader = TinyImageNetLoader(train_path, label_file, transform, batch_size=batch_size)
    testLoader = TinyImageNetLoader(test_path, label_file, transform, batch_size=batch_size)
    net_act = [Activation.relu for i in range(len(net_arc) - 1)]
    net_act.append(Activation.softmax)
    assert len(net_arc) == len(net_act)
    n_layers = len(net_arc)
    net = Network(n_input, net_arc, net_act, sigma)
    trainLoss = 0.
    testLoss = 0.
    print('epoch to run:{} learning rate:{}'.format(epoches, learning_rate))
    start = time.time()
    print('模型信息:\narc:{}\nact:{}\nK:{}'.format(net_arc, net_act, repeat_n))
    best = 0.
    for epoch in range(start_epoch, start_epoch + epoches):
        loss = 0.
        nbatch = 0.
        N = 0.
        n = 0.
        trainLoss = 0.
        train_estimation_relative_error = 0
        for batch, [trainX, trainY] in enumerate(tqdm(trainLoader, ncols=10)):
            # break

            nbatch += 1
            trainX = trainX.to(device)
            trainY = trainY.to(device)
            trainY = OneHotLabel(trainY, n_output)
            batch_train_repeatX, batch_train_repeatY = KMeansRepeatX(trainX, repeat_n), KMeansRepeatY(trainY,
                                                                                                      repeat_n)
            pre = net.forward(batch_train_repeatX, train=True, BP=BP_train)

            loss = CELoss(pre, batch_train_repeatY)
            trainLoss += torch.mean(loss).detach().cpu().numpy()
            if BP_train:
                net.backward(batch_train_repeatY, BP_train)
                net.update_params(learning_rate, BP_train, method=method)
            else:
                net.backward(loss, BP_train)
                net.update_params(learning_rate, BP_train)
        trainLoss /= nbatch
        train_loss.append(trainLoss)
        print('train epoch:{} loss:{}'.format(epoch, trainLoss))
        if ((epoch + 1) % 20 == 0):
            learning_rate *= 0.8
            print('学习率衰减至{}'.format(learning_rate))
        loss = 0.
        N = 0.
        n = 0.
        nbatch = 0.
        test_estimation_relative_error = 0
        for batch, [testX, testY] in enumerate(tqdm(testLoader, ncols=10)):
            nbatch += 1
            testX = testX.to(device)
            testY = testY.to(device)
            testX = KMeansRepeatX(testX, 1, False)
            testY = OneHotLabel(testY, n_output)

            pre = net.forward(testX, train=False)
            testLoss += torch.mean(CELoss(pre, testY)).detach().cpu().numpy()
            N += len(testX)
            n += evalHot(testY, pre)
        testLoss /= nbatch
        test_loss.append(testLoss)
        testAcc = n / N

        acc.append(testAcc)
        print('test epoch:{} loss:{} acc:{}'.format(epoch, testLoss, n / N))
        time_list.append(time.time() - start)
        # net.save_weights(os.path.join('./models_mnist', model_path), epoch, learning_rate)

    print('train_loss:{}\n test_loss:{}\n acc:{}'.format(train_loss, test_loss, acc))
    print('time:{}'.format(time_list))
    with open('BP_2.pkl'.format(n_layers, repeat_n, h), 'wb') as file:
        pkl.dump(
            [train_loss, test_loss, acc], file)
    net.saveModel('model/BP_2.pkl'.format(n_layers, repeat_n, h))