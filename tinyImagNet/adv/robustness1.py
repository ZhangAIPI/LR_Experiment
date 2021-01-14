import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torchvision import transforms
from tqdm import *
from skimage import transform, data
import pickle
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import pickle as pkl

device = torch.device('cpu')


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
        repeatX = torch.cat([X] * repeat, dim=0)
        one_shape = tuple(repeatX.shape[:-1]) + (1,)
        one = torch.ones(size=one_shape, dtype=torch.float)
        return torch.cat([repeatX, one], dim=-1)
    else:
        one = torch.ones(tuple(X.shape[:-1]) + (1,), dtype=torch.float)
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
        self.batchSize = None
        self.momentum = 0
        self.momentum_beta = 0.9

    def get_params(self):
        return self.w

    def forward(self, x, train=False, BP=False):
        self.input = x
        self.batchSize = len(x)
        if BP:
            # print(self.input.shape)
            # print(self.w.shape)
            self.output = self.input.matmul(self.w)
            if self.activation:
                self.output = self.activation(self.output)
            return self.output
        else:
            if not train:
                self.output = self.input.matmul(self.w)
                if self.activation:
                    self.output = self.activation(self.output)
                return self.output
            else:
                self.noise = torch.randn([len(self.input), self.n_output]) * self.sigma
                self.noise = self.noise.to(device)
                self.output = self.input.matmul(self.w) + self.noise
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
            else:
                print('尚未注册！\n')
                exit()
            batch_size = len(self.input)
            grad = self.input.T.matmul(eta)
            self.bp_grad = grad / batch_size
            return torch.einsum('ij,kj->ik', eta, self.w)
        else:
            term = self.input * target[:, np.newaxis]
            batch_grad = torch.einsum('ni, nj->ij', term, self.noise)
            batch_grad /= self.sigma ** 2
            batch_grad /= len(self.input)  # torch.mean(batch_grad, dim=0)
            self.lr_grad = batch_grad
            return target

    def update_params(self, learning_rate, BP=True, method='sgd', weight_decay=1e-4):
        if BP:
            grad = self.bp_grad
        else:
            grad = self.lr_grad
        # grad += weight_decay * self.w / self.batchSize
        if method == 'sgd':
            self.t += 1
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
            self.t += 1
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
            layer.update_params(learning_rate, BP, method=method, weight_decay=weight_decay)

    def saveModel(self, path):
        weights_list = []
        for layer in self.layers:
            weights_list.append(layer.w.cpu().detach().numpy())
        with open(path, 'wb') as file:
            pkl.dump(weights_list, file)

    def load_state_dict(self, path):
        with open(path, 'rb') as file:
            weight_list = pkl.load(file)
        for index, layer in enumerate(self.layers):
            layer.w = torch.from_numpy(weight_list[index]).to(device)


if __name__ == '__main__':
    net_arc = [300, 100, 10]
    # 3 layers 100 50 10
    # 4 layers 100 50 30 10
    h = 32
    net_act = [Activation.relu, Activation.relu, Activation.relu, Activation.softmax]
    net_acts = [Activation.relu, Activation.relu, Activation.relu, Activation.softmax]
    bp_model = Network(h * h + 1, [100, 10],
                       [Activation.relu, Activation.softmax], 1.0)
    bp_with_noise_model = Network(h * h + 1, [100, 50, 10], [Activation.relu, Activation.relu, Activation.softmax], 1.0)
    bp_plus = Network(h * h + 1, [100, 50, 10], [Activation.relu, Activation.relu, Activation.softmax], 1.0)
    bp_plus_z = Network(h * h + 1, [100, 50, 10], [Activation.relu, Activation.relu, Activation.softmax], 1.0)
    bp_with_noise_model_z = Network(h * h + 1, [100, 50, 10],
                                    [Activation.relu, Activation.relu, Activation.softmax],
                                    1.0)
    bp_static = '../model/BP_1.pkl'
    bp_with_noise_static = '../model/BP_2.pkl'
    bp_plus_static = '../model/LR_1.pkl'
    bp_with_noise_static_z = '../model/LR_2.pkl'
    bp_plus_static_z = '../model/LR_3.pkl'
    bp_model.load_state_dict(bp_static)
    bp_with_noise_model.load_state_dict(bp_with_noise_static)
    bp_plus.load_state_dict(bp_plus_static)
    bp_with_noise_model_z.load_state_dict(bp_with_noise_static_z)
    bp_plus_z.load_state_dict(bp_plus_static_z)

    import config

    noise = ['gaussian', 'impulse', 'glass_blur', 'contrast']
    strength = [1, 2, 3, 4, 5]
    epsilon = config.epsilon
    for i in range(len(noise)):
        avg_bp_plus_z_base_acc = 0.
        avg_bp_with_noise_z_base_acc = 0.
        avg_bp_with_noise_base_acc = 0.
        avg_bp_base_acc = 0.
        avg_bp_plus_base_acc = 0.
        avg_bp_with_noise_adv_acc = 0.
        avg_bp_adv_acc = 0.
        avg_bp_plus_adv_acc = 0.
        avg_bp_plus_z_adv_acc = 0.
        avg_bp_with_noise_z_adv_acc = 0.
        for j in range(len(strength)):
            with open("Generate_adversarial_sample_epsilon=" + str(epsilon) + "by_{}_{}.pkl".format(noise[i],
                                                                                                    strength[j]),
                      "rb") as f:
                adv_data_dict2 = pickle.load(f)
                xs_clean = adv_data_dict2['xs']
                y_true_clean = adv_data_dict2['y_trues']
                y_preds = adv_data_dict2['y_preds']
                adv_x = adv_data_dict2['noises']
                y_preds_adversarial = adv_data_dict2['y_preds_adversarial']
            print("load adv samples Successful!!")

            bp_plus_z_base_acc = 0.
            bp_with_noise_z_base_acc = 0.
            bp_with_noise_base_acc = 0.
            bp_base_acc = 0.
            bp_plus_base_acc = 0.
            bp_with_noise_adv_acc = 0.
            bp_adv_acc = 0.
            bp_plus_adv_acc = 0.
            bp_plus_z_adv_acc = 0.
            bp_with_noise_z_adv_acc = 0.
            N = 0.
            for img_index in tqdm(range(len(adv_x))):
                N += 1
                adv_img = adv_x[img_index].reshape(-1, h * h)
                clean_img = xs_clean[img_index].reshape(-1, h * h)
                label = y_true_clean[img_index][0]
                adv_img = torch.from_numpy(adv_img)
                clean_img = torch.from_numpy(clean_img)
                adv_img = KMeansRepeatX(adv_img, 1, False)
                clean_img = KMeansRepeatX(clean_img, 1, False)
                advpred_based_bp_with_noise_z = bp_with_noise_model_z.forward(adv_img, train=False)
                advpred_based_bp_with_noise = bp_with_noise_model.forward(adv_img, train=False)
                advpred_based_bp = bp_model.forward(adv_img, train=False)
                advpred_based_bp_plus = bp_plus.forward(adv_img, train=False)
                advpred_based_bp_plus_z = bp_plus_z.forward(adv_img, train=False)
                advpred_based_bp_with_noise_z = torch.argmax(advpred_based_bp_with_noise_z, dim=-1)[0]
                advpred_based_bp_with_noise = torch.argmax(advpred_based_bp_with_noise, dim=-1)[0]
                advpred_based_bp = torch.argmax(advpred_based_bp, dim=-1)[0]
                advpred_based_bp_plus = torch.argmax(advpred_based_bp_plus, dim=-1)[0]
                advpred_based_bp_plus_z = torch.argmax(advpred_based_bp_plus_z, dim=-1)[0]

                cleanpred_based_bp_with_noise_z = bp_with_noise_model_z.forward(clean_img, train=False)
                cleanpred_based_bp_with_noise = bp_with_noise_model.forward(clean_img, train=False)
                cleanpred_based_bp = bp_model.forward(clean_img, train=False)
                cleanpred_based_bp_plus = bp_plus.forward(clean_img, train=False)
                cleanpred_based_bp_plus_z = bp_plus_z.forward(clean_img, train=False)
                cleanpred_based_bp_with_noise_z = torch.argmax(cleanpred_based_bp_with_noise_z, dim=-1)[0]
                cleanpred_based_bp_with_noise = torch.argmax(cleanpred_based_bp_with_noise, dim=-1)[0]
                cleanpred_based_bp = torch.argmax(cleanpred_based_bp, dim=-1)[0]
                cleanpred_based_bp_plus = torch.argmax(cleanpred_based_bp_plus, dim=-1)[0]
                cleanpred_based_bp_plus_z = torch.argmax(cleanpred_based_bp_plus_z, dim=-1)[0]

                if advpred_based_bp_plus_z == label:
                    bp_plus_z_adv_acc += 1
                if advpred_based_bp_with_noise_z == label:
                    bp_with_noise_z_adv_acc += 1
                if advpred_based_bp_with_noise == label:
                    bp_with_noise_adv_acc += 1
                if advpred_based_bp == label:
                    bp_adv_acc += 1
                if advpred_based_bp_plus == label:
                    bp_plus_adv_acc += 1
                if cleanpred_based_bp_with_noise == label:
                    bp_with_noise_base_acc += 1
                if cleanpred_based_bp == label:
                    bp_base_acc += 1
                if cleanpred_based_bp_plus == label:
                    bp_plus_base_acc += 1
                if cleanpred_based_bp_plus_z == label:
                    bp_plus_z_base_acc += 1
                if cleanpred_based_bp_with_noise_z == label:
                    bp_with_noise_z_base_acc += 1

            bp_with_noise_z_base_acc /= N
            bp_with_noise_z_adv_acc /= N
            bp_with_noise_base_acc /= N
            bp_with_noise_adv_acc /= N
            bp_base_acc /= N
            bp_adv_acc /= N
            bp_plus_base_acc /= N
            bp_plus_adv_acc /= N
            bp_plus_z_base_acc /= N
            bp_plus_z_adv_acc /= N

            # print('NOISE:{} STRENGTH:{}'.format(noise[i], strength[j]))
            # print('bp: base:{}  adv:{}'.format(bp_base_acc, bp_adv_acc))
            # print('bp+: base:{}  adv:{}'.format(bp_plus_base_acc, bp_plus_adv_acc))
            # print('bp_with_noise: base:{} adv:{}'.format(bp_with_noise_base_acc, bp_with_noise_adv_acc))
            # print('bp+z: base:{}  adv:{}'.format(bp_plus_z_base_acc, bp_plus_z_adv_acc))
            # print('bp_with_noisez: base:{} adv:{}'.format(bp_with_noise_z_base_acc, bp_with_noise_z_adv_acc))

            avg_bp_plus_z_base_acc += bp_plus_z_base_acc
            avg_bp_with_noise_z_base_acc += bp_with_noise_z_base_acc
            avg_bp_with_noise_base_acc += bp_with_noise_base_acc
            avg_bp_base_acc += bp_base_acc
            avg_bp_plus_base_acc += bp_plus_base_acc
            avg_bp_with_noise_adv_acc += bp_with_noise_adv_acc
            avg_bp_adv_acc += bp_adv_acc
            avg_bp_plus_adv_acc += bp_plus_adv_acc
            avg_bp_plus_z_adv_acc += bp_plus_z_adv_acc
            avg_bp_with_noise_z_adv_acc += bp_with_noise_z_adv_acc

        avg_bp_with_noise_z_base_acc /= len(strength)
        avg_bp_with_noise_z_adv_acc /= len(strength)
        avg_bp_with_noise_base_acc /= len(strength)
        avg_bp_with_noise_adv_acc /= len(strength)
        avg_bp_base_acc /= len(strength)
        avg_bp_adv_acc /= len(strength)
        avg_bp_plus_base_acc /= len(strength)
        avg_bp_plus_adv_acc /= len(strength)
        avg_bp_plus_z_base_acc /= len(strength)
        avg_bp_plus_z_adv_acc /= len(strength)

        print('NOISE:{}'.format(noise[i]))
        print('bp: base:{}  adv:{}'.format(avg_bp_base_acc, avg_bp_adv_acc))
        print('bp_with_noise: base:{} adv:{}'.format(avg_bp_with_noise_base_acc, avg_bp_with_noise_adv_acc))
        print('bp+: base:{}  adv:{}'.format(avg_bp_plus_base_acc, avg_bp_plus_adv_acc))
        print('bp_with_noisez: base:{} adv:{}'.format(avg_bp_with_noise_z_base_acc, avg_bp_with_noise_z_adv_acc))
        print('bp+z: base:{}  adv:{}'.format(avg_bp_plus_z_base_acc, avg_bp_plus_z_adv_acc))
