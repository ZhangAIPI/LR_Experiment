import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torchvision import transforms
from tqdm import *
import matplotlib.pyplot as plt
import pickle
from skimage import transform, data
import random
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image

os.environ["CUDA_VISIBLE_DEVICES"] = '1'


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


# DEFINE NETWORK
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # my network is composed of only affine layers
        self.fc1 = nn.Linear(32 * 32, 300)
        self.fc2 = nn.Linear(300, 100)
        self.fc3 = nn.Linear(100, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def classify(self, x):
        outputs = self.forward(x)
        outputs = outputs / torch.norm(outputs)
        max_val, max_idx = torch.max(outputs, 1)
        return int(max_idx.data.numpy()), float(max_val.data.numpy())


if __name__ == '__main__':
    net = Net()
    with open("weights_size=48_protocol=2.pkl", "rb") as f:
        weights_dict = pickle.load(f)
    for param in net.named_parameters():
        if param[0] in weights_dict.keys():
            print("Copying: ", param[0])
            param[1].data = weights_dict[param[0]].data
    print("Weights Loaded!")
    SoftmaxWithXent = nn.CrossEntropyLoss()
    label_file = '../../data/IMagenet-master/tiny-imagenet-200/label.txt'
    train_path = '../../data/IMagenet-master/tiny-imagenet-200/train_LR'
    test_path = '../../data/IMagenet-master/tiny-imagenet-200/test_LR'
    batch_size = 128
    h = 32
    transform = transforms.Compose([
        transforms.Resize((h, h)),
        transforms.Grayscale(),
        transforms.ToTensor(),
    ])
    test_dataloader = TinyImageNetLoader(test_path, label_file, transform, batch_size=batch_size)
    xs = []
    y_trues = []
    for data in tqdm(test_dataloader):
        inputs, labels = data
        inputs = inputs.reshape((len(inputs), -1))
        if len(xs) == 0:
            xs = inputs
            y_trues = labels
        else:
            xs = torch.cat([xs, inputs], dim=0)
            y_trues = torch.cat([y_trues, labels], dim=0)
    xs = np.array(xs)
    y_trues = np.array(y_trues).reshape(-1)

    noises = []
    y_preds = []
    y_preds_adversarial = []
    totalMisclassifications = 0
    xs_clean = []
    y_trues_clean = []
    num_adv = 0
    import config

    epsilon = config.epsilon
    for x, y_true in tqdm(zip(xs, y_trues)):

        # Wrap x as a variable
        xs_clean.append(np.array(x))
        xv = torch.Tensor(x.reshape(1, h * h))
        xv = nn.Parameter(xv, requires_grad=True)
        # xv = nn.Parameter(torch.FloatTensor(x.reshape(1, 28 * 28)), requires_grad=True)
        y_true = Variable(torch.LongTensor(np.array([y_true])), requires_grad=False)
        method = optim.LBFGS([xv], lr=5e-2)
        # Classification before Adv
        y_pred = np.argmax(net(xv).data.numpy())

        # Generate Adversarial Image
        def closure():
            method.zero_grad()
            output = net(xv)
            loss = -SoftmaxWithXent(output, y_true)
            loss.backward()
            return loss
        method.step(closure)
        # method = optim.LBFGS(list(xv), lr=1e-1)
        # Add perturbation
        # x_grad = torch.sign(x.grad.data)
        x_adversarial = torch.clamp(xv.data, 0, 1)

        # Classification after optimization
        y_pred_adversarial = np.argmax(net(Variable(x_adversarial)).data.numpy())
        # print "Before: {} | after: {}".format(y_pred, y_pred_adversarial)

        # print "Y_TRUE: {} | Y_PRED: {}".format(_y_true, y_pred)
        if y_true.data.numpy() != y_pred:
            print("WARNING: MISCLASSIFICATION ERROR")
            totalMisclassifications += 1
        else:
            if y_pred_adversarial != y_pred:
                num_adv += 1
        y_preds.append(y_pred)
        y_preds_adversarial.append(y_pred_adversarial)
        noises.append(x_adversarial.numpy())
        y_trues_clean.append(y_true.data.numpy())

    print("Total totalMisclassifications :{}/{} ".format(totalMisclassifications, len(xs)))  # 1221/1797
    print("the amount of adv samples is : {}".format(num_adv))  # 576

    print("Successful!!")

    with open("Generate_adversarial_sample_epsilon=" + str(epsilon) + "by_LBFGS.pkl", "wb") as f:
        adv_data_dict2 = {
            "xs": xs_clean,
            "y_trues": y_trues_clean,
            "y_preds": y_preds,
            "noises": noises,
            "y_preds_adversarial": y_preds_adversarial
        }
        pickle.dump(adv_data_dict2, f, protocol=3)
    print("Successful!!")