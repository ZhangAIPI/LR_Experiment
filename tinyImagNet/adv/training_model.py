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

device = torch.device('cuda:1')


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


# define nn
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # my network is composed of only affine layers
        self.fc1 = nn.Linear(h * h, 500)
        self.fc2 = nn.Linear(500, 200)
        self.fc3 = nn.Linear(200, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# DATA LOADERS
def flat_trans(x):
    x.resize_(h * h)  # 图片缩放
    return x


if __name__ == '__main__':
    label_file = '../../data/IMagenet-master/tiny-imagenet-200/label.txt'
    train_path = '../../data/IMagenet-master/tiny-imagenet-200/train_LR'
    test_path = '../../data/IMagenet-master/tiny-imagenet-200/test_LR'
    h = 32
    batch_size = 64
    transform = transforms.Compose([
        transforms.Resize((h, h)),
        transforms.Grayscale(),
        transforms.ToTensor(),
    ])
    net = Net()
    net = net.to(device)
    # define the loss function
    SoftmaxWithXent = nn.CrossEntropyLoss()
    # define optimization algorithm
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-04)
    # load the data
    trainLoader = TinyImageNetLoader(train_path, label_file, transform, batch_size=batch_size)
    testLoader = TinyImageNetLoader(test_path, label_file, transform, batch_size=batch_size)

    # strat to train the nn
    for epoch in range(500):
        print("Epoch: {}".format(epoch))
        running_loss = 0.0
        n = 0
        N = 0
        # import ipdb; ipdb.set_trace()
        for data in tqdm(trainLoader):
            # get the inputs
            inputs, labels = data
            n = len(inputs)
            N += n
            inputs = inputs.to(device)
            labels = labels.to(device)
            inputs = inputs.reshape((-1, h * h))
            # wrap them in a variable
            # zero the gradients
            optimizer.zero_grad()
            # print(inputs.shape)

            # forward + loss + backward
            outputs = net(inputs)  # forward pass
            loss = SoftmaxWithXent(outputs, labels)  # compute softmax -> loss
            loss.backward()  # get gradients on params
            optimizer.step()  # SGD update

            # print statistics
            running_loss += loss.cpu().detach().numpy()
        running_loss = running_loss / N
        print('Epoch: {} | Loss: {}'.format(epoch, running_loss))
        # print("Finished Training")
        # TEST
        correct = 0.0
        total = 0
        for data in testLoader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            images = images.reshape((-1, h * h))
            outputs = net(Variable(images.float()))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
        print("test set Accuracy: {}".format(float(correct) / total))


    print("Dumping weights to disk")
    weights_dict = {}
    # save the weights of nn with protocol=2
    for param in list(net.named_parameters()):
        print("Serializing Param", param[0])
        weights_dict[param[0]] = param[1].cpu().detach()
    with open("weights_size=48_protocol=2.pkl", "wb") as f:
        import pickle

        pickle.dump(weights_dict, f, protocol=2)

    weights_dict2 = {}
    # save the weights of nn with protocol=3
    for param in list(net.named_parameters()):
        print("Serializing Param", param[0])
        weights_dict2[param[0]] = param[1].cpu().detach()
    with open("weights_size=48_protocol=3.pkl", "wb") as f2:
        pickle.dump(weights_dict2, f2, protocol=3)
    print("Finished dumping to disk..")
