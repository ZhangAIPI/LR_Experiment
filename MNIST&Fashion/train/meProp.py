import numpy
import torch
from torchvision import datasets, transforms
import torch
from torch.autograd import Function
import math
import torch
import torch.nn as nn
from collections import OrderedDict
import sys
import time
from statistics import mean

import torch
import torch.cuda
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

from torch.nn.parameter import Parameter

import os

os.environ['CUDA_VISIBLE_DEVICES'] = '6'
import sys
from argparse import ArgumentParser


class PartDataset(torch.utils.data.Dataset):
    '''
    Partial Dataset:
        Extract the examples from the given dataset,
        starting from the offset.
        Stop if reach the length.
    '''

    def __init__(self, dataset, offset, length):
        self.dataset = dataset
        self.offset = offset
        self.length = length
        super(PartDataset, self).__init__()

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        return self.dataset[i + self.offset]


def get_mnist(datapath='../data/MNIST', download=True):
    '''
    The MNIST dataset in PyTorch does not have a development set, and has its own format.
    We use the first 5000 examples from the training dataset as the development dataset. (the same with TensorFlow)
    Assuming 'datapath/processed/training.pt' and 'datapath/processed/test.pt' exist, if download is set to False.
    '''
    trn = datasets.MNIST(
        datapath,
        train=True,
        download=download,
        transform=transforms.ToTensor())
    dev = PartDataset(trn, 0, 5000)
    trnn = PartDataset(trn, 5000, 55000)
    tst = datasets.MNIST(
        datapath, train=False, transform=transforms.ToTensor())
    return trnn, dev, tst


def get_artificial_dataset(nsample, ninfeature, noutfeature):
    '''
    Generate a synthetic dataset.
    '''
    data = torch.randn(nsample, ninfeature).cuda()
    target = torch.LongTensor(
        numpy.random.randint(noutfeature, size=(nsample, 1))).cuda()
    return torch.utils.data.TensorDataset(data, target)


class linearUnified(Function):
    '''
    linear function with meProp, unified top-k across minibatch
    y = f(x, w, b) = xw + b
    '''

    def __init__(self, k):
        '''
        k is top-k in the backprop of meprop
        '''
        super(linearUnified, self).__init__()
        self.k = k

    @staticmethod
    def forward(ctx, *args, **kwargs):
        '''
        forward propagation
        x should be of size [minibatch, input feature]
        w should be of size [input feature, output feature]
        b should be of size [output feature]

        This is slightly different from the default linear function in PyTorch.
        In that implementation, w is of size [output feature, input feature].
        We find that that implementation is slower in both forward and backward propagation on our devices.
        '''
        x, w, b, k = args
        ctx.save_for_backward(x, w, b, k)
        y = x.new(x.size(0), w.size(1))
        y.addmm_(0, 1, x, w)
        ctx.add_buffer = x.new(x.size(0)).fill_(1)
        y.addr_(ctx.add_buffer, b)
        return y

    @staticmethod
    def backward(ctx, *grad_outputs):
        '''
        backprop with meprop
        if k is invalid (<=0 or > output feature), no top-k selection is applied
        '''
        dy = grad_outputs[0]
        x, w, b, k = ctx.saved_tensors
        dx = dw = db = None

        if k > 0 and k < w.size(1):  # backprop with top-k selection
            _, inds = dy.abs().sum(0).topk(
                int(k))  # get top-k across examples in magnitude
            inds = inds.view(-1)  # flat
            pdy = dy.index_select(
                -1, inds
            )  # get the top-k values (k column) from dy and form a smaller dy matrix

            # compute the gradients of x, w, and b, using the smaller dy matrix
            if ctx.needs_input_grad[0]:
                dx = torch.mm(pdy, w.index_select(-1, inds).t_())
            if ctx.needs_input_grad[1]:
                dw = w.new(w.size()).zero_().index_copy_(
                    -1, inds, torch.mm(x.t(), pdy))
            if ctx.needs_input_grad[2]:
                db = torch.mv(dy.t(), ctx.add_buffer)
        else:  # backprop without top-k selection
            if ctx.needs_input_grad[0]:
                dx = torch.mm(dy, w.t())
            if ctx.needs_input_grad[1]:
                dw = torch.mm(x.t(), dy)
            if ctx.needs_input_grad[2]:
                db = torch.mv(dy.t(), ctx.add_buffer)

        return dx, dw, db, None


class linear(Function):
    '''
    linear function with meProp, top-k selection with respect to each example in minibatch
    y = f(x, w, b) = xw + b
    '''

    def __init__(self, k, sparse=True):
        '''
        k is top-k in the backprop of meprop
        '''
        super(linear, self).__init__()
        self.k = k
        self.sparse = sparse

    @staticmethod
    def forward(ctx, *args, **kwargs):
        '''
        forward propagation
        x should be of size [minibatch, input feature]
        w should be of size [input feature, output feature]
        b should be of size [output feature]

        This is slightly different from the default linear function in PyTorch.
        In that implementation, w is of size [output feature, input feature].
        We find that that implementation is slower in both forward and backward propagation on our devices.
        '''
        x, w, b, k, sparse = args
        ctx.save_for_backward(x, w, b, )
        y = x.new(x.size(0), w.size(1))
        y.addmm_(0, 1, x, w)
        ctx.add_buffer = x.new(x.size(0)).fill_(1)
        y.addr_(ctx.add_buffer, b)
        return y

    @staticmethod
    def backward(ctx, *grad_outputs):
        '''
        backprop with meprop
        if k is invalid (<=0 or > output feature), no top-k selection is applied
        '''
        dy = grad_outputs[0]
        x, w, b, k, sparse = ctx.saved_tensors
        dx = dw = db = None

        if k > 0 and k < w.size(1):  # backprop with top-k selection
            _, indices = dy.abs().topk(k)
            if sparse:  # using sparse matrix multiplication
                values = dy.gather(-1, indices).view(-1)
                row_indices = torch.arange(
                    0, dy.size()[0]).long().cuda().unsqueeze_(-1).repeat(
                    1, k)
                indices = torch.stack([row_indices.view(-1), indices.view(-1)])
                pdy = torch.cuda.sparse.FloatTensor(indices, values, dy.size())
                if ctx.needs_input_grad[0]:
                    dx = torch.dsmm(pdy, w.t())
                if ctx.needs_input_grad[1]:
                    dw = torch.dsmm(pdy.t(), x).t()
            else:
                pdy = torch.cuda.FloatTensor(dy.size()).zero_().scatter_(
                    -1, indices, dy.gather(-1, indices))
                if ctx.needs_input_grad[0]:
                    dx = torch.mm(pdy, w.t())
                if ctx.needs_input_grad[1]:
                    dw = torch.mm(x.t(), pdy)
        else:  # backprop without top-k selection
            if ctx.needs_input_grad[0]:
                dx = torch.mm(dy, w.t())
            if ctx.needs_input_grad[1]:
                dw = torch.mm(x.t(), dy)

        if ctx.needs_input_grad[2]:
            db = torch.mv(dy.t(), ctx.add_buffer)

        return dx, dw, db, dx-dx, dx-dx


class Linear(nn.Module):
    '''
    A linear module (layer without activation) with meprop
    The initialization of w and b is the same with the default linear module.
    '''

    def __init__(self, in_, out_, k, unified=False):
        super(Linear, self).__init__()
        self.in_ = in_
        self.out_ = out_
        self.k = k
        self.unified = unified

        self.w = Parameter(torch.Tensor(self.in_, self.out_))
        self.b = Parameter(torch.Tensor(self.out_))
        self.k = Parameter(torch.Tensor(1))
        self.k.data.fill_(k)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_)
        self.w.data.uniform_(-stdv, stdv)
        self.b.data.uniform_(-stdv, stdv)

    def forward(self, x):
        if self.unified:
            return linearUnified(self.k).apply(x, self.w, self.b, self.k)
        else:
            return linear(self.k).apply(x, self.w, self.b, self.k, True)

    def __repr__(self):
        return '{} ({} -> {} <- {}{})'.format(self.__class__.__name__,
                                              self.in_, self.out_, 'unified'
                                              if self.unified else '', self.k)


class NetLayer(nn.Module):
    '''
    A complete network(MLP) for MNSIT classification.

    Input feature is 28*28=784
    Output feature is 10
    Hidden features are of hidden size

    Activation is ReLU
    '''

    def __init__(self, hidden, k, layer, dropout=None, unified=False):
        super(NetLayer, self).__init__()
        self.k = k
        self.layer = layer
        self.dropout = dropout
        self.unified = unified
        self.model = nn.Sequential(self._create(hidden, k, layer, dropout))

    def _create(self, hidden, k, layer, dropout=None):
        if layer == 1:
            return OrderedDict([Linear(784, 10, 0)])
        d = OrderedDict()
        for i in range(layer):
            if i == 0:
                d['linear' + str(i)] = Linear(784, hidden, k, self.unified)
                d['sigmoid' + str(i)] = nn.Sigmoid()
                if dropout:
                    d['dropout' + str(i)] = nn.Dropout(p=dropout)
            elif i == layer - 1:
                d['linear' + str(i)] = Linear(hidden, 10, 0, self.unified)
            else:
                d['linear' + str(i)] = Linear(hidden, hidden, k, self.unified)
                d['sigmoid' + str(i)] = nn.Sigmoid()
                if dropout:
                    d['dropout' + str(i)] = nn.Dropout(p=dropout)
        return d

    def forward(self, x):
        return F.log_softmax(self.model(x.view(-1, 784)))

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, type(Linear)):
                m.reset_parameters()


class TestGroup(object):
    '''
    A network and k in meporp form a test group.
    Test groups differ in minibatch size, hidden features, layer number and dropout rate.
    '''

    def __init__(self,
                 args,
                 trnset,
                 mb,
                 hidden,
                 layer,
                 dropout,
                 unified,
                 devset=None,
                 tstset=None,
                 cudatensor=False,
                 file=sys.stdout):
        self.args = args
        self.mb = mb
        self.hidden = hidden
        self.layer = layer
        self.dropout = dropout
        self.file = file
        self.trnset = trnset
        self.unified = unified

        if cudatensor:  # dataset is on GPU
            self.trainloader = torch.utils.data.DataLoader(
                trnset, batch_size=mb, num_workers=0)
            if tstset:
                self.testloader = torch.utils.data.DataLoader(
                    tstset, batch_size=mb, num_workers=0)
            else:
                self.testloader = None
        else:  # dataset is on CPU, using prefetch and pinned memory to shorten the data transfer time
            self.trainloader = torch.utils.data.DataLoader(
                trnset,
                batch_size=mb,
                shuffle=True,
                num_workers=1,
                pin_memory=True)
            if devset:
                self.devloader = torch.utils.data.DataLoader(
                    devset,
                    batch_size=mb,
                    shuffle=False,
                    num_workers=1,
                    pin_memory=True)
            else:
                self.devloader = None
            if tstset:
                self.testloader = torch.utils.data.DataLoader(
                    tstset,
                    batch_size=mb,
                    shuffle=False,
                    num_workers=1,
                    pin_memory=True)
            else:
                self.testloader = None
        self.basettime = None
        self.basebtime = None

    def reset(self):
        '''
        Reinit the trainloader at the start of each run,
        so that the traning examples is in the same random order
        '''
        torch.manual_seed(self.args.random_seed)
        self.trainloader = torch.utils.data.DataLoader(
            self.trnset,
            batch_size=self.mb,
            shuffle=True,
            num_workers=1,
            pin_memory=True)

    def _train(self, model, opt):
        '''
        Train the given model using the given optimizer
        Record the time and loss
        '''
        model.train()
        ftime = 0
        btime = 0
        utime = 0
        tloss = 0
        for bid, (data, target) in enumerate(self.trainloader):
            data, target = Variable(data).cuda(), Variable(
                target.view(-1)).cuda()
            start = torch.cuda.Event(True)
            end = torch.cuda.Event(True)

            start.record()
            opt.zero_grad()
            end.record()
            end.synchronize()
            utime += start.elapsed_time(end)

            start.record()
            output = model(data)
            loss = F.nll_loss(output, target)
            end.record()
            end.synchronize()
            ftime += start.elapsed_time(end)

            start.record()
            loss.backward()
            end.record()
            end.synchronize()
            btime += start.elapsed_time(end)

            start.record()
            opt.step()
            end.record()
            end.synchronize()
            utime += start.elapsed_time(end)

            tloss += loss.data.item()
        tloss /= len(self.trainloader)
        return tloss, ftime, btime, utime

    def _evaluate(self, model, loader, name='test'):
        '''
        Use the given model to classify the examples in the given data loader
        Record the loss and accuracy.
        '''
        model.eval()
        test_loss = 0
        correct = 0
        for data, target in loader:
            data, target = Variable(
                data, volatile=True).cuda(), Variable(target).cuda()
            output = model(data)
            test_loss += (output, target).data.item()
            pred = output.data.max(1)[
                1]  # get the index of the max log-probability
            correct += pred.eq(target.data).cpu().sum()

        test_loss = test_loss
        test_loss /= len(
            loader)  # loss function already averages over batch size
        print(
            '{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
                name, test_loss, correct,
                len(loader.dataset), 100. * correct / len(loader.dataset)),
            file=self.file,
            flush=True)
        return 100. * correct / len(loader.dataset)

    def run(self, k=None, epoch=None):
        '''
        Run a training loop.
        '''
        if k is None:
            k = self.args.k
        if epoch is None:
            epoch = self.args.n_epoch
        print(
            'mbsize: {}, hidden size: {}, layer: {}, dropout: {}, k: {}'.
                format(self.mb, self.hidden, self.layer, self.dropout, k),
            file=self.file)
        # Init the model, the optimizer and some structures for logging
        self.reset()

        model = NetLayer(self.hidden, k, self.layer, self.dropout,
                         self.unified)
        # print(model)
        model.reset_parameters()
        model.cuda()

        opt = optim.Adam(model.parameters())

        acc = 0  # best dev. acc.
        accc = 0  # test acc. at the time of best dev. acc.
        e = -1  # best dev iteration/epoch

        times = []
        losses = []
        ftime = []
        btime = []
        utime = []

        # training loop
        for t in range(epoch):
            print('{}：'.format(t), end='', file=self.file, flush=True)
            # train
            start = torch.cuda.Event(True)
            end = torch.cuda.Event(True)
            start.record()
            loss, ft, bt, ut = self._train(model, opt)
            end.record()
            end.synchronize()
            ttime = start.elapsed_time(end)

            times.append(ttime)
            losses.append(loss)
            ftime.append(ft)
            btime.append(bt)
            utime.append(ut)
            # predict
            curacc = self._evaluate(model, self.devloader, 'dev')
            if curacc > acc:
                e = t
                acc = curacc
                accc = self._evaluate(model, self.testloader, '    test')
        etime = [sum(t) for t in zip(ftime, btime, utime)]
        print(
            '${:.2f}|{:.2f} at {}'.format(acc, accc, e),
            file=self.file,
            flush=True)
        print('', file=self.file)
        torch.save(model.state_dict(), 'mnist_model/meProp_{}.pth'.format(k))

    def _stat(self, name, t, agg=mean):
        return '{:<5}:\t{:8.3f}; {}'.format(
            name, agg(t), ', '.join(['{:8.2f}'.format(x) for x in t]))


def get_args():
    # a simple use example (not unified)
    parser = ArgumentParser()
    parser.add_argument(
        '--n_epoch', type=int, default=50, help='number of training epochs')
    parser.add_argument(
        '--d_hidden', type=int, default=50, help='dimension of hidden layers')
    parser.add_argument(
        '--n_layer',
        type=int,
        default=2,
        help='number of layers, including the output layer')
    parser.add_argument(
        '--d_minibatch', type=int, default=32, help='size of minibatches')
    parser.add_argument(
        '--dropout', type=float, default=0., help='dropout rate')
    parser.add_argument(
        '--k',
        type=int,
        default=10,
        help='k in meProp (if invalid, e.g. 0, do not use meProp)')
    parser.add_argument(
        '--unified',
        dest='unified',
        action='store_true',
        help='use unified meProp')
    parser.add_argument(
        '--no-unified',
        dest='unified',
        action='store_false',
        help='do not use unified meProp')
    parser.add_argument(
        '--random_seed', type=int, default=12976, help='random seed')
    parser.set_defaults(unified=False)
    return parser.parse_args()


def get_args_unified():
    # a simple use example (unified)
    parser = ArgumentParser()
    parser.add_argument(
        '--n_epoch', type=int, default=20, help='number of training epochs')
    parser.add_argument(
        '--d_hidden', type=int, default=50, help='dimension of hidden layers')
    parser.add_argument(
        '--n_layer',
        type=int,
        default=2,
        help='number of layers, including the output layer')
    parser.add_argument(
        '--d_minibatch', type=int, default=50, help='size of minibatches')
    parser.add_argument(
        '--dropout', type=float, default=0., help='dropout rate')
    parser.add_argument(
        '--k',
        type=int,
        default=1,
        help='k in meProp (if invalid, e.g. 0, do not use meProp)')
    parser.add_argument(
        '--unified',
        dest='unified',
        action='store_true',
        help='use unified meProp')
    parser.add_argument(
        '--no-unified',
        dest='unified',
        action='store_false',
        help='do not use unified meProp')
    parser.add_argument(
        '--random_seed', type=int, default=12976, help='random seed')
    parser.set_defaults(unified=True)
    return parser.parse_args()


def main():
    args = get_args()
    trn, dev, tst = get_mnist()

    group = TestGroup(
        args,
        trn,
        args.d_minibatch,
        args.d_hidden,
        args.n_layer,
        args.dropout,
        args.unified,
        dev,
        tst,
        file=sys.stdout)

    # results may be different at each run
    # group.run(0, args.n_epoch)

    group.run(args.k, args.n_epoch)


def main_unified():
    args = get_args_unified()
    trn, dev, tst = get_mnist()

    # change the sys.stdout to a file object to write the results to the file
    group = TestGroup(
        args,
        trn,
        args.d_minibatch,
        args.d_hidden,
        args.n_layer,
        args.dropout,
        args.unified,
        dev,
        tst,
        file=sys.stdout)

    # results may be different at each run
    # group.run(0)

    group.run(args.k)


if __name__ == '__main__':
    # uncomment to run meprop
    # main()
    # run unified meprop
    main_unified()
