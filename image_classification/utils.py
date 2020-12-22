#*
# @file Different utility functions
# Copyright (c) Zhewei Yao, Amir Gholami, Sheng Shen
# All rights reserved.
# This file is part of AdaHessian library.
#
# AdaHessian is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# AdaHessian is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with adahessian.  If not, see <http://www.gnu.org/licenses/>.
#*



import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable


def getData(
        name='cifar10',
        train_bs=128,
        test_bs=1000,
        shuffle_not=True,
        train_index=None):

    if name == 'mnist':

        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=train_bs, shuffle=True)
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(
                '../data',
                train=False,
                transform=transforms.Compose(
                    [
                        transforms.ToTensor(),
                        transforms.Normalize(
                            (0.1307,
                             ),
                            (0.3081,
                             ))])),
            batch_size=test_bs,
            shuffle=False)

    if name == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        trainset = datasets.CIFAR10(
            root='../data',
            train=True,
            download=True,
            transform=transform_train)
        train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=train_bs, shuffle=True)

        testset = datasets.CIFAR10(
            root='../data',
            train=False,
            download=False,
            transform=transform_test)
        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=test_bs, shuffle=False)

    if name == 'cifar100':

        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        trainset = datasets.CIFAR100(
            root='../data',
            train=True,
            download=True,
            transform=transform_train)
        train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=train_bs, shuffle=True)

        testset = datasets.CIFAR100(
            root='../data',
            train=False,
            download=False,
            transform=transform_test)
        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=test_bs, shuffle=False)

    if name == 'svhn':
        train_loader = torch.utils.data.DataLoader(
            datasets.SVHN('../data', split='train', download=True,
                          transform=transforms.Compose([
                              transforms.ToTensor()
                          ])),
            batch_size=train_bs, shuffle=True)
        test_loader = torch.utils.data.DataLoader(
            datasets.SVHN(
                '../data',
                split='test',
                download=True,
                transform=transforms.Compose(
                    [
                        transforms.ToTensor()])),
            batch_size=test_bs,
            shuffle=False)

    return train_loader, test_loader


def validate(model, test_loader):
    # print('Testing')
    model.eval()
    correct = 0
    total_num = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            output = model(data)
            # get the index of the max log-probability
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()
            total_num += len(data)
    # print('testing_correct: ', correct / total_num, '\n')
    return correct / total_num


def get_params_grad(model):
    """
    get model parameters and corresponding gradients
    """
    params = []
    grads = []
    for param in model.parameters():
        if not param.requires_grad:
            continue
        params.append(param)
        grads.append(0. if param.grad is None else param.grad + 0.)
    return params, grads
