import argparse
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import time
import os
import wandb
from utils import *
from models import *
from optim_adahessian import Adahessian

class AverageMeter(object):
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix="", postfix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        self.postfix = postfix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        entries += self.postfix
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def train(train_loader,model,criterion,optimizer,epoch,device):
    batch_time = AverageMeter('Time', ':.4f')
    train_loss = AverageMeter('Loss', ':.6f')
    train_acc = AverageMeter('Accuracy', ':.6f')
    progress = ProgressMeter(
        len(train_loader),
        [train_loss, train_acc, batch_time],
        prefix="Epoch: [{}]".format(epoch))
    model.train()
    t = time.perf_counter()
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = target.to(device)
        output = model(data)
        loss = criterion(output, target)
        train_loss.update(loss.item(), data.size(0))
        pred = output.data.max(1)[1]
        acc = 100. * pred.eq(target.data).cpu().sum() / target.size(0)
        train_acc.update(acc, data.size(0))
        optimizer.zero_grad()
        loss.backward(create_graph=True)
        optimizer.step()
        if batch_idx % 20 == 0:
            batch_time.update(time.perf_counter() - t)
            t = time.perf_counter()
            progress.display(batch_idx)
    return train_loss.avg, train_acc.avg

def validate(val_loader,model,criterion,device):
    val_loss = AverageMeter('Loss', ':.6f')
    val_acc = AverageMeter('Accuracy', ':.1f')
    progress = ProgressMeter(
        len(val_loader),
        [val_loss, val_acc],
        prefix='\nValidation: ',
        postfix='\n')
    model.eval()
    for data, target in val_loader:
        data = data.to(device)
        target = target.to(device)
        output = model(data)
        loss = criterion(output, target)
        val_loss.update(loss.item(), data.size(0))
        pred = output.data.max(1)[1]
        acc = 100. * pred.eq(target.data).cpu().sum() / target.size(0)
        val_acc.update(acc, data.size(0))
    progress.display(len(val_loader))
    return val_loss.avg, val_acc.avg

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Example')
    parser.add_argument('--bs', '--batch-size', type=int, default=256, metavar='B',
                        help='input batch size for training (default: 256)')
    parser.add_argument('--epochs', type=int, default=140, metavar='E',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.15, metavar='LR',
                        help='learning rate (default: 0.15)')
    #parser.add_argument('--lr-decay', type=float, default=0.1,
    #                    help='learning rate ratio')
    #parser.add_argument('--lr-decay-epoch', type=int, nargs='+', default=[80, 120],
    #                    help='decrease learning rate at these epochs.')
    parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                        metavar='W', help='weight decay (default: 5e-4)')
    parser.add_argument('--optimizer', type=str, default='adahessian',
                        help='choose optim')
    args = parser.parse_args()

    device = torch.device('cuda',0)

    wandb.init()
    wandb.config.update(args)

    # get dataset
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_dataset = datasets.CIFAR10('./data',
                                     train=True,
                                     download=True,
                                     transform=transform_train)
    val_dataset = datasets.CIFAR10('./data',
                                   train=False,
                                   transform=transform_val)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=args.bs)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                             batch_size=args.bs,
                                             shuffle=False)

    # make sure to use cudnn.benchmark for second backprop
    #cudnn.benchmark = True

    # get model and optimizer
    model = resnet(num_classes=10, depth=20).to(device)
    wandb.config.update({"model": model.__class__.__name__, "dataset": "CIFAR10"})
    model = torch.nn.DataParallel(model)

    criterion = nn.CrossEntropyLoss()
    args.weight_decay = args.weight_decay / args.lr
    optimizer = Adahessian(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay)

    # learning rate schedule
    #scheduler = torch.lr_scheduler.MultiStepLR(
    #    optimizer,
    #    args.lr_decay_epoch,
    #    gamma=args.lr_decay,
    #    last_epoch=-1)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train(train_loader,model,criterion,optimizer,epoch,device)
        val_loss, val_acc = validate(val_loader,model,criterion,device)
        # scheduler.step()
        wandb.log({
            'train_loss': train_loss,
            'val_acc': val_acc*0.01
            })

if __name__ == '__main__':
    main()
