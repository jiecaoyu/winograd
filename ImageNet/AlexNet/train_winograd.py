#!/usr/bin/env python2
from __future__ import absolute_import, division, print_function, unicode_literals
import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
# import torchvision.transforms as transforms
# import torchvision.datasets as datasets
import model_list
import subprocess

# set the seed
torch.manual_seed(1)
torch.cuda.manual_seed(1)

import sys
import gc
cwd = os.getcwd()
sys.path.append(cwd+'/../')
import datasets as datasets
import datasets.transforms as transforms

sys.path.append(cwd+'/../')
from utils import *

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--arch', '-a', metavar='ARCH', default='alexnet',
                    help='model architecture (default: alexnet)')
parser.add_argument('--data', metavar='DATA_PATH', default='./data/',
                    help='path to imagenet data (default: ./data/)')
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 16)')
parser.add_argument('--epochs', default=120, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.90, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('--print-freq', '-p', default=100, type=int,
                    metavar='N', help='print frequency (default: 100)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained_normal', default='', type=str, metavar='PRE',
                    help='pretrained_normal model')

best_prec1 = 0


def main():
    global args, best_prec1
    args = parser.parse_args()

    # create model
    if args.arch=='alexnet':
        model = model_list.alexnet(winograd=True)
        input_size = 227
    else:
        raise Exception('Model not supported yet')

    if args.arch.startswith('alexnet'):
        model.features = torch.nn.DataParallel(model.features)
        model.cuda()
    else:
        model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    elif args.pretrained_normal:
        if os.path.isfile(args.pretrained_normal):
            print("=> loading pretrained_normal model '{}'".format(args.pretrained_normal))
            checkpoint = torch.load(args.pretrained_normal)
            load_state_normal(model, checkpoint['state_dict'])
        else:
            print("=> no pretrained_normal model found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    if not os.path.exists(args.data+'/imagenet_mean.binaryproto'):
        if os.path.exists(cwd+'/../datasets/imagenet_mean.binaryproto'):
            normalize = transforms.Normalize(
                    meanfile=cwd+'/../datasets/imagenet_mean.binaryproto')
        else:
            print("==> Data directory"+args.data+"does not exits")
            print("==> Please specify the correct data path by")
            print("==>     --data <DATA_PATH>")
            return
    else:
        normalize = transforms.Normalize(
                meanfile=args.data+'/imagenet_mean.binaryproto')


    train_dataset = datasets.ImageFolder(
        args.data,
        transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            transforms.RandomSizedCrop(input_size),
        ]),
        Train=True)

    train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(args.data, transforms.Compose([
            transforms.ToTensor(),
            normalize,
            transforms.CenterCrop(input_size),
        ]),
        Train=False),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    print(model)

    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, is_best)


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))
    gc.collect()


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5))

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg


def save_checkpoint(state, is_best, filename='saved_models/checkpoint.pth.tar'):
    torch.save(state, filename)
    subprocess.call('mkdir saved_models -p', shell=True)
    if is_best:
        shutil.copyfile(filename, 'saved_models/model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
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


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 35 epochs"""
    lr = args.lr * (0.1 ** (epoch // 35))
    print('Learning rate:', lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def load_state(model, state_dict):
    state_dict_keys = state_dict.keys()
    for key in state_dict_keys:
        if 'module' in key:
            state_dict[key.replace('module.', '')] = state_dict[key]
    state_dict_keys = state_dict.keys()
    cur_state_dict = model.state_dict()
    for key in cur_state_dict:
        if (key in state_dict_keys) or (key.replace('module.', '') in state_dict_keys):
            cur_state_dict[key].copy_(state_dict[key.replace('module.', '')])

def load_state_normal(model, state_dict):
    state_dict_keys = state_dict.keys()
    for key in state_dict_keys:
        if 'module' in key:
            state_dict[key.replace('module.', '')] = state_dict[key]
    state_dict_keys = state_dict.keys()
    cur_state_dict = model.state_dict()
    for key in cur_state_dict:
        if (key in state_dict_keys) or (key.replace('module.', '') in state_dict_keys):
            old_key = key.replace('module.', '')
            print(key)
            target_data = state_dict[old_key]
            if cur_state_dict[key].shape == state_dict[old_key].shape:
                # original weights will be kept
                cur_state_dict[key].copy_(state_dict[old_key])
            else:
                # original weights will be transfered into Winograd domain
                print('Winograd')
                kernel_size = state_dict[key].shape[3]
                if kernel_size == 5:
                    G = torch.from_numpy(G_4x4_5x5).float()
                    BT = torch.from_numpy(BT_4x4_5x5).float()
                elif kernel_size == 3:
                    G = torch.from_numpy(G_4x4_3x3).float()
                    BT = torch.from_numpy(BT_4x4_3x3).float()
                else:
                    raise Exception ('Kernel size of ' + str(kernel_size) + " is not supported.")
                weight = state_dict[key]
                weight_t = weight.view(weight.shape[0] * weight.shape[1],
                        kernel_size, kernel_size)
                if weight.is_cuda:
                    G = G.cuda()
                weight_t = torch.bmm(G.unsqueeze(0).expand(weight_t.size(0), *G.size()),
                        weight_t)
                GT = G.transpose(0, 1)
                weight_t = torch.bmm(weight_t,
                        GT.unsqueeze(0).expand(weight_t.size(0), *GT.size()))
                weight_t = weight_t.view(weight.shape[0], weight.shape[1],
                        BT.shape[0], BT.shape[1])
                cur_state_dict[key].copy_(weight_t)

if __name__ == '__main__':
    main()
