#!/usr/bin/env python2
from __future__ import absolute_import, division, print_function, unicode_literals
import argparse
import os
import shutil
import time
import subprocess

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
# import torchvision.models as models

# import utils
import os
import sys
cwd = os.getcwd()
sys.path.append(cwd + '/../../')
import utils
import newLayers
import self_models

# model_names = sorted(name for name in models.__dict__
#     if name.islower() and not name.startswith("__")
#     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', metavar='DIR', default='/data/tmp/imagenet_raw/',
                    help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18_winograd',
                    help='model architecture: (default: resnet18_winograd)')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='gloo', type=str,
                    help='distributed backend')

# analysis arguments
parser.add_argument('--pretrained', action='store', default=None,
        help='pretrained model')
parser.add_argument('--pretrained-normal', action='store', default=None,
        help='pretrained_normal model')
parser.add_argument('--index', default=19, type=int,
        help='index of targeting layer')
best_prec1 = 0


def main():
    global args, best_prec1
    args = parser.parse_args()

    args.distributed = args.world_size > 1

    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size)

    # create model
    if args.pretrained:
        if os.path.isfile(args.pretrained):
            print("=> loading pretrained model '{}'".format(args.pretrained))
            model = self_models.__dict__[args.arch]()
            checkpoint = torch.load(args.pretrained)
            load_state(model, checkpoint['state_dict'])
        else:
            print("=> no pretrained model found at '{}'".format(args.resume))
    elif args.pretrained_normal:
        if os.path.isfile(args.pretrained_normal):
            print("=> loading pretrained_normal model '{}'".format(args.pretrained_normal))
            model = self_models.__dict__[args.arch]()
            checkpoint = torch.load(args.pretrained_normal)
            load_state_normal(model, checkpoint['state_dict'])
        else:
            print("=> no pretrained_normal model found at '{}'".format(args.resume))
    else:
        print("=> creating model '{}'".format(args.arch))
        model = self_models.__dict__[args.arch]()

    if not args.distributed:
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()
    else:
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model)

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

    if (not args.resume) and (not args.pretrained) and (not args.pretrained_normal) and args.prune:
        raise Exception ('Pruning requires pretrained model.')

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    
    print(model)

    if args.evaluate:
        validate(val_loader, model, criterion, mask=None)
        return
    
    print('-' * 80)
    prune_list = [1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 13, 14, 15, 16, 18, 19]
    assert(args.index in prune_list), 'Targeting layer must be in the prune_list'
    count = 0
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, newLayers.Winograd2d.Winograd2d):
            if count == args.index:
                weight = m.weight.data
                break
            count += 1
    print('Weighs of layer [{}] shape: {}'.format(args.index, weight.shape))
    print('-' * 80)
    pruned = weight.eq(0.0)
    pruned_count = int(pruned.sum())
    totol_count = int(pruned.nelement())
    print('Overall sparsity: {}/{} = {:.4f}'.format(pruned_count, totol_count, float(pruned_count) / totol_count))
    print('-' * 80)
    print('Sparsity distribution across 2d kernels')
    count_list = {}
    pruned_count_flatten = pruned.sum(3).sum(2).view(-1)
    for i in range(int(pruned_count_flatten.max()) + 1):
        count_list[i] = 0
    for i in range(int(pruned_count_flatten.shape[0])):
        count_list[int(pruned_count_flatten[i])] += 1
    print(count_list)
    for i in range(int(pruned_count_flatten.max()) + 1):
        print('{},{}'.format(i, count_list[i]))

    print('-' * 80)
    # sparsity heatmap
    print('Sparsity heatmap:')
    pruned_heatmap = pruned.float().sum(1).sum(0).div(pruned.shape[0] * pruned.shape[1])
    print(pruned_heatmap)
    print('-' * 80)
    print('Sparsity heatmap for kernels with at least 1 weight remaining:')
    complete_removed_percentage = float(count_list[int(weight[0][0].nelement())]) / (weight.shape[0] * weight.shape[1])
    pruned_heatmap_one_weight = pruned_heatmap.sub(complete_removed_percentage).div(1.0 - complete_removed_percentage)
    print(pruned_heatmap_one_weight)
    print('-' * 80)
    print('Sparsity heatmap for kernels with 20 weight pruned:')
    count = 0
    pruned_flatten = pruned.view(-1, pruned.shape[2], pruned.shape[3])
    count_heatmap = pruned_flatten[0].clone().float().zero_()
    for i in range(int(pruned_flatten.shape[0])):
        kernel = pruned_flatten[i]
        if int(kernel.sum()) == 20:
            count += 1
            count_heatmap = count_heatmap.add(kernel.float())
    print(count)
    count_heatmap = count_heatmap.div(count)
    print(count_heatmap)

def validate(val_loader, model, criterion, mask):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda(non_blocking=True)

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
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
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 60))
    print(lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
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
    return

def load_state_normal(model, state_dict):
    state_dict_keys = state_dict.keys()
    for key in state_dict_keys:
        if 'module' in key:
            state_dict[key.replace('module.', '')] = state_dict[key]
    state_dict_keys = state_dict.keys()
    cur_state_dict = model.state_dict()
    for key in cur_state_dict:
        if (key in state_dict_keys) or (key.replace('module.', '') in state_dict_keys):
            loaded_weight = state_dict[key.replace('module.', '')]
            if cur_state_dict[key].shape != loaded_weight.shape:
                print(loaded_weight.shape)
                kernel_size = state_dict[key].shape[3]
                if kernel_size == 5:
                    G = torch.from_numpy(utils.para.G_4x4_5x5).float()
                    BT = torch.from_numpy(utils.para.BT_4x4_5x5).float()
                elif kernel_size == 3:
                    G = torch.from_numpy(utils.para.G_4x4_3x3).float()
                    BT = torch.from_numpy(utils.para.BT_4x4_3x3).float()
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
            else:
                cur_state_dict[key].copy_(state_dict[key])
    return

if __name__ == '__main__':
    main()
