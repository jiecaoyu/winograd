#!/usr/bin/env python2
from __future__ import absolute_import, division, print_function, unicode_literals
import argparse
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
# import data
import torchvision
# import torchvision.models as models
from torch.autograd import Variable

# import utils
import os
import sys
cwd = os.getcwd()
sys.path.append(cwd + '/../')
import utils
import self_models
import newLayers

def save_state(model, acc):
    print('==> Saving model ...')
    state = {
        'acc': acc,
        'state_dict': model.state_dict(),
        }
    for key in list(state['state_dict'].keys()):
        if 'module' in key:
            state['state_dict'][key.replace('module.', '')] = \
                    state['state_dict'].pop(key)
    subprocess.call('mkdir saved_models/ -p', shell=True)
    if args.prune:
        torch.save(state, 'saved_models/'+args.arch+'.prune.'+str(args.stage)+'.pth.tar')
    else:
        torch.save(state, 'saved_models/'+args.arch+'.best_origin.pth.tar')
    return

def load_state(model, state_dict):
    state_dict_keys = state_dict.keys()
    cur_state_dict = model.state_dict()
    for key in cur_state_dict:
        if key in state_dict_keys:
            cur_state_dict[key].copy_(state_dict[key])
        elif key.replace('module.','') in state_dict_keys:
            cur_state_dict[key].copy_(state_dict[key.replace('module.','')])
        elif 'module.'+key in state_dict_keys:
            cur_state_dict[key].copy_(state_dict['module.'+key])
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

def train(epoch):
    model.train()
    loss_avg = 0.0
    for batch_idx, (data, target) in enumerate(trainloader):
        data, target = Variable(data.cuda()), Variable(target.cuda())
        if args.prune:
            mask.apply()
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss_avg += loss
        loss.backward()
        grad_optimizer.step()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(trainloader.dataset),
                len(data) * batch_idx / len(trainloader), loss.data.item()))
            sys.stdout.flush()
    loss_avg /= len(trainloader.dataset)
    print('Avg train loss: {:.4f}'.format(loss_avg * args.batch_size))

def test(evaluate=False):
    global best_acc
    model.eval()
    test_loss = 0
    correct = 0
    if args.prune:
        mask.apply()
    for data, target in testloader:
        data, target = Variable(data.cuda()), Variable(target.cuda())
        output = model(data)
        test_loss += criterion(output, target).data.item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    acc = 100. * float(correct) / len(testloader.dataset)
    if (acc > best_acc) or (args.prune):
        best_acc = acc
        if not evaluate:
            save_state(model, best_acc)
            
    test_loss /= len(testloader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        test_loss * args.batch_size, correct, len(testloader.dataset),
        100. * float(correct) / len(testloader.dataset)))
    print('Best Accuracy: {:.2f}%\n'.format(best_acc))

def adjust_learning_rate(optimizer, epoch):
    if args.prune:
        S = [200, 350]
    else:
        S = [200, 350, 500, 600]
    if epoch in S:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1
    return optimizer.param_groups[0]['lr']

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Example')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
            help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=350, metavar='N',
            help='number of epochs to train (default: 350)')
    parser.add_argument('--lr-epochs', type=int, default=0, metavar='N',
            help='number of epochs to decay the lr (default: 0)')
    parser.add_argument('--lr', type=float, default=0.05, metavar='LR',
            help='learning rate (default: 0.05)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
            help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', '--wd', default=3e-4, type=float,
            metavar='W', help='weight decay (default: 3e-4)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
            help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
            help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
            help='how many batches to wait before logging training status')
    parser.add_argument('--arch', action='store', default='vgg_nagadomi_winograd',
            help='the MNIST network structure: vgg_nagadomi_winograd')
    parser.add_argument('--pretrained', action='store', default=None,
            help='pretrained model')
    parser.add_argument('--pretrained-normal', action='store', default=None,
            help='pretrained normal model')
    parser.add_argument('--evaluate', action='store_true', default=False,
            help='whether to run evaluation')

    # pruning arguments
    parser.add_argument('--prune', action='store_true', default=False,
            help='enable pruning')
    parser.add_argument('--stage', type=int, default=0,
            help='pruning stage')
    parser.add_argument('--winograd-structured', action='store_true', default=False,
            help='enable winograd-driven structured pruning')
    parser.add_argument('--percentage', type=float, default=0.0,
            help='pruning percentage')
    parser.add_argument('--wd-power', type=float, default=0.1,
            help='weight_decay power')
    parser.add_argument('--wd-count', type=int, default=3,
            help='weight_decay count')
    parser.add_argument('--target', action='store', default=None,
            help='pruning target')
    parser.add_argument('--thresholds', action='store', default='',
            help='thresholds for targeting layers')
    parser.add_argument('--threshold-multi', type=float, default=0.0,
            help='pruning threshold-multi')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    print(args)

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    normalize = transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])
    train_dataset = torchvision.datasets.CIFAR10(
            root='./data',
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.RandomCrop(32, padding=5),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
                ]))

    trainloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=16)

    test_dataset = torchvision.datasets.CIFAR10(
            root='./data',
            train=False,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
                ]))

    testloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=100,
            shuffle=False,
            num_workers=2)

    if args.arch == 'vgg_nagadomi_winograd':
        model = self_models.vgg_nagadomi_winograd()
    else:
        print('ERROR: specified arch is not suppported')
        exit()
    
    if (not args.pretrained) and (not args.pretrained_normal) and args.prune:
        raise Exception ('Pruning requires pretrained model.')

    # initialization
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.05)

    if args.pretrained:
        pretrained_model = torch.load(args.pretrained)
        best_acc = pretrained_model['acc']
        load_state(model, pretrained_model['state_dict'])
    elif args.pretrained_normal:
        pretrained_model = torch.load(args.pretrained_normal)
        best_acc = pretrained_model['acc']
        load_state_normal(model, pretrained_model['state_dict'])
    else:
        best_acc = 0.0

    if args.cuda:
        model.cuda()
        model.feature = torch.nn.DataParallel(model.feature, device_ids=range(torch.cuda.device_count()))

    param_dict = dict(model.named_parameters())
    params = []
    for key, value in param_dict.items():
        if 'feature' in key:
            print(key)
            params += [{'params':[value], 'lr': args.lr, 'momentum': args.momentum,
                'weight_decay': args.weight_decay}]
        else:
            params += [{'params':[value], 'lr': args.lr, 'momentum': args.momentum,
                'weight_decay': args.weight_decay}]

    optimizer = torch.optim.SGD(params, lr=0.0, weight_decay=args.weight_decay,
            momentum=args.momentum)

    criterion = nn.CrossEntropyLoss()
    print(model)
    
    if args.prune:
        # adjust dropout
        # count = 0
        # for m in model.modules():
        #     if isinstance(m, nn.Dropout):
        #         m.p *= ((1. - args.percentage) ** args.wd_power)
        #         count += 1
        #         print(m)
        #         if count >= args.wd_count:
        #             break
        # print(model)
        if not args.target:
            mask = utils.mask.Mask(model,
                    prune_list=[1,2,3,4,5,6,7],
                    winograd_domain=True,
                    percentage=args.percentage)
            print('Insert sparsity into the first layer with fixed sparsity of 20% ...')
            mask.prune_list.insert(0, 0)
            for m in model.modules():
                if isinstance(m, newLayers.Winograd2d.Winograd2d):
                    left = 0.0
                    right = m.weight.data.abs().max()
                    tmp_percentage = -1.0
                    count_limit = 100
                    while True:
                        threshold = (left + right) / 2.0
                        tmp_weight = m.weight.data.abs()
                        tmp_mask = tmp_weight.lt(-1.0)
                        if m.kernel_size == 5:
                            threshold_tensor = torch.from_numpy(utils.para.mask_multi_4x4_5x5).float()
                        elif m.kernel_size == 3:
                            threshold_tensor = torch.from_numpy(utils.para.mask_multi_4x4_3x3).float()
                        else:
                            raise Exception ('kernel_size currently not supported')
                        if m.weight.data.is_cuda:
                            threshold_tensor = threshold_tensor.cuda()
                        threshold_tensor = threshold_tensor / threshold_tensor.min()
                        tmp_mask = m.weight.data.clone().abs()\
                                .lt(threshold_tensor.pow(-1.0).mul(threshold)).float()
                        pruned = tmp_mask.sum()
                        total = tmp_mask.nelement()
                        tmp_percentage = float(pruned) / total
                        percentage = 0.2
                        if abs(percentage - tmp_percentage) < 0.0001:
                            break
                        elif tmp_percentage > percentage:
                            right = threshold
                        else:
                            left = threshold
                        print(tmp_percentage)
                        count_limit -= 1
                        if count_limit < 0:
                            break
                    mask.mask_list[0] = tmp_mask.float()
                    break
        else:
            prune_list = [int(x) for x in args.target.split(',')]
            if args.thresholds != '':
                thresholds = [float(x) for x in args.thresholds.split(',')]
                if len(prune_list) != len(thresholds):
                    raise Exception ('prune_list and threshold_list should have the same size')
                threshold_list = {}
                for index in range(len(prune_list)):
                    threshold_list[prune_list[index]] = thresholds[index]
                mask = utils.mask.Mask(model,
                        prune_list=prune_list,
                        winograd_domain=True,
                        threshold_multi=args.threshold_multi,
                        threshold_list=threshold_list)
            else:
                mask = utils.mask.Mask(model,
                        prune_list=[1,2,3,4,5,6,7],
                        winograd_domain=True,
                        percentage=args.percentage)
        mask.print_mask_info()
    else:
        mask = None

    if args.evaluate:
        test(evaluate=True)
        exit()
    
    grad_optimizer = utils.grad_compute.GradOptimizer(model)

    for epoch in range(args.epochs):
        lr_cur = adjust_learning_rate(optimizer, epoch)
        print('Current learing rate: {}'.format(lr_cur))
        train(epoch)
        test()
