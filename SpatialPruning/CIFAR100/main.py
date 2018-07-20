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

def save_state(model, acc):
    print('==> Saving model ...')
    state = {
        'acc': acc,
        'state_dict': model.state_dict(),
        }
    for key in state['state_dict'].keys():
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
    param_dict = dict(model.named_parameters())
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
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(trainloader.dataset),
                len(data) * batch_idx / len(trainloader), loss.data.item()))
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
        S = [50, 100, 150]
    else:
        S = [200, 250, 300]
    if epoch in S:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1
            lr_cur = param_group['lr']
    return optimizer.param_groups[0]['lr']

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='PyTorch CIFAR100 Example')
    parser.add_argument('--batch-size', type=int, default=100, metavar='N',
            help='input batch size for training (default: 100)')
    parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
            help='input batch size for testing (default: 100)')
    parser.add_argument('--epochs', type=int, default=350, metavar='N',
            help='number of epochs to train (default: 350)')
    parser.add_argument('--lr-epochs', type=int, default=0, metavar='N',
            help='number of epochs to decay the lr (default: 0)')
    parser.add_argument('--lr', type=float, default=0.05, metavar='LR',
            help='learning rate (default: 0.05)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
            help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
            metavar='W', help='weight decay (default: 5e-4)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
            help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
            help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
            help='how many batches to wait before logging training status')
    parser.add_argument('--arch', action='store', default='conv_pool_cnn_c',
            help='the MNIST network structure: conv_pool_cnn_c')
    parser.add_argument('--pretrained', action='store', default=None,
            help='pretrained model')
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

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    print(args)

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    normalize = transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
    train_dataset = torchvision.datasets.CIFAR100(
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
    
    test_dataset = torchvision.datasets.CIFAR100(
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

    if args.arch == 'conv_pool_cnn_c':
        model = self_models.conv_pool_cnn_c()
    else:
        print('ERROR: specified arch is not suppported')
        exit()
    
    if (not args.pretrained) and args.prune:
        raise Exception ('Pruning requires pretrained model.')

    if not args.pretrained:
        best_acc = 0.0
    else:
        pretrained_model = torch.load(args.pretrained)
        best_acc = pretrained_model['acc']
        load_state(model, pretrained_model['state_dict'])

    if args.cuda:
        model.cuda()
        model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    
    optimizer = torch.optim.SGD(model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay)

    criterion = nn.CrossEntropyLoss()
    print(model)
    
    if args.prune:
        assert(args.winograd_structured), 'Please trun on --winograd-structured'
        mask = utils.mask.Mask(model,
                prune_list=[1,2,3,4,5,6],
                winograd_structured=args.winograd_structured,
                percentage=args.percentage)
        count = 0
        for m in model.modules():
            if isinstance(m, nn.Dropout):
                if count == 0:
                    m.p *= ((1. - 0.2) ** 0.75)
                else:
                    m.p *= ((1. - args.percentage) ** 0.75)
                count += 1
        print('Insert sparsity into the first layer with fixed sparsity of 20% ...')
        mask.prune_list.insert(0, 0)
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                left = 0.0
                right = m.weight.data.abs().max()
                tmp_percentage = -1.0
                while True:
                    threshold = (left + right) / 2.0
                    tmp_weight = m.weight.data.abs()
                    tmp_mask = tmp_weight.lt(-1.0)
                    if tmp_weight.shape[2] == 5:
                        S = torch.from_numpy(utils.para.S_4x4_5x5).float()
                    elif tmp_weight.shape[2] == 3:
                        S = torch.from_numpy(utils.para.S_4x4_3x3).float()
                    else:
                        raise Exception ('The kernel size is not supported.')
                    if tmp_weight.is_cuda:
                        S = S.cuda()
                    for i in range(S.shape[0]):
                        S_piece = S[i].view(tmp_weight.shape[2],
                                tmp_weight.shape[2])
                        mask_piece = S_piece.abs().gt(0.0)
                        tmp_weight_masked = tmp_weight.mul(
                                mask_piece.unsqueeze(0).unsqueeze(0).float())
                        tmp_weight_masked = tmp_weight_masked.view(
                                tmp_weight_masked.shape[0],
                                tmp_weight_masked.shape[1],
                                -1)
                        tmp_weight_masked,_ = torch.max(
                                tmp_weight_masked, dim=2)
                        tmp_weight_masked = tmp_weight_masked.lt(threshold)
                        mask_piece = tmp_weight_masked.unsqueeze(2).unsqueeze(3)\
                                .mul(mask_piece.unsqueeze(0).unsqueeze(1))
                        tmp_mask = tmp_mask | mask_piece

                    # test winograd sparsity
                    tmp_weight = m.weight.data.clone()
                    tmp_weight = tmp_weight.mul(1.0 - tmp_mask.float())
                    if tmp_weight.shape[2] == 5:
                        G = torch.from_numpy(utils.para.G_4x4_5x5).float()
                    elif tmp_weight.shape[2] == 3:
                        G = torch.from_numpy(utils.para.G_4x4_3x3).float()
                    else:
                        raise Exception ('The kernel size is not supported.')
                    tmp_weight = tmp_weight.view(-1, tmp_weight.shape[2], tmp_weight.shape[3])
                    if tmp_weight.is_cuda:
                        G = G.cuda()
                    tmp_weight_t = torch.bmm(
                            G.unsqueeze(0).expand(tmp_weight.size(0), *G.size()), tmp_weight)
                    GT = G.transpose(0, 1)
                    tmp_weight_t = torch.bmm(
                            tmp_weight_t,
                            GT.unsqueeze(0).expand(tmp_weight_t.size(0), *GT.size()))
                    pruned = tmp_weight_t.eq(0.0).sum()
                    total = tmp_weight_t.nelement()
                    del tmp_weight
                    tmp_percentage = float(pruned) / total
                    percentage = 0.2
                    if abs(percentage - tmp_percentage) < 0.0001:
                        break
                    elif tmp_percentage > percentage:
                        right = threshold
                    else:
                        left = threshold
                    print(tmp_percentage)
                mask.mask_list[0] = tmp_mask.float()
                break
        mask.print_mask_info()
        mask.print_mask_info_winograd()
    else:
        mask = None

    if args.evaluate:
        test(evaluate=True)
        exit()

    for epoch in range(args.epochs):
        lr_cur = adjust_learning_rate(optimizer, epoch)
        print('Current learing rate: {}'.format(lr_cur))
        train(epoch)
        test()