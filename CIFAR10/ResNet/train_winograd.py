from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
cwd = os.getcwd()
sys.path.append(cwd+'/../')
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import subprocess
sys.path.append(cwd+'/../../')
from utils import *

from torch.autograd import Variable
from models import resnet_winograd

def save_state(model, best_acc):
    print('==> Saving model ...')
    state = {
            'best_acc': best_acc,
            'state_dict': model.state_dict(),
            }
    for key in state['state_dict'].keys():
        if 'module' in key:
            state['state_dict'][key.replace('module.', '')] = \
                    state['state_dict'].pop(key)
    subprocess.call('mkdir saved_models/ -p', shell=True)
    torch.save(state, 'saved_models/'+args.arch+'.winograd.pth.tar')
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
            old_key = key.replace('module.', '')
            target_data = state_dict[old_key]
            if cur_state_dict[key].shape == state_dict[old_key].shape:
                # original weights will be kept
                cur_state_dict[key].copy_(state_dict[old_key])
            else:
                # original weights will be transfered into Winograd domain
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
    return

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(trainloader):
        # forwarding
        data, target = Variable(data.cuda()), Variable(target.cuda())
        optimizer.zero_grad()
        output = model(data)
        
        # backwarding
        loss = criterion(output, target)
        loss.backward()
        
        grad_optimizer.step()
        
        # restore weights
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tLR: {}'.format(
                epoch, batch_idx * len(data), len(trainloader.dataset),
                100. * float(batch_idx) / len(trainloader), loss.data.item(),
                optimizer.param_groups[0]['lr']))
    return

def test():
    global best_acc
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in testloader:
        data, target = Variable(data.cuda()), Variable(target.cuda())
                                    
        output = model(data)
        test_loss += criterion(output, target).data.item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    acc = 100. * float(correct) / len(testloader.dataset)

    if acc > best_acc:
        best_acc = acc
        save_state(model, best_acc)
    
    test_loss /= len(testloader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        test_loss * 128., correct, len(testloader.dataset),
        100. * float(correct) / len(testloader.dataset)))
    print('Best Accuracy: {:.2f}%\n'.format(best_acc))
    return

def adjust_learning_rate(optimizer, epoch):
    update_list = [50, 80, 100, 120]
    if epoch in update_list:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1
    return

if __name__=='__main__':
    # prepare the options
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', action='store_true',
            help='set if only CPU is available')
    parser.add_argument('--data', action='store', default='./data/',
            help='dataset path')
    parser.add_argument('--arch', action='store', default='resnet18',
            help='the architecture for the network: resnet18')
    parser.add_argument('--lr', action='store', default='0.1',
            help='the intial learning rate')
    parser.add_argument('--pretrained', action='store', default=None,
            help='the path to the pretrained model')
    parser.add_argument('--pretrained_normal', action='store', default=None,
            help='the path to the pretrained_normal model')
    parser.add_argument('--evaluate', action='store_true',
            help='evaluate the model')
    parser.add_argument('--weight-decay', action='store', type=float, default=0.0001,
            help='weight_decay value')
    args = parser.parse_args()
    print('==> Options:',args)

    # set the seed
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)

    # prepare the data
    # Data
    print('==> Preparing data ...')
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
    
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
    
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    # define classes
    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # define the model
    print('==> building model',args.arch,'...')
    if args.arch == 'resnet18':
        model = resnet_winograd.resnet18_winograd()
    else:
        raise Exception(args.arch+' is currently not supported')
    
    # initialize the model
    if args.pretrained_normal:
        print('==> Load pretrained_normal model form', args.pretrained_normal, '...')
        pretrained_model = torch.load(args.pretrained_normal)
        best_acc = pretrained_model['best_acc']
        load_state_normal(model, pretrained_model['state_dict'])
    elif args.pretrained:
        print('==> Load pretrained model form', args.pretrained, '...')
        pretrained_model = torch.load(args.pretrained)
        best_acc = pretrained_model['best_acc']
        model.load_state_dict(pretrained_model['state_dict'])
    else:
        print('==> Initializing model parameters ...')
        best_acc = 0

    if not args.cpu:
        model.cuda()
        model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    print(model)

    # define solver and criterion
    base_lr = float(args.lr)
    param_dict = dict(model.named_parameters())
    params = []
    for key, value in param_dict.items():
        params += [{'params':[value], 'lr': base_lr, 'momentum': 0.9,
            'weight_decay': args.weight_decay}]

    optimizer = optim.SGD(params, lr=0.10,weight_decay=args.weight_decay, momentum=0.9)
    
    criterion = nn.CrossEntropyLoss()
    # print(optimizer)

    # do the evaluation if specified
    if args.evaluate:
        test()
        exit(0)

    grad_optimizer = GradOptimizer(model)

    # start training
    for epoch in range(1, 130):
        adjust_learning_rate(optimizer, epoch)
        train(epoch)
        test()
