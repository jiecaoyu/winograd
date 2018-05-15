from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import sys
cwd = os.getcwd()
sys.path.append(cwd+'/../')
import models
from torchvision import datasets, transforms
from torch.autograd import Variable
from newLayers import *
from utils import *


def load_state_normal(model, state_dict):
    param_dict = dict(model.named_parameters())
    state_dict_keys = state_dict.keys()
    cur_state_dict = model.state_dict()
    for key in cur_state_dict:
        if key in state_dict_keys:
            if ('conv' in key) and ('weight' in key):
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
            else:
                cur_state_dict[key].copy_(state_dict[key])
        elif key.replace('module.','') in state_dict_keys:
            cur_state_dict[key].copy_(state_dict[key.replace('module.','')])
        elif 'module.'+key in state_dict_keys:
            cur_state_dict[key].copy_(state_dict['module.'+key])
    return

def load_state_winograd(model, state_dict):
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
    if args.prune:
        torch.save(state, 'saved_models/'+args.arch+'.winograd.prune.pth.tar')
    else:
        torch.save(state, 'saved_models/'+args.arch+'.winograd.best_origin.pth.tar')
    return

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.prune:
            mask.apply()
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        grad_optimizer.step()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data.item()))
    return

def test(evaluate=False):
    global best_acc
    model.eval()
    test_loss = 0
    correct = 0

    if args.prune:
        mask.apply()
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, requires_grad=False), Variable(target)
        output = model(data)
        test_loss += criterion(output, target).data.item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    
    acc = 100. * float(correct) / len(test_loader.dataset)
    if (acc > best_acc) or args.prune:
        best_acc = acc
        if not evaluate:
            save_state(model, best_acc)

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        test_loss * args.batch_size, correct, len(test_loader.dataset),
        100. * float(correct) / len(test_loader.dataset)))
    print('Best Accuracy: {:.2f}%\n'.format(best_acc))
    return

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 15 epochs"""
    lr = args.lr * (0.1 ** (epoch // args.lr_epochs))
    print('Learning rate:', lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

if __name__=='__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
            help='input batch size for training (default: 128)')
    parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
            help='input batch size for testing (default: 128)')
    parser.add_argument('--epochs', type=int, default=60, metavar='N',
            help='number of epochs to train (default: 60)')
    parser.add_argument('--lr-epochs', type=int, default=15, metavar='N',
            help='number of epochs to decay the lr (default: 15)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
            help='learning rate (default: 0.1)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
            help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
            metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
            help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
            help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
            help='how many batches to wait before logging training status')
    parser.add_argument('--arch', action='store', default='LeNet_5',
            help='the MNIST network structure: LeNet_5 | LeNet_5_3x3')
    parser.add_argument('--pretrained_normal', action='store', default=None,
            help='pretrained_normal model')
    parser.add_argument('--pretrained_winograd', action='store', default=None,
            help='pretrained_winograd model')
    parser.add_argument('--evaluate', action='store_true', default=False,
            help='whether to run evaluation')

    parser.add_argument('--prune', action='store_true', default=False,
            help='enable pruning')
    parser.add_argument('--threshold', type=float, default=0.0,
            help='pruning threshold')
    parser.add_argument('--stage', type=int, default=0,
            help='pruning stage')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    print(args)
    
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    
    # load data
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('data', train=True, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                    ])),
                batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
                ])),
            batch_size=args.test_batch_size, shuffle=False, **kwargs)
    
    # generate the model
    if args.arch == 'LeNet_5':
        model = models.LeNet_5_Winograd()
    elif args.arch == 'LeNet_5_3x3':
        model = models.LeNet_5_3x3_Winograd()
    else:
        print('ERROR: specified arch is not suppported')
        exit()

    if args.pretrained_normal:
        pretrained_model = torch.load(args.pretrained_normal)
        best_acc = pretrained_model['acc']
        load_state_normal(model, pretrained_model['state_dict'])
    elif args.pretrained_winograd:
        pretrained_model = torch.load(args.pretrained_winograd)
        best_acc = pretrained_model['acc']
        load_state_winograd(model, pretrained_model['state_dict'])
    else:
        if args.prune:
            raise Exception ('Pruning requires a pretrained model.')
        best_acc = 0.0

    criterion = nn.CrossEntropyLoss()
    if args.cuda:
        model.cuda()
        model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
        criterion.cuda()
    
    print(model)
    param_dict = dict(model.named_parameters())
    params = []
    
    for key, value in param_dict.items():
        params += [{'params':[value], 'lr': args.lr,
            'momentum':args.momentum,
            'weight_decay': args.weight_decay,
            'key':key}]
    
    optimizer = winogradSGD(params, lr=args.lr, momentum=args.momentum,
            weight_decay=args.weight_decay)


    grad_optimizer = GradOptimizer(model)
    if args.evaluate:
        test(evaluate=True)
        exit()

    if args.prune:
        mask = Mask(model, args.threshold)
        mask.print_info()
    for epoch in range(1, args.epochs + 1):
        adjust_learning_rate(optimizer, epoch)
        train(epoch)
        test()
