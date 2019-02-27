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
import numpy
# import torchvision.models as models
from torch.autograd import Variable

# import utils
import os
import sys
cwd = os.getcwd()
relative_path = '/../../SpatialPruning/CIFAR10/'
sys.path.append(cwd + relative_path)
import self_models

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

def write_data(path, data, data_type):
    fp = open(path, 'a+b')
    if isinstance(data, torch.Size):
        data = numpy.array(data).astype(data_type)
        data.tofile(fp)
    elif isinstance(data, torch.Tensor):
        data = data.cpu().numpy().astype(data_type)
        if len(data.shape) == 2 and data.shape[1] == 4096:
            # weird bug
            data[0:512].tofile(fp)
            data[512:1024].tofile(fp)
        else:
            data.tofile(fp)
    else:
        raise Exception('Unsupported data type: {}'.format(type(data)))
    fp.close()
    return

def test():
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

    test_loss /= len(testloader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        test_loss * args.batch_size, correct, len(testloader.dataset),
        100. * float(correct) / len(testloader.dataset)))
    print('Best Accuracy: {:.2f}%\n'.format(best_acc))

def generate_data(target_dir):
    model.eval()
    for data, target in testloader:
        data, target = Variable(data.cuda()), Variable(target.cuda())
        break

    single_input = data[0]
    write_data(target_dir + '/input_img', single_input.shape, numpy.int32)
    write_data(target_dir + '/input_img', single_input, numpy.float32)
    output = model(single_input.unsqueeze(0))
    return

def generate_model_para(model, target_dir):
    index_list = [0, 1, 2, 3, 4, 5, 6, 7]
    for index in index_list:
        if index < 2:
            conv = model.feature.module[index * 3]
            bn = model.feature.module[index * 3 + 1]
        elif index < 4:
            conv = model.feature.module[index * 3 + 1]
            bn = model.feature.module[index * 3 + 2]
        else:
            conv = model.feature.module[index * 3 + 2]
            bn = model.feature.module[index * 3 + 3]
        conv_weight = conv.weight.data
        conv_weight *= bn.weight.data.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        conv_weight /= bn.running_var.data.unsqueeze(1).unsqueeze(2).unsqueeze(3)\
                .add(bn.eps).pow(0.5)
        conv_bias = -bn.running_mean.data
        if hasattr(conv, 'bias'):
            conv_bias += conv.bias.data
        conv_bias *= bn.weight.data
        conv_bias /= bn.running_var.data.add(bn.eps).pow(0.5)
        conv_bias += bn.bias.data

        write_data(target_dir + 'model_para/conv' + str(index), conv_weight, numpy.float32)
        write_data(target_dir + 'model_para/conv' + str(index), conv_bias, numpy.float32)

    fc_list = ['fc0', 'fc1', 'fc2']
    for fc_name in fc_list:
        fc = getattr(model, fc_name)
        write_data(target_dir + 'model_para/' + fc_name, fc.weight.data, numpy.float32)
        write_data(target_dir + 'model_para/' + fc_name, fc.bias.data, numpy.float32)
    return

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Example')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
            help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=700, metavar='N',
            help='number of epochs to train (default: 700)')
    parser.add_argument('--lr-epochs', type=int, default=0, metavar='N',
            help='number of epochs to decay the lr (default: 0)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
            help='learning rate (default: 0.1)')
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
    parser.add_argument('--arch', action='store', default='vgg_nagadomi',
            help='the MNIST network structure: vgg_nagadomi')
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

    test_dataset = torchvision.datasets.CIFAR10(
            root= './' + relative_path + '/data',
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


    if args.arch == 'vgg_nagadomi':
        model = self_models.vgg_nagadomi()
    else:
        print('ERROR: specified arch is not suppported')
        exit()

    if args.pretrained:
        pretrained_model = torch.load(args.pretrained)
        best_acc = pretrained_model['acc']
        load_state(model, pretrained_model['state_dict'])
    else:
        raise Exception ('Requires pretrained model.')

    if args.cuda:
        model.cuda()
        model.feature =\
                torch.nn.DataParallel(model.feature, device_ids=range(torch.cuda.device_count()))

    criterion = nn.CrossEntropyLoss()
    print(model)

    if args.evaluate:
        test()
        exit()

    target_dir = 'test_para_vgg_nagadomi/dense_spatial/'
    subprocess.call('rm -rf ' + target_dir + '&& mkdir -p ' + target_dir + '/model_para',
            shell=True)
    generate_data(target_dir)
    generate_model_para(model, target_dir)
