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
import torchvision
# import torchvision.models as models
from torch.autograd import Variable
import numpy

# import utils
import os
import sys
cwd = os.getcwd()
relative_path = '/../../SpatialPruning/CIFAR100/'
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

def test(evaluate=False):
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

    test_loss /= len(testloader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        test_loss * args.test_batch_size, correct, len(testloader.dataset),
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
    index_list = numpy.arange(9)
    for index in index_list:
        if index < 3:
            conv = model.classifer[index * 3 + 1]
            bn = model.classifer[index * 3 + 2]
        elif index < 6:
            conv = model.classifer[index * 3 + 3]
            bn = model.classifer[index * 3 + 4]
        else:
            conv = model.classifer[index * 3 + 5]
            bn = model.classifer[index * 3 + 6]

        if isinstance(bn, nn.ReLU):
            conv_weight = conv.weight.data
            conv_bias = conv.bias.data
        else:
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

    return

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='PyTorch CIFAR100 Example')
    parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
            help='input batch size for testing (default: 100)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
            help='disables CUDA training')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
            help='how many batches to wait before logging training status')
    parser.add_argument('--arch', action='store', default='conv_pool_cnn_c',
            help='the MNIST network structure: conv_pool_cnn_c')
    parser.add_argument('--pretrained', action='store', default=None,
            help='pretrained model')
    parser.add_argument('--evaluate', action='store_true', default=False,
            help='whether to run evaluation')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    print(args)

    normalize = transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
    
    test_dataset = torchvision.datasets.CIFAR100(
            root='../../SpatialPruning/CIFAR100/data',
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

    if not args.pretrained:
        raise Exception ('Requires pretrained model.')
    else:
        pretrained_model = torch.load(args.pretrained)
        best_acc = pretrained_model['acc']
        load_state(model, pretrained_model['state_dict'])

    if args.cuda:
        model.cuda()

    criterion = nn.CrossEntropyLoss()
    print(model)

    if args.evaluate:
        test(evaluate=True)
        exit()

    target_dir = 'test_para_conv_pool_cnn_c/dense_spatial/'
    subprocess.call('rm -rf ' + target_dir + '&& mkdir -p ' + target_dir + '/model_para',
            shell=True)
    generate_data(target_dir)
    generate_model_para(model, target_dir)
