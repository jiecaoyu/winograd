#!/usr/bin/env python2
from __future__ import absolute_import, division, print_function, unicode_literals
import argparse
import shutil
import time
import subprocess

import numpy
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
sys.path.append(cwd + '/../../SpatialPruning/')
sys.path.append(cwd + '/../../SpatialPruning/ImageNet/ResNet.max/')
import utils
import self_models

# model_names = sorted(name for name in models.__dict__
#     if name.islower() and not name.startswith("__")
#     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', metavar='DIR', default='/home/jiecaoyu/work/data/imagenet',
                    help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    help='model architecture: (default: resnet18)')
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=1, type=int,
                    metavar='N', help='mini-batch size (default: 1)')
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

# pruning arguments
parser.add_argument('--prune', action='store_true', default=False,
        help='enable pruning')
parser.add_argument('--threshold-multi', type=float, default=0.0,
        help='pruning threshold-multi')
parser.add_argument('--stage', type=int, default=0,
        help='pruning stage')
parser.add_argument('--winograd-structured', action='store_true', default=False,
        help='enable winograd-driven structured pruning')
parser.add_argument('--pretrained', action='store', default=None,
        help='pretrained model')
parser.add_argument('--percentage', type=float, default=0.0,
        help='pruning percentage')
parser.add_argument('--generate-mask', action='store_true', default=False,
        help='whether save the spatial mask')
parser.add_argument('--skip-layers', action='store', default=None,
        help='the list of layers not pruned')
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

    if (not args.resume) and (not args.pretrained) and args.prune:
        raise Exception ('Pruning requires pretrained model.')

    cudnn.benchmark = True

    # Data loading code
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

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

    if args.prune:
        # skip layers when applying pruning to part of the layers
        # only useful when using threshold to prune layers
        if args.skip_layers:
            print(args.skip_layers)
            skip_layers_list = args.skip_layers.split(',')
            for index in skip_layers_list:
                utils.para.resnet18_threshold_dict[int(index)] *= 0.0
        # generate the mask for each layer
        mask = utils.mask.Mask(model, args.threshold_multi,
                [1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 13, 14, 15, 16, 18, 19],
                winograd_structured=args.winograd_structured,
                percentage=args.percentage, generate_mask=args.generate_mask)
    else:
        mask = None

    if args.evaluate:
        validate(val_loader, model, criterion, mask)
        return
    
    subprocess.call("mkdir -p test_para/", shell=True)
    generate_data(val_loader, model, criterion, mask)

    generate_model_para(val_loader, model, criterion, mask)


def generate_data(val_loader, model, criterion, mask):
    model.eval()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            import subprocess
            import numpy
            subprocess.call("rm test_para/input_img", shell=True)
            
            fp = open("test_para/input_img", "wb")
            
            single_input = input[0]
            single_input_size = numpy.array(single_input.shape).astype(numpy.int32)
            single_input_size.tofile(fp)

            data = single_input.numpy().astype(numpy.float32)
            data.tofile(fp)

            fp.close()
            
            break
    output = model(single_input.unsqueeze(0))
    return

def save_para(name, data_list):
    fp_path = "test_para/model_para/" + name
    subprocess.call("rm " + fp_path, shell=True)
    fp = open(fp_path, "wb")
    for data in data_list:
        data.cpu().numpy().astype(numpy.float32).tofile(fp)

    fp.close()
    return

def generate_layer_para(layer, start_index):
    block_count = 0
    for block in [layer[0], layer[1]]:
        conv1 = block.conv1
        bn1 = block.bn1
        conv1_weight = conv1.weight.data
        conv1_weight *= bn1.weight.data.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        conv1_weight /= bn1.running_var.data.unsqueeze(1).unsqueeze(2).unsqueeze(3)\
                .add(bn1.eps).pow(0.5)
        conv1_bias = -bn1.running_mean.data
        conv1_bias *= bn1.weight.data
        conv1_bias /= bn1.running_var.data.add(bn1.eps).pow(0.5)
        conv1_bias += bn1.bias.data
        conv1_name = "conv" + str(start_index + block_count) + "_a"

        conv2 = block.conv2
        bn2 = block.bn2
        conv2_weight = conv2.weight.data
        conv2_weight *= bn2.weight.data.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        conv2_weight /= bn2.running_var.data.unsqueeze(1).unsqueeze(2).unsqueeze(3)\
                .add(bn2.eps).pow(0.5)
        conv2_bias = -bn2.running_mean.data
        conv2_bias *= bn2.weight.data
        conv2_bias /= bn2.running_var.data.add(bn2.eps).pow(0.5)
        conv2_bias += bn2.bias.data
        conv2_name = "conv" + str(start_index + block_count) + "_b"

        save_para(conv1_name, [conv1_weight, conv1_bias])
        save_para(conv2_name, [conv2_weight, conv2_bias])
        if hasattr(block, "downsample") and (block.downsample != None):
            downsample_name = "conv" + str(start_index + block_count) + "_downsample"
            downsample_conv = block.downsample[0]
            downsample_bn = block.downsample[1]
            downsample_weight = downsample_conv.weight.data
            downsample_weight *= downsample_bn.weight.data\
                    .unsqueeze(1).unsqueeze(2).unsqueeze(3)
            downsample_weight /= downsample_bn.running_var.data\
                    .unsqueeze(1).unsqueeze(2).unsqueeze(3)\
                    .add(bn2.eps).pow(0.5)
            downsample_bias = -downsample_bn.running_mean.data
            downsample_bias *= downsample_bn.weight.data
            downsample_bias /= downsample_bn.running_var.data\
                    .add(downsample_bn.eps).pow(0.5)
            downsample_bias += downsample_bn.bias.data
            downsample_name = "conv" + str(start_index + block_count) + "_downsample"
            save_para(downsample_name, [downsample_weight, downsample_bias])
        block_count += 1
    return

def generate_model_para(val_loader, model, criterion, mask):
    subprocess.call("mkdir test_para/model_para/ -p", shell=True)
    # conv1
    conv1 = model.module.conv1
    bn1 = model.module.bn1
    conv1_weight = conv1.weight.data
    conv1_weight *= bn1.weight.data.unsqueeze(1).unsqueeze(2).unsqueeze(3)
    conv1_weight /= bn1.running_var.data.unsqueeze(1).unsqueeze(2).unsqueeze(3)\
            .add(bn1.eps).pow(0.5)

    conv1_bias = -bn1.running_mean.data
    conv1_bias *= bn1.weight.data
    conv1_bias /= bn1.running_var.data.add(bn1.eps).pow(0.5)
    conv1_bias += bn1.bias.data
    save_para("conv1", [conv1_weight, conv1_bias])

    generate_layer_para(model.module.layer1, 0)
    generate_layer_para(model.module.layer2, 2)
    generate_layer_para(model.module.layer3, 4)
    generate_layer_para(model.module.layer4, 6)

    # fc
    fc_weight = model.module.fc.weight.data
    fc_bias = model.module.fc.bias.data
    save_para("fc", [fc_weight, fc_bias])
    return

def validate(val_loader, model, criterion, mask):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    if args.prune:
        mask.apply()
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


def save_checkpoint(state, is_best):
    if args.prune:
        filename='saved_models/checkpoint.prune.' + str(args.stage) + '.pth.tar'
    else:
        filename='saved_models/checkpoint.pth.tar'
    subprocess.call('mkdir saved_models -p', shell=True)
    torch.save(state, filename)
    if is_best and (not args.prune):
        shutil.copyfile(filename, 'saved_models/model_best.pth.tar')
    return


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
    lr = args.lr * (0.1 ** (epoch // 30))
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

if __name__ == '__main__':
    main()
