#!/usr/bin/env bash
python winograd.py --pretrained-normal ../../SpatialPruning/CIFAR10/saved_models/vgg_nagadomi.prune.7.pth.tar --prune --percentage  0.73 --lr 0.000001 --stage 0 --weight-decay 0.0003 &> step0
python winograd.py --pretrained-normal saved_models/vgg_nagadomi_winograd.prune.0.pth.tar --prune --percentage 0.76 --lr 0.000001 --stage 1 --weight-decay 0.0003 &> step1
python winograd.py --pretrained-normal saved_models/vgg_nagadomi_winograd.prune.1.pth.tar --prune --percentage 0.79 --lr 0.000001 --stage 2 --weight-decay 0.0003 &> step2
python winograd.py --pretrained-normal saved_models/vgg_nagadomi_winograd.prune.2.pth.tar --prune --percentage 0.82 --lr 0.000001 --stage 3 --weight-decay 0.0003 &> step3
