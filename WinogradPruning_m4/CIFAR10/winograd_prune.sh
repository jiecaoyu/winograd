#!/usr/bin/env bash
python winograd.py --pretrained-normal ../../SpatialPruning_m4/CIFAR10/saved_models/vgg_nagadomi.prune.8.pth.tar --prune --percentage 0.78 --lr 0.001 --stage 0  &> step0
python winograd.py --pretrained saved_models/vgg_nagadomi_winograd.prune.0.pth.tar --prune --percentage 0.81 --lr 0.001 --stage 1  &> step1
python winograd.py --pretrained saved_models/vgg_nagadomi_winograd.prune.1.pth.tar --prune --percentage 0.84 --lr 0.0001 --stage 2  &> step2
