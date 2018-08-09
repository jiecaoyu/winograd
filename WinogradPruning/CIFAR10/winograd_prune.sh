#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=4,5 python winograd.py --pretrained-normal ../../SpatialPruning/CIFAR10/saved_models/vgg_nagadomi.prune.4.pth.tar --prune --percentage 0.60 --lr 0.000001 --stage 0 --weight-decay 0.0005 &> step1
