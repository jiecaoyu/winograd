#!/usr/bin/env bash

python winograd.py --pretrained-normal ../../SpatialPruning/CIFAR100/saved_models/conv_pool_cnn_c.prune.7.pth.tar --prune --percentage 0.79 --epochs 50 --stage 0 --lr 1e-6 &> step0
