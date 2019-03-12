#!/usr/bin/env bash

python winograd.py --pretrained-normal ../../SpatialPruning/CIFAR100/saved_models/conv_pool_cnn_c.prune.9.pth.tar --prune --percentage 0.87 --epochs 150 --stage 0 --lr 1e-6 &> step0
python winograd.py --pretrained saved_models/conv_pool_cnn_c_winograd.winograd.prune.0.pth.tar --prune --percentage 0.89 --epochs 150 --stage 1 --lr 1e-6 &> step1
python winograd.py --pretrained saved_models/conv_pool_cnn_c_winograd.winograd.prune.1.pth.tar --prune --percentage 0.91 --epochs 150 --stage 2 --lr 1e-6 &> step2
