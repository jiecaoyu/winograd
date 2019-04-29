#!/usr/bin/env bash

python winograd.py --pretrained-normal ../../SpatialPruning/CIFAR100/saved_models/conv_pool_cnn_c.prune.5.pth.tar --prune --percentage 0.63 --epochs 150 --stage 0 --lr 1e-7 &> step0
python winograd.py --pretrained saved_models/conv_pool_cnn_c_winograd.winograd.prune.0.pth.tar --prune --percentage 0.66 --epochs 150 --stage 1 --lr 1e-7 &> step1
python winograd.py --pretrained saved_models/conv_pool_cnn_c_winograd.winograd.prune.1.pth.tar --prune --percentage 0.69 --epochs 150 --stage 2 --lr 1e-6 &> step2
python winograd.py --pretrained saved_models/conv_pool_cnn_c_winograd.winograd.prune.2.pth.tar --prune --percentage 0.72 --epochs 150 --stage 3 --lr 1e-6 &> step3
python winograd.py --pretrained saved_models/conv_pool_cnn_c_winograd.winograd.prune.3.pth.tar --prune --percentage 0.75 --epochs 150 --stage 4 --lr 1e-6 &> step4
python winograd.py --pretrained saved_models/conv_pool_cnn_c_winograd.winograd.prune.4.pth.tar --prune --percentage 0.78 --epochs 150 --stage 5 --lr 1e-6 &> step5
python winograd.py --pretrained saved_models/conv_pool_cnn_c_winograd.winograd.prune.5.pth.tar --prune --percentage 0.81 --epochs 150 --stage 6 --lr 1e-6 &> step6
