#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=5,6,7 python winograd.py --pretrained-normal ../../SpatialPruning/CIFAR100/saved_models/conv_pool_cnn_c.prune.1.pth.tar --prune --percentage 0.40 --lr 0.0000001 --epochs  50 --stage 0 &> step1.step0.40

CUDA_VISIBLE_DEVICES=5,6,7 python winograd.py --pretrained-normal saved_models/conv_pool_cnn_c_winograd.winograd.prune.0.pth.tar             --prune --percentage 0.45 --lr 0.000001  --epochs  50 --stage 1 &> step1.step1.45

CUDA_VISIBLE_DEVICES=5,6,7 python winograd.py --pretrained-normal saved_models/conv_pool_cnn_c_winograd.winograd.prune.1.pth.tar             --prune --percentage 0.50 --lr 0.000001  --epochs  50 --stage 2 &> step1.step2.50

CUDA_VISIBLE_DEVICES=5,6,7 python winograd.py --pretrained-normal saved_models/conv_pool_cnn_c_winograd.winograd.prune.2.pth.tar             --prune --percentage 0.55 --lr 0.000001  --epochs 100 --stage 3 &> step1.step3.55

CUDA_VISIBLE_DEVICES=5,6,7 python winograd.py --pretrained-normal saved_models/conv_pool_cnn_c_winograd.winograd.prune.3.pth.tar             --prune --percentage 0.60 --lr 0.000001  --epochs 100 --stage 4 &> step1.step4.60
