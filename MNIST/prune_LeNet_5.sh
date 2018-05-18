#!/bin/bash
# train the original model -- 99.43%
CUDA_VISIBLE_DEVICES=4,5,6,7 python train_winograd.py --weight-decay 0.0001 --seed 100

# 51.85% sparsity in conv2
CUDA_VISIBLE_DEVICES=4,5,6,7 python train_winograd.py --weight-decay 0.0001 --seed 100 --pretrained_winograd saved_models/LeNet_5.winograd.best_origin.pth.tar --prune --lr 0.01 --epochs 40 --lr-epochs 30 --stage 0 --threshold 1.0
