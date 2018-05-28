#!/usr/bin/env bash

# prune stage 0 -- 0.7683 -- 99.42%
CUDA_VISIBLE_DEVICES=0 python spatial.py --pretrained saved_models/LeNet_5.best_origin.pth.tar  --prune --threshold 0.06 --lr 0.001 --stage 0 --lr-epochs 25

# prune stage 1 -- 0.8098 -- 99.41%
CUDA_VISIBLE_DEVICES=0 python spatial.py --pretrained saved_models/LeNet_5.prune.0.pth.tar  --prune --threshold 0.07 --lr 0.001 --stage 1 --lr-epochs 25

# prune stage 2 -- 0.8286 -- 0.6321 -- 99.40%
CUDA_VISIBLE_DEVICES=0 python spatial.py --pretrained saved_models/LeNet_5.prune.1.pth.tar --prune --threshold 0.075 --lr 0.001 --stage 2 --lr-epochs 25
