#!/usr/bin/env bash

# prune 0 -- 0.6960 -- 99.41%
CUDA_VISIBLE_DEVICES=0 python winograd.py --pretrained_normal saved_models/LeNet_5.prune.2.pth.tar  --prune --threshold 0.6 --lr 0.001 --lr-epochs 25 --stage 0

# prune 1 -- 0.7118 -- 99.40%
CUDA_VISIBLE_DEVICES=0 python winograd.py --pretrained saved_models/LeNet_5.winograd.prune0.pth.tar  --prune --threshold 0.75 --lr 0.001 --lr-epochs 45 --stage 1
