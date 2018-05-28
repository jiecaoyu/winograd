#!/usr/bin/env bash

# prune stage 0 -- 0.5311
CUDA_VISIBLE_DEVICES=0 python spatial.py --pretrained saved_models/LeNet_5.best_origin.pth.tar  --prune --threshold 0.06 --lr 0.001 --stage 0 --lr-epochs 25 --winograd-structured

# prune stage 1 -- 0.6041
CUDA_VISIBLE_DEVICES=0 python spatial.py --pretrained saved_models/LeNet_5.prune.0.pth.tar  --prune --threshold 0.07 --lr 0.001 --stage 1 --lr-epochs 25 --winograd-structured

# prune stage 2 -- 0.6320
CUDA_VISIBLE_DEVICES=0 python spatial.py --pretrained saved_models/LeNet_5.prune.1.pth.tar  --prune --threshold 0.075 --lr 0.001 --stage 2 --lr-epochs 25 --winograd-structured
