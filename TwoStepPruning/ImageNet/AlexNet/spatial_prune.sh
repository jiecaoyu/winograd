#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0,1,2,3 python spatial.py --pretrained ../../../ImageNet/AlexNet/saved_models/model_best.pth.tar --prune --threshold 0.013 --stage 0 --winograd-structured --lr-epochs 15 --epochs 40 --lr 0.001 &> step0

CUDA_VISIBLE_DEVICES=0,1,2,3 python spatial.py --pretrained saved_models/checkpoint.prune.0.pth.tar --prune --threshold 0.016 --stage 1 --winograd-structured --lr-epochs 15 --epochs 40 --lr 0.001  &> step1

CUDA_VISIBLE_DEVICES=0,1,2,3 python spatial.py --pretrained saved_models/checkpoint.prune.1.pth.tar --prune --threshold 0.020 --stage 2 --winograd-structured --lr-epochs 15 --epochs 40 --lr 0.001 &> step2

CUDA_VISIBLE_DEVICES=0,1,2,3 python spatial.py --pretrained saved_models/checkpoint.prune.2.pth.tar --prune --threshold 0.022 --stage 3 --winograd-structured --lr-epochs 15 --epochs 40 --lr 0.001 &> step3
