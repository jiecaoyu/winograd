#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0,1,2,3 python winograd.py --pretrained-normal saved_models/checkpoint.prune.4.pth.tar --prune --lr 0.01 --epochs 60 --stage 5 --percentage 0.60 --spatial-mask spatial_mask.pth.tar &> step5
