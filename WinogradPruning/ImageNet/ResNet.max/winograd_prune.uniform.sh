#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0,1,2,3 python winograd.py --pretrained-normal ../../../SpatialPruning/ImageNet/ResNet.max/max.uniform.bkp/checkpoint.prune.6.pth.tar --prune --percentage 0.72 --lr 0.000001 --epochs 60 --stage 0 &> step6.step0.72

CUDA_VISIBLE_DEVICES=0,1,2,3 python winograd.py --pretrained saved_models/checkpoint.winograd.prune.0.pth.tar --prune --percentage 0.74 --lr 0.000001 --epochs 60 --stage 1 &> step6.step1.74

CUDA_VISIBLE_DEVICES=0,1,2,3 python winograd.py --pretrained saved_models/checkpoint.winograd.prune.1.pth.tar --prune --percentage 0.76 --lr 0.000001 --epochs 60 --stage 2 &> step6.step2.76

CUDA_VISIBLE_DEVICES=0,1,2,3 python winograd.py --pretrained saved_models/checkpoint.winograd.prune.2.pth.tar --prune --percentage 0.78 --lr 0.000001 --epochs 60 --stage 3 &> step6.step3.78
