#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --pretrained saved_models/model_best.pth.tar         --prune --winograd-structured --lr 0.001 --epochs 30 --percentage 0.2  --stage 0 &> step0

CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --pretrained saved_models/checkpoint.prune.0.pth.tar --prune --winograd-structured --lr 0.001 --epochs 30 --percentage 0.3  --stage 1 &> step1

CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --pretrained saved_models/checkpoint.prune.1.pth.tar --prune --winograd-structured --lr 0.001 --epochs 30 --percentage 0.4  --stage 2 &> step2

CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --pretrained saved_models/checkpoint.prune.2.pth.tar --prune --winograd-structured --lr 0.01  --epochs 60 --percentage 0.5  --stage 3 &> step3

CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --pretrained saved_models/checkpoint.prune.3.pth.tar --prune --winograd-structured --lr 0.01  --epochs 60 --percentage 0.6  --stage 4 &> step4

CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --pretrained saved_models/checkpoint.prune.4.pth.tar --prune --winograd-structured --lr 0.01  --epochs 60 --percentage 0.65 --stage 5 &> step5

CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --pretrained saved_models/checkpoint.prune.5.pth.tar --prune --winograd-structured --lr 0.01  --epochs 60 --percentage 0.7  --stage 6 &> step6
