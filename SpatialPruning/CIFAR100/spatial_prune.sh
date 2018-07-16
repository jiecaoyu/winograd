#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=5,6 python main.py  --pretrained saved_models/conv_pool_cnn_c.best_origin.pth.tar --winograd-structured --prune --percentage 0.20 --lr 0.0005 --epochs 100 --stage 0 &> step0

CUDA_VISIBLE_DEVICES=5,6 python main.py  --pretrained saved_models/conv_pool_cnn_c.prune.0.pth.tar     --winograd-structured --prune --percentage 0.30 --lr 0.0005 --epochs 100 --stage 1 &> step1

CUDA_VISIBLE_DEVICES=5,6 python main.py  --pretrained saved_models/conv_pool_cnn_c.prune.1.pth.tar     --winograd-structured --prune --percentage 0.40 --lr 0.0005 --epochs 100 --stage 2 &> step2

CUDA_VISIBLE_DEVICES=5,6 python main.py  --pretrained saved_models/conv_pool_cnn_c.prune.2.pth.tar     --winograd-structured --prune --percentage 0.50 --lr 0.0005 --epochs 100 --stage 3 &> step3
