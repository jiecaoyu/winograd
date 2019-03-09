#!/usr/bin/env bash

python main.py  --pretrained saved_models/conv_pool_cnn_c.best_origin.pth.tar --winograd-structured --prune --percentage 0.20 --lr 0.00005 --epochs 050 --stage 0 &> step0

python main.py  --pretrained saved_models/conv_pool_cnn_c.prune.0.pth.tar     --winograd-structured --prune --percentage 0.30 --lr 0.0005 --epochs 100 --stage 1 &> step1

python main.py  --pretrained saved_models/conv_pool_cnn_c.prune.1.pth.tar     --winograd-structured --prune --percentage 0.40 --lr 0.005 --epochs 150 --stage 2 &> step2

python main.py  --pretrained saved_models/conv_pool_cnn_c.prune.2.pth.tar     --winograd-structured --prune --percentage 0.50 --lr 0.0005 --epochs 100 --stage 3 &> step3

python main.py  --pretrained saved_models/conv_pool_cnn_c.prune.3.pth.tar     --winograd-structured --prune --percentage 0.55 --lr 0.005 --epochs 150 --stage 4 &> step4

python main.py  --pretrained saved_models/conv_pool_cnn_c.prune.4.pth.tar     --winograd-structured --prune --percentage 0.60 --lr 0.005 --epochs 150 --stage 5 &> step5
