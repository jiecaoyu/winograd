#!/usr/bin/env bash

python main.py  --pretrained saved_models/conv_pool_cnn_c.best_origin.pth.tar --winograd-structured --prune --percentage 0.20 --lr 0.0005 --epochs 100 --stage 0 &> step0

python main.py  --pretrained saved_models/conv_pool_cnn_c.prune.0.pth.tar     --winograd-structured --prune --percentage 0.30 --lr 0.0005 --epochs 100 --stage 1 &> step1

python main.py  --pretrained saved_models/conv_pool_cnn_c.prune.1.pth.tar     --winograd-structured --prune --percentage 0.40 --lr 0.005 --epochs 150 --stage 2 &> step2

python main.py  --pretrained saved_models/conv_pool_cnn_c.prune.2.pth.tar     --winograd-structured --prune --percentage 0.50 --lr 0.005 --epochs 150 --stage 3 &> step3

python main.py  --pretrained saved_models/conv_pool_cnn_c.prune.3.pth.tar     --winograd-structured --prune --percentage 0.60 --lr 0.005 --epochs 150 --stage 4 &> step4

python main.py  --pretrained saved_models/conv_pool_cnn_c.prune.4.pth.tar     --winograd-structured --prune --percentage 0.65 --lr 0.005 --epochs 150 --stage 5 &> step5

python main.py  --pretrained saved_models/conv_pool_cnn_c.prune.5.pth.tar     --winograd-structured --prune --percentage 0.70 --lr 0.005 --epochs 150 --stage 6 &> step6

python main.py  --pretrained saved_models/conv_pool_cnn_c.prune.6.pth.tar     --winograd-structured --prune --percentage 0.75 --lr 0.005 --epochs 150 --stage 7 &> step7

python main.py  --pretrained saved_models/conv_pool_cnn_c.prune.7.pth.tar     --winograd-structured --prune --percentage 0.80 --lr 0.005 --epochs 150 --stage 8 &> step8

python main.py  --pretrained saved_models/conv_pool_cnn_c.prune.8.pth.tar     --winograd-structured --prune --percentage 0.85 --lr 0.005 --epochs 150 --stage 9 &> step9
