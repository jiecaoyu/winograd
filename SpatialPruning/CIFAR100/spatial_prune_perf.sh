#!/usr/bin/env bash

PRUNING_OPT="--target 1,2,4,5,6 --thresholds 0.032,0.045,0.031,0.033,0.034"

python main.py  &> log

# python main.py  --pretrained saved_models/conv_pool_cnn_c.best_origin.pth.tar --winograd-structured --prune --lr 0.0000 --epochs 150 --stage 0 --threshold-multi 0.00 ${PRUNING_OPT} &> step0
