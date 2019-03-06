#!/usr/bin/env bash

PRUNING_OPT="--target 1,2,4,5,6 --thresholds 0.025,0.036,0.03,0.03,0.026"

# python main.py  &> log

python main.py  --pretrained saved_models/conv_pool_cnn_c.best_origin.pth.tar --winograd-structured --prune --lr 0.0005 --epochs 150 --stage 0 --threshold-multi 0.47 ${PRUNING_OPT} &> step0
