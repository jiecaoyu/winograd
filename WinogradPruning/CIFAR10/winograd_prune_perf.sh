#!/usr/bin/env bash
PRUNING_OPT="--target 1,2,3,5,6,7 --thresholds 0.012,0.011,0.008,0.006,0.007,0.016"

python winograd.py --pretrained-normal ../../SpatialPruning/CIFAR10/saved_models/vgg_nagadomi.prune.7.pth.tar --prune --lr 0.000001 --stage 0 ${PRUNING_OPT} --threshold-multi 0.15 &> step0
python winograd.py --pretrained saved_models/vgg_nagadomi_winograd.prune.0.pth.tar --prune --lr 0.000001 --stage 1 ${PRUNING_OPT} --threshold-multi 0.20 &> step1
