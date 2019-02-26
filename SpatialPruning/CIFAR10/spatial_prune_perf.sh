#!/usr/bin/env bash
# python main.py &> baseline.log

# thresholds are the thresholds for each layer to cause a 10% accuracy loss
PRUNING_OPT="--target 1,2,3,5,6,7 --thresholds 0.058,0.044,0.036,0.024,0.027,0.023"

DEVICES="0,3"

CUDA_VISIBLE_DEVICES=${DEVICES} python main.py --pretrained saved_models/vgg_nagadomi.best_origin.pth.tar --prune --winograd-structured --lr 0.0001 --epochs 150 --stage 0 --threshold-multi 0.35 ${PRUNING_OPT} &> step0
CUDA_VISIBLE_DEVICES=${DEVICES} python main.py --pretrained saved_models/vgg_nagadomi.prune.0.pth.tar --prune --winograd-structured --lr 0.0001 --epochs 150 --stage 1 --threshold-multi 0.47 ${PRUNING_OPT} &> step1
CUDA_VISIBLE_DEVICES=${DEVICES} python main.py --pretrained saved_models/vgg_nagadomi.prune.1.pth.tar --prune --winograd-structured --lr 0.0100 --epochs 350 --stage 2 --threshold-multi 0.57 ${PRUNING_OPT} &> step2
CUDA_VISIBLE_DEVICES=${DEVICES} python main.py --pretrained saved_models/vgg_nagadomi.prune.2.pth.tar --prune --winograd-structured --lr 0.0010 --epochs 250 --stage 3 --threshold-multi 0.62 ${PRUNING_OPT} &> step3
CUDA_VISIBLE_DEVICES=${DEVICES} python main.py --pretrained saved_models/vgg_nagadomi.prune.3.pth.tar --prune --winograd-structured --lr 0.0100 --epochs 350 --stage 4 --threshold-multi 0.77 ${PRUNING_OPT} &> step4
CUDA_VISIBLE_DEVICES=${DEVICES} python main.py --pretrained saved_models/vgg_nagadomi.prune.4.pth.tar --prune --winograd-structured --lr 0.0010 --epochs 250 --stage 5 --threshold-multi 0.74 ${PRUNING_OPT} &> step5
CUDA_VISIBLE_DEVICES=${DEVICES} python main.py --pretrained saved_models/vgg_nagadomi.prune.5.pth.tar --prune --winograd-structured --lr 0.0100 --epochs 350 --stage 6 --threshold-multi 0.91 ${PRUNING_OPT} &> step6
CUDA_VISIBLE_DEVICES=${DEVICES} python main.py --pretrained saved_models/vgg_nagadomi.prune.6.pth.tar --prune --winograd-structured --lr 0.0100 --epochs 350 --stage 7 --threshold-multi 1.13 ${PRUNING_OPT} &> step7
