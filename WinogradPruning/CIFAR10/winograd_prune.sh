#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=4,5 python winograd.py --pretrained-normal ../../SpatialPruning/CIFAR10/saved_models/vgg_nagadomi.prune.5.pth.tar --prune --percentage 0.64 --lr 0.0000001 --stage 0 --weight-decay 0.0003 &> step0
CUDA_VISIBLE_DEVICES=6,7 python winograd.py --pretrained-normal saved_models/vgg_nagadomi_winograd.prune.0.pth.tar --prune --percentage 0.68 --lr 0.0000001 --stage 1 --weight-decay 0.0003 &> step1
CUDA_VISIBLE_DEVICES=6,7 python winograd.py --pretrained-normal saved_models/vgg_nagadomi_winograd.prune.1.pth.tar --prune --percentage 0.72 --lr 0.0000001 --stage 2 --weight-decay 0.0003 &> step2
CUDA_VISIBLE_DEVICES=6,7 python winograd.py --pretrained-normal saved_models/vgg_nagadomi_winograd.prune.2.pth.tar --prune --percentage 0.76 --lr 0.0000001 --stage 3 --weight-decay 0.0003 &> step3
CUDA_VISIBLE_DEVICES=6,7 python winograd.py --pretrained-normal saved_models/vgg_nagadomi_winograd.prune.2.pth.tar --prune --percentage 0.80 --lr 0.0000001 --stage 4 --weight-decay 0.0003 &> step4
