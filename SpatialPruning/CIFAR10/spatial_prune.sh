#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=4,5 python main.py &> baseline.log

CUDA_VISIBLE_DEVICES=4,5 python main.py --pretrained saved_models/vgg_nagadomi.best_origin.pth.tar --prune --winograd-structured --lr 0.00005 --epochs 150 --percentage 0.2 --stage 0 --wd-power 0.3 &> step0
CUDA_VISIBLE_DEVICES=4,5 python main.py --pretrained saved_models/vgg_nagadomi.prune.0.pth.tar   --prune --winograd-structured --lr 0.00005 --epochs 150 --percentage 0.3 --stage 1 --wd-power 0.3 &> step1
CUDA_VISIBLE_DEVICES=4,5 python main.py --pretrained saved_models/vgg_nagadomi.prune.1.pth.tar   --prune --winograd-structured --lr 0.005   --epochs 350 --percentage 0.4 --stage 2 --wd-power 0.5 &> step2
CUDA_VISIBLE_DEVICES=4,5 python main.py --pretrained saved_models/vgg_nagadomi.prune.2.pth.tar   --prune --winograd-structured --lr 0.0005  --epochs 250 --percentage 0.5 --stage 3 --wd-power 0.3 &> step3
