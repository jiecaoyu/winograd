#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=4,5 python main.py &> baseline.log

CUDA_VISIBLE_DEVICES=4,5 python main.py --pretrained saved_models/vgg_nagadomi.best_origin.pth.tar --prune --winograd-structured --lr 0.0001 --epochs 150 --percentage 0.2  --stage 0 &> step0
CUDA_VISIBLE_DEVICES=4,5 python main.py --pretrained saved_models/vgg_nagadomi.prune.0.pth.tar     --prune --winograd-structured --lr 0.01   --epochs 350 --percentage 0.3  --stage 1 &> step1
CUDA_VISIBLE_DEVICES=4,5 python main.py --pretrained saved_models/vgg_nagadomi.prune.1.pth.tar     --prune --winograd-structured --lr 0.0001 --epochs 150 --percentage 0.4  --stage 2 &> step2
CUDA_VISIBLE_DEVICES=4,5 python main.py --pretrained saved_models/vgg_nagadomi.prune.2.pth.tar     --prune --winograd-structured --lr 0.01   --epochs 350 --percentage 0.5  --stage 3 &> step3
CUDA_VISIBLE_DEVICES=4,5 python main.py --pretrained saved_models/vgg_nagadomi.prune.3.pth.tar     --prune --winograd-structured --lr 0.0001 --epochs 150 --percentage 0.55 --stage 4 &> step4
