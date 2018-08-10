#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=4,5 python main.py &> baseline.log

CUDA_VISIBLE_DEVICES=4,5 python main.py --pretrained saved_models/vgg_nagadomi.best_origin.pth.tar --prune --winograd-structured --lr 0.0001 --epochs 150 --percentage 0.2  --stage 0 &> step0
CUDA_VISIBLE_DEVICES=4,5 python main.py --pretrained saved_models/vgg_nagadomi.prune.0.pth.tar     --prune --winograd-structured --lr 0.01   --epochs 350 --percentage 0.3  --stage 1 &> step1
CUDA_VISIBLE_DEVICES=4,5 python main.py --pretrained saved_models/vgg_nagadomi.prune.1.pth.tar     --prune --winograd-structured --lr 0.0001 --epochs 150 --percentage 0.4  --stage 2 &> step2
CUDA_VISIBLE_DEVICES=4,5 python main.py --pretrained saved_models/vgg_nagadomi.prune.2.pth.tar     --prune --winograd-structured --lr 0.01   --epochs 350 --percentage 0.5  --stage 3 &> step3
CUDA_VISIBLE_DEVICES=6,7 python main.py --pretrained saved_models/vgg_nagadomi.prune.3.pth.tar     --prune --winograd-structured --lr 0.001  --epochs 250 --percentage 0.55 --stage 4 &> step4

CUDA_VISIBLE_DEVICES=6,7 python main.py --pretrained saved_models/vgg_nagadomi.prune.4.pth.tar     --prune --winograd-structured --lr 0.001  --epochs 250 --percentage 0.60 --stage 5 &> step5
CUDA_VISIBLE_DEVICES=6,7 python main.py --pretrained saved_models/vgg_nagadomi.prune.5.pth.tar     --prune --winograd-structured --lr 0.001  --epochs 250 --percentage 0.65 --stage 6 &> step6
CUDA_VISIBLE_DEVICES=6,7 python main.py --pretrained saved_models/vgg_nagadomi.prune.6.pth.tar     --prune --winograd-structured --lr 0.001  --epochs 250 --percentage 0.70 --stage 7 &> step7
CUDA_VISIBLE_DEVICES=6,7 python main.py --pretrained saved_models/vgg_nagadomi.prune.7.pth.tar     --prune --winograd-structured --lr 0.001  --epochs 250 --percentage 0.75 --stage 8 &> step8
CUDA_VISIBLE_DEVICES=6,7 python main.py --pretrained saved_models/vgg_nagadomi.prune.8.pth.tar     --prune --winograd-structured --lr 0.001  --epochs 250 --percentage 0.80 --stage 9 &> step9
