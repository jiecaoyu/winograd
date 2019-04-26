#!/usr/bin/env bash
python main.py &> baseline.log

python main.py --pretrained saved_models/vgg_nagadomi.best_origin.pth.tar --prune --winograd-structured --lr 0.0001 --epochs 150 --percentage 0.2  --stage 0 &> step0
python main.py --pretrained saved_models/vgg_nagadomi.prune.0.pth.tar     --prune --winograd-structured --lr 0.01   --epochs 350 --percentage 0.3  --stage 1 &> step1
python main.py --pretrained saved_models/vgg_nagadomi.prune.1.pth.tar     --prune --winograd-structured --lr 0.01   --epochs 350 --percentage 0.4  --stage 2 &> step2
python main.py --pretrained saved_models/vgg_nagadomi.prune.2.pth.tar     --prune --winograd-structured --lr 0.01   --epochs 350 --percentage 0.5  --stage 3 &> step3
python main.py --pretrained saved_models/vgg_nagadomi.prune.3.pth.tar     --prune --winograd-structured --lr 0.01   --epochs 350 --percentage 0.55 --stage 4 &> step4

python main.py --pretrained saved_models/vgg_nagadomi.prune.4.pth.tar     --prune --winograd-structured --lr 0.01   --epochs 350 --percentage 0.60 --stage 5 &> step5
python main.py --pretrained saved_models/vgg_nagadomi.prune.5.pth.tar     --prune --winograd-structured --lr 0.01   --epochs 350 --percentage 0.65 --stage 6 &> step6
python main.py --pretrained saved_models/vgg_nagadomi.prune.6.pth.tar     --prune --winograd-structured --lr 0.01   --epochs 350 --percentage 0.70 --stage 7 &> step7
python main.py --pretrained saved_models/vgg_nagadomi.prune.7.pth.tar     --prune --winograd-structured --lr 0.01   --epochs 350 --percentage 0.75 --stage 8 &> step8
python main.py --pretrained saved_models/vgg_nagadomi.prune.8.pth.tar     --prune --winograd-structured --lr 0.01   --epochs 350 --percentage 0.80 --stage 9 &> step9
python main.py --pretrained saved_models/vgg_nagadomi.prune.9.pth.tar     --prune --winograd-structured --lr 0.01   --epochs 350 --percentage 0.85 --stage 10 &> step10
python main.py --pretrained saved_models/vgg_nagadomi.prune.10.pth.tar    --prune --winograd-structured --lr 0.01   --epochs 350 --percentage 0.90 --stage 11 &> step11
