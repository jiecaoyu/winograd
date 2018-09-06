#!/usr/bin/env bash
python winograd.py --pretrained-normal vgg_nagadomi.prune.5.pth.tar                       --prune --percentage 0.63 --lr 0.000001 --stage 0 --weight-decay 0.0003 &> step0
python winograd.py --pretrained-normal saved_models/vgg_nagadomi_winograd.prune.0.pth.tar --prune --percentage 0.66 --lr 0.000001 --stage 1 --weight-decay 0.0003 &> step1
python winograd.py --pretrained-normal saved_models/vgg_nagadomi_winograd.prune.1.pth.tar --prune --percentage 0.69 --lr 0.000001 --stage 2 --weight-decay 0.0003 &> step2
python winograd.py --pretrained-normal saved_models/vgg_nagadomi_winograd.prune.2.pth.tar --prune --percentage 0.72 --lr 0.000001 --stage 3 --weight-decay 0.0003 &> step3
python winograd.py --pretrained-normal saved_models/vgg_nagadomi_winograd.prune.3.pth.tar --prune --percentage 0.75 --lr 0.000001 --stage 4 --weight-decay 0.0003 &> step4
