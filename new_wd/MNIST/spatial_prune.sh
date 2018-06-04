#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0 python spatial.py --pretrained saved_models/LeNet_5.best_origin.pth.tar  --prune --threshold 0.17 --lr 0.01 --stage 0 --lr-epochs 25 --winograd-structured
