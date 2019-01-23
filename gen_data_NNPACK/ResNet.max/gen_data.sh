#!/usr/bin/env bash

python rpi_data_spatial.py --resume ../../SpatialPruning/ImageNet/ResNet.max/saved_models/model_best.pth.tar
python rpi_data_winograd.py --pretrained ../../WinogradPruning/ImageNet/ResNet.max/saved_models/checkpoint.winograd.prune.1.pth.tar --prune --percentage 0.74
scp -r test_para/* pi@141.212.111.13:/home/pi/LIBS/jiecaoyu_NNPACK/other/test_para
