#!/usr/bin/env bash

python rpi_data_spatial.py --resume ../../SpatialPruning/ImageNet/ResNet.max/saved_models/model_best.pth.tar
scp -r test_para/* pi@141.212.111.13:/home/pi/LIBS/jiecaoyu_NNPACK/other/test_para
