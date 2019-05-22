#!/usr/bin/env bash

python rpi_data_spatial.py --pretrained ../../SpatialPruning/CIFAR100/saved_models/conv_pool_cnn_c.best_origin.pth.tar
python rpi_data_winograd.py --pretrained ../../WinogradPruning/CIFAR100/saved_models/conv_pool_cnn_c_winograd.winograd.prune.2.pth.tar
scp -r test_para_conv_pool_cnn_c/ pi@141.212.111.13:/home/pi/LIBS/jiecaoyu_NNPACK/other/
