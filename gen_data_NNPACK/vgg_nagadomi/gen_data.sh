#!/usr/bin/env bash

python rpi_data_spatial.py --pretrained ../../SpatialPruning/CIFAR10/saved_models/vgg_nagadomi.best_origin.pth.tar
python rpi_data_winograd.py --pretrained ../../WinogradPruning/CIFAR10/saved_models/vgg_nagadomi_winograd.prune.1.pth.tar
scp -r test_para_vgg_nagadomi/ pi@141.212.111.13:/home/pi/LIBS/jiecaoyu_NNPACK/other/
