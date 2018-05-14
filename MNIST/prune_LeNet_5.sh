CUDA_VISIBLE_DEVICES=6,7 python train_winograd.py --pretrained_winograd saved_models/LeNet_5.winograd.best_origin.pth.tar --prune --lr 0.001 --epochs 30 --lr-epochs 20 --stage 0 --threshold 0.0002
