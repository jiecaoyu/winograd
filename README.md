# Spatial-Winograd Pruning

This project is for Jiecao Yu's summer intern at Facebook.

## ResNet-18

To run the spatial structured pruning on ResNet-18 model:

```bash
$ cd SpatialPruning/ImageNet/ResNet.max/
$ bash spatial_prune.uniform.sh
```

To run the Winograd direct pruning on ResNet-18 model:

```bash
$ cd ../../../WinogradPruning/ImageNet/ResNet.max/
$ bash winograd_prune.uniform.sh
```
# winograd_scalpel
